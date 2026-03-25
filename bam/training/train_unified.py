"""Unified JAX training script for single-head and multihead RACE models.

Supports single-node (1-8 GPUs) and multi-node (SLURM or manual) distributed training.
Controlled by JSON config: "multihead": true/false.

Single-head mode: Epoch -> File (IPKL) -> Batch training loop.
Multihead mode: Epoch -> Step (draw from all head streams) with gradient accumulation.

Usage:
    python -m bam.training.train_unified config.json
"""

# =============================================================================
# Environment Setup (must be before JAX import)
# =============================================================================
import os
os.environ.setdefault('NCCL_TIMEOUT', '3600')
os.environ.setdefault('NCCL_DEBUG', 'WARN')
os.environ.setdefault('JAX_COORDINATION_SERVICE_CONNECT_TIMEOUT', '600')
os.environ.setdefault('JAX_COORDINATION_SERVICE_HEARTBEAT_INTERVAL', '30')
os.environ.setdefault('JAX_COORDINATION_SERVICE_SHUTDOWN_TIMEOUT', '120')

# Disable XLA CUDA command buffers to prevent OOM from accumulated CUDA graphs.
# Each unique input shape creates a new CUDA graph; with bucketed batching this
# leads to thousands of graphs consuming GPU memory.
xla_flags = os.environ.get('XLA_FLAGS', '')
if '--xla_gpu_enable_command_buffer=' not in xla_flags:
    os.environ['XLA_FLAGS'] = xla_flags + ' --xla_gpu_enable_command_buffer='

# =============================================================================
# Imports
# =============================================================================
from typing import Callable, Dict, Tuple, List, Any, Optional
from functools import partial
import re
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils, multihost_utils
try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map
from flax import nnx
import optax
import jraph
import pickle
import time
import gc
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import signal
import shutil
import sys
import traceback
import threading

from bam.data.data_nnx import Dataset, BucketedDataLoader, MultiDeviceDataLoader
from bam.data.data_multihead_nnx import DatasetWithHead
from bam.models.race_nnx import RACE
from bam.models.race_multihead_nnx import RACEMultihead, load_foundation_as_multihead
from bam.training.losses import LOSS_FUNCTIONS
from bam.training.sharding import (
    setup_mesh, replicate, replicate_pytree,
    unreplicate, unreplicate_pytree, squeeze_batch,
)

jax.config.update("jax_enable_x64", False)


# =============================================================================
# Debug Utilities
# =============================================================================

class DebugLogger:
    """Thread-safe logger for multi-host debugging."""

    def __init__(self, process_id: int, log_file: str = None):
        self.process_id = process_id
        self.lock = threading.Lock()
        self.log_file = log_file
        self._fh = None
        if log_file and process_id >= 0:
            self._fh = open(f"debug_process_{process_id}.log", 'w', buffering=1)

    def log(self, msg: str, level: str = "INFO"):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        formatted = f"[{timestamp}] [P{self.process_id}] [{level}] {msg}"
        with self.lock:
            print(formatted, flush=True)
            if self._fh:
                self._fh.write(formatted + "\n")
                self._fh.flush()

    def debug(self, msg): self.log(msg, "DEBUG")
    def info(self, msg): self.log(msg, "INFO")
    def warn(self, msg): self.log(msg, "WARN")
    def error(self, msg): self.log(msg, "ERROR")

    def close(self):
        if self._fh:
            self._fh.close()


_debug_logger = None


def get_debug_logger() -> DebugLogger:
    global _debug_logger
    if _debug_logger is None:
        try:
            pid = jax.process_index()
        except:
            pid = -1
        _debug_logger = DebugLogger(pid)
    return _debug_logger


def debug_log(msg: str, level: str = "INFO"):
    get_debug_logger().log(msg, level)


def sync_and_check(mesh, name: str = "sync_point"):
    """Synchronize all hosts and check for stragglers."""
    debug_log(f"Entering sync point: {name}")
    start_time = time.time()
    try:
        multihost_utils.sync_global_devices(name)
        elapsed = time.time() - start_time
        debug_log(f"Sync point '{name}' completed in {elapsed:.2f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        debug_log(f"Sync point '{name}' FAILED after {elapsed:.2f}s: {e}", "ERROR")
        return False


class EmergencyCheckpointer:
    """Handles emergency checkpoint saving on signals."""

    def __init__(self, save_path: str = "ckpt_emergency.pkl"):
        self.state = None
        self.save_path = save_path
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGUSR1, self._checkpoint_signal_handler)

    def _signal_handler(self, sig, frame):
        sig_name = signal.Signals(sig).name
        debug_log(f"Received signal {sig_name}, attempting emergency save...", "WARN")
        self._save_emergency_checkpoint()
        debug_log(f"Emergency checkpoint saved, exiting...")
        sys.exit(0)

    def _checkpoint_signal_handler(self, sig, frame):
        debug_log(f"Received SIGUSR1, saving manual checkpoint...", "INFO")
        self._save_emergency_checkpoint()
        debug_log(f"Manual checkpoint saved, continuing...")

    def _save_emergency_checkpoint(self):
        if self.state is not None:
            try:
                with open(self.save_path, 'wb') as f:
                    pickle.dump(self.state, f)
                debug_log(f"Emergency checkpoint saved to {self.save_path}")
            except Exception as e:
                debug_log(f"Failed to save emergency checkpoint: {e}", "ERROR")

    def update_state(self, state: dict):
        self.state = state


_emergency_checkpointer = None


def get_emergency_checkpointer(save_path: str = "ckpt_emergency.pkl") -> EmergencyCheckpointer:
    global _emergency_checkpointer
    if _emergency_checkpointer is None:
        _emergency_checkpointer = EmergencyCheckpointer(save_path=save_path)
    return _emergency_checkpointer


# =============================================================================
# Batch Logger (Multi-Host Aware)
# =============================================================================

class BatchLogger:
    """Logger for tracking batch sizes per GPU (multi-host aware)."""

    def __init__(self, process_id: int, n_processes: int,
                 n_local_devices: int = 8, log_root: str = "batch_logs"):
        self.process_id = process_id
        self.n_processes = n_processes
        self.n_local_devices = n_local_devices
        self.n_total_devices = n_processes * n_local_devices
        self.log_root = Path(log_root)

        self.node_id = os.environ.get('SLURM_NODEID',
                       os.environ.get('NODE_RANK', str(process_id)))
        self.global_rank_start = process_id * n_local_devices

        self.cumulative_graphs = {i: 0 for i in range(n_local_devices)}
        self.global_base_offset = {i: 0 for i in range(n_local_devices)}

        self.gpu_logs = {}
        self.batch_logs = {}

        for local_gpu in range(n_local_devices):
            global_rank = self.global_rank_start + local_gpu
            log_dir = self.log_root / f"rank_{global_rank}"
            log_dir.mkdir(parents=True, exist_ok=True)

            gpu_log_name = f"node{self.node_id}_gpu{local_gpu}_global{global_rank}.log"
            self.gpu_logs[local_gpu] = open(log_dir / gpu_log_name, 'w')

            batch_log_path = log_dir / "batch_sizes.log"
            self.batch_logs[local_gpu] = open(batch_log_path, 'w')
            self.batch_logs[local_gpu].write(
                f"# Batch size log for global rank {global_rank} "
                f"(node {self.node_id}, local GPU {local_gpu})\n"
            )
            self.batch_logs[local_gpu].write(
                "# timestamp,epoch,mode,head,file,batch_idx,n_graphs,n_nodes,n_edges\n"
            )
            self.batch_logs[local_gpu].flush()

            self._log_gpu_single(local_gpu,
                f"Initialized - Node: {self.node_id}, Local GPU: {local_gpu}, "
                f"Global Rank: {global_rank}")

        print(f"[Process {process_id}] Created batch logs for global ranks "
              f"{self.global_rank_start}-{self.global_rank_start + n_local_devices - 1}")

    def _log_gpu_single(self, local_gpu: int, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.gpu_logs[local_gpu].write(f"[{timestamp}] {message}\n")
        self.gpu_logs[local_gpu].flush()

    def log_gpu(self, message: str):
        for local_gpu in range(self.n_local_devices):
            self._log_gpu_single(local_gpu, message)

    def log_batch(self, epoch: int, mode: str, file_name: str, batch_idx: int,
                  per_device_metadata: List[Tuple[int, int, int]],
                  head_name: str = None,
                  graph_indices_per_device: List[List[int]] = None):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        head_str = head_name if head_name else "-"

        for local_gpu in range(self.n_local_devices):
            if local_gpu < len(per_device_metadata):
                n_graphs, n_nodes, n_edges = per_device_metadata[local_gpu]
            else:
                n_graphs, n_nodes, n_edges = 0, 0, 0

            if graph_indices_per_device and local_gpu < len(graph_indices_per_device):
                indices = graph_indices_per_device[local_gpu]
                graph_indices_str = "[" + ",".join(map(str, indices)) + "]" if indices else "[]"
            else:
                global_start = self.global_base_offset[local_gpu] + self.cumulative_graphs[local_gpu]
                global_end = global_start + n_graphs - 1
                self.cumulative_graphs[local_gpu] += n_graphs
                graph_indices_str = f"[{global_start}-{global_end}]"

            line = (f"{timestamp},{epoch},{mode},{head_str},{file_name},{batch_idx},"
                    f"{n_graphs},{n_nodes},{n_edges},{graph_indices_str}\n")
            self.batch_logs[local_gpu].write(line)
            self.batch_logs[local_gpu].flush()

    def reset_counters(self):
        for i in range(self.n_local_devices):
            self.cumulative_graphs[i] = 0

    def set_total_graphs(self, total_graphs: int):
        graphs_per_gpu = total_graphs // self.n_total_devices
        for local_gpu in range(self.n_local_devices):
            global_rank = self.global_rank_start + local_gpu
            self.global_base_offset[local_gpu] = global_rank * graphs_per_gpu

    def close(self):
        for local_gpu in range(self.n_local_devices):
            if local_gpu in self.gpu_logs and not self.gpu_logs[local_gpu].closed:
                self.gpu_logs[local_gpu].close()
            if local_gpu in self.batch_logs and not self.batch_logs[local_gpu].closed:
                self.batch_logs[local_gpu].close()
        print(f"[Process {self.process_id}] Closed all batch log files")


def extract_per_device_batch_metadata(
    batch, n_local_devices: int
) -> List[Tuple[int, int, int]]:
    """Extract per-device batch metadata (multi-host safe via addressable_shards)."""
    per_device_metadata = []
    n_node_arr = batch.n_node
    n_edge_arr = batch.n_edge

    if hasattr(n_node_arr, 'addressable_shards') and len(n_node_arr.addressable_shards) > 0:
        n_node_shards = n_node_arr.addressable_shards
        n_edge_shards = n_edge_arr.addressable_shards
        for i in range(min(n_local_devices, len(n_node_shards))):
            device_n_node = np.asarray(n_node_shards[i].data).reshape(-1)
            device_n_edge = np.asarray(n_edge_shards[i].data).reshape(-1)
            n_graphs = int(device_n_node.shape[0]) - 1
            n_nodes = int(np.sum(device_n_node[:-1]))
            n_edges = int(np.sum(device_n_edge[:-1]))
            per_device_metadata.append((n_graphs, n_nodes, n_edges))
    else:
        n_node_array = np.asarray(n_node_arr)
        n_edge_array = np.asarray(n_edge_arr)
        for i in range(n_local_devices):
            device_n_node = n_node_array[i]
            device_n_edge = n_edge_array[i]
            n_graphs = int(device_n_node.shape[0]) - 1
            n_nodes = int(np.sum(device_n_node[:-1]))
            n_edges = int(np.sum(device_n_edge[:-1]))
            per_device_metadata.append((n_graphs, n_nodes, n_edges))

    return per_device_metadata


# =============================================================================
# Distributed Initialization
# =============================================================================

def initialize_distributed():
    """Initialize JAX distributed runtime (SLURM / manual / single-host)."""
    global _debug_logger

    if 'SLURM_PROCID' in os.environ:
        process_id = int(os.environ['SLURM_PROCID'])
        num_processes = int(os.environ['SLURM_NPROCS'])
        coordinator_address = os.environ.get('JAX_COORDINATOR_ADDRESS')
        if coordinator_address is None:
            import subprocess
            result = subprocess.run(
                ['scontrol', 'show', 'hostnames', os.environ['SLURM_JOB_NODELIST']],
                capture_output=True, text=True
            )
            head_node = result.stdout.strip().split('\n')[0]
            coordinator_address = f"{head_node}:29500"
        print(f"[SLURM Process {process_id}/{num_processes}] "
              f"Initializing with coordinator: {coordinator_address}")
        init_start = time.time()
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
        )
        print(f"[SLURM Process {process_id}] Distributed init completed "
              f"in {time.time() - init_start:.2f}s")

    elif 'JAX_NUM_PROCESSES' in os.environ:
        coordinator_address = os.environ.get('JAX_COORDINATOR_ADDRESS')
        num_processes = int(os.environ['JAX_NUM_PROCESSES'])
        process_id = int(os.environ.get('JAX_PROCESS_INDEX', '0'))
        if coordinator_address is None:
            raise ValueError(
                "JAX_COORDINATOR_ADDRESS must be set for manual multi-host setup."
            )
        print(f"[Manual Process {process_id}/{num_processes}] "
              f"Initializing with coordinator: {coordinator_address}")
        init_start = time.time()
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
        )
        print(f"[Manual Process {process_id}] Distributed init completed "
              f"in {time.time() - init_start:.2f}s")

    else:
        print("[Single-host mode] Skipping distributed initialization")

    _debug_logger = DebugLogger(jax.process_index())
    debug_log("Debug logger initialized")
    print(f"Process {jax.process_index()} of {jax.process_count()}")
    print(f"Local devices: {jax.local_device_count()}")


# =============================================================================
# File Discovery
# =============================================================================

def _get_pkl_files(path: str) -> List[str]:
    """Get naturally-sorted list of pkl files from a directory path."""
    def natural_key(s):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', str(s))]
    p = Path(path)
    if p.is_file():
        return [str(p)]
    files = sorted(list(p.glob('*.pkl')), key=natural_key)
    return [str(f) for f in files]


# =============================================================================
# Loss Functions
# =============================================================================

def compute_loss_singlehead(
    graphdef: nnx.GraphDef,
    params: nnx.State,
    batch: jraph.GraphsTuple,
    energy_weight: float,
    force_weight: float,
    stress_weight: float,
    loss_fn: Callable,
) -> Tuple[jnp.ndarray, Dict]:
    """Compute loss for single-head RACE model."""
    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask = jraph.get_node_padding_mask(batch)

    n_graphs = jnp.maximum(graph_mask.sum(), 1)
    n_atoms = jnp.maximum(node_mask.sum(), 1)

    model = nnx.merge(graphdef, params)
    energy, forces, stress = model(batch)

    energy_diff = (energy - batch.globals["energy"]) / batch.n_node
    energy_loss = loss_fn(energy_diff)
    energy_loss = jnp.sum(energy_loss * graph_mask) / n_graphs
    energy_mse = jnp.sum(energy_diff ** 2 * graph_mask) / n_graphs

    forces_diff = forces - batch.nodes["forces"]
    force_loss = loss_fn(forces_diff)
    force_loss = jnp.sum(force_loss * node_mask[:, None]) / (3 * n_atoms)
    force_mse = jnp.sum(forces_diff ** 2 * node_mask[:, None]) / (3 * n_atoms)

    stress_diff = (stress - batch.globals["stress"]) * graph_mask[:, None]
    stress_loss = jnp.sum(loss_fn(stress_diff) * graph_mask[:, None]) / n_graphs
    stress_mse = jnp.sum(stress_diff ** 2 * graph_mask[:, None]) / n_graphs

    total_loss = (energy_weight * energy_loss
                  + force_weight * force_loss
                  + stress_weight * stress_loss)

    aux = {
        'energy_loss': energy_loss, 'force_loss': force_loss,
        'stress_loss': stress_loss, 'energy_mse': energy_mse,
        'force_mse': force_mse, 'stress_mse': stress_mse,
        'n_graphs': n_graphs, 'n_atoms': n_atoms,
    }
    return total_loss, aux


def compute_loss_multihead(
    graphdef: nnx.GraphDef,
    params: nnx.State,
    batch: jraph.GraphsTuple,
    energy_weight: float,
    force_weight: float,
    stress_weight: float,
    loss_fn: Callable,
    num_heads: int,
) -> Tuple[jnp.ndarray, Dict]:
    """Compute loss for multihead RACE model with per-head metrics."""
    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask = jraph.get_node_padding_mask(batch)

    n_graphs = jnp.maximum(graph_mask.sum(), 1)
    n_atoms = jnp.maximum(node_mask.sum(), 1)

    model = nnx.merge(graphdef, params)
    energy, forces, stress = model(batch)

    energy_diff = (energy - batch.globals["energy"]) / batch.n_node
    energy_loss = loss_fn(energy_diff)
    energy_loss = jnp.sum(energy_loss * graph_mask) / n_graphs
    energy_mse = jnp.sum(energy_diff ** 2 * graph_mask) / n_graphs

    forces_diff = forces - batch.nodes["forces"]
    force_loss = loss_fn(forces_diff)
    force_loss = jnp.sum(force_loss * node_mask[:, None]) / (3 * n_atoms)
    force_mse = jnp.sum(forces_diff ** 2 * node_mask[:, None]) / (3 * n_atoms)

    stress_diff = (stress - batch.globals["stress"]) * graph_mask[:, None]
    stress_loss = jnp.sum(loss_fn(stress_diff) * graph_mask[:, None]) / n_graphs
    stress_mse = jnp.sum(stress_diff ** 2 * graph_mask[:, None]) / n_graphs

    total_loss = (energy_weight * energy_loss
                  + force_weight * force_loss
                  + stress_weight * stress_loss)

    # Per-head metrics (vectorized with one_hot)
    head_per_graph = batch.globals["head"]
    head_one_hot = jax.nn.one_hot(head_per_graph, num_heads)
    head_graph_counts = jnp.sum(head_one_hot * graph_mask[:, None], axis=0)

    per_head_energy_ae = jnp.sum(
        jnp.abs(energy_diff)[:, None] * head_one_hot * graph_mask[:, None], axis=0
    )

    sum_n_node = node_mask.shape[0]
    n_graphs_total = batch.n_node.shape[0]
    graph_idx = jnp.arange(n_graphs_total)
    node_graph = jnp.repeat(
        graph_idx, batch.n_node, axis=0, total_repeat_length=sum_n_node
    )
    node_head = head_per_graph[node_graph]
    node_head_one_hot = jax.nn.one_hot(node_head, num_heads)
    head_atom_counts = jnp.sum(node_head_one_hot * node_mask[:, None], axis=0)
    per_head_force_ae = jnp.sum(
        jnp.sum(jnp.abs(forces_diff), axis=-1, keepdims=True)
        * node_head_one_hot * node_mask[:, None], axis=0,
    )

    aux = {
        'energy_loss': energy_loss, 'force_loss': force_loss,
        'stress_loss': stress_loss, 'energy_mse': energy_mse,
        'force_mse': force_mse, 'stress_mse': stress_mse,
        'n_graphs': n_graphs, 'n_atoms': n_atoms,
        'per_head_energy_ae': per_head_energy_ae,
        'per_head_force_ae': per_head_force_ae,
        'per_head_graph_counts': head_graph_counts,
        'per_head_atom_counts': head_atom_counts,
    }
    return total_loss, aux


# =============================================================================
# Training Step Factories
# =============================================================================

def make_sharded_train_step(
    optimizer, mesh, energy_weight, force_weight, stress_weight,
    loss_fn, ema_decay=0.99,
):
    """Create sharded training step for single-head (1D DP)."""

    def per_device_step(graphdef, params, opt_state, schedule_state,
                        ema_params, step, batch):
        batch = squeeze_batch(batch)

        def _loss_fn(params):
            return compute_loss_singlehead(
                graphdef, params, batch,
                energy_weight, force_weight, stress_weight, loss_fn,
            )

        (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(params)

        grads = jax.lax.pmean(grads, axis_name='dp')
        loss = jax.lax.pmean(loss, axis_name='dp')
        aux = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name='dp'), aux)

        grad_norm = optax.global_norm(grads)

        updates, new_opt = optimizer.update(grads, opt_state, params)
        updates = optax.tree_utils.tree_scale(schedule_state.scale, updates)
        new_p = optax.apply_updates(params, updates)
        new_ema = jax.tree.map(
            lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
            ema_params, new_p,
        )

        metrics = {'loss': loss, 'grad_norm': grad_norm, **aux}
        return new_p, new_opt, new_ema, step + 1, metrics

    sharded_step = shard_map(
        per_device_step, mesh=mesh,
        in_specs=(P(), P(), P(), P(), P(), P(), P('dp')),
        out_specs=(P(), P(), P(), P(), P()),
    )
    return jax.jit(sharded_step)


def make_per_head_grad_step(mesh, loss_fn, num_heads):
    """Create a sharded gradient step for a single head.

    Returns (grads, loss, aux, grad_norm) with pmean already applied.
    Each head call is a separate JIT, so different heads can have different
    batch shapes without causing cross-head compilation mismatches.
    """

    def per_device_grad(graphdef, params, batch, energy_w, force_w, stress_w):
        batch = squeeze_batch(batch)

        def _loss_fn(params):
            return compute_loss_multihead(
                graphdef, params, batch,
                energy_w, force_w, stress_w, loss_fn, num_heads,
            )

        (loss, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(params)

        grads = jax.lax.pmean(grads, axis_name='dp')
        loss = jax.lax.pmean(loss, axis_name='dp')
        aux = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name='dp'), aux)
        grad_norm = optax.global_norm(grads)

        return grads, loss, aux, grad_norm

    sharded_fn = shard_map(
        per_device_grad, mesh=mesh,
        in_specs=(P(), P(), P('dp'), P(), P(), P()),
        out_specs=(P(), P(), P(), P()),
    )
    return jax.jit(sharded_fn)


def make_accumulate_and_update_step(optimizer, mesh, num_head_configs, ema_decay=0.99):
    """Accumulate per-head gradients and apply optimizer, all inside shard_map.

    Takes pre-computed per-head grads (already pmean'd) and their grad_weights,
    accumulates them with weighting, normalizes, and applies the optimizer update.
    All operations stay inside shard_map to avoid multi-host sharding issues.

    Args: fixed = (graphdef_unused, params, opt_state, schedule_state, ema_params, step)
          per_head = (grads_0, grad_weight_0, grads_1, grad_weight_1, ...)
    Returns: (new_params, new_opt_state, new_ema_params, new_step, combined_grad_norm)
    """
    N = num_head_configs

    def per_device_fn(params, opt_state, schedule_state, ema_params, step,
                      *grad_and_weight_args):
        acc = None
        total_w = jnp.float32(0.0)
        for i in range(N):
            grads_i = grad_and_weight_args[i * 2]
            w_i = grad_and_weight_args[i * 2 + 1]
            weighted = jax.tree.map(lambda g: g * w_i, grads_i)
            if acc is None:
                acc = weighted
            else:
                acc = jax.tree.map(lambda a, g: a + g, acc, weighted)
            total_w = total_w + w_i

        acc = jax.tree.map(lambda g: g / total_w, acc)
        combined_norm = optax.global_norm(acc)

        updates, new_opt = optimizer.update(acc, opt_state, params)
        updates = optax.tree_utils.tree_scale(schedule_state.scale, updates)
        new_p = optax.apply_updates(params, updates)
        new_ema = jax.tree.map(
            lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
            ema_params, new_p,
        )
        return new_p, new_opt, new_ema, step + 1, combined_norm

    fixed_specs = (P(), P(), P(), P(), P())
    # Each head contributes (grads=P(), grad_weight=P())
    per_head_specs = tuple(P() for _ in range(N * 2))

    sharded_fn = shard_map(
        per_device_fn, mesh=mesh,
        in_specs=fixed_specs + per_head_specs,
        out_specs=(P(), P(), P(), P(), P()),
    )
    return jax.jit(sharded_fn)


def make_simple_optimizer_step(optimizer, mesh, ema_decay=0.99):
    """Apply optimizer update with pre-accumulated gradients.

    This is used with online accumulation where gradients are accumulated
    in the training loop itself, reducing peak memory usage.

    Args:
        optimizer: Optax optimizer
        mesh: JAX mesh for sharding
        ema_decay: EMA decay rate

    Returns:
        JIT-compiled function that takes (params, opt_state, schedule_state,
        ema_params, step, accumulated_grads) and returns updated state.
    """
    def per_device_fn(params, opt_state, schedule_state, ema_params, step, acc_grads):
        combined_norm = optax.global_norm(acc_grads)

        updates, new_opt = optimizer.update(acc_grads, opt_state, params)
        updates = optax.tree_utils.tree_scale(schedule_state.scale, updates)
        new_p = optax.apply_updates(params, updates)
        new_ema = jax.tree.map(
            lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
            ema_params, new_p,
        )
        return new_p, new_opt, new_ema, step + 1, combined_norm

    sharded_fn = shard_map(
        per_device_fn, mesh=mesh,
        in_specs=(P(), P(), P(), P(), P(), P()),
        out_specs=(P(), P(), P(), P(), P()),
    )
    return jax.jit(sharded_fn)


# =============================================================================
# Evaluation
# =============================================================================

def make_sharded_evaluate_step(mesh: Mesh, loss_fn: Callable):
    """Create sharded evaluation function (shared by both modes)."""

    def eval_single_device(graphdef, params, batch):
        batch = squeeze_batch(batch)
        graph_mask = jraph.get_graph_padding_mask(batch)
        node_mask = jraph.get_node_padding_mask(batch)

        model = nnx.merge(graphdef, params)
        energy, forces, stress = model(batch)

        energy_diff = (energy - batch.globals["energy"]) / batch.n_node
        energy_loss = loss_fn(energy_diff)
        forces_diff = forces - batch.nodes["forces"]
        force_loss = loss_fn(forces_diff)
        stress_diff = (stress - batch.globals["stress"]) * graph_mask[:, None]
        stress_loss = loss_fn(stress_diff)

        return {
            'energy_loss': jnp.sum(energy_loss * graph_mask),
            'force_loss': jnp.sum(force_loss * node_mask[:, None]),
            'stress_loss': jnp.sum(stress_loss * graph_mask[:, None]),
            'energy_ae': jnp.sum(jnp.abs(energy_diff) * graph_mask),
            'force_ae': jnp.sum(jnp.abs(forces_diff) * node_mask[:, None]),
            'stress_ae': jnp.sum(jnp.abs(stress_diff) * graph_mask[:, None]),
            'energy_se': jnp.sum(energy_diff ** 2 * graph_mask),
            'force_se': jnp.sum(forces_diff ** 2 * node_mask[:, None]),
            'stress_se': jnp.sum(stress_diff ** 2 * graph_mask[:, None]),
            'n_graphs': graph_mask.sum(),
            'n_atoms': node_mask.sum(),
        }

    def eval_and_reduce(graphdef, params, batch):
        local_metrics = eval_single_device(graphdef, params, batch)
        return jax.tree.map(
            lambda x: jax.lax.psum(x, axis_name='dp'), local_metrics
        )

    sharded_eval = shard_map(
        eval_and_reduce, mesh=mesh,
        in_specs=(P(), P(), P('dp')),
        out_specs=P(),
    )
    return jax.jit(sharded_eval)


def evaluate_sharded(
    graphdef, params, files_pkl, batch_size, n_devices, mesh,
    energy_weight, force_weight, stress_weight, loss_fn,
    batch_logger=None, epoch=0,
) -> Dict:
    """Evaluate single-head model on validation files."""
    params_replicated = replicate_pytree(params, mesh)
    eval_step = make_sharded_evaluate_step(mesh, loss_fn)

    totals = {
        'energy_loss': 0.0, 'force_loss': 0.0, 'stress_loss': 0.0,
        'energy_ae': 0.0, 'force_ae': 0.0, 'stress_ae': 0.0,
        'energy_se': 0.0, 'force_se': 0.0, 'stress_se': 0.0,
        'n_graphs': 0.0, 'n_atoms': 0.0,
    }

    is_first_eval = True
    for fidx, fname in enumerate(files_pkl):
        debug_log(f"Evaluating file {fidx+1}/{len(files_pkl)}: {Path(fname).name}")

        if fidx > 0:
            jax.clear_caches()
            is_first_eval = True

        if batch_logger is not None:
            batch_logger.reset_counters()

        file_name = Path(fname).name
        dataset = Dataset(
            file_path=fname,
            process_id=jax.process_index(),
            n_processes=jax.process_count(),
        )
        if batch_logger is not None:
            total_graphs = len(dataset) * jax.process_count()
            batch_logger.set_total_graphs(total_graphs)

        base_loader = BucketedDataLoader(
            dataset=dataset, batch_size=batch_size, n_buckets=8,
            shuffle=False, drop_last=False, rngs=nnx.Rngs(0),
        )
        loader = MultiDeviceDataLoader(
            base_loader=base_loader, n_devices=n_devices,
            mesh=mesh, drop_incomplete=False,
        )

        # Sync batch count across hosts
        if jax.process_count() > 1:
            local_n = len(base_loader) // jax.local_device_count()
            local_arr = jnp.array([local_n], dtype=jnp.int32)
            all_counts = multihost_utils.process_allgather(local_arr, tiled=False)
            all_counts = jax.device_get(all_counts).flatten().tolist()
            synced_max_batches = min(all_counts)
            multihost_utils.sync_global_devices(f"eval_batch_sync_f{fidx}")
        else:
            synced_max_batches = float('inf')

        batch_idx = 0
        for batch, info in loader:
            if batch_idx >= synced_max_batches:
                break
            batch_idx += 1
            graph_indices = getattr(info, 'graph_indices_per_device', None)

            if is_first_eval:
                debug_log("Starting eval_step JIT compilation")
                compile_start = time.time()

            metrics = eval_step(graphdef, params_replicated, batch)

            if is_first_eval:
                debug_log(f"eval_step JIT done in {time.time() - compile_start:.1f}s")
                is_first_eval = False

            if batch_logger is not None:
                per_device_metadata = extract_per_device_batch_metadata(
                    batch, jax.local_device_count()
                )
                batch_logger.log_batch(
                    epoch=epoch, mode='valid', file_name=file_name,
                    batch_idx=batch_idx, per_device_metadata=per_device_metadata,
                    graph_indices_per_device=graph_indices,
                )

            for k in totals:
                totals[k] += float(metrics[k])

    n_g = max(totals['n_graphs'], 1)
    n_a = max(totals['n_atoms'], 1)
    total_loss = (energy_weight * totals['energy_loss'] / n_g
                  + force_weight * totals['force_loss'] / (3 * n_a)
                  + stress_weight * totals['stress_loss'] / n_g)

    return {
        'energy_mae': totals['energy_ae'] / n_g,
        'force_mae': totals['force_ae'] / (3 * n_a),
        'stress_mae': totals['stress_ae'] / n_g,
        'energy_rmse': np.sqrt(totals['energy_se'] / n_g),
        'force_rmse': np.sqrt(totals['force_se'] / (3 * n_a)),
        'stress_rmse': np.sqrt(totals['stress_se'] / n_g),
        'energy_loss': totals['energy_loss'] / n_g,
        'force_loss': totals['force_loss'] / (3 * n_a),
        'stress_loss': totals['stress_loss'] / n_g,
        'total_loss': total_loss,
        'n_graphs': n_g,
        'n_atoms': n_a,
    }


def evaluate_per_head(
    graphdef, params, head_configs, batch_size, n_devices, mesh,
    energy_weight, force_weight, stress_weight, loss_fn, num_heads,
    batch_logger=None, epoch=0,
    eval_local_mesh=None, eval_step_fn=None,
    head_weights=None,
) -> Tuple[Dict, float]:
    """Evaluate multihead model on each head's validation set separately.

    Uses a LOCAL mesh (per-host devices only) so that eval_step's psum
    communicates only within each host's GPUs.  This eliminates cross-host
    collective operations during the batch loop, preventing deadlocks when
    hosts have different numbers of batches.  Metrics are gathered across
    hosts once per head after all files are processed.
    """
    # --- Local mesh: reuse pre-built or create on first call ---
    if eval_local_mesh is None or eval_step_fn is None:
        local_devices = jax.local_devices()
        n_local = len(local_devices)
        eval_local_mesh = Mesh(np.array(local_devices), ('dp',))
        eval_step_fn = make_sharded_evaluate_step(eval_local_mesh, loss_fn)
    else:
        n_local = len(eval_local_mesh.devices.flat)

    local_mesh = eval_local_mesh
    eval_step = eval_step_fn

    params_replicated = replicate_pytree(params, local_mesh)

    if jax.process_index() == 0:
        print(f"  [eval] using local mesh ({n_local} devices per host, "
              f"no cross-host sync during batches)", flush=True)

    all_head_metrics = {}
    total_val_loss = 0.0
    total_weight = 0.0

    for head_cfg in head_configs:
        head_idx = head_cfg["head_idx"]
        head_name = head_cfg["name"]
        valid_path = head_cfg.get("valid_path")

        if valid_path is None:
            continue

        valid_files = _get_pkl_files(valid_path)
        if not valid_files:
            continue

        totals = {
            'energy_loss': 0.0, 'force_loss': 0.0, 'stress_loss': 0.0,
            'energy_ae': 0.0, 'force_ae': 0.0, 'stress_ae': 0.0,
            'energy_se': 0.0, 'force_se': 0.0, 'stress_se': 0.0,
            'n_graphs': 0.0, 'n_atoms': 0.0,
        }

        n_valid_files = len(valid_files)
        if jax.process_index() == 0:
            print(f"  [eval] head={head_name}: {n_valid_files} valid files",
                  flush=True)

        for fi, fname in enumerate(valid_files):
            file_name = Path(fname).name
            if batch_logger is not None:
                batch_logger.reset_counters()

            # --- step 1: load data ---
            _diag_t0 = time.time()
            dataset = DatasetWithHead(
                file_path=fname, head_idx=head_idx,
                process_id=jax.process_index(),
                n_processes=jax.process_count(),
            )

            if batch_logger is not None:
                batch_logger.set_total_graphs(len(dataset) * jax.process_count())

            # --- step 2: create loader (LOCAL mesh, no cross-host sync) ---
            base_loader = BucketedDataLoader(
                dataset=dataset, batch_size=batch_size, n_buckets=8,
                shuffle=False, drop_last=False, rngs=nnx.Rngs(0),
            )
            loader = MultiDeviceDataLoader(
                base_loader=base_loader, n_devices=n_local,
                mesh=local_mesh, drop_incomplete=False,
            )
            total_batches = len(base_loader) // max(n_local, 1)

            if jax.process_index() == 0:
                print(f"  [eval] head={head_name} file {fi+1}/{n_valid_files} "
                      f"({file_name}): {len(dataset)} graphs, "
                      f"~{total_batches} batches, "
                      f"loaded in {time.time()-_diag_t0:.1f}s", flush=True)

            # --- step 3: process batches (LOCAL psum only) ---
            eval_start_time = time.time()
            batch_idx = 0
            for batch, info in loader:
                batch_idx += 1
                graph_indices = getattr(info, 'graph_indices_per_device', None)
                metrics = eval_step(graphdef, params_replicated, batch)
                for k in totals:
                    totals[k] += float(metrics[k])

                if batch_idx % 100 == 0 and jax.process_index() == 0:
                    print(f"  [eval] head={head_name} "
                          f"batch {batch_idx}/{total_batches} "
                          f"({time.time()-eval_start_time:.1f}s)", flush=True)

            if jax.process_index() == 0:
                elapsed = time.time() - eval_start_time
                print(f"  [eval] head={head_name} file {fi+1}/{n_valid_files} "
                      f"({file_name}): {batch_idx} batches in {elapsed:.1f}s",
                      flush=True)

                if batch_logger is not None:
                    per_device_metadata = extract_per_device_batch_metadata(
                        batch, n_local
                    )
                    batch_logger.log_batch(
                        epoch=epoch, mode='valid', file_name=file_name,
                        batch_idx=batch_idx, per_device_metadata=per_device_metadata,
                        head_name=head_name,
                        graph_indices_per_device=graph_indices,
                    )

        # --- Gather metrics across all hosts (one sync per head) ---
        if jax.process_count() > 1:
            if jax.process_index() == 0:
                print(f"  [eval] head={head_name} gathering metrics "
                      f"across {jax.process_count()} hosts...", flush=True)
            for k in totals:
                local_val = jnp.array([totals[k]], dtype=jnp.float32)
                all_vals = multihost_utils.process_allgather(
                    local_val, tiled=False)
                totals[k] = float(jnp.sum(all_vals))
            multihost_utils.sync_global_devices(
                f"eval_head_{head_name}_complete")
            if jax.process_index() == 0:
                print(f"  [eval] head={head_name} sync complete", flush=True)

        n_g = max(totals['n_graphs'], 1)
        n_a = max(totals['n_atoms'], 1)
        # Use per-head weights if available, otherwise fall back to global
        if head_weights is not None and head_idx in head_weights:
            hw = head_weights[head_idx]
            h_ew = float(hw['energy_weight'])
            h_fw = float(hw['force_weight'])
            h_sw = float(hw['stress_weight'])
        else:
            h_ew = energy_weight
            h_fw = force_weight
            h_sw = stress_weight
        head_loss = (
            h_ew * totals['energy_loss'] / n_g
            + h_fw * totals['force_loss'] / (3 * n_a)
            + h_sw * totals['stress_loss'] / n_g
        )
        head_metrics = {
            'energy_mae': totals['energy_ae'] / n_g,
            'force_mae': totals['force_ae'] / (3 * n_a),
            'stress_mae': totals['stress_ae'] / n_g,
            'energy_rmse': np.sqrt(totals['energy_se'] / n_g),
            'force_rmse': np.sqrt(totals['force_se'] / (3 * n_a)),
            'stress_rmse': np.sqrt(totals['stress_se'] / n_g),
            'energy_loss': totals['energy_loss'] / n_g,
            'force_loss': totals['force_loss'] / (3 * n_a),
            'stress_loss': totals['stress_loss'] / n_g,
            'total_loss': head_loss,
            'n_graphs': n_g, 'n_atoms': n_a,
        }
        all_head_metrics[head_name] = head_metrics
        total_val_loss += head_loss * n_g
        total_weight += n_g

    # --- Free replicated params to reclaim GPU memory before training resumes ---
    del params_replicated
    gc.collect()

    total_val_loss = total_val_loss / max(total_weight, 1)
    return all_head_metrics, total_val_loss


# =============================================================================
# Multihead Utilities
# =============================================================================

def parse_head_weights(head_configs, global_e_w, global_f_w, global_s_w):
    """Parse per-head E/F/S/grad weights. Falls back to global values."""
    weights = {}
    for hc in head_configs:
        weights[hc["head_idx"]] = {
            'energy_weight': jnp.array(hc.get('energy_weight', global_e_w), dtype=jnp.float32),
            'force_weight': jnp.array(hc.get('force_weight', global_f_w), dtype=jnp.float32),
            'stress_weight': jnp.array(hc.get('stress_weight', global_s_w), dtype=jnp.float32),
            'grad_weight': jnp.array(hc.get('grad_weight', 1.0), dtype=jnp.float32),
        }
    return weights


class HeadBatchStream:
    """Per-head batch stream with automatic file transitions and cycling."""

    def __init__(self, head_config, seed, epoch, batch_size, n_devices, mesh,
                 max_nodes_per_batch=256, max_edges_per_batch=None):
        self.head_idx = head_config["head_idx"]
        self.head_name = head_config["name"]
        self.all_files = _get_pkl_files(head_config["train_path"])
        self.n_files = len(self.all_files)
        if self.n_files == 0:
            raise FileNotFoundError(
                f"Head '{self.head_name}': no .pkl files in '{head_config['train_path']}'")
        self.seed = seed
        self.epoch = epoch
        self.batch_size = batch_size
        self.n_devices = n_devices
        self.mesh = mesh
        self.max_nodes_per_batch = max_nodes_per_batch
        self.max_edges_per_batch = max_edges_per_batch
        self.cycle_count = 0
        self.file_idx = 0
        self.files_consumed = 0
        self.batches_yielded = 0
        self._current_iter = None
        self._current_file = None
        self._shuffle_files()

    def _shuffle_files(self):
        rng = np.random.RandomState(
            self.seed + 1000 * self.epoch + 100 * self.head_idx + self.cycle_count
        )
        self.shuffled_files = list(self.all_files)
        rng.shuffle(self.shuffled_files)
        self.file_idx = 0

    def _open_next_file(self):
        if self.file_idx >= self.n_files:
            self.cycle_count += 1
            self._shuffle_files()

        file_path = self.shuffled_files[self.file_idx]
        self.file_idx += 1
        self.files_consumed += 1

        dataset = DatasetWithHead(
            file_path=file_path, head_idx=self.head_idx,
            process_id=jax.process_index(),
            n_processes=jax.process_count(),
        )
        rngs = nnx.Rngs(
            self.seed + 100 * self.epoch + 10 * self.files_consumed + self.head_idx
        )
        base_loader = BucketedDataLoader(
            dataset=dataset, batch_size=self.batch_size,
            n_buckets=8, shuffle=True, drop_last=True,
            rngs=rngs, max_nodes_per_batch=self.max_nodes_per_batch,
            max_edges_per_batch=self.max_edges_per_batch,
        )
        loader = MultiDeviceDataLoader(
            base_loader=base_loader, n_devices=self.n_devices,
            mesh=self.mesh, drop_incomplete=True,
        )
        self._current_iter = iter(loader)
        self._current_file = file_path

    def next_batch(self):
        while True:
            if self._current_iter is not None:
                try:
                    batch, info = next(self._current_iter)
                    self.batches_yielded += 1
                    return batch, info
                except StopIteration:
                    self._current_iter = None
            self._open_next_file()

    @property
    def completed_first_pass(self):
        return self.files_consumed >= self.n_files


# =============================================================================
# Checkpoint Management
# =============================================================================

def save_checkpoint_safe(state_dict: Dict, path: str):
    """Save checkpoint (only on process 0)."""
    if jax.process_index() == 0:
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)
        print(f"Checkpoint saved to {path}")


def load_checkpoint(path: str) -> Dict:
    with open(path, 'rb') as f:
        return pickle.load(f)


# =============================================================================
# Model Initialization
# =============================================================================

def create_model(config: Dict, is_multihead: bool):
    """Create model and return (graphdef, params, load_info).

    load_info is a dict with foundation model metadata (multihead only),
    or None for single-head / no foundation.
    """
    seed = config.get('seed', 42)
    model_rngs = nnx.Rngs(seed)
    load_info = None

    if is_multihead:
        num_heads = len(config["heads"])
        foundation_ckpt = config.get("foundation_ckpt")

        if foundation_ckpt:
            graphdef, params, load_info = load_foundation_as_multihead(
                ckpt_path=foundation_ckpt,
                num_heads=num_heads,
                config=config,
                rngs=model_rngs,
                use_checkpoint=config.get('use_checkpoint', False),
            )
        else:
            model = RACEMultihead(
                n_species=len(config["atom_energies"]),
                num_heads=num_heads,
                lmax=config["lmax"],
                hidden_irreps=config["hidden_irreps"],
                n_layers=config["n_layers"],
                radial_basis_size=config["radial_basis_size"],
                radial_mlp_size=config["radial_mlp_size"],
                radial_mlp_layers=config["radial_mlp_layers"],
                radial_polynomial_p=config["radial_polynomial_p"],
                mlp_init_scale=config["mlp_init_scale"],
                shift=0.0, scale=1.0, avg_n_neighbors=25.0,
                atom_energies=config["atom_energies"],
                l_train=True, periodic=True,
                use_checkpoint=config.get('use_checkpoint', False),
                rngs=model_rngs,
            )
            graphdef, params = nnx.split(model, nnx.Param)
    else:
        model = RACE(
            n_species=len(config["atom_energies"]),
            lmax=config["lmax"],
            hidden_irreps=config["hidden_irreps"],
            n_layers=config["n_layers"],
            radial_basis_size=config["radial_basis_size"],
            radial_mlp_size=config["radial_mlp_size"],
            radial_mlp_layers=config["radial_mlp_layers"],
            radial_polynomial_p=config["radial_polynomial_p"],
            mlp_init_scale=config["mlp_init_scale"],
            shift=0.0, scale=1.0, avg_n_neighbors=25.0,
            atom_energies=config["atom_energies"],
            l_train=True, periodic=True,
            use_checkpoint=config.get('use_checkpoint', False),
            rngs=model_rngs,
        )
        graphdef, params = nnx.split(model, nnx.Param)

    return graphdef, params, load_info


# =============================================================================
# Optimizer Setup
# =============================================================================

def create_optimizer(config: Dict):
    """Create optimizer chain and LR schedule."""
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.get('grad_clip', 1.0)),
        optax.amsgrad(config["learning_rate"]),
    )
    schedule = optax.contrib.reduce_on_plateau(
        factor=config["lr_schedule_factor"],
        patience=config["lr_schedule_patience"],
        min_scale=config["lr_schedule_min_scale"],
    )
    return optimizer, schedule


# =============================================================================
# Single-Head Training Loop
# =============================================================================

def _train_loop_singlehead(
    config, mesh, n_devices, graphdef, params, ema_params,
    opt_state, schedule_state, step, optimizer, schedule, loss_fn,
    train_files, valid_files, batch_logger, fout,
    start_epoch, start_ipkl, best_val_loss, ckpt_dir,
):
    """File-based epoch -> ipkl -> batch training loop for single-head."""
    seed = config.get('seed', 42)
    energy_weight = config.get('energy_weight', 1.0)
    force_weight = config.get('force_weight', 1.0)
    stress_weight = config.get('stress_weight', 1.0)

    train_step = make_sharded_train_step(
        optimizer=optimizer, mesh=mesh,
        energy_weight=energy_weight, force_weight=force_weight,
        stress_weight=stress_weight, loss_fn=loss_fn,
        ema_decay=config.get('ema_decay', 0.99),
    )

    best_state = None
    is_first_train_step = True
    is_first_eval_step = True
    log_every = config.get('log_every', 100)

    emergency_ckpt = get_emergency_checkpointer(
        save_path=str(ckpt_dir / 'ckpt_emergency.pkl'))

    if not sync_and_check(mesh, "training_start"):
        debug_log("Initial sync failed, aborting", "ERROR")
        return

    for epoch in range(start_epoch, config["n_epochs"]):
        epoch_start = time.time()
        debug_log(f"Starting epoch {epoch+1}/{config['n_epochs']}")

        epoch_start_ipkl = start_ipkl if epoch == start_epoch else 0

        for ipkl, train_pkl in enumerate(train_files):
            if ipkl < epoch_start_ipkl:
                continue

            batch_logger.reset_counters()

            train_dataset = Dataset(
                file_path=train_pkl,
                process_id=jax.process_index(),
                n_processes=jax.process_count(),
            )
            total_graphs = len(train_dataset) * jax.process_count()
            batch_logger.set_total_graphs(total_graphs)

            loader_rngs = nnx.Rngs(seed + 100 * epoch + ipkl)
            base_loader = BucketedDataLoader(
                dataset=train_dataset, batch_size=config['batch_size'],
                n_buckets=8, shuffle=True, drop_last=True, rngs=loader_rngs,
            )
            loader = MultiDeviceDataLoader(
                base_loader=base_loader, n_devices=n_devices,
                mesh=mesh, drop_incomplete=True,
            )

            # Sync batch count across hosts
            if jax.process_count() > 1:
                local_n_batches = len(base_loader) // jax.local_device_count()
                local_arr = jnp.array([local_n_batches], dtype=jnp.int32)
                all_counts = multihost_utils.process_allgather(local_arr, tiled=False)
                all_counts = jax.device_get(all_counts).flatten().tolist()
                synced_max_batches = min(all_counts)
                multihost_utils.sync_global_devices(f"train_sync_e{epoch}_f{ipkl}")
            else:
                synced_max_batches = float('inf')

            acc_energy_loss, acc_force_loss, acc_stress_loss = 0., 0., 0.
            acc_graphs, acc_atoms, acc_grad = 0., 0., 0.
            dataset_start = time.time()
            batch_idx = 0
            file_name = Path(train_pkl).name

            for batch_data in loader:
                if batch_idx >= synced_max_batches:
                    break
                batch_idx += 1

                if isinstance(batch_data, tuple):
                    batch, batch_info = batch_data
                    graph_indices = getattr(batch_info, 'graph_indices_per_device', None)
                else:
                    batch = batch_data
                    graph_indices = None

                if is_first_train_step:
                    print("JIT Compiling train_step...", file=fout, flush=True)
                    compile_start = time.time()

                try:
                    params, opt_state, ema_params, step, metrics = train_step(
                        graphdef, params, opt_state, schedule_state,
                        ema_params, step, batch,
                    )
                except Exception as e:
                    debug_log(f"train_step FAILED: {e}", "ERROR")
                    sync_and_check(mesh, f"error_sync_batch_{batch_idx}")
                    raise

                if is_first_train_step:
                    compile_time = time.time() - compile_start
                    print(f"train_step compilation done! ({compile_time:.1f}s)",
                          file=fout, flush=True)
                    is_first_train_step = False
                    sync_and_check(mesh, "post_first_train_step")

                n_g = float(metrics['n_graphs'])
                n_a = float(metrics['n_atoms'])
                acc_energy_loss += float(metrics['energy_loss']) * n_g
                acc_force_loss += float(metrics['force_loss']) * 3 * n_a
                acc_stress_loss += float(metrics['stress_loss']) * n_g
                acc_graphs += n_g
                acc_atoms += n_a
                acc_grad += float(metrics['grad_norm']) * n_g

                per_device_metadata = extract_per_device_batch_metadata(
                    batch, jax.local_device_count()
                )
                batch_logger.log_batch(
                    epoch=epoch, mode='train', file_name=file_name,
                    batch_idx=batch_idx, per_device_metadata=per_device_metadata,
                    graph_indices_per_device=graph_indices,
                )

                current_step = int(np.asarray(step))

                if current_step % log_every == 0:
                    el = acc_energy_loss / max(acc_graphs, 1)
                    fl = acc_force_loss / max(3 * acc_atoms, 1)
                    sl = acc_stress_loss / max(acc_graphs, 1)
                    tl = energy_weight * el + force_weight * fl + stress_weight * sl
                    print(f"step {current_step} LOSS: {tl:.6f} E: {el:.6f} "
                          f"F: {fl:.6f} S: {sl:.6f}", file=fout)

                if current_step % 500 == 0 and current_step > 0:
                    sync_and_check(mesh, f"periodic_sync_{current_step}")

            # End-of-file summary
            unrep_schedule = unreplicate_pytree(schedule_state)
            lr = config['learning_rate'] * float(unrep_schedule.scale)
            dataset_time = time.time() - dataset_start
            el = acc_energy_loss / max(acc_graphs, 1)
            fl = acc_force_loss / max(3 * acc_atoms, 1)
            sl = acc_stress_loss / max(acc_graphs, 1)
            tl = energy_weight * el + force_weight * fl + stress_weight * sl
            print(f"Epoch {epoch+1} IPKL {ipkl+1} ({dataset_time:.1f}s) "
                  f"Step {current_step} LR: {lr:.2e}", file=fout)
            print(f"  Train | Loss: {tl:.6f} | E: {el:.6f} | "
                  f"F: {fl:.6f} | S: {sl:.6f}", file=fout)

            acc_energy_loss, acc_force_loss, acc_stress_loss = 0., 0., 0.
            acc_graphs, acc_atoms, acc_grad = 0., 0., 0.

            # Validation check
            val_every_ipkl = config.get('val_every_ipkl', 10)
            is_val_time = ((ipkl + 1) % val_every_ipkl == 0) or \
                          (ipkl == len(train_files) - 1)

            if is_val_time:
                # Save latest checkpoint
                latest_unrep_params = unreplicate_pytree(params)
                latest_unrep_params = multihost_utils.broadcast_one_to_all(
                    latest_unrep_params)
                latest_unrep_ema = unreplicate_pytree(ema_params)
                latest_unrep_ema = multihost_utils.broadcast_one_to_all(
                    latest_unrep_ema)
                latest_unrep_schedule = unreplicate_pytree(schedule_state)
                latest_unrep_schedule = multihost_utils.broadcast_one_to_all(
                    latest_unrep_schedule)
                latest_state = {
                    'params': latest_unrep_params,
                    'ema_params': latest_unrep_ema,
                    'opt_state': unreplicate_pytree(opt_state),
                    'schedule_state': latest_unrep_schedule,
                    'step': int(np.asarray(step)),
                    'best_val_loss': best_val_loss,
                    'epoch': epoch, 'ipkl': ipkl,
                    'atom_energies': config['atom_energies'],
                    'atom_energies_path': config.get('atom_energies_path'),
                    'atomic_number_to_index': config.get('atomic_number_to_index'),
                    'config': config,
                }
                save_checkpoint_safe(latest_state, str(ckpt_dir / 'ckpt_latest.pkl'))

                sync_and_check(mesh, f"pre_val_e{epoch+1}_i{ipkl+1}")

                val_start = time.time()
                unrep_ema = unreplicate_pytree(ema_params)
                unrep_ema = multihost_utils.broadcast_one_to_all(unrep_ema)

                val_metrics = evaluate_sharded(
                    graphdef=graphdef, params=unrep_ema,
                    files_pkl=valid_files, batch_size=config['batch_size'],
                    n_devices=n_devices, mesh=mesh,
                    energy_weight=energy_weight, force_weight=force_weight,
                    stress_weight=stress_weight, loss_fn=loss_fn,
                    batch_logger=batch_logger, epoch=epoch,
                )
                val_time = time.time() - val_start
                val_loss = val_metrics['total_loss']

                if jax.process_count() > 1:
                    val_loss = float(multihost_utils.broadcast_one_to_all(
                        jnp.array(val_loss)))

                print(f"  Valid ({val_time:.1f}s) | Loss: {val_loss:.6f} | "
                      f"E: {val_metrics['energy_loss']:.6f} | "
                      f"F: {val_metrics['force_loss']:.6f} | "
                      f"S: {val_metrics['stress_loss']:.6f} | "
                      f"E_MAE: {val_metrics['energy_mae']:.6f} | "
                      f"F_MAE: {val_metrics['force_mae']:.6f} | "
                      f"S_MAE: {val_metrics['stress_mae']:.6f} | "
                      f"E_RMSE: {val_metrics['energy_rmse']:.6f} | "
                      f"F_RMSE: {val_metrics['force_rmse']:.6f} | "
                      f"S_RMSE: {val_metrics['stress_rmse']:.6f}", file=fout)

                # Update LR schedule
                unrep_params = unreplicate_pytree(params)
                unrep_params = multihost_utils.broadcast_one_to_all(unrep_params)
                unrep_sched = unreplicate_pytree(schedule_state)
                unrep_sched = multihost_utils.broadcast_one_to_all(unrep_sched)
                _, new_schedule = schedule.update(
                    updates=unrep_params, state=unrep_sched, value=val_loss)
                new_schedule = multihost_utils.broadcast_one_to_all(new_schedule)
                schedule_state = replicate_pytree(new_schedule, mesh)

                if val_loss < best_val_loss:
                    improvement = best_val_loss - val_loss
                    best_val_loss = val_loss
                    best_state = {
                        'params': unrep_params,
                        'ema_params': unrep_ema,
                        'opt_state': unreplicate_pytree(opt_state),
                        'schedule_state': new_schedule,
                        'step': int(np.asarray(step)),
                        'best_val_loss': best_val_loss,
                        'metrics': val_metrics,
                        'epoch': epoch, 'ipkl': ipkl,
                        'atom_energies': config['atom_energies'],
                        'atom_energies_path': config.get('atom_energies_path'),
                        'atomic_number_to_index': config.get('atomic_number_to_index'),
                    'config': config,
                    }
                    print(f"  *** New best: {best_val_loss:.6f} "
                          f"(+{improvement:.6f}) ***", file=fout)
                    save_checkpoint_safe(best_state,
                                         str(ckpt_dir / 'ckpt_best.pkl'))
                else:
                    print(f"  No improvement (best: {best_val_loss:.6f})",
                          file=fout)

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config['n_epochs']} completed in "
              f"{epoch_time:.1f}s\n", file=fout)

    if best_state is not None:
        save_checkpoint_safe(best_state, str(ckpt_dir / 'ckpt_best.pkl'))


# =============================================================================
# Multihead Training Loop
# =============================================================================

def _train_loop_multihead(
    config, mesh, n_devices, graphdef, params, ema_params,
    opt_state, schedule_state, step, optimizer, schedule, loss_fn,
    head_configs, head_weights, batch_logger, fout,
    start_epoch, best_val_loss, ckpt_dir,
):
    """Stream-based multihead training loop with gradient accumulation."""
    seed = config.get('seed', 42)
    num_heads = len(head_configs)
    energy_weight = config.get('energy_weight', 1.0)
    force_weight = config.get('force_weight', 1.0)
    stress_weight = config.get('stress_weight', 1.0)
    log_every = config.get('log_every', 50)
    val_every_steps = config.get('val_every_steps', 0)  # 0 = epoch-end only
    max_nodes_per_batch = config.get('max_nodes_per_batch', 256)
    max_edges_per_batch = config.get('max_edges_per_batch', None)

    grad_step = make_per_head_grad_step(mesh, loss_fn, num_heads)
    # Use simple optimizer step with online accumulation for memory efficiency
    simple_optimizer_step = make_simple_optimizer_step(
        optimizer, mesh, ema_decay=config.get('ema_decay', 0.99),
    )
    # Pre-compute total grad weight for normalization
    total_grad_weight = sum(
        hc.get('grad_weight', 1.0) for hc in head_configs
    )

    # Pre-build eval resources (local mesh + JIT-compiled eval_step)
    # so that evaluate_per_head does NOT recompile every call.
    local_devices = jax.local_devices()
    n_local = len(local_devices)
    eval_local_mesh = Mesh(np.array(local_devices), ('dp',))
    eval_step_fn = make_sharded_evaluate_step(eval_local_mesh, loss_fn)

    # Replicate per-head weights across devices
    for h in head_weights:
        head_weights[h] = {k: replicate(v, mesh)
                           for k, v in head_weights[h].items()}

    best_state = None
    last_val_step = int(jax.device_get(unreplicate(step)))  # skip eval already done before restart
    emergency_ckpt = get_emergency_checkpointer(
        save_path=str(ckpt_dir / 'ckpt_emergency.pkl'))

    if not sync_and_check(mesh, "training_start"):
        debug_log("Initial sync failed, aborting", "ERROR")
        return

    for epoch in range(start_epoch, config["n_epochs"]):
        epoch_start = time.time()
        debug_log(f"Starting epoch {epoch+1}/{config['n_epochs']}")

        streams = {
            hc["head_idx"]: HeadBatchStream(
                hc, seed, epoch, config['batch_size'], n_devices, mesh,
                max_nodes_per_batch=max_nodes_per_batch,
                max_edges_per_batch=max_edges_per_batch,
            )
            for hc in head_configs
        }
        longest_head = max(streams.keys(), key=lambda h: streams[h].n_files)

        step_in_epoch = 0
        epoch_step_start = time.time()
        is_multihost = jax.process_count() > 1

        is_first_step = True
        while True:
            # Online accumulation: compute grad → accumulate → free, one head at a time
            # This keeps only 1 gradient in memory instead of num_heads gradients
            acc_grads = None
            per_head_metrics_list = []
            head_log_meta = []
            step_start_time = time.time()

            for i, head_cfg in enumerate(head_configs):
                h = head_cfg["head_idx"]
                stream = streams[h]
                hw = head_weights[h]

                # Load batch for this head only
                batch, info = stream.next_batch()
                graph_indices = getattr(info, 'graph_indices_per_device', None)

                # Extract logging metadata (CPU-side) before gradient computation
                per_device_metadata = extract_per_device_batch_metadata(
                    batch, jax.local_device_count()
                )
                head_log_meta.append((
                    head_cfg, stream, per_device_metadata, graph_indices,
                ))

                # Compute gradient for this head (separate JIT → frees activations)
                if is_first_step and jax.process_index() == 0:
                    print(f"  [JIT] grad_step head={head_cfg['name']} compiling...",
                          flush=True)
                    t0 = time.time()

                grads, loss_i, aux_i, grad_norm_i = grad_step(
                    graphdef, params, batch,
                    hw['energy_weight'], hw['force_weight'], hw['stress_weight'],
                )

                if is_first_step and jax.process_index() == 0:
                    print(f"  [JIT] grad_step head={head_cfg['name']} done "
                          f"in {time.time() - t0:.1f}s", flush=True)

                del batch

                # Online accumulation: accumulate weighted gradients immediately
                grad_weight = head_cfg.get('grad_weight', 1.0)
                if acc_grads is None:
                    acc_grads = jax.tree.map(lambda g: g * grad_weight, grads)
                else:
                    acc_grads = jax.tree.map(
                        lambda acc, g: acc + g * grad_weight, acc_grads, grads
                    )
                del grads  # Free gradient immediately after accumulation

                per_head_metrics_list.append({
                    'loss': loss_i,
                    'energy_loss': aux_i['energy_loss'],
                    'force_loss': aux_i['force_loss'],
                    'stress_loss': aux_i['stress_loss'],
                    'grad_norm': grad_norm_i,
                    'n_graphs': aux_i['n_graphs'],
                    'n_atoms': aux_i['n_atoms'],
                })

            # Normalize accumulated gradients by total weight
            acc_grads = jax.tree.map(lambda g: g / total_grad_weight, acc_grads)

            if is_first_step and jax.process_index() == 0:
                print(f"  [JIT] simple_optimizer_step compiling...", flush=True)
                t0 = time.time()

            params, opt_state, ema_params, step, combined_grad_norm = \
                simple_optimizer_step(
                    params, opt_state, schedule_state, ema_params, step,
                    acc_grads,
                )
            del acc_grads

            if is_first_step and jax.process_index() == 0:
                print(f"  [JIT] simple_optimizer_step done in {time.time() - t0:.1f}s",
                      flush=True)
                print(f"  [JIT] First step total: {time.time() - step_start_time:.1f}s",
                      flush=True)
                is_first_step = False

            # Build metrics dict (compatible with existing logging code)
            metrics = {
                'combined_grad_norm': combined_grad_norm,
                'per_head_loss': jnp.stack([m['loss'] for m in per_head_metrics_list]),
                'per_head_energy_loss': jnp.stack([m['energy_loss'] for m in per_head_metrics_list]),
                'per_head_force_loss': jnp.stack([m['force_loss'] for m in per_head_metrics_list]),
                'per_head_stress_loss': jnp.stack([m['stress_loss'] for m in per_head_metrics_list]),
                'per_head_grad_norm': jnp.stack([m['grad_norm'] for m in per_head_metrics_list]),
                'per_head_n_graphs': jnp.stack([m['n_graphs'] for m in per_head_metrics_list]),
                'per_head_n_atoms': jnp.stack([m['n_atoms'] for m in per_head_metrics_list]),
            }

            step_in_epoch += 1
            current_step = int(np.asarray(step))

            # Log batch info (uses pre-extracted CPU metadata, no GPU tensors)
            for head_cfg, stream, per_device_metadata, graph_indices in head_log_meta:
                batch_logger.log_batch(
                    epoch, 'train', Path(stream._current_file).name,
                    stream.batches_yielded, per_device_metadata,
                    head_name=head_cfg["name"],
                    graph_indices_per_device=graph_indices,
                )

            if current_step % log_every == 0:
                elapsed = time.time() - epoch_step_start
                unrep_schedule = unreplicate_pytree(schedule_state)
                lr = config['learning_rate'] * float(unrep_schedule.scale)
                print(f"step {current_step} (epoch {epoch+1}/{config['n_epochs']}, "
                      f"{elapsed:.1f}s, LR={lr:.2e})", file=fout)
                for i, head_cfg in enumerate(head_configs):
                    stream = streams[head_cfg["head_idx"]]
                    cycle_str = f" cyc={stream.cycle_count}" if stream.cycle_count > 0 else ""
                    file_str = f" file={stream.files_consumed}/{stream.n_files}"
                    print(
                        f"  [{head_cfg['name']}{cycle_str}{file_str}] "
                        f"loss={float(metrics['per_head_loss'][i]):.6f} "
                        f"E={float(metrics['per_head_energy_loss'][i]):.6f} "
                        f"F={float(metrics['per_head_force_loss'][i]):.6f} "
                        f"S={float(metrics['per_head_stress_loss'][i]):.6f} "
                        f"grad={float(metrics['per_head_grad_norm'][i]):.4f}",
                        file=fout,
                    )
                print(f"  [combined] grad_norm="
                      f"{float(metrics['combined_grad_norm']):.4f}", file=fout)

            if current_step % 500 == 0 and current_step > 0:
                sync_and_check(mesh, f"periodic_sync_{current_step}")

            # Mid-epoch validation every val_every_steps
            if (val_every_steps > 0 and current_step > 0 and
                    current_step - last_val_step >= val_every_steps):
                last_val_step = current_step
                print(f"\n--- Mid-epoch validation at step {current_step} ---",
                      file=fout)

                # Save latest checkpoint
                mid_unrep_params = unreplicate_pytree(params)
                mid_unrep_params = multihost_utils.broadcast_one_to_all(
                    mid_unrep_params)
                mid_unrep_ema = unreplicate_pytree(ema_params)
                mid_unrep_ema = multihost_utils.broadcast_one_to_all(
                    mid_unrep_ema)
                mid_unrep_schedule = unreplicate_pytree(schedule_state)
                mid_unrep_schedule = multihost_utils.broadcast_one_to_all(
                    mid_unrep_schedule)
                latest_state = {
                    'params': mid_unrep_params,
                    'ema_params': mid_unrep_ema,
                    'opt_state': unreplicate_pytree(opt_state),
                    'schedule_state': mid_unrep_schedule,
                    'step': current_step,
                    'best_val_loss': best_val_loss,
                    'epoch': epoch, 'step_in_epoch': step_in_epoch,
                    'epoch_completed': False,
                    'num_heads': num_heads,
                    'head_configs': head_configs,
                    'foundation_ckpt': config.get('foundation_ckpt'),
                    'atom_energies': config['atom_energies'],
                    'atom_energies_path': config.get('atom_energies_path'),
                    'atomic_number_to_index': config.get('atomic_number_to_index'),
                    'config': config,
                }
                save_checkpoint_safe(latest_state,
                                     str(ckpt_dir / 'ckpt_latest.pkl'))

                # Run validation
                sync_and_check(mesh, f"pre_midval_step_{current_step}")
                val_start = time.time()

                mid_per_head_metrics, mid_val_loss = evaluate_per_head(
                    graphdef=graphdef, params=mid_unrep_ema,
                    head_configs=head_configs, batch_size=config['batch_size'],
                    n_devices=n_devices, mesh=mesh,
                    energy_weight=energy_weight, force_weight=force_weight,
                    stress_weight=stress_weight, loss_fn=loss_fn,
                    num_heads=num_heads, batch_logger=batch_logger,
                    epoch=epoch,
                    eval_local_mesh=eval_local_mesh,
                    eval_step_fn=eval_step_fn,
                    head_weights=head_weights,
                )
                val_time = time.time() - val_start

                if jax.process_count() > 1:
                    mid_val_loss = float(multihost_utils.broadcast_one_to_all(
                        jnp.array(mid_val_loss)))

                print(f"  Validation ({val_time:.1f}s) | "
                      f"Total Loss: {mid_val_loss:.6f}", file=fout)
                for hname, hmetrics in mid_per_head_metrics.items():
                    print(f"    [{hname}] Loss: {hmetrics['total_loss']:.6f} | "
                          f"E: {hmetrics['energy_loss']:.6f} | "
                          f"F: {hmetrics['force_loss']:.6f} | "
                          f"S: {hmetrics['stress_loss']:.6f} | "
                          f"E_MAE: {hmetrics['energy_mae']:.6f} | "
                          f"F_MAE: {hmetrics['force_mae']:.6f} | "
                          f"S_MAE: {hmetrics['stress_mae']:.6f} | "
                          f"E_RMSE: {hmetrics['energy_rmse']:.6f} | "
                          f"F_RMSE: {hmetrics['force_rmse']:.6f} | "
                          f"S_RMSE: {hmetrics['stress_rmse']:.6f}", file=fout)

                # Update LR schedule
                _, new_schedule = schedule.update(
                    updates=mid_unrep_params, state=mid_unrep_schedule,
                    value=mid_val_loss)
                new_schedule = multihost_utils.broadcast_one_to_all(
                    new_schedule)
                schedule_state = replicate_pytree(new_schedule, mesh)

                # Save best checkpoint
                if mid_val_loss < best_val_loss:
                    improvement = best_val_loss - mid_val_loss
                    best_val_loss = mid_val_loss
                    best_state = {
                        'params': mid_unrep_params,
                        'ema_params': mid_unrep_ema,
                        'opt_state': unreplicate_pytree(opt_state),
                        'schedule_state': new_schedule,
                        'step': current_step,
                        'epoch': epoch,
                        'best_val_loss': best_val_loss,
                        'metrics': mid_per_head_metrics,
                        'num_heads': num_heads,
                        'head_configs': head_configs,
                        'foundation_ckpt': config.get('foundation_ckpt'),
                        'atom_energies': config['atom_energies'],
                        'atom_energies_path': config.get('atom_energies_path'),
                        'atomic_number_to_index': config.get('atomic_number_to_index'),
                        'config': config,
                    }
                    print(f"  *** New best: {best_val_loss:.6f} "
                          f"(+{improvement:.6f}) ***", file=fout)
                    save_checkpoint_safe(best_state,
                                         str(ckpt_dir / 'ckpt_best.pkl'))
                else:
                    print(f"  No improvement (best: {best_val_loss:.6f})",
                          file=fout)

                print(f"--- End mid-epoch validation ---\n", file=fout)
                del mid_unrep_params, mid_unrep_ema, mid_unrep_schedule

            # Collective epoch-done check: all nodes must agree to stop
            local_done = streams[longest_head].completed_first_pass
            if is_multihost:
                done_arr = multihost_utils.process_allgather(
                    jnp.array([int(local_done)], dtype=jnp.int32),
                    tiled=False,
                )
                # Stop when ANY node has finished (prevents desync)
                if int(jax.device_get(done_arr).max()) > 0:
                    break
            else:
                if local_done:
                    break

        # Stream statistics
        for h, stream in streams.items():
            hname = next(hc["name"] for hc in head_configs if hc["head_idx"] == h)
            print(f"  [{hname}] files={stream.files_consumed}/{stream.n_files} "
                  f"cycles={stream.cycle_count} batches={stream.batches_yielded}",
                  file=fout)

        # Validation
        sync_and_check(mesh, f"pre_val_epoch_{epoch+1}")
        val_start = time.time()
        unrep_ema = unreplicate_pytree(ema_params)
        unrep_ema = multihost_utils.broadcast_one_to_all(unrep_ema)

        per_head_metrics, val_loss = evaluate_per_head(
            graphdef=graphdef, params=unrep_ema,
            head_configs=head_configs, batch_size=config['batch_size'],
            n_devices=n_devices, mesh=mesh,
            energy_weight=energy_weight, force_weight=force_weight,
            stress_weight=stress_weight, loss_fn=loss_fn,
            num_heads=num_heads, batch_logger=batch_logger, epoch=epoch,
            eval_local_mesh=eval_local_mesh,
            eval_step_fn=eval_step_fn,
            head_weights=head_weights,
        )
        val_time = time.time() - val_start

        if jax.process_count() > 1:
            val_loss = float(multihost_utils.broadcast_one_to_all(
                jnp.array(val_loss)))

        print(f"\n  Validation ({val_time:.1f}s) | Total Loss: {val_loss:.6f}",
              file=fout)
        for hname, hmetrics in per_head_metrics.items():
            print(f"    [{hname}] Loss: {hmetrics['total_loss']:.6f} | "
                  f"E: {hmetrics['energy_loss']:.6f} | "
                  f"F: {hmetrics['force_loss']:.6f} | "
                  f"S: {hmetrics['stress_loss']:.6f} | "
                  f"E_MAE: {hmetrics['energy_mae']:.6f} | "
                  f"F_MAE: {hmetrics['force_mae']:.6f} | "
                  f"S_MAE: {hmetrics['stress_mae']:.6f} | "
                  f"E_RMSE: {hmetrics['energy_rmse']:.6f} | "
                  f"F_RMSE: {hmetrics['force_rmse']:.6f} | "
                  f"S_RMSE: {hmetrics['stress_rmse']:.6f}", file=fout)

        # Update LR schedule
        unrep_params = unreplicate_pytree(params)
        unrep_params = multihost_utils.broadcast_one_to_all(unrep_params)
        unrep_sched = unreplicate_pytree(schedule_state)
        unrep_sched = multihost_utils.broadcast_one_to_all(unrep_sched)
        _, new_schedule = schedule.update(
            updates=unrep_params, state=unrep_sched, value=val_loss)
        new_schedule = multihost_utils.broadcast_one_to_all(new_schedule)
        schedule_state = replicate_pytree(new_schedule, mesh)

        if val_loss < best_val_loss:
            improvement = best_val_loss - val_loss
            best_val_loss = val_loss
            best_state = {
                'params': unrep_params,
                'ema_params': unrep_ema,
                'opt_state': unreplicate_pytree(opt_state),
                'schedule_state': new_schedule,
                'step': int(np.asarray(step)),
                'epoch': epoch,
                'best_val_loss': best_val_loss,
                'metrics': per_head_metrics,
                'num_heads': num_heads,
                'head_configs': head_configs,
                'foundation_ckpt': config.get('foundation_ckpt'),
                'atom_energies': config['atom_energies'],
                'atom_energies_path': config.get('atom_energies_path'),
                'atomic_number_to_index': config.get('atomic_number_to_index'),
                'config': config,
            }
            print(f"  *** New best: {best_val_loss:.6f} "
                  f"(+{improvement:.6f}) ***", file=fout)
            save_checkpoint_safe(best_state,
                                 str(ckpt_dir / 'ckpt_best.pkl'))
        else:
            print(f"  No improvement (best: {best_val_loss:.6f})", file=fout)

        # Save latest checkpoint at epoch end with epoch_completed flag
        epoch_end_state = {
            'params': unrep_params,
            'ema_params': unrep_ema,
            'opt_state': unreplicate_pytree(opt_state),
            'schedule_state': new_schedule,
            'step': int(np.asarray(step)),
            'best_val_loss': best_val_loss,
            'epoch': epoch,
            'epoch_completed': True,
            'num_heads': num_heads,
            'head_configs': head_configs,
            'foundation_ckpt': config.get('foundation_ckpt'),
            'atom_energies': config['atom_energies'],
            'atom_energies_path': config.get('atom_energies_path'),
            'atomic_number_to_index': config.get('atomic_number_to_index'),
            'config': config,
        }
        save_checkpoint_safe(epoch_end_state,
                             str(ckpt_dir / 'ckpt_latest.pkl'))

        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config['n_epochs']} completed in "
              f"{epoch_time:.1f}s\n", file=fout)

    if best_state is not None:
        save_checkpoint_safe(best_state, str(ckpt_dir / 'ckpt_best.pkl'))


# =============================================================================
# Main Orchestrator
# =============================================================================

def train_unified(config: Dict):
    """Main entry point. Dispatches to single-head or multihead training loop."""
    is_multihead = config.get('multihead', False)
    ckpt_dir = Path(config.get('ckpt_dir', './checkpoints'))
    if jax.process_index() == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Setup mesh (1D DP)
    mesh, n_devices = setup_mesh()

    # Logging (process 0 only)
    if jax.process_index() == 0:
        fout = open(config.get('fname_log', 'loss.out'), 'w', 1)
    else:
        fout = open('/dev/null', 'w')

    batch_logger = BatchLogger(
        process_id=jax.process_index(),
        n_processes=jax.process_count(),
        n_local_devices=jax.local_device_count(),
        log_root=config.get('batch_log_root', 'batch_logs'),
    )
    batch_logger.log_gpu(f"Process {jax.process_index()}/{jax.process_count()} started")

    # Print header
    mode_str = "Multihead" if is_multihead else "Single-head"
    print("=" * 70, file=fout)
    print(f"JAX Unified {mode_str} Training", file=fout)
    print(f"  Total devices: {n_devices}", file=fout)
    print(f"  Processes: {jax.process_count()}", file=fout)
    print(f"  Batch size per device: {config['batch_size']}", file=fout)
    print(f"  Effective batch size: {config['batch_size'] * n_devices}", file=fout)
    if is_multihead:
        for hc in config["heads"]:
            print(f"  Head {hc['head_idx']}: {hc['name']}", file=fout)
    print("=" * 70, file=fout)

    # Skip foundation_ckpt if restarting from existing checkpoint
    skip_foundation = False
    if config.get('restart', False):
        resume_from = Path(config.get('resume_from', ckpt_dir))
        for ckpt_name in ['ckpt_latest.pkl', 'ckpt_best.pkl', 'ckpt_multihead_best.pkl']:
            if (resume_from / ckpt_name).exists():
                skip_foundation = True
                print(f"Restart mode: skipping foundation_ckpt (will load from {resume_from})", file=fout)
                break

    # Create model
    config_for_model = config.copy()
    if skip_foundation:
        config_for_model['foundation_ckpt'] = None
    graphdef, params, load_info = create_model(config_for_model, is_multihead)
    ema_params = params
    n_params = sum(x.size for x in jax.tree.leaves(params)) # 총 파라미터 개수 계산
    print(f"Model parameters: {n_params:,}", file=fout)

    # Foundation model provenance
    if is_multihead and load_info is not None:
        total = load_info['total_size']
        copied = load_info['copied_size']
        expanded = load_info['repeated_size']
        skipped = load_info['skipped_size']
        cp = 100 * copied / total if total > 0 else 0
        ep = 100 * expanded / total if total > 0 else 0
        sp = 100 * skipped / total if total > 0 else 0
        print("-" * 70, file=fout)
        print("Foundation model loaded:", file=fout)
        print(f"  Path: {load_info['ckpt_path']}", file=fout)
        if load_info['source_epoch'] is not None:
            print(f"  Trained epochs: {load_info['source_epoch']}", file=fout)
        if load_info['source_step'] is not None:
            print(f"  Trained steps: {load_info['source_step']}", file=fout)
        if load_info['source_val_loss'] is not None:
            print(f"  Best val_loss: {load_info['source_val_loss']:.6f}", file=fout)
        print(f"  Used EMA params: {load_info['used_ema']}", file=fout)
        num_heads = len(config['heads'])
        print(f"  Param transfer (total {total:,}):", file=fout)
        print(f"    Backbone (direct copy):      {copied:>13,} ({cp:.2f}%)", file=fout)
        print(f"    Readout  (1-head -> {num_heads}-head): {expanded:>13,} ({ep:.2f}%)", file=fout)
        if skipped > 0:
            print(f"    NOT matched (random init):   {skipped:>13,} ({sp:.2f}%) *** WARNING ***", file=fout)
        print("-" * 70, file=fout)
    elif is_multihead:
        print("Model initialized from scratch (no foundation_ckpt)", file=fout)

    # Optimizer
    optimizer, schedule = create_optimizer(config)
    opt_state = optimizer.init(params)
    schedule_state = schedule.init(params)

    # Broadcast initial state (multi-host)
    # broadcast_one_to_all : 동일한 가중치를 각 노드에 복사
    if jax.process_count() > 1:
        params = multihost_utils.broadcast_one_to_all(params)
        ema_params = multihost_utils.broadcast_one_to_all(ema_params)
        opt_state = multihost_utils.broadcast_one_to_all(opt_state)
        schedule_state = multihost_utils.broadcast_one_to_all(schedule_state)

    # Replicate across devices
    # Replicate : 동일한 가중치를 각 지피유에 복사
    params = replicate_pytree(params, mesh)
    ema_params = replicate_pytree(ema_params, mesh)
    opt_state = replicate_pytree(opt_state, mesh)
    schedule_state = replicate_pytree(schedule_state, mesh)
    step = replicate(jnp.array(0), mesh)

    # Loss function
    loss_type = config.get('loss_type', 'mse')
    if loss_type == 'huber':
        huber_delta = config.get('huber_delta', 0.02)
        loss_fn = partial(LOSS_FUNCTIONS[loss_type], delta=huber_delta)
    else:
        loss_fn = LOSS_FUNCTIONS[loss_type]

    # Load checkpoint if restarting
    best_val_loss = float('inf')
    start_epoch = 0
    start_ipkl = 0

    if config.get('restart', False):
        ckpt_loaded = False
        ckpt_data = None
        # resume_from: checkpoint directory to load from (defaults to ckpt_dir)
        resume_from = Path(config.get('resume_from', ckpt_dir))

        if jax.process_index() == 0:
            if config.get('reset_best_loss', False):
                ckpt_order = ['ckpt_best.pkl', 'ckpt_latest.pkl',
                              'ckpt_multihead_best.pkl']
            else:
                ckpt_order = ['ckpt_latest.pkl', 'ckpt_best.pkl',
                              'ckpt_multihead_best.pkl']
            for ckpt_name in ckpt_order:
                ckpt_path = resume_from / ckpt_name
                if ckpt_path.exists():
                    ckpt_data = load_checkpoint(str(ckpt_path))
                    ckpt_loaded = True
                    print(f"Loaded checkpoint from {ckpt_path}", file=fout)
                    break

        if jax.process_count() > 1:
            ckpt_flag = multihost_utils.broadcast_one_to_all(
                jnp.array(1 if ckpt_loaded else 0))
            ckpt_loaded = int(ckpt_flag) > 0

        if ckpt_loaded:
            if jax.process_index() != 0:
                # Non-coordinator: create placeholder
                ckpt_data = {
                    'params': unreplicate_pytree(params),
                    'ema_params': unreplicate_pytree(ema_params),
                    'opt_state': unreplicate_pytree(opt_state),
                    'schedule_state': unreplicate_pytree(schedule_state),
                }

            # Broadcast checkpoint from process 0
            loaded_params = multihost_utils.broadcast_one_to_all(
                ckpt_data['params'])
            loaded_ema = multihost_utils.broadcast_one_to_all(
                ckpt_data['ema_params'])
            loaded_opt = multihost_utils.broadcast_one_to_all(
                ckpt_data['opt_state'])
            loaded_schedule = multihost_utils.broadcast_one_to_all(
                ckpt_data['schedule_state'])

            params = replicate_pytree(loaded_params, mesh)
            ema_params = replicate_pytree(loaded_ema, mesh)
            opt_state = replicate_pytree(loaded_opt, mesh)
            schedule_state = replicate_pytree(loaded_schedule, mesh)

            # Broadcast scalar values
            ckpt_step = int(multihost_utils.broadcast_one_to_all(
                jnp.array(ckpt_data.get('step', 0) if jax.process_index() == 0 else 0)))
            best_val_loss = float(multihost_utils.broadcast_one_to_all(
                jnp.array(ckpt_data.get('best_val_loss', float('inf')) if jax.process_index() == 0 else 0.0)))
            ckpt_epoch = int(multihost_utils.broadcast_one_to_all(
                jnp.array(ckpt_data.get('epoch', 0) if jax.process_index() == 0 else 0)))
            ckpt_ipkl = int(multihost_utils.broadcast_one_to_all(
                jnp.array(ckpt_data.get('ipkl', 0) if jax.process_index() == 0 else 0)))

            step = replicate(jnp.array(ckpt_step), mesh)
            start_epoch = ckpt_epoch

            if not is_multihead:
                start_ipkl = ckpt_ipkl + 1
                if start_ipkl >= len(_get_pkl_files(config['train_path'])):
                    start_epoch += 1
                    start_ipkl = 0
            else:
                # If epoch was completed, start next epoch;
                # otherwise restart the same epoch from the beginning
                epoch_completed = ckpt_data.get('epoch_completed', False) if jax.process_index() == 0 else False
                if jax.process_count() > 1:
                    epoch_completed = bool(int(multihost_utils.broadcast_one_to_all(
                        jnp.array(1 if epoch_completed else 0))))
                if epoch_completed:
                    start_epoch = ckpt_epoch + 1
                else:
                    start_epoch = ckpt_epoch
                    print(f"Epoch {ckpt_epoch+1} was incomplete, restarting from beginning of epoch",
                          file=fout)

            print(f"Resumed from step {ckpt_step} (epoch {start_epoch})",
                  file=fout)

            # Reset best_val_loss when loss weights change across restarts
            if config.get('reset_best_loss', False):
                if jax.process_index() == 0:
                    best_ckpt = ckpt_dir / 'ckpt_best.pkl'
                    if best_ckpt.exists():
                        backup_path = ckpt_dir / 'ckpt_best_backup.pkl'
                        shutil.copy2(str(best_ckpt), str(backup_path))
                        print(f"Backed up {best_ckpt} -> {backup_path}",
                              file=fout)
                best_val_loss = float('inf')
                print(f"Reset best_val_loss to inf (reset_best_loss=true)",
                      file=fout)

    # Dispatch to training loop
    if is_multihead:
        head_configs = config['heads']
        energy_weight = config.get('energy_weight', 1.0)
        force_weight = config.get('force_weight', 1.0)
        stress_weight = config.get('stress_weight', 1.0)
        head_weights = parse_head_weights(
            head_configs, energy_weight, force_weight, stress_weight)

        _train_loop_multihead(
            config, mesh, n_devices, graphdef, params, ema_params,
            opt_state, schedule_state, step, optimizer, schedule, loss_fn,
            head_configs, head_weights, batch_logger, fout,
            start_epoch, best_val_loss, ckpt_dir,
        )
    else:
        train_files = _get_pkl_files(config['train_path'])
        valid_files = _get_pkl_files(config['valid_path'])
        print(f"Training files: {len(train_files)}", file=fout)
        print(f"Validation files: {len(valid_files)}", file=fout)

        _train_loop_singlehead(
            config, mesh, n_devices, graphdef, params, ema_params,
            opt_state, schedule_state, step, optimizer, schedule, loss_fn,
            train_files, valid_files, batch_logger, fout,
            start_epoch, start_ipkl, best_val_loss, ckpt_dir,
        )

    # Cleanup
    print("\nTraining completed!", file=fout)
    sync_and_check(mesh, "training_complete")
    batch_logger.log_gpu("Training completed")
    batch_logger.close()
    get_debug_logger().close()
    fout.close()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import json

    # Initialize distributed runtime FIRST
    try:
        initialize_distributed()
    except Exception as e:
        print(f"FATAL: Failed to initialize distributed runtime: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Only process 0 prints startup info
    if jax.process_index() == 0:
        print("=" * 70)
        print("JAX Unified Training (Single-head / Multihead)")
        print("=" * 70)
        print(f"JAX version: {jax.__version__}")
        print(f"Process: {jax.process_index()} / {jax.process_count()}")
        print(f"Local devices: {jax.local_device_count()}")
        print(f"Total devices: {jax.device_count()}")
        print("=" * 70)

    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'input.json'
    if jax.process_index() == 0:
        print(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    # Load atom energies
    atom_energies_path = config.get('atom_energies_path', None)
    if atom_energies_path and Path(atom_energies_path).exists():
        if jax.process_index() == 0:
            print(f"Loading atom energies from {atom_energies_path}")
        with open(atom_energies_path) as f:
            ae_data = json.load(f)
        if not isinstance(ae_data, dict):
            raise ValueError(
                f"atom_energies file {atom_energies_path} is old format (list). "
                "Regenerate with updated build_pkl_multihead.py using --fit-energies"
            )
        if 'atomic_number_to_index' not in ae_data:
            raise ValueError(
                f"atom_energies file {atom_energies_path} missing 'atomic_number_to_index'. "
                "Regenerate with updated build_pkl_multihead.py using --fit-energies"
            )
        config['atom_energies'] = ae_data['atom_energies']
        config['atomic_number_to_index'] = {
            int(k): v for k, v in ae_data['atomic_number_to_index'].items()
        }
    else:
        raise ValueError(
            "atom_energies_path not specified or file does not exist. "
            "Set 'atom_energies_path' in config JSON to the path of atom_energies.json"
        )

    mode_str = "Multihead" if config.get('multihead', False) else "Single-head"
    if jax.process_index() == 0:
        print(f"Mode: {mode_str}")

    try:
        train_unified(config)
    except Exception as e:
        debug_log(f"FATAL: {e}", "ERROR")
        debug_log(f"Traceback:\n{traceback.format_exc()}", "ERROR")
        try:
            multihost_utils.sync_global_devices("fatal_error_sync")
        except:
            pass
        get_debug_logger().close()
        sys.exit(1)
