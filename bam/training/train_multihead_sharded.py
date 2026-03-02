"""Multihead RACE multi-GPU training with gradient accumulation.

Fine-tunes a pre-trained RACE model with multiple heads for different datasets.
Each head corresponds to a dataset (e.g., OMat24 target + MPTrj replay).

Every optimizer step draws one batch from each head, computes per-head gradients
with per-head E/F/S weights, accumulates them with configurable grad_weight,
and applies a single optimizer update. This prevents gradient oscillation
and catastrophic forgetting compared to file-level interleaving.

Validation is per-head. Based on train_sharded.py with multihead extensions.
"""

from typing import Callable, Dict, Tuple, List, Any, Optional
from functools import partial
import os
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map
from flax import nnx
import optax
import jraph
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime

from bam.data.data_nnx import BucketedDataLoader, MultiDeviceDataLoader
from bam.data.data_multihead_nnx import DatasetWithHead
from bam.models.race_multihead_nnx import RACEMultihead, load_foundation_as_multihead
from bam.training.losses import LOSS_FUNCTIONS
from bam.training.sharding import (
    setup_mesh, replicate, replicate_pytree,
    unreplicate, unreplicate_pytree, squeeze_batch,
    save_checkpoint, load_checkpoint,
)

jax.config.update("jax_enable_x64", False)


# =============================================================================
# Batch Logging Utilities
# =============================================================================

class BatchLogger:
    """Logger for tracking batch sizes per GPU."""

    def __init__(self, n_devices: int, log_root: str = "batch_logs"):
        self.n_devices = n_devices
        self.log_root = Path(log_root)
        self.cumulative_graphs = {i: 0 for i in range(n_devices)}
        self.global_base_offset = {i: 0 for i in range(n_devices)}

        self.gpu_logs = {}
        self.batch_logs = {}

        for gpu_id in range(n_devices):
            log_dir = self.log_root / f"rank_{gpu_id}"
            log_dir.mkdir(parents=True, exist_ok=True)

            gpu_log_path = log_dir / f"gpu{gpu_id}.log"
            self.gpu_logs[gpu_id] = open(gpu_log_path, 'w')

            batch_log_path = log_dir / "batch_sizes.log"
            self.batch_logs[gpu_id] = open(batch_log_path, 'w')
            self.batch_logs[gpu_id].write(
                f"# Batch size log for GPU {gpu_id}\n"
            )
            self.batch_logs[gpu_id].write(
                "# timestamp,epoch,mode,head,file,batch_idx,n_graphs,n_nodes,n_edges\n"
            )
            self.batch_logs[gpu_id].flush()

            self._log_gpu_single(gpu_id, f"Initialized - GPU: {gpu_id}")

        print(f"Created batch logs for {n_devices} GPUs in {log_root}/")

    def _log_gpu_single(self, gpu_id: int, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.gpu_logs[gpu_id].write(f"[{timestamp}] {message}\n")
        self.gpu_logs[gpu_id].flush()

    def log_gpu(self, message: str):
        for gpu_id in range(self.n_devices):
            self._log_gpu_single(gpu_id, message)

    def log_batch(self, epoch: int, mode: str, head_name: str, file_name: str,
                  batch_idx: int,
                  per_device_metadata: List[Tuple[int, int, int]]):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        for gpu_id in range(self.n_devices):
            if gpu_id < len(per_device_metadata):
                n_graphs, n_nodes, n_edges = per_device_metadata[gpu_id]
            else:
                n_graphs, n_nodes, n_edges = 0, 0, 0

            line = (f"{timestamp},{epoch},{mode},{head_name},{file_name},{batch_idx},"
                    f"{n_graphs},{n_nodes},{n_edges}\n")
            self.batch_logs[gpu_id].write(line)
            self.batch_logs[gpu_id].flush()

    def reset_counters(self):
        for i in range(self.n_devices):
            self.cumulative_graphs[i] = 0

    def set_total_graphs(self, total_graphs: int):
        graphs_per_gpu = total_graphs // self.n_devices
        for gpu_id in range(self.n_devices):
            self.global_base_offset[gpu_id] = gpu_id * graphs_per_gpu

    def close(self):
        for gpu_id in range(self.n_devices):
            if gpu_id in self.gpu_logs and not self.gpu_logs[gpu_id].closed:
                self.gpu_logs[gpu_id].close()
            if gpu_id in self.batch_logs and not self.batch_logs[gpu_id].closed:
                self.batch_logs[gpu_id].close()
        print("Closed all batch log files")


def extract_per_device_batch_metadata(
    batch, n_devices: int
) -> List[Tuple[int, int, int]]:
    """Extract per-device (n_graphs, n_nodes, n_edges) from addressable shards only."""
    per_device_metadata = []

    # Multi-host: only access addressable (local) shards
    if hasattr(batch.n_node, 'addressable_shards'):
        for shard_n, shard_e in zip(
            batch.n_node.addressable_shards, batch.n_edge.addressable_shards
        ):
            device_n_node = np.asarray(shard_n.data).reshape(-1)
            device_n_edge = np.asarray(shard_e.data).reshape(-1)

            n_graphs = int(device_n_node.shape[0]) - 1
            n_nodes = int(np.sum(device_n_node[:-1]))
            n_edges = int(np.sum(device_n_edge[:-1]))

            per_device_metadata.append((n_graphs, n_nodes, n_edges))
    else:
        n_node_array = np.asarray(batch.n_node)
        n_edge_array = np.asarray(batch.n_edge)

        for i in range(n_devices):
            device_n_node = n_node_array[i]
            device_n_edge = n_edge_array[i]

            n_graphs = int(device_n_node.shape[0]) - 1
            n_nodes = int(np.sum(device_n_node[:-1]))
            n_edges = int(np.sum(device_n_edge[:-1]))

            per_device_metadata.append((n_graphs, n_nodes, n_edges))

    return per_device_metadata


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
    """Compute loss for a multihead batch.

    Same unified loss as base RACE, plus per-head metrics computed
    using one_hot masking (JIT-compatible).
    """
    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask = jraph.get_node_padding_mask(batch)

    n_graphs = jnp.maximum(graph_mask.sum(), 1)
    n_atoms = jnp.maximum(node_mask.sum(), 1)

    model = nnx.merge(graphdef, params)
    energy, forces, stress = model(batch)

    # Energy loss (per-graph, normalized by n_node)
    energy_diff = (energy - batch.globals["energy"]) / batch.n_node
    energy_loss = loss_fn(energy_diff)
    energy_loss = jnp.sum(energy_loss * graph_mask) / n_graphs
    energy_mse = jnp.sum(energy_diff ** 2 * graph_mask) / n_graphs

    # Force loss
    forces_diff = forces - batch.nodes["forces"]
    force_loss = loss_fn(forces_diff)
    force_loss = jnp.sum(force_loss * node_mask[:, None]) / (3 * n_atoms)
    force_mse = jnp.sum(forces_diff ** 2 * node_mask[:, None]) / (3 * n_atoms)

    # Stress loss
    stress_diff = (stress - batch.globals["stress"]) * graph_mask[:, None]
    stress_loss = jnp.sum(
        loss_fn(stress_diff) * graph_mask[:, None]
    ) / n_graphs
    stress_mse = jnp.sum(stress_diff ** 2 * graph_mask[:, None]) / n_graphs

    total_loss = (
        energy_weight * energy_loss
        + force_weight * force_loss
        + stress_weight * stress_loss
    )

    # Per-head metrics (vectorized with one_hot, JIT-compatible)
    head_per_graph = batch.globals["head"]  # (n_graphs,)
    head_one_hot = jax.nn.one_hot(head_per_graph, num_heads)  # (n_graphs, num_heads)
    head_graph_counts = jnp.sum(
        head_one_hot * graph_mask[:, None], axis=0
    )  # (num_heads,)

    # Per-head energy MAE
    per_head_energy_ae = jnp.sum(
        jnp.abs(energy_diff)[:, None] * head_one_hot * graph_mask[:, None], axis=0
    )  # (num_heads,)

    # Per-head force MAE requires node-level head info
    sum_n_node = node_mask.shape[0]
    n_graphs_total = batch.n_node.shape[0]
    graph_idx = jnp.arange(n_graphs_total)
    node_graph = jnp.repeat(
        graph_idx, batch.n_node, axis=0, total_repeat_length=sum_n_node
    )
    node_head = head_per_graph[node_graph]  # (n_atoms,)
    node_head_one_hot = jax.nn.one_hot(node_head, num_heads)  # (n_atoms, num_heads)
    head_atom_counts = jnp.sum(
        node_head_one_hot * node_mask[:, None], axis=0
    )  # (num_heads,)
    per_head_force_ae = jnp.sum(
        jnp.sum(jnp.abs(forces_diff), axis=-1, keepdims=True)
        * node_head_one_hot
        * node_mask[:, None],
        axis=0,
    )  # (num_heads,)

    aux = {
        'energy_loss': energy_loss,
        'force_loss': force_loss,
        'stress_loss': stress_loss,
        'energy_mse': energy_mse,
        'force_mse': force_mse,
        'stress_mse': stress_mse,
        'n_graphs': n_graphs,
        'n_atoms': n_atoms,
        # Per-head sums (divide by counts later for MAE)
        'per_head_energy_ae': per_head_energy_ae,
        'per_head_force_ae': per_head_force_ae,
        'per_head_graph_counts': head_graph_counts,
        'per_head_atom_counts': head_atom_counts,
    }

    return total_loss, aux


# =============================================================================
# Fused Multihead Train Step (gradient accumulation + optimizer in one XLA graph)
# =============================================================================

def make_fused_multihead_train_step(
    optimizer, mesh, loss_fn, num_heads, num_head_configs, ema_decay=0.99,
):
    """Fused gradient accumulation + optimizer inside one shard_map.

    Takes one batch per head, computes per-head gradients with per-head E/F/S
    weights, accumulates with grad_weight, and applies a single optimizer step.
    All within one XLA computation so XLA can pipeline gradient computation
    with optimizer updates and reuse activation memory across heads.

    Args:
        num_heads: Number of heads in the model (for one_hot in loss).
        num_head_configs: Number of head configs (= number of batches per step).
    """
    N_ARGS_PER_HEAD = 5  # batch, energy_w, force_w, stress_w, grad_w

    def per_device_step(graphdef, params, opt_state, schedule_state,
                        ema_params, step, *head_args):
        acc_grads = None
        total_gw = jnp.float32(0.0)
        per_head_losses = []
        per_head_e_losses = []
        per_head_f_losses = []
        per_head_s_losses = []
        per_head_grad_norms = []
        per_head_n_graphs = []
        per_head_n_atoms = []

        for i in range(num_head_configs):
            base = i * N_ARGS_PER_HEAD
            batch_i = squeeze_batch(head_args[base])
            e_w = head_args[base + 1]
            f_w = head_args[base + 2]
            s_w = head_args[base + 3]
            g_w = head_args[base + 4]

            def loss_fn_i(params, _b=batch_i, _ew=e_w, _fw=f_w, _sw=s_w):
                return compute_loss_multihead(
                    graphdef, params, _b, _ew, _fw, _sw, loss_fn, num_heads,
                )

            (loss_i, aux_i), grads_i = jax.value_and_grad(
                loss_fn_i, has_aux=True,
            )(params)

            # Weighted gradient accumulation
            weighted = jax.tree.map(lambda g: g * g_w, grads_i)
            if acc_grads is None:
                acc_grads = weighted
            else:
                acc_grads = jax.tree.map(lambda a, g: a + g, acc_grads, weighted)
            total_gw = total_gw + g_w

            per_head_losses.append(loss_i)
            per_head_e_losses.append(aux_i['energy_loss'])
            per_head_f_losses.append(aux_i['force_loss'])
            per_head_s_losses.append(aux_i['stress_loss'])
            per_head_grad_norms.append(optax.global_norm(grads_i))
            per_head_n_graphs.append(aux_i['n_graphs'])
            per_head_n_atoms.append(aux_i['n_atoms'])

        # Normalize and sync across devices
        acc_grads = jax.tree.map(lambda g: g / total_gw, acc_grads)
        acc_grads = jax.lax.pmean(acc_grads, axis_name='dp')
        combined_grad_norm = optax.global_norm(acc_grads)

        # Optimizer step
        updates, new_opt = optimizer.update(acc_grads, opt_state, params)
        updates = optax.tree_utils.tree_scale(schedule_state.scale, updates)
        new_p = optax.apply_updates(params, updates)
        new_ema = jax.tree.map(
            lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
            ema_params, new_p,
        )

        # Per-head metrics (stacked arrays, shape=(num_head_configs,))
        metrics = {
            'combined_grad_norm': combined_grad_norm,
            'per_head_loss': jax.lax.pmean(
                jnp.stack(per_head_losses), axis_name='dp'),
            'per_head_energy_loss': jax.lax.pmean(
                jnp.stack(per_head_e_losses), axis_name='dp'),
            'per_head_force_loss': jax.lax.pmean(
                jnp.stack(per_head_f_losses), axis_name='dp'),
            'per_head_stress_loss': jax.lax.pmean(
                jnp.stack(per_head_s_losses), axis_name='dp'),
            'per_head_grad_norm': jax.lax.pmean(
                jnp.stack(per_head_grad_norms), axis_name='dp'),
            'per_head_n_graphs': jax.lax.pmean(
                jnp.stack(per_head_n_graphs), axis_name='dp'),
            'per_head_n_atoms': jax.lax.pmean(
                jnp.stack(per_head_n_atoms), axis_name='dp'),
        }

        return new_p, new_opt, new_ema, step + 1, metrics

    # Build in_specs dynamically
    fixed_specs = (P(), P(), P(), P(), P(), P())  # graphdef,params,opt,sched,ema,step
    head_specs = tuple(
        spec
        for _ in range(num_head_configs)
        for spec in (P('dp'), P(), P(), P(), P())  # batch, e_w, f_w, s_w, g_w
    )

    sharded_fn = shard_map(
        per_device_step,
        mesh=mesh,
        in_specs=fixed_specs + head_specs,
        out_specs=(P(), P(), P(), P(), P()),
    )
    return jax.jit(sharded_fn)


# =============================================================================
# Sharded Evaluation (per-head)
# =============================================================================

def make_sharded_evaluate_step(mesh: Mesh, loss_fn: Callable, num_heads: int):
    """Create sharded evaluation function for multihead model."""

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
            'n_graphs': graph_mask.sum(),
            'n_atoms': node_mask.sum(),
        }

    def eval_and_reduce(graphdef, params, batch):
        local_metrics = eval_single_device(graphdef, params, batch)
        return jax.tree.map(
            lambda x: jax.lax.psum(x, axis_name='dp'), local_metrics
        )

    sharded_eval = shard_map(
        eval_and_reduce,
        mesh=mesh,
        in_specs=(P(), P(), P('dp')),
        out_specs=P(),
    )

    return jax.jit(sharded_eval)


def evaluate_per_head(
    graphdef: nnx.GraphDef,
    params: nnx.State,
    head_configs: List[Dict],
    batch_size: int,
    n_devices: int,
    mesh: Mesh,
    energy_weight: float,
    force_weight: float,
    stress_weight: float,
    loss_fn: Callable,
    num_heads: int,
    batch_logger: Optional[BatchLogger] = None,
    epoch: int = 0,
) -> Tuple[Dict, float]:
    """Evaluate model on each head's validation set separately.

    Returns:
        Tuple of (per_head_metrics_dict, total_val_loss).
    """
    params_replicated = replicate_pytree(params, mesh)
    eval_step = make_sharded_evaluate_step(mesh, loss_fn, num_heads)

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
            print(f"  [Head {head_name}] No validation files found in {valid_path}")
            continue

        totals = {
            'energy_loss': 0.0, 'force_loss': 0.0, 'stress_loss': 0.0,
            'energy_ae': 0.0, 'force_ae': 0.0, 'stress_ae': 0.0,
            'n_graphs': 0.0, 'n_atoms': 0.0,
        }

        for fname in tqdm(valid_files, desc=f"  Eval {head_name}", leave=False):
            file_name = Path(fname).name

            if batch_logger is not None:
                batch_logger.reset_counters()

            dataset = DatasetWithHead(file_path=fname, head_idx=head_idx)

            if batch_logger is not None:
                batch_logger.set_total_graphs(len(dataset))

            base_loader = BucketedDataLoader(
                dataset=dataset,
                batch_size=batch_size,
                n_buckets=8,
                shuffle=False,
                drop_last=False,
                rngs=nnx.Rngs(0),
            )
            loader = MultiDeviceDataLoader(
                base_loader=base_loader,
                n_devices=n_devices,
                mesh=mesh,
                drop_incomplete=False,
            )

            batch_idx = 0
            for batch, info in loader:
                batch_idx += 1
                metrics = eval_step(graphdef, params_replicated, batch)
                for k in totals:
                    totals[k] += float(metrics[k])

                if batch_logger is not None:
                    per_device_metadata = extract_per_device_batch_metadata(
                        batch, n_devices
                    )
                    batch_logger.log_batch(
                        epoch=epoch,
                        mode='valid',
                        head_name=head_name,
                        file_name=file_name,
                        batch_idx=batch_idx,
                        per_device_metadata=per_device_metadata,
                    )

        n_g = max(totals['n_graphs'], 1)
        n_a = max(totals['n_atoms'], 1)
        head_loss = (
            energy_weight * totals['energy_loss'] / n_g
            + force_weight * totals['force_loss'] / (3 * n_a)
            + stress_weight * totals['stress_loss'] / n_g
        )

        head_metrics = {
            'energy_mae': totals['energy_ae'] / n_g,
            'force_mae': totals['force_ae'] / (3 * n_a),
            'stress_mae': totals['stress_ae'] / n_g,
            'energy_loss': totals['energy_loss'] / n_g,
            'force_loss': totals['force_loss'] / (3 * n_a),
            'stress_loss': totals['stress_loss'] / n_g,
            'total_loss': head_loss,
            'n_graphs': n_g,
            'n_atoms': n_a,
        }

        all_head_metrics[head_name] = head_metrics

        # Weighted sum for total val loss (weight by n_graphs)
        total_val_loss += head_loss * n_g
        total_weight += n_g

    total_val_loss = total_val_loss / max(total_weight, 1)
    return all_head_metrics, total_val_loss


# =============================================================================
# Utilities
# =============================================================================

def parse_head_weights(
    head_configs: List[Dict],
    global_e_w: float,
    global_f_w: float,
    global_s_w: float,
) -> Dict[int, Dict[str, jnp.ndarray]]:
    """Parse per-head E/F/S/grad weights. Falls back to global values if absent."""
    weights = {}
    for hc in head_configs:
        weights[hc["head_idx"]] = {
            'energy_weight': jnp.array(hc.get('energy_weight', global_e_w), dtype=jnp.float32),
            'force_weight': jnp.array(hc.get('force_weight', global_f_w), dtype=jnp.float32),
            'stress_weight': jnp.array(hc.get('stress_weight', global_s_w), dtype=jnp.float32),
            'grad_weight': jnp.array(hc.get('grad_weight', 1.0), dtype=jnp.float32),
        }
    return weights


def _get_pkl_files(path: str) -> List[str]:
    """Get sorted list of pkl files from a directory path."""
    import re
    def natural_key(s):
        return [int(t) if t.isdigit() else t.lower()
                for t in re.split(r'(\d+)', str(s))]

    p = Path(path)
    if p.is_file():
        return [str(p)]
    files = sorted(list(p.glob('*.pkl')), key=natural_key)
    return [str(f) for f in files]


class HeadBatchStream:
    """Per-head batch stream with automatic file transitions and cycling.

    Shuffles file list, creates DataLoaders one file at a time, and yields
    batches. When all files are exhausted, reshuffles and cycles.
    """

    def __init__(self, head_config, seed, epoch, batch_size, n_devices, mesh,
                 max_nodes_per_batch=256):
        self.head_idx = head_config["head_idx"]
        self.head_name = head_config["name"]
        self.all_files = _get_pkl_files(head_config["train_path"])
        self.n_files = len(self.all_files)
        if self.n_files == 0:
            raise FileNotFoundError(
                f"Head '{self.head_name}' (idx={self.head_idx}): "
                f"no .pkl files found in '{head_config['train_path']}'. "
                f"Run data preprocessing first."
            )
        self.seed = seed
        self.epoch = epoch
        self.batch_size = batch_size
        self.n_devices = n_devices
        self.mesh = mesh
        self.max_nodes_per_batch = max_nodes_per_batch

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
        """Create DataLoader for the next file. Cycles if exhausted."""
        if self.file_idx >= self.n_files:
            self.cycle_count += 1
            self._shuffle_files()

        file_path = self.shuffled_files[self.file_idx]
        self.file_idx += 1
        self.files_consumed += 1

        dataset = DatasetWithHead(file_path=file_path, head_idx=self.head_idx)
        rngs = nnx.Rngs(
            self.seed + 100 * self.epoch + 10 * self.files_consumed + self.head_idx
        )
        base_loader = BucketedDataLoader(
            dataset=dataset, batch_size=self.batch_size,
            n_buckets=8, shuffle=True, drop_last=True,
            rngs=rngs, max_nodes_per_batch=self.max_nodes_per_batch,
        )
        loader = MultiDeviceDataLoader(
            base_loader=base_loader, n_devices=self.n_devices,
            mesh=self.mesh, drop_incomplete=True,
        )
        self._current_iter = iter(loader)
        self._current_file = file_path

    def next_batch(self):
        """Return next batch, automatically transitioning files and cycling."""
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
        """Whether the first full pass through all files is complete."""
        return self.files_consumed >= self.n_files



# =============================================================================
# Main Training Loop
# =============================================================================

def train_multihead_sharded(config: Dict):
    """Main multihead training function with true data sharding."""

    mesh, n_devices = setup_mesh()
    fout = open(config.get('fname_log', 'loss_multihead.out'), 'w', 1)

    # Initialize batch logger
    batch_logger = BatchLogger(
        n_devices=n_devices,
        log_root=config.get('batch_log_root', 'batch_logs'),
    )

    head_configs = config["heads"]
    num_heads = len(head_configs)

    print("=" * 70, file=fout)
    print(f"Multihead Sharded Training on {n_devices} device(s)", file=fout)
    print(f"Number of heads: {num_heads}", file=fout)
    for hc in head_configs:
        print(f"  Head {hc['head_idx']}: {hc['name']}", file=fout)
    print(f"Effective batch size: {config['batch_size']} x {n_devices} = "
          f"{config['batch_size'] * n_devices}", file=fout)
    print("=" * 70, file=fout)

    batch_logger.log_gpu(f"Multihead training started on {n_devices} device(s)")
    batch_logger.log_gpu(f"Heads: {[hc['name'] for hc in head_configs]}")

    seed = config.get('seed', 42)
    model_rngs = nnx.Rngs(seed)

    # Model initialization
    foundation_ckpt = config.get("foundation_ckpt")

    if foundation_ckpt:
        graphdef, params, load_info = load_foundation_as_multihead(
            ckpt_path=foundation_ckpt,
            num_heads=num_heads,
            config=config,
            rngs=model_rngs,
            use_checkpoint=config.get('use_checkpoint', False),
        )
        total = load_info['total_size']
        copied = load_info['copied_size']
        expanded = load_info['repeated_size']
        skipped = load_info['skipped_size']
        cp = 100 * copied / total if total > 0 else 0
        ep = 100 * expanded / total if total > 0 else 0
        print(f"Foundation model loaded: {foundation_ckpt}", file=fout)
        if load_info.get('source_epoch') is not None:
            print(f"  Trained epochs: {load_info['source_epoch']}", file=fout)
        if load_info.get('source_step') is not None:
            print(f"  Trained steps: {load_info['source_step']}", file=fout)
        if load_info.get('source_val_loss') is not None:
            print(f"  Best val_loss: {load_info['source_val_loss']:.6f}", file=fout)
        print(f"  Param transfer (total {total:,}):", file=fout)
        print(f"    Backbone (direct copy):      {copied:>13,} ({cp:.2f}%)", file=fout)
        print(f"    Readout  (1-head -> {num_heads}-head): {expanded:>13,} ({ep:.2f}%)", file=fout)
        if skipped > 0:
            print(f"    NOT matched (random init):   {skipped:>13,} ({sp:.2f}%) *** WARNING ***", file=fout)
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
            shift=0.0,
            scale=1.0,
            avg_n_neighbors=25.0,
            atom_energies=config["atom_energies"],
            l_train=True,
            periodic=True,
            use_checkpoint=config.get('use_checkpoint', False),
            rngs=model_rngs,
        )
        graphdef, params = nnx.split(model, nnx.Param)

    ema_params = params

    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {n_params:,}", file=fout)

    # Optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.amsgrad(config["learning_rate"]),
    )
    schedule = optax.contrib.reduce_on_plateau(
        factor=config["lr_schedule_factor"],
        patience=config["lr_schedule_patience"],
        min_scale=config["lr_schedule_min_scale"],
    )
    opt_state = optimizer.init(params)
    schedule_state = schedule.init(params)

    # Replicate across devices
    params = replicate_pytree(params, mesh)
    ema_params = replicate_pytree(ema_params, mesh)
    opt_state = replicate_pytree(opt_state, mesh)
    schedule_state = replicate_pytree(schedule_state, mesh)
    step = replicate(jnp.array(0), mesh)

    energy_weight = config.get('energy_weight', 1.0)
    force_weight = config.get('force_weight', 1.0)
    stress_weight = config.get('stress_weight', 1.0)
    loss_type = config.get('loss_type', 'huber')
    huber_delta = config.get('huber_delta', 0.02)
    loss_fn = partial(LOSS_FUNCTIONS[loss_type], delta=huber_delta)

    # Per-head weights (JAX arrays, will be replicated after checkpoint load)
    head_weights = parse_head_weights(head_configs, energy_weight, force_weight, stress_weight)

    # Fused multihead train step: grad accumulation + optimizer in one XLA graph
    train_step = make_fused_multihead_train_step(
        optimizer=optimizer,
        mesh=mesh,
        loss_fn=loss_fn,
        num_heads=num_heads,
        num_head_configs=len(head_configs),
        ema_decay=config.get('ema_decay', 0.99),
    )

    log_every = config.get('log_every', 50)
    max_nodes_per_batch = config.get('max_nodes_per_batch', 256)

    # Load checkpoint if restarting
    best_val_loss = float('inf')
    start_epoch = 0
    if config.get('restart', False) and Path('ckpt_multihead_best.pkl').exists():
        ckpt = load_checkpoint('ckpt_multihead_best.pkl')
        params = replicate_pytree(ckpt['params'], mesh)
        ema_params = replicate_pytree(ckpt['ema_params'], mesh)
        opt_state = replicate_pytree(ckpt['opt_state'], mesh)
        schedule_state = replicate_pytree(ckpt['schedule_state'], mesh)
        step = replicate(jnp.array(ckpt.get('step', 0)), mesh)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from epoch {start_epoch}, step {ckpt.get('step', 0)}", file=fout)

    # Replicate per-head weights across devices
    for h in head_weights:
        head_weights[h] = {k: replicate(v, mesh) for k, v in head_weights[h].items()}

    # Training loop
    best_state = None

    for epoch in range(start_epoch, config["n_epochs"]):
        epoch_start = time.time()

        # Per-head batch streams
        streams = {
            hc["head_idx"]: HeadBatchStream(
                hc, seed, epoch, config['batch_size'], n_devices, mesh,
                max_nodes_per_batch=max_nodes_per_batch,
            )
            for hc in head_configs
        }
        longest_head = max(streams.keys(), key=lambda h: streams[h].n_files)

        step_in_epoch = 0
        epoch_step_start = time.time()

        while not streams[longest_head].completed_first_pass:
            step_in_epoch += 1

            # === Collect one batch from each head ===
            head_batches = []
            head_batch_meta = []  # for logging
            for head_cfg in head_configs:
                h = head_cfg["head_idx"]
                stream = streams[h]
                batch, info = stream.next_batch()
                head_batches.append(batch)
                head_batch_meta.append((head_cfg, stream, batch))

            # === Build fused step arguments ===
            # Layout: (batch_0, e_w_0, f_w_0, s_w_0, g_w_0, batch_1, ...)
            head_args = []
            for i, head_cfg in enumerate(head_configs):
                hw = head_weights[head_cfg["head_idx"]]
                head_args.extend([
                    head_batches[i],
                    hw['energy_weight'], hw['force_weight'],
                    hw['stress_weight'], hw['grad_weight'],
                ])

            # === Fused train step: grad accumulation + optimizer in one XLA call ===
            params, opt_state, ema_params, step, metrics = train_step(
                graphdef, params, opt_state, schedule_state,
                ema_params, step, *head_args,
            )

            current_step = int(np.asarray(step))

            # === Batch logging ===
            for head_cfg, stream, batch in head_batch_meta:
                per_device_metadata = extract_per_device_batch_metadata(batch, n_devices)
                batch_logger.log_batch(
                    epoch, 'train', head_cfg["name"],
                    Path(stream._current_file).name, stream.batches_yielded,
                    per_device_metadata,
                )

            # === Logging ===
            if current_step % log_every == 0:
                elapsed = time.time() - epoch_step_start
                unrep_schedule = unreplicate_pytree(schedule_state)
                lr = config['learning_rate'] * float(unrep_schedule.scale)
                n_epochs = config["n_epochs"]
                print(f"step {current_step} (epoch {epoch+1}/{n_epochs}, {elapsed:.1f}s, LR={lr:.2e})",
                      file=fout)
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
                print(f"  [combined] grad_norm={float(metrics['combined_grad_norm']):.4f}",
                      file=fout)

        # Log stream statistics
        for h, stream in streams.items():
            hname = next(hc["name"] for hc in head_configs if hc["head_idx"] == h)
            print(f"  [{hname}] files={stream.files_consumed}/{stream.n_files} "
                  f"cycles={stream.cycle_count} batches={stream.batches_yielded}",
                  file=fout)

        # End-of-epoch validation (per-head)
        val_start = time.time()
        unrep_ema = unreplicate_pytree(ema_params)

        per_head_metrics, val_loss = evaluate_per_head(
            graphdef=graphdef,
            params=unrep_ema,
            head_configs=head_configs,
            batch_size=config['batch_size'],
            n_devices=n_devices,
            mesh=mesh,
            energy_weight=energy_weight,
            force_weight=force_weight,
            stress_weight=stress_weight,
            loss_fn=loss_fn,
            num_heads=num_heads,
            batch_logger=batch_logger,
            epoch=epoch,
        )
        val_time = time.time() - val_start

        print(f"\n  Validation ({val_time:.1f}s) | Total Loss: {val_loss:.6f}",
              file=fout)
        for hname, hmetrics in per_head_metrics.items():
            print(
                f"    [{hname}] Loss: {hmetrics['total_loss']:.6f} | "
                f"E_MAE: {hmetrics['energy_mae']:.6f} | "
                f"F_MAE: {hmetrics['force_mae']:.6f} | "
                f"S_MAE: {hmetrics['stress_mae']:.6f}",
                file=fout,
            )

        # Update LR schedule
        unrep_params = unreplicate_pytree(params)
        unrep_schedule = unreplicate_pytree(schedule_state)
        _, new_schedule = schedule.update(
            updates=unrep_params,
            state=unrep_schedule,
            value=val_loss,
        )
        schedule_state = replicate_pytree(new_schedule, mesh)

        # Save best model
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
            }
            print(
                f"  *** New best val loss: {best_val_loss:.6f} "
                f"(improved by {improvement:.6f}) ***",
                file=fout,
            )
            save_checkpoint(best_state, 'ckpt_multihead_best.pkl')
            print("  Checkpoint saved to ckpt_multihead_best.pkl", file=fout)
        else:
            print(
                f"  No improvement for {new_schedule.plateau_count} evaluations "
                f"(best: {best_val_loss:.6f})",
                file=fout,
            )

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch+1}/{config['n_epochs']} completed in {epoch_time:.1f}s\n",
            file=fout,
        )

    # Save final checkpoint
    if best_state is not None:
        save_checkpoint(best_state, 'ckpt_multihead_best.pkl')
        print("\nFinal checkpoint saved", file=fout)

    print("\nMultihead training completed!", file=fout)

    # Close batch logger
    batch_logger.log_gpu("Training completed")
    batch_logger.close()

    fout.close()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path
    from bam.data.atom_energies import ATOM_ENERGIES

    # Multi-node distributed initialization
    num_processes = int(os.environ.get("JAX_NUM_PROCESSES", 1))
    if num_processes > 1:
        coordinator_address = os.environ["JAX_COORDINATOR_ADDRESS"]
        process_id = int(os.environ["JAX_PROCESS_INDEX"])
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
        )

    print("=" * 70)
    print("JAX Multihead Sharded Training (Multi-GPU)")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Device count: {jax.device_count()} (local: {jax.local_device_count()})")
    print(f"Process index: {jax.process_index()} / {jax.process_count()}")
    print("=" * 70)

    # Load config
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'input.json'
    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    # Load atom energies: from JSON file or fallback to built-in ATOM_ENERGIES
    atom_energies_path = config.get('atom_energies_path', None)
    if atom_energies_path and Path(atom_energies_path).exists():
        print(f"Loading atom energies from {atom_energies_path}")
        with open(atom_energies_path) as f:
            ae_data = json.load(f)
        if isinstance(ae_data, dict):
            # New format: {"atom_energies": [...], "atomic_numbers": [...], ...}
            config['atom_energies'] = ae_data['atom_energies']
        else:
            # Old format: plain list
            config['atom_energies'] = ae_data
    else:
        print("Using built-in ATOM_ENERGIES")
        config['atom_energies'] = ATOM_ENERGIES.tolist()

    train_multihead_sharded(config)
