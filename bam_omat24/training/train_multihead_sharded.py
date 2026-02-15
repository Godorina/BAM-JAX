"""Multihead RACE multi-GPU training with true data sharding.

Fine-tunes a pre-trained RACE model with multiple heads for different datasets.
Each head corresponds to a dataset (e.g., OMat24 target + MPTrj replay).
Data from all heads is interleaved during training. Validation is per-head.

Based on train_sharded.py with multihead extensions.
"""

from typing import Callable, Dict, Tuple, List, Any
from functools import partial
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

from bam_omat24.data.data_nnx import BucketedDataLoader, MultiDeviceDataLoader
from bam_omat24.data.data_multihead_nnx import DatasetWithHead
from bam_omat24.models.race_multihead_nnx import RACEMultihead, load_foundation_as_multihead
from bam_omat24.training.losses import LOSS_FUNCTIONS
from bam_omat24.training.sharding import (
    setup_mesh, replicate, replicate_pytree,
    unreplicate, unreplicate_pytree, squeeze_batch,
    save_checkpoint, load_checkpoint,
)

jax.config.update("jax_enable_x64", False)



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
# Sharded Training Step
# =============================================================================

def make_sharded_train_step(
    optimizer: optax.GradientTransformation,
    mesh: Mesh,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    stress_weight: float = 1.0,
    loss_fn: Callable = None,
    ema_decay: float = 0.99,
    num_heads: int = 2,
):
    """Create sharded training step for multihead model."""

    def per_device_step(
        graphdef, params, opt_state, schedule_state, ema_params, step, batch
    ):
        batch = squeeze_batch(batch)

        def loss_fn_inner(params):
            return compute_loss_multihead(
                graphdef, params, batch,
                energy_weight, force_weight, stress_weight,
                loss_fn, num_heads,
            )

        (loss, aux), grads = jax.value_and_grad(loss_fn_inner, has_aux=True)(params)

        # Synchronize across devices
        grads = jax.lax.pmean(grads, axis_name='dp')
        loss = jax.lax.pmean(loss, axis_name='dp')
        aux = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name='dp'), aux)

        grad_norm = optax.global_norm(grads)

        # Apply update
        updates, new_opt = optimizer.update(grads, opt_state, params)
        updates = optax.tree_utils.tree_scale(schedule_state.scale, updates)
        new_p = optax.apply_updates(params, updates)
        new_ema = jax.tree.map(
            lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
            ema_params, new_p,
        )

        metrics = {
            'loss': loss,
            'grad_norm': grad_norm,
            **aux,
        }

        return new_p, new_opt, new_ema, step + 1, metrics

    sharded_step = shard_map(
        per_device_step,
        mesh=mesh,
        in_specs=(
            P(),       # graphdef
            P(),       # params
            P(),       # opt_state
            P(),       # schedule_state
            P(),       # ema_params
            P(),       # step
            P('dp'),   # batch
        ),
        out_specs=(
            P(),       # new_params
            P(),       # new_opt_state
            P(),       # new_ema_params
            P(),       # step
            P(),       # metrics
        ),
    )

    return jax.jit(sharded_step)


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
            dataset = DatasetWithHead(file_path=fname, head_idx=head_idx)
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

            for batch, info in loader:
                metrics = eval_step(graphdef, params_replicated, batch)
                for k in totals:
                    totals[k] += float(metrics[k])

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


def _interleave_file_lists(
    head_configs: List[Dict],
    epoch: int,
    seed: int,
) -> List[Tuple[str, int]]:
    """Build interleaved list of (pkl_file, head_idx) for an epoch.

    Shuffles files within each head, then round-robin interleaves heads.

    Returns:
        List of (file_path, head_idx) tuples.
    """
    rng = np.random.RandomState(seed + epoch)

    per_head_files = []
    for head_cfg in head_configs:
        train_path = head_cfg["train_path"]
        files = _get_pkl_files(train_path)
        rng.shuffle(files)
        per_head_files.append(
            [(f, head_cfg["head_idx"]) for f in files]
        )

    # Round-robin interleave
    result = []
    max_len = max(len(fl) for fl in per_head_files)
    for i in range(max_len):
        for fl in per_head_files:
            if i < len(fl):
                result.append(fl[i])

    return result


# =============================================================================
# Main Training Loop
# =============================================================================

def train_multihead_sharded(config: Dict):
    """Main multihead training function with true data sharding."""

    mesh, n_devices = setup_mesh()
    fout = open(config.get('fname_log', 'loss_multihead.out'), 'w', 1)

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

    seed = config.get('seed', 42)
    model_rngs = nnx.Rngs(seed)

    # Model initialization
    foundation_ckpt = config.get("foundation_ckpt")

    if foundation_ckpt and not config.get('restart', False):
        print(f"Loading foundation model from {foundation_ckpt}", file=fout)
        graphdef, params = load_foundation_as_multihead(
            ckpt_path=foundation_ckpt,
            num_heads=num_heads,
            config=config,
            rngs=model_rngs,
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
            shift=0.0,
            scale=1.0,
            avg_n_neighbors=25.0,
            atom_energies=config["atom_energies"],
            l_train=True,
            periodic=True,
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

    train_step = make_sharded_train_step(
        optimizer=optimizer,
        mesh=mesh,
        energy_weight=energy_weight,
        force_weight=force_weight,
        stress_weight=stress_weight,
        loss_fn=loss_fn,
        ema_decay=config.get('ema_decay', 0.99),
        num_heads=num_heads,
    )

    # Load checkpoint if restarting
    best_val_loss = float('inf')
    if config.get('restart', False) and Path('ckpt_multihead_best.pkl').exists():
        ckpt = load_checkpoint('ckpt_multihead_best.pkl')
        params = replicate_pytree(ckpt['params'], mesh)
        ema_params = replicate_pytree(ckpt['ema_params'], mesh)
        opt_state = replicate_pytree(ckpt['opt_state'], mesh)
        schedule_state = replicate_pytree(ckpt['schedule_state'], mesh)
        step = replicate(jnp.array(ckpt.get('step', 0)), mesh)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed from step {ckpt.get('step', 0)}", file=fout)

    # Training loop
    best_state = None

    for epoch in range(config["n_epochs"]):
        epoch_start = time.time()

        # Interleave files from all heads
        file_list = _interleave_file_lists(head_configs, epoch, seed)

        pbar_files = tqdm(
            enumerate(file_list),
            total=len(file_list),
            desc=f"Epoch {epoch+1} files",
            leave=False,
        )

        for ipkl, (train_pkl, head_idx) in pbar_files:
            head_name = head_configs[head_idx]["name"]
            pbar_files.set_postfix({"file": Path(train_pkl).name, "head": head_name})

            train_dataset = DatasetWithHead(
                file_path=train_pkl, head_idx=head_idx
            )

            loader_rngs = nnx.Rngs(seed + 100 * epoch + ipkl)
            base_loader = BucketedDataLoader(
                dataset=train_dataset,
                batch_size=config['batch_size'],
                n_buckets=8,
                shuffle=True,
                drop_last=True,
                rngs=loader_rngs,
            )
            loader = MultiDeviceDataLoader(
                base_loader=base_loader,
                n_devices=n_devices,
                mesh=mesh,
                drop_incomplete=True,
            )

            acc_energy_loss, acc_force_loss, acc_stress_loss = 0., 0., 0.
            acc_graphs, acc_atoms, acc_grad = 0., 0., 0.
            dataset_start = time.time()
            current_step = int(np.asarray(step))

            pbar_batches = tqdm(
                loader,
                desc="  Batches",
                leave=False,
                total=len(loader) if hasattr(loader, '__len__') else None,
            )

            for batch, valid_device_count in pbar_batches:
                params, opt_state, ema_params, step, metrics = train_step(
                    graphdef, params, opt_state, schedule_state,
                    ema_params, step, batch,
                )

                n_g = float(metrics['n_graphs'])
                n_a = float(metrics['n_atoms'])
                acc_energy_loss += float(metrics['energy_loss']) * n_g
                acc_force_loss += float(metrics['force_loss']) * 3 * n_a
                acc_stress_loss += float(metrics['stress_loss']) * n_g
                acc_graphs += n_g
                acc_atoms += n_a
                acc_grad += float(metrics['grad_norm']) * n_g

                current_step = int(np.asarray(step))

                if acc_graphs > 0:
                    pbar_batches.set_postfix({
                        "loss": f"{(energy_weight*acc_energy_loss/acc_graphs + force_weight*acc_force_loss/(3*acc_atoms) + stress_weight*acc_stress_loss/acc_graphs):.4f}",
                        "step": current_step,
                    })

                if current_step % 50 == 0:
                    e_loss = acc_energy_loss / max(acc_graphs, 1)
                    f_loss = acc_force_loss / max(3 * acc_atoms, 1)
                    s_loss = acc_stress_loss / max(acc_graphs, 1)
                    total_loss = (
                        energy_weight * e_loss
                        + force_weight * f_loss
                        + stress_weight * s_loss
                    )
                    print(
                        f"step {current_step} [{head_name}] LOSS: {total_loss:.6f} "
                        f"E_LOSS: {e_loss:.6f} F_LOSS: {f_loss:.6f}",
                        file=fout,
                    )

            unrep_schedule = unreplicate_pytree(schedule_state)
            lr = config['learning_rate'] * float(unrep_schedule.scale)
            dataset_time = time.time() - dataset_start

            e_loss = acc_energy_loss / max(acc_graphs, 1)
            f_loss = acc_force_loss / max(3 * acc_atoms, 1)
            s_loss = acc_stress_loss / max(acc_graphs, 1)
            total_loss = (
                energy_weight * e_loss
                + force_weight * f_loss
                + stress_weight * s_loss
            )

            print(
                f"Epoch {epoch+1}/{config['n_epochs']} IPKL {ipkl+1} "
                f"[{head_name}] ({dataset_time:.1f}s) "
                f"Step {current_step} LR: {lr:.2e}",
                file=fout,
            )
            print(
                f"  Train | Loss: {total_loss:.6f} | "
                f"E_LOSS: {e_loss:.6f} | "
                f"F_LOSS: {f_loss:.6f} | "
                f"S_LOSS: {s_loss:.6f} | "
                f"Grad: {acc_grad/max(acc_graphs,1):.4f}",
                file=fout,
            )

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
    fout.close()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    import json
    from pathlib import Path
    from bam_omat24.data.atom_energies import ATOM_ENERGIES

    print("=" * 70)
    print("JAX Multihead Sharded Training (Multi-GPU)")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Device count: {jax.local_device_count()}")
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
            config['atom_energies'] = json.load(f)
    else:
        print("Using built-in ATOM_ENERGIES")
        config['atom_energies'] = ATOM_ENERGIES.tolist()

    train_multihead_sharded(config)
