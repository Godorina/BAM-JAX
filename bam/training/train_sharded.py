"""NequIP/RACE multi-GPU training with true data sharding.

This version implements true data parallelism where each GPU processes
different batches simultaneously and gradients are averaged.

Both training AND evaluation are sharded across all GPUs.
"""

# ============================================================================
# DETERMINISM: Must be set BEFORE importing JAX
# ============================================================================
#import os
# Force XLA to use deterministic scatter/gather ops on GPU.
# Without this, segment_sum (used in GNN message passing) and
# other scatter-based ops are non-deterministic across runs.
#os.environ["XLA_FLAGS"] = os.environ.get("XLA_FLAGS", "") + " --xla_gpu_deterministic_ops=true"
# Force deterministic cuDNN algorithms
#os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

from typing import Callable, Dict, Tuple, List, Any
from functools import partial
import json
import pickle
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

from bam.data.data_nnx import Dataset, BucketedDataLoader, MultiDeviceDataLoader
from bam.models.race_nnx import RACE
from bam.training.losses import LOSS_FUNCTIONS
from bam.training.sharding import (
    setup_mesh, replicate, replicate_pytree,
    unreplicate, unreplicate_pytree, squeeze_batch,
    save_checkpoint, load_checkpoint,
)

jax.config.update("jax_enable_x64", False)
# Disable XLA autotuning cache (autotuning can pick different algorithms between runs)
#jax.config.update("jax_xla_profile_enabled", False)



def compute_loss(
    graphdef: nnx.GraphDef,
    params: nnx.State,
    batch: jraph.GraphsTuple,
    energy_weight: float,
    force_weight: float,
    stress_weight: float,
    loss_fn: Callable
) -> Tuple[jnp.ndarray, Dict]:
    """Compute loss for a single batch (no device dimension)."""
    graph_mask = jraph.get_graph_padding_mask(batch)
    node_mask = jraph.get_node_padding_mask(batch)

    n_graphs = jnp.maximum(graph_mask.sum(), 1)
    n_atoms = jnp.maximum(node_mask.sum(), 1)

    model = nnx.merge(graphdef, params)
    energy, forces, stress = model(batch)

    # Energy loss
    energy_diff = (energy - batch.globals["energy"])/batch.n_node
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
    stress_loss = jnp.sum(loss_fn(stress_diff) * graph_mask[:, None]) / n_graphs
    stress_mse = jnp.sum(stress_diff ** 2 * graph_mask[:, None]) / n_graphs

    total_loss = energy_weight * energy_loss + force_weight * force_loss + stress_weight * stress_loss

    aux = {
        'energy_loss': energy_loss,
        'force_loss': force_loss,
        'stress_loss': stress_loss,
        'energy_mse': energy_mse,
        'force_mse': force_mse,
        'stress_mse': stress_mse,
        'n_graphs': n_graphs,
        'n_atoms': n_atoms,
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
    ema_decay: float = 0.99
):
    """Create sharded training step with gradient synchronization.

    Uses shard_map to explicitly control SPMD execution:
    - Each device processes its own batch (indexed by leading dimension)
    - Gradients are synchronized via pmean across the 'dp' axis
    - Parameters are updated identically on all devices
    """

    def per_device_step(graphdef, params, opt_state, schedule_state, ema_params, step, batch):
        """Training step for a single device's batch."""
        # Squeeze the leading device dimension: (1, ...) -> (...)
        batch = squeeze_batch(batch)
        def loss_fn(params):
            return compute_loss(
                graphdef, params, batch,
                energy_weight, force_weight, stress_weight, loss_fn
            )

        # Compute gradients locally
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)

        # Synchronize gradients and metrics across devices
        grads = jax.lax.pmean(grads, axis_name='dp')
        loss = jax.lax.pmean(loss, axis_name='dp')
        aux = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name='dp'), aux)

        grad_norm = optax.global_norm(grads)

        # JAX-style conditional: skip update when grad_norm >= threshold
        def apply_update(_):
            """Apply gradient update and EMA when grad_norm is small."""
            updates, new_opt = optimizer.update(grads, opt_state, params)
            updates = optax.tree_utils.tree_scale(schedule_state.scale, updates)
            new_p = optax.apply_updates(params, updates)
            new_ema = jax.tree.map(
                lambda ema, p: ema_decay * ema + (1 - ema_decay) * p,
                ema_params, new_p
            )
            return new_p, new_opt, new_ema

        # Use jax.lax.cond for JAX-traceable conditional
        new_params, new_opt_state, new_ema_params = apply_update(None)
        '''
        jax.lax.cond(
            #grad_norm < 1.0,  # condition
            frc_mse < 10.0,
            apply_update,     # true branch
            skip_update,      # false branch
            operand=None      # dummy operand
        )
        '''

        metrics = {
            'loss': loss,
            'grad_norm': grad_norm,
            'update_applied': (grad_norm < 1.0).astype(jnp.float32),  # track skipped updates
            **aux
        }

        return new_params, new_opt_state, new_ema_params, step + 1, metrics

    # Create sharded version using shard_map
    sharded_step = shard_map(
        per_device_step,
        mesh=mesh,
        in_specs=(
            P(),       # graphdef - replicated
            P(),       # params - replicated
            P(),       # opt_state - replicated
            P(),       # schedule_state - replicated
            P(),       # ema_params - replicated
            P(),       # step - replicated
            P('dp'),   # batch - sharded along device axis
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
# Sharded Evaluation
# =============================================================================

def make_sharded_evaluate_step(mesh: Mesh, loss_fn: Callable):
    """Create sharded evaluation function that runs on all GPUs.

    Each device evaluates its own batch, then results are summed across devices.
    """

    def eval_single_device(graphdef, params, batch):
        """Evaluate a single batch on one device.

        Returns raw sums (not averages) for proper aggregation across devices.
        """
        # Squeeze the leading device dimension: (1, ...) -> (...)
        batch = squeeze_batch(batch)
        graph_mask = jraph.get_graph_padding_mask(batch)
        node_mask = jraph.get_node_padding_mask(batch)
        #n_graphs = jnp.maximum(graph_mask.sum(), 1)
        #n_atoms = jnp.maximum(node_mask.sum(), 1)

        model = nnx.merge(graphdef, params)
        energy, forces, stress = model(batch)

        energy_diff = (energy - batch.globals["energy"])/batch.n_node
        energy_loss = loss_fn(energy_diff)
        forces_diff = forces - batch.nodes["forces"]
        force_loss = loss_fn(forces_diff)
        stress_diff = (stress - batch.globals["stress"]) * graph_mask[:, None]
        stress_loss = loss_fn(stress_diff)
        # Return sums weighted by actual counts for proper aggregation
        return {
            'energy_loss': jnp.sum(energy_loss * graph_mask),
            'force_loss': jnp.sum(force_loss * node_mask[:, None]),
            'stress_loss': jnp.sum(stress_loss * graph_mask[:, None]),
            'energy_se': jnp.sum(energy_diff ** 2 * graph_mask),
            'force_se': jnp.sum(forces_diff ** 2 * node_mask[:, None]),
            'stress_se': jnp.sum(stress_diff ** 2 * graph_mask[:, None]),
            'energy_ae': jnp.sum(jnp.abs(energy_diff) * graph_mask),
            'force_ae': jnp.sum(jnp.abs(forces_diff) * node_mask[:, None]),
            'stress_ae': jnp.sum(jnp.abs(stress_diff) * graph_mask[:, None]),
            'n_graphs': graph_mask.sum(),  # Actual count (not max with 1)
            'n_atoms': node_mask.sum(),    # Actual count
        }

    def eval_and_reduce(graphdef, params, batch):
        """Evaluate on each device and sum results across devices."""
        local_metrics = eval_single_device(graphdef, params, batch)
        # Sum across all devices using psum
        return jax.tree.map(lambda x: jax.lax.psum(x, axis_name='dp'), local_metrics)

    # Create sharded version
    sharded_eval = shard_map(
        eval_and_reduce,
        mesh=mesh,
        in_specs=(
            P(),       # graphdef - replicated
            P(),       # params - replicated
            P('dp'),   # batch - sharded along device axis
        ),
        out_specs=P(),  # metrics - reduced (same on all devices after psum)
    )

    return jax.jit(sharded_eval)


def evaluate_sharded(
    graphdef: nnx.GraphDef,
    params: nnx.State,
    files_pkl: List[str],
    batch_size: int,
    n_devices: int,
    mesh: Mesh,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    stress_weight: float = 1.0,
    loss_fn: Callable = None
) -> Dict:
    """Evaluate model on validation set using all GPUs.

    Args:
        graphdef: Model graph definition
        params: Model parameters (unreplicated)
        files_pkl: List of validation data files
        batch_size: Batch size per device
        n_devices: Number of devices
        mesh: Device mesh
        energy_weight: Weight for energy in loss
        force_weight: Weight for forces in loss
        stress_weight: Weight for stress in loss
        loss_fn: Loss function (e.g., partial(huber_loss, delta=0.02))

    Returns:
        Dictionary with evaluation metrics
    """
    # Replicate params for sharded computation
    params_replicated = replicate_pytree(params, mesh)

    # Initialize accumulators
    totals = {
        'energy_loss': 0.0,
        'force_loss': 0.0,
        'stress_loss': 0.0,
        'energy_se': 0.0,
        'force_se': 0.0,
        'stress_se': 0.0,
        'energy_ae': 0.0,
        'force_ae': 0.0,
        'stress_ae': 0.0,
        'n_graphs': 0.0,
        'n_atoms': 0.0
    }

    # Create sharded evaluation function
    eval_step = make_sharded_evaluate_step(mesh, loss_fn=loss_fn)

    for fname in tqdm(files_pkl, desc="Evaluating files", leave=False):
        dataset = Dataset(file_path=fname)
        base_loader = BucketedDataLoader (
            dataset=dataset,
            batch_size=batch_size,
            n_buckets=8,
            shuffle=False,       # No shuffle for deterministic evaluation
            drop_last=False,     # Evaluate ALL data
            rngs=nnx.Rngs(0)    # Deterministic rngs (not used when shuffle=False)
        )
        # Use multi-device loader for evaluation
        loader = MultiDeviceDataLoader(
            base_loader=base_loader,
            n_devices=n_devices,
            mesh=mesh,
            drop_incomplete=False  # Don't drop last batch during evaluation
        )

        for batch, info in tqdm(loader, desc="    Eval batches", leave=False):
            # Run sharded evaluation
            metrics = eval_step(graphdef, params_replicated, batch)

            # Accumulate results (metrics are already summed across devices)
            for k in totals:
                totals[k] += float(metrics[k])

    # Compute final metrics
    n_g = max(totals['n_graphs'], 1)
    n_a = max(totals['n_atoms'], 1)
    total_loss = (energy_weight * totals['energy_loss'] / n_g +
                  force_weight * totals['force_loss'] / (3* n_a) +
                  stress_weight * totals['stress_loss'] / n_g )
    return {
        #'energy_rmse': np.sqrt(totals['energy_se'] / n_g),
        #'force_rmse': np.sqrt(totals['force_se'] / (3 * n_a)),
        #'stress_rmse': np.sqrt(totals['stress_se'] / n_g),
        'energy_mae': totals['energy_ae'] / n_g,
        'force_mae': totals['force_ae'] / (3 * n_a),
        'stress_mae': totals['stress_ae'] / n_g,
        'energy_loss': totals['energy_loss'] / n_g,
        'force_loss': totals['force_loss'] / (3 * n_a),
        'stress_loss': totals['stress_loss'] / n_g,
        'total_loss': total_loss,
        'n_graphs': n_g,
        'n_atoms': n_a,
    }


# =============================================================================
# Prediction / Evaluation
# =============================================================================

def predict_sharded(config: Dict):
    """Run evaluation using checkpoint and config."""
    predict_config = config.get('predict', {})
    ckpt_path = Path(predict_config.get('model', 'ckpt_best.pkl'))
    test_path = Path(predict_config.get('test_path', config.get('test_path', '')))

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return
    if not test_path.exists():
        print(f"Test data not found: {test_path}")
        return

    # Setup
    mesh, n_devices = setup_mesh()

    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}...")
    ckpt = load_checkpoint(str(ckpt_path))

    # Use config from checkpoint if available
    if 'config' in ckpt:
        print("Using config from checkpoint")
        model_config = ckpt['config']
    else:
        model_config = config

    # Create model (for graphdef)
    print("Creating model...")
    model = RACE(
        n_species=len(model_config["atom_energies"]),
        lmax=model_config["lmax"],
        hidden_irreps=model_config["hidden_irreps"],
        n_layers=model_config["n_layers"],
        radial_basis_size=model_config["radial_basis_size"],
        radial_mlp_size=model_config["radial_mlp_size"],
        radial_mlp_layers=model_config["radial_mlp_layers"],
        radial_polynomial_p=model_config["radial_polynomial_p"],
        mlp_init_scale=model_config["mlp_init_scale"],
        shift=0.0,
        scale=1.0,
        avg_n_neighbors=model_config.get("avg_n_neighbors", 25.0),
        atom_energies=model_config["atom_energies"],
        l_train=False,
        rngs=nnx.Rngs(42)
    )
    graphdef, _ = nnx.split(model, nnx.Param)

    # Use EMA params if available
    if 'ema_params' in ckpt:
        print("Using EMA parameters")
        params = ckpt['ema_params']
    else:
        print("Using regular parameters")
        params = ckpt['params']

    print(f"Checkpoint step: {ckpt.get('step', 'N/A')}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')}")
    print(f"Checkpoint best_val_loss: {ckpt.get('best_val_loss', 'N/A')}")
    del ckpt

    # Load test dataset
    print(f"\nLoading test dataset from {test_path}...")
    test_dataset = Dataset(file_path=str(test_path))

    print(f"Test dataset: {len(test_dataset)} graphs")

    # Run evaluation
    print("\n" + "=" * 70)
    print("Running Evaluation")
    print("=" * 70)

    loss_type = model_config.get('loss_type', 'huber')
    huber_delta = model_config.get('huber_delta', 0.02)
    loss_fn = partial(LOSS_FUNCTIONS[loss_type], delta=huber_delta)

    metrics = evaluate_sharded(
        graphdef=graphdef,
        params=params,
        files_pkl=[str(test_path)],
        batch_size=model_config.get('batch_size', 4),
        n_devices=n_devices,
        mesh=mesh,
        energy_weight=model_config.get('energy_weight', 1.0),
        force_weight=model_config.get('force_weight', 1.0),
        stress_weight=model_config.get('stress_weight', 1.0),
        loss_fn=loss_fn
    )

    # Print results
    print("\n" + "=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Total graphs: {metrics['n_graphs']}")
    print(f"Total atoms:  {metrics['n_atoms']}")
    print()
    print(f"Total Loss:   {metrics['total_loss']:.6f}")
    print()
    print(f"Energy MAE: {metrics['energy_mae']:.6f} eV/atom")
    print(f"Force  MAE: {metrics['force_mae']:.6f} eV/A")
    print(f"Stress MAE: {metrics['stress_mae']:.6f} eV/A^3")
    print("=" * 70)

    # Save results
    results_path = Path('eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


# =============================================================================
# Main Training Loop
# =============================================================================

def train_sharded(config: Dict, train_files_pkl: List[str], valid_files_pkl: List[str]):
    """Main training function with true data sharding on all GPUs."""

    # Setup
    mesh, n_devices = setup_mesh()
    fout = open(config.get('fname_log', 'loss_sharded.out'), 'w', 1)

    print("=" * 70, file=fout)
    print(f"Sharded Training on {n_devices} device(s)", file=fout)
    print(f"Effective batch size: {config['batch_size']} x {n_devices} = {config['batch_size'] * n_devices}", file=fout)
    print(f"Training AND evaluation are parallelized across all GPUs", file=fout)
    print("=" * 70, file=fout)

    # Create model - use SEPARATE rngs for model init and data loading
    seed = config.get('seed', 42)
    model_rngs = nnx.Rngs(seed)      # For model parameter initialization only
    rngs = nnx.Rngs(seed + 1)        # For data loading (shuffling) only

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
        shift=0.0,
        scale=1.0,
        avg_n_neighbors=25.0,
        atom_energies=config["atom_energies"],
        l_train=True,
        rngs=model_rngs  # Dedicated rngs for model init
    )

    graphdef, params = nnx.split(model, nnx.Param)
    ema_params = params

    n_params = sum(x.size for x in jax.tree.leaves(params))
    print(f"Model parameters: {n_params:,}", file=fout)

    # Optimizer
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.amsgrad(config["learning_rate"])
    )
    schedule = optax.contrib.reduce_on_plateau(
        factor=config["lr_schedule_factor"],
        patience=config["lr_schedule_patience"],
        min_scale=config["lr_schedule_min_scale"],
    )
    opt_state = optimizer.init(params)
    schedule_state = schedule.init(params)

    # Load checkpoint if exists, otherwise replicate initial params
    best_val_loss = float('inf')
    if config.get('restart', False) and Path('ckpt_best.pkl').exists():
        ckpt = load_checkpoint('ckpt_best.pkl')
        # Free initial params before loading checkpoint to save GPU memory
        del params, ema_params, opt_state, schedule_state
        params = replicate_pytree(ckpt['params'], mesh)
        ema_params = replicate_pytree(ckpt['ema_params'], mesh)
        opt_state = replicate_pytree(ckpt['opt_state'], mesh)
        schedule_state = replicate_pytree(ckpt['schedule_state'], mesh)
        step = replicate(jnp.array(ckpt.get('step', 0)), mesh)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        start_epoch = ckpt.get('epoch', 0)
        del ckpt
        print(f"Resumed from epoch {start_epoch+1}, step {jax.device_get(step).item()}", file=fout)
    else:
        # Replicate across devices
        params = replicate_pytree(params, mesh)
        ema_params = replicate_pytree(ema_params, mesh)
        opt_state = replicate_pytree(opt_state, mesh)
        schedule_state = replicate_pytree(schedule_state, mesh)
        step = replicate(jnp.array(0), mesh)
        start_epoch = 0

    energy_weight=config.get('energy_weight', 1.0)
    force_weight=config.get('force_weight', 1.0)
    stress_weight=config.get('stress_weight', 1.0)
    loss_type = config.get('loss_type', 'huber')
    huber_delta = config.get('huber_delta', 0.02)
    loss_fn = partial(LOSS_FUNCTIONS[loss_type], delta=huber_delta)
    # Training step
    train_step = make_sharded_train_step(
        optimizer=optimizer,
        mesh=mesh,
        energy_weight=energy_weight,
        force_weight=force_weight,
        stress_weight=stress_weight,
        loss_fn=loss_fn,
        ema_decay=config.get('ema_decay', 0.99)
    )

    # Training loop
    best_state = None
    #n_val = 0

    for epoch in range(start_epoch, config["n_epochs"]):
        epoch_start = time.time()

        pbar_files = tqdm(enumerate(train_files_pkl), total=len(train_files_pkl),
                          desc=f"Epoch {epoch+1} files", leave=False)
        for ipkl, train_pkl in pbar_files:
            pbar_files.set_postfix({"file": Path(train_pkl).name})
            train_dataset = Dataset(file_path=train_pkl)
            # Create multi-device dataloader for training
            # Use deterministic seed based on epoch + file index
            loader_rngs = nnx.Rngs(seed + 100 * epoch + ipkl)
            base_loader = BucketedDataLoader (
                dataset=train_dataset,
                batch_size=config['batch_size'],
                n_buckets=8,
                shuffle=True,
                drop_last=True,
                rngs=loader_rngs  # Deterministic per epoch + file
            )
            loader = MultiDeviceDataLoader(
                base_loader=base_loader,
                n_devices=n_devices,
                mesh=mesh,
                drop_incomplete=True
            )

            # Metrics accumulators
            acc_energy_loss, acc_force_loss, acc_stress_loss = 0., 0., 0.
            acc_graphs, acc_atoms, acc_grad = 0., 0., 0.
            dataset_start = time.time()
            current_step = int(np.asarray(step))

            pbar_batches = tqdm(loader, desc="  Batches", leave=False,
                               total=len(loader) if hasattr(loader, '__len__') else None)
            for batch, valid_device_count in pbar_batches:
                # Training step (sharded across all GPUs)
                params, opt_state, ema_params, step, metrics = train_step(
                    graphdef, params, opt_state, schedule_state, ema_params, step, batch
                )

                # Accumulate (metrics are already averaged across devices via pmean)
                n_g = float(metrics['n_graphs'])
                n_a = float(metrics['n_atoms'])
                acc_energy_loss += float(metrics['energy_loss']) * n_g
                acc_force_loss += float(metrics['force_loss']) * 3 * n_a
                acc_stress_loss += float(metrics['stress_loss']) * n_g
                acc_graphs += n_g
                acc_atoms += n_a
                acc_grad += float(metrics['grad_norm']) * n_g

                current_step = int(np.asarray(step))

                # Update progress bar with current metrics
                if acc_graphs > 0:
                    pbar_batches.set_postfix({
                        "loss": f"{(energy_weight*acc_energy_loss/acc_graphs + force_weight*acc_force_loss/(3*acc_atoms) + stress_weight*acc_stress_loss/acc_graphs):.4f}",
                        "step": current_step
                    })

                if current_step % 50 == 0:
                    energy_loss = acc_energy_loss/max(acc_graphs,1)
                    force_loss = acc_force_loss/max(3*acc_atoms,1)
                    stress_loss = acc_stress_loss/max(acc_graphs,1)
                    total_loss = energy_weight*energy_loss + force_weight*force_loss + stress_weight*stress_loss
                    print(f"step {current_step} LOSS: {total_loss:.6f} E_LOSS: {energy_loss:.6f} "
                          f"F_LOSS: {force_loss:.6f}", file=fout)


            unrep_schedule = unreplicate_pytree(schedule_state)
            lr = config['learning_rate'] * float(unrep_schedule.scale)
            dataset_time = time.time() - dataset_start

            energy_loss = acc_energy_loss/max(acc_graphs,1)
            force_loss = acc_force_loss/max(3*acc_atoms,1)
            stress_loss = acc_stress_loss/max(acc_graphs,1)
            total_loss = energy_weight*energy_loss + force_weight*force_loss + stress_weight*stress_loss

            print(f"Epoch {epoch+1}/{config['n_epochs']} IPKL {ipkl+1} ({dataset_time:.1f}s) "
                  f"Step {current_step} LR: {lr:.2e}", file=fout)
            print(f"  Train | Epoch {epoch+1}/{ipkl+1} | Loss: {total_loss:.6f} | "
                  f"E_LOSS: {energy_loss:.6f} | "
                  f"F_LOSS: {force_loss:.6f} | "
                  f"S_LOSS: {stress_loss:.6f} | "
                  f"Grad: {acc_grad/max(acc_graphs,1):.4f}", file=fout)

            # Reset accumulators
            acc_energy_loss, acc_force_loss, acc_stress_loss = 0., 0., 0.
            acc_graphs, acc_atoms, acc_grad = 0., 0., 0.
            dataset_start = time.time()


            # Validation (sharded across all GPUs)
            val_start = time.time()
            unrep_ema = unreplicate_pytree(ema_params)

            val_metrics = evaluate_sharded(
                graphdef=graphdef,
                params=unrep_ema,
                files_pkl=valid_files_pkl,
                batch_size=config['batch_size'],
                n_devices=n_devices,
                mesh=mesh,
                energy_weight=energy_weight,
                force_weight=force_weight,
                stress_weight=stress_weight,
                loss_fn=loss_fn
            )
            val_time = time.time() - val_start

            val_loss = val_metrics['total_loss']
            print(f"  Valid ({val_time:.1f}s) | Epoch {epoch+1}/{ipkl+1} | Loss: {val_loss:.6f} | "
                  f"E_LOSS: {val_metrics['energy_loss']:.6f} | "
                  f"F_LOSS: {val_metrics['force_loss']:.6f} | "
                  f"S_LOSS: {val_metrics['stress_loss']:.6f} | "
                  f"E_MAE: {val_metrics['energy_mae']:.6f} | "
                  f"F_MAE: {val_metrics['force_mae']:.6f} | "
                  f"S_MAE: {val_metrics['stress_mae']:.6f}", file=fout)

            # Update LR schedule
            unrep_params = unreplicate_pytree(params)
            unrep_schedule = unreplicate_pytree(schedule_state)
            _, new_schedule = schedule.update(
                updates=unrep_params,
                state=unrep_schedule,
                value=val_loss
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
                    'metrics': val_metrics,
                    'config': config,
                }
                print(f"  *** New best val loss: {best_val_loss:.6f} "
                      f"(improved by {improvement:.6f}) ***", file=fout)
                save_checkpoint(best_state, 'ckpt_best.pkl')
                print("  Checkpoint saved to ckpt_best.pkl", file=fout)
            else:
                print(f"  No improvement for {new_schedule.plateau_count} evaluations "
                      f"(best: {best_val_loss:.6f})", file=fout)


        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1}/{config['n_epochs']} completed in {epoch_time:.1f}s", file=fout)

    # Save final checkpoint
    if best_state is not None:
        save_checkpoint(best_state, 'ckpt_best.pkl')
        print("\nFinal checkpoint saved", file=fout)

    print("\nTraining completed!", file=fout)
    fout.close()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import sys
    import json
    from bam.data.atom_energies import ATOM_ENERGIES

    import re
    def natsorted(lst):
        def natural_key(s):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]
        return sorted(lst, key=natural_key)

    def resolve_data_files(data_path_str):
        """Resolve a data path to a list of files.

        - Directory → glob for *.pkl files
        - Single .pkl file → [file]
        - Single .traj/.xyz/etc → [file]
        """
        p = Path(data_path_str)
        if p.is_dir():
            files = natsorted(list(p.glob('*.pkl')))
            if not files:
                raise FileNotFoundError(f"No .pkl files found in {p}")
            return [str(f) for f in files]
        elif p.exists():
            return [str(p)]
        else:
            raise FileNotFoundError(f"Data path not found: {p}")


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

    # Branch: predict (eval) or train
    predict_config = config.get('predict', {})
    if predict_config.get('evaluate_tag', False):
        print("Mode: Evaluation")
        print("=" * 70)
        predict_sharded(config)
    else:
        print("Mode: Training")
        print("=" * 70)

        train_path = config.get('train_path', '')
        valid_path = config.get('valid_path', '')

        if not train_path or not valid_path:
            print("ERROR: 'train_path' and 'valid_path' must be set in config")
            sys.exit(1)

        train_files = resolve_data_files(train_path)
        valid_files = resolve_data_files(valid_path)

        print(f"Training files: {len(train_files)}")
        for i, f in enumerate(train_files[:5]):
            print(f"  [{i}] {Path(f).name}")
        if len(train_files) > 5:
            print(f"  ... and {len(train_files) - 5} more files")

        print(f"Validation files: {len(valid_files)}")
        for i, f in enumerate(valid_files[:3]):
            print(f"  [{i}] {Path(f).name}")
        if len(valid_files) > 3:
            print(f"  ... and {len(valid_files) - 3} more files")

        train_sharded(config, train_files, valid_files)
