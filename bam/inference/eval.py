"""Evaluation script for trained NequIP/RACE model.

Supports multi-GPU parallel evaluation with per-GPU logs.

Usage:
    python -m bam.inference.eval input.json
"""

from typing import Dict, List
import sys
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding
from jax.experimental import mesh_utils
try:
    from jax import shard_map
except ImportError:
    from jax.experimental.shard_map import shard_map
from flax import nnx
import pickle
import jraph
import numpy as np
from pathlib import Path
import json

from ase.io import iread
from scipy.optimize import minimize
from tqdm import tqdm

from bam.models.race_nnx import RACE
from bam.data.data_nnx import Dataset, BucketedDataLoader, MultiDeviceDataLoader, atoms_to_graph_with_targets
from bam.training.losses import huber_loss, mae_loss, mse_loss, LOSS_FUNCTIONS
from bam.training.sharding import (
    setup_mesh, replicate, replicate_pytree, squeeze_batch,
)

jax.config.update("jax_enable_x64", False)


# =============================================================================
# Sharded Evaluation (aggregate metrics only)
# =============================================================================

def make_sharded_evaluate_step(mesh: Mesh, loss_type: str = 'mae', loss_delta: float = 0.02):
    loss_fn = LOSS_FUNCTIONS.get(loss_type, mae_loss)

    def eval_single_device(graphdef, params, batch):
        batch = squeeze_batch(batch)
        graph_mask = jraph.get_graph_padding_mask(batch)
        node_mask = jraph.get_node_padding_mask(batch)

        model = nnx.merge(graphdef, params)
        energy, forces, stress = model(batch)

        # Per-atom energy difference
        energy_diff = (energy - batch.globals["energy"]) / batch.n_node
        energy_loss = loss_fn(energy_diff, delta=loss_delta)

        forces_diff = forces - batch.nodes["forces"]
        force_loss = loss_fn(forces_diff, delta=loss_delta)

        stress_diff = (stress - batch.globals["stress"]) * graph_mask[:, None]
        stress_loss = loss_fn(stress_diff, delta=loss_delta)

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
            'n_graphs': graph_mask.sum(),
            'n_atoms': node_mask.sum(),
        }

    def eval_and_reduce(graphdef, params, batch):
        local_metrics = eval_single_device(graphdef, params, batch)
        return jax.tree.map(lambda x: jax.lax.psum(x, axis_name='dp'), local_metrics)

    sharded_eval = shard_map(
        eval_and_reduce,
        mesh=mesh,
        in_specs=(P(), P(), P('dp')),
        out_specs=P(),
    )

    return jax.jit(sharded_eval)


# =============================================================================
# Sharded Predict (per-device raw predictions for verbose output)
# =============================================================================

def make_sharded_predict_step(mesh: Mesh):
    """Create sharded prediction step that returns per-device raw predictions."""

    def per_device_predict(graphdef, params, batch):
        batch = squeeze_batch(batch)
        model = nnx.merge(graphdef, params)
        energy, forces, stress = model(batch)
        graph_mask = jraph.get_graph_padding_mask(batch)
        return {
            'pred_energy': energy,
            'pred_forces': forces,
            'pred_stress': stress,
            'exact_energy': batch.globals['energy'],
            'exact_forces': batch.nodes['forces'],
            'exact_stress': batch.globals['stress'],
            'n_node': batch.n_node,
            'graph_mask': graph_mask,
        }

    sharded = shard_map(
        per_device_predict,
        mesh=mesh,
        in_specs=(P(), P(), P('dp')),
        out_specs=P('dp'),
    )
    return jax.jit(sharded)


# =============================================================================
# Aggregate Evaluation (parallel)
# =============================================================================

def evaluate(
    graphdef,
    params,
    files_pkl: List[str],
    batch_size: int,
    n_devices: int,
    mesh: Mesh,
    energy_weight: float = 1.0,
    force_weight: float = 1.0,
    stress_weight: float = 1.0,
    loss_type: str = 'mae',
    loss_delta: float = 0.02
) -> Dict:
    """Evaluate model on test set (aggregate metrics, multi-GPU)."""
    params_replicated = replicate_pytree(params, mesh)

    totals = {
        'energy_loss': 0.0, 'force_loss': 0.0, 'stress_loss': 0.0,
        'energy_se': 0.0, 'force_se': 0.0, 'stress_se': 0.0,
        'energy_ae': 0.0, 'force_ae': 0.0, 'stress_ae': 0.0,
        'n_graphs': 0.0, 'n_atoms': 0.0
    }

    eval_step = make_sharded_evaluate_step(mesh, loss_type=loss_type, loss_delta=loss_delta)

    print("Evaluating...")
    batch_count = 0

    for file_idx, fname in enumerate(files_pkl):
        print(f"  File [{file_idx+1}/{len(files_pkl)}]: {Path(fname).name}")
        dataset = Dataset(file_path=str(fname))
        base_loader = BucketedDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            n_buckets=8,
            shuffle=False,
            drop_last=False,
            rngs=nnx.Rngs(0)
        )
        loader = MultiDeviceDataLoader(
            base_loader=base_loader,
            n_devices=n_devices,
            mesh=mesh,
            drop_incomplete=False
        )

        for batch, info in loader:
            metrics = eval_step(graphdef, params_replicated, batch)
            for k in totals:
                totals[k] += float(metrics[k])
            batch_count += 1
            if batch_count % 50 == 0:
                print(f"    Processed {batch_count} batches...")

    n_g = max(totals['n_graphs'], 1)
    n_a = max(totals['n_atoms'], 1)

    total_loss = (energy_weight * totals['energy_loss'] / n_g +
                  force_weight * totals['force_loss'] / (3 * n_a) +
                  stress_weight * totals['stress_loss'] / n_g)

    return {
        'total_loss': total_loss,
        'energy_loss': totals['energy_loss'] / n_g,
        'force_loss': totals['force_loss'] / (3 * n_a),
        'stress_loss': totals['stress_loss'] / n_g,
        'energy_rmse': np.sqrt(totals['energy_se'] / n_g),
        'force_rmse': np.sqrt(totals['force_se'] / (3 * n_a)),
        'stress_rmse': np.sqrt(totals['stress_se'] / n_g),
        'energy_mae': totals['energy_ae'] / n_g,
        'force_mae': totals['force_ae'] / (3 * n_a),
        'stress_mae': totals['stress_ae'] / n_g,
        'n_graphs': int(n_g),
        'n_atoms': int(n_a),
    }


# =============================================================================
# Verbose Parallel Evaluation (per-structure + per-GPU logs)
# =============================================================================

def evaluate_verbose_parallel(
    graphdef, params, test_files, config,
    n_devices, mesh,
    log_dir='.',
):
    """Multi-GPU per-structure evaluation with per-GPU logs and summary.

    Outputs:
      - eval_gpu{i}.log : Per-structure results for each GPU
      - eval_summary.log : Aggregate metrics (MAE, RMSE, etc.)
      - eval_results.json : Aggregate metrics as JSON

    Args:
        graphdef: Model graph definition
        params: Model parameters (from checkpoint)
        test_files: List of paths to test pkl files
        config: Model configuration dict
        n_devices: Number of GPUs
        mesh: JAX mesh
        log_dir: Directory to write log files
    """
    from datetime import datetime

    log_dir = Path(log_dir)
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))
    batch_size = config.get('batch_size', 16)

    # Replicate params and create predict step
    params_rep = replicate_pytree(params, mesh)
    predict_step = make_sharded_predict_step(mesh)

    # Open per-GPU log files
    gpu_files = []
    gpu_accumulators = []
    header = (
        f"{'':30s}| {'PREDICT__________________________________________':42s}| {'EXACT____________':17s}\n"
        f"{'MM/DD/YYYY HH/MM/SS':20s} {'DATA':7s} | {'ENERGY':16s} {'MAE_E':16s} {'MAE_F':17s}| {'ENERGY':17s}|\n"
        + "-" * 102
    )

    for dev in range(n_devices):
        fpath = log_dir / f'eval_gpu{dev}.log'
        gf = open(fpath, 'w', buffering=1)
        gf.write(header + '\n')
        gpu_files.append(gf)
        gpu_accumulators.append({
            'mae_e': [], 'mae_f': [], 'se_e': [], 'se_f': [],
            'ae_stress': [], 'se_stress': [], 'n_atoms': 0, 'n_structures': 0,
        })

    # Global counters per GPU
    gpu_data_idx = [0] * n_devices

    # Process batches
    total_batches = 0
    for file_idx, fname in enumerate(tqdm(test_files, desc="Evaluating files")):
        dataset = Dataset(file_path=str(fname))
        base_loader = BucketedDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            n_buckets=8,
            shuffle=False,
            drop_last=False,
            rngs=nnx.Rngs(0),
            max_nodes_per_batch=config.get('max_nodes_per_batch', 4000),
        )
        loader = MultiDeviceDataLoader(
            base_loader=base_loader,
            n_devices=n_devices,
            mesh=mesh,
            drop_incomplete=False,
        )

        for batch, info in tqdm(loader, desc=f"  Batches ({Path(fname).name})", leave=False):
            results = predict_step(graphdef, params_rep, batch)

            # Convert to numpy: shape (n_devices, ...)
            pred_e = np.array(results['pred_energy'])
            exact_e = np.array(results['exact_energy'])
            pred_f = np.array(results['pred_forces'])
            exact_f = np.array(results['exact_forces'])
            pred_s = np.array(results['pred_stress'])
            exact_s = np.array(results['exact_stress'])
            n_node = np.array(results['n_node'])
            g_mask = np.array(results['graph_mask'])

            timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

            # Extract per-structure results for each device
            for dev in range(n_devices):
                node_offset = 0
                n_graphs_dev = len(g_mask[dev])
                acc = gpu_accumulators[dev]

                for g in range(n_graphs_dev):
                    na = int(n_node[dev][g])
                    if not g_mask[dev][g] or na == 0:
                        node_offset += na
                        continue

                    pe = float(pred_e[dev][g])
                    ee = float(exact_e[dev][g])
                    pf = pred_f[dev][node_offset:node_offset + na]
                    ef = exact_f[dev][node_offset:node_offset + na]
                    ps = pred_s[dev][g]
                    es = exact_s[dev][g]
                    node_offset += na

                    # Per-atom energy error
                    e_diff = (pe - ee) / na
                    mae_e = abs(e_diff)

                    # Force error
                    f_diff = pf - ef
                    mae_f = float(np.mean(np.abs(f_diff)))

                    # Stress error
                    s_diff = ps - es
                    ae_stress = float(np.mean(np.abs(s_diff)))

                    # Accumulate
                    acc['mae_e'].append(mae_e)
                    acc['mae_f'].append(mae_f)
                    acc['se_e'].append(e_diff ** 2)
                    acc['se_f'].append(float(np.mean(f_diff ** 2)))
                    acc['ae_stress'].append(ae_stress)
                    acc['se_stress'].append(float(np.mean(s_diff ** 2)))
                    acc['n_atoms'] += na
                    acc['n_structures'] += 1

                    # Write to GPU log
                    idx = gpu_data_idx[dev]
                    gpu_files[dev].write(
                        f"{timestamp}    {idx:<7d}"
                        f"| {pe:<16.7f} {mae_e:<16.11f} {mae_f:<17.11f}"
                        f"| {ee:<17.7f}|\n"
                    )
                    gpu_data_idx[dev] += 1

            total_batches += 1

    # Close GPU log files
    for dev in range(n_devices):
        acc = gpu_accumulators[dev]
        gpu_files[dev].write("-" * 102 + '\n')
        if acc['n_structures'] > 0:
            gpu_files[dev].write(
                f"GPU {dev} | Structures: {acc['n_structures']} | "
                f"E_MAE: {np.mean(acc['mae_e']):.6f} | "
                f"F_MAE: {np.mean(acc['mae_f']):.6f}\n"
            )
        gpu_files[dev].close()
        print(f"  GPU {dev} log saved: eval_gpu{dev}.log ({acc['n_structures']} structures)")

    # Merge all GPU accumulators for summary
    all_mae_e = []
    all_mae_f = []
    all_se_e = []
    all_se_f = []
    all_ae_stress = []
    all_se_stress = []
    total_atoms = 0
    total_structures = 0

    for acc in gpu_accumulators:
        all_mae_e.extend(acc['mae_e'])
        all_mae_f.extend(acc['mae_f'])
        all_se_e.extend(acc['se_e'])
        all_se_f.extend(acc['se_f'])
        all_ae_stress.extend(acc['ae_stress'])
        all_se_stress.extend(acc['se_stress'])
        total_atoms += acc['n_atoms']
        total_structures += acc['n_structures']

    # Compute aggregate metrics
    mean_mae_e = float(np.mean(all_mae_e)) if all_mae_e else 0.0
    mean_mae_f = float(np.mean(all_mae_f)) if all_mae_f else 0.0
    energy_rmse = float(np.sqrt(np.mean(all_se_e))) if all_se_e else 0.0
    force_rmse = float(np.sqrt(np.mean(all_se_f))) if all_se_f else 0.0
    stress_mae = float(np.mean(all_ae_stress)) if all_ae_stress else 0.0
    stress_rmse = float(np.sqrt(np.mean(all_se_stress))) if all_se_stress else 0.0

    e_w = config.get('energy_weight', 1.0)
    f_w = config.get('force_weight', 1.0)
    s_w = config.get('stress_weight', 1.0)
    total_loss = e_w * mean_mae_e + f_w * mean_mae_f + s_w * stress_mae

    # Write summary log
    summary_lines = [
        "=" * 70,
        f"Evaluation Summary ({n_devices} GPUs, {total_structures} structures)",
        "=" * 70,
        f"  Total graphs : {total_structures}",
        f"  Total atoms  : {total_atoms}",
        "",
        f"  Total Loss   : {total_loss:.6f}",
        "",
        "  Per-atom Energy (eV/atom):",
        f"    MAE  : {mean_mae_e:.6f}",
        f"    RMSE : {energy_rmse:.6f}",
        "",
        "  Force (eV/A):",
        f"    MAE  : {mean_mae_f:.6f}",
        f"    RMSE : {force_rmse:.6f}",
        "",
        "  Stress (eV/A^3):",
        f"    MAE  : {stress_mae:.6f}",
        f"    RMSE : {stress_rmse:.6f}",
        "=" * 70,
        "",
        "* NUMBER OF PARAMETERS:",
        f" - MODEL(TOTAL)   {n_params}",
        f" --- HIDDEN.      {config['hidden_irreps']}",
        f" --- N_LAYERS.    {config['n_layers']}",
        f" --- RADI. BASIS. {config['radial_basis_size']}",
        "=" * 70,
    ]

    summary_path = log_dir / 'eval_summary.log'
    with open(summary_path, 'w') as f:
        f.write('\n'.join(summary_lines) + '\n')

    # Print summary
    for line in summary_lines:
        print(line)
    print(f"\nSummary saved to {summary_path}")

    # Save results JSON
    metrics = {
        'total_loss': total_loss,
        'energy_mae': mean_mae_e,
        'energy_rmse': energy_rmse,
        'force_mae': mean_mae_f,
        'force_rmse': force_rmse,
        'stress_mae': stress_mae,
        'stress_rmse': stress_rmse,
        'n_structures': total_structures,
        'n_atoms': total_atoms,
        'n_params': n_params,
    }

    results_path = log_dir / 'eval_results.json'
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {results_path}")

    return metrics


# =============================================================================
# Helpers
# =============================================================================

def compute_atom_energies(data_path: str, element: list) -> np.ndarray:
    """Compute per-element reference energies from any ASE-readable file."""
    print(f"Computing atom energies from: {data_path}")
    traj = list(tqdm(iread(data_path, index=":"), desc="Reading structures"))
    print(f"  Loaded {len(traj)} structures")

    tgt_enr = np.array([atoms.get_potential_energy() for atoms in traj])

    uniq_element = {int(e): i for i, e in enumerate(element)}
    element_counts = {i: np.array([(atoms.numbers == e).sum() for atoms in traj])
                      for e, i in uniq_element.items()}
    c0 = np.array([element_counts[i] for i in range(len(element))])

    m0 = tgt_enr.sum() / c0.sum()
    w0 = np.array([m0 for _ in element])

    def loss_fn(weight, count):
        prd_enr = np.einsum('i,ij->j', weight, count)
        diff = tgt_enr - prd_enr
        return (diff * diff).mean()

    results = minimize(loss_fn, x0=w0, args=(c0,), method='BFGS')
    atom_energies = results.x

    from ase.data import chemical_symbols
    print(f"  Computed atom energies:")
    for idx, z in enumerate(element):
        print(f"    {chemical_symbols[z]} (Z={z}): {atom_energies[idx]:.6f} eV")

    return atom_energies


# =============================================================================
# Main
# =============================================================================

def main():
    # ── Load config ──
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('input.json')

    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    predict_config = config['predict']
    ckpt_path = Path(predict_config['model'])
    test_path = Path(predict_config['test_path'])

    # Load checkpoint
    print(f"Loading checkpoint from {ckpt_path}...")
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    # Load atom energies
    if 'config' in ckpt:
        print("Using config from checkpoint")
        model_config = ckpt['config']
        model_config['predict'] = predict_config
        config = model_config
    elif 'atom_energies_path' in config:
        ae_path = Path(config['atom_energies_path'])
        print(f"Loading atom energies from {ae_path}")
        with open(ae_path) as f:
            ae_data = json.load(f)
        config['atom_energies'] = ae_data['atom_energies']
        config['atomic_numbers'] = ae_data['atomic_numbers']
    else:
        data_path = config['data']
        element = config.get('element', config.get('atomic_numbers'))
        if element is None:
            print("  Auto-detecting elements from data file...")
            traj_peek = list(iread(data_path, index=":"))
            all_elements = set()
            for atoms in traj_peek:
                all_elements.update(atoms.numbers)
            element = sorted(list(all_elements))
            print(f"  Found elements: {element}")
        config['atom_energies'] = compute_atom_energies(data_path, element)
        config['atomic_numbers'] = element

    # ── Setup mesh ──
    mesh, n_devices = setup_mesh()

    print("=" * 70)
    print("Model Evaluation (Parallel)")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Device count: {n_devices}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test data: {test_path}")
    print("=" * 70)

    # ── Create model ──
    print("Creating model...")
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
        avg_n_neighbors=config.get("avg_n_neighbors", 25.0),
        atom_energies=config["atom_energies"],
        l_train=True,
        rngs=nnx.Rngs(42)
    )
    graphdef, _ = nnx.split(model, nnx.Param)

    # ── Load params ──
    if 'ema_params' in ckpt:
        print("Using EMA parameters")
        params = ckpt['ema_params']
    else:
        print("Using regular parameters")
        params = ckpt['params']

    print(f"Checkpoint step: {ckpt.get('step', 'N/A')}")
    if ckpt.get('best_val_loss') is not None:
        print(f"Checkpoint best_val_loss: {ckpt['best_val_loss']:.6f}")

    # ── Load test data ──
    print(f"\nLoading test data from {test_path}...")
    PKL_SUFFIXES = {'.pkl', '.pickle'}

    if test_path.is_dir():
        import re
        def natsorted(lst):
            def natural_key(s):
                return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]
            return sorted(lst, key=natural_key)
        test_files = natsorted(list(test_path.glob('*.pkl')))
        print(f"Found {len(test_files)} test pkl files")

    elif test_path.suffix in PKL_SUFFIXES:
        test_files = [test_path]
        print(f"Single test file: {test_path}")

    else:
        import tempfile
        print(f"  Converting {test_path.suffix} format to graph data...")
        traj = list(tqdm(iread(str(test_path), index=":"), desc="Reading test structures"))
        cutoff = config.get('cutoff', 6.0)
        atom_energies_arr = np.array(config['atom_energies'])
        atomic_numbers = config.get('atomic_numbers', config.get('element'))
        atom_indices = {int(z): i for i, z in enumerate(atomic_numbers)}

        graphs = []
        for atoms in tqdm(traj, desc="Converting to graphs"):
            g = atoms_to_graph_with_targets(
                atoms, cutoff=cutoff,
                atom_energies=atom_energies_arr, atom_indices=atom_indices,
            )
            if g is not None:
                graphs.append(g)

        if len(graphs) < len(traj):
            print(f"  Warning: {len(traj) - len(graphs)} structures had no neighbors (skipped)")

        tmp_path = Path(tempfile.mktemp(suffix='.pkl'))
        with open(tmp_path, 'wb') as f:
            pickle.dump(graphs, f)
        test_files = [tmp_path]
        print(f"  Converted {len(graphs)} structures -> {tmp_path}")

    # ── Run evaluation ──
    print("\n" + "=" * 70)
    print(f"Running Evaluation ({n_devices} GPUs)")
    print("=" * 70 + "\n")

    loss_type = config.get('loss_type', 'mae')
    loss_delta = config.get('huber_delta', 0.02)
    energy_weight = config.get('energy_weight', 1.0)
    force_weight = config.get('force_weight', 1.0)
    stress_weight = config.get('stress_weight', 1.0)
    batch_size = config.get('batch_size', 16)

    metrics = evaluate(
        graphdef=graphdef,
        params=params,
        files_pkl=[str(f) for f in test_files],
        batch_size=batch_size,
        n_devices=n_devices,
        mesh=mesh,
        energy_weight=energy_weight,
        force_weight=force_weight,
        stress_weight=stress_weight,
        loss_type=loss_type,
        loss_delta=loss_delta,
    )

    # ── Print summary ──
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    print("=" * 70)
    print("Evaluation Results")
    print("=" * 70)
    print(f"Total graphs: {metrics['n_graphs']}")
    print(f"Total atoms:  {metrics['n_atoms']}")
    print()
    print(f"Total Loss:   {metrics['total_loss']:.6f}")
    print()
    print("Per-atom Energy (eV/atom):")
    print(f"  MAE:  {metrics['energy_mae']:.6f}")
    print(f"  RMSE: {metrics['energy_rmse']:.6f}")
    print()
    print("Force (eV/A):")
    print(f"  MAE:  {metrics['force_mae']:.6f}")
    print(f"  RMSE: {metrics['force_rmse']:.6f}")
    print()
    print("Stress (eV/A^3):")
    print(f"  MAE:  {metrics['stress_mae']:.6f}")
    print(f"  RMSE: {metrics['stress_rmse']:.6f}")
    print("=" * 70)
    print()
    print("* NUMBER OF PARAMETERS:")
    print(f" - MODEL(TOTAL)   {n_params:,}")
    print(f" --- HIDDEN.      {config['hidden_irreps']}")
    print(f" --- N_LAYERS.    {config['n_layers']}")
    print(f" --- RADI. BASIS. {config['radial_basis_size']}")
    print("=" * 70)

    # Save results to JSON
    results_path = Path('eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
