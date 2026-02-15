"""Evaluation script for trained NequIP/RACE model.

All settings are read from input_omat24.json.
Required keys in JSON: ckpt, test, data

Usage:
    python eval.py
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

from bam_omat24.models.race_nnx import RACE
from bam_omat24.data.data_nnx import Dataset, BucketedDataLoader, MultiDeviceDataLoader, atoms_to_graph
from bam_omat24.training.losses import huber_loss, mae_loss, mse_loss, LOSS_FUNCTIONS
from bam_omat24.training.sharding import (
    setup_mesh, replicate, replicate_pytree, squeeze_batch,
)

jax.config.update("jax_enable_x64", False)


# =============================================================================
# Sharded Evaluation
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
    """Evaluate model on test set.

    Supports both single file and multiple files (directory of .pkl).
    Files are loaded one at a time to save memory.
    """
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
# Main
# =============================================================================

def compute_atom_energies(data_path: str, element: list) -> np.ndarray:
    """Compute per-element reference energies from any ASE-readable file.

    Uses least-squares optimization (BFGS) to find per-element energies
    that best reproduce total energies. Same logic as calc_energy_per_element.py.

    Args:
        data_path: Path to ASE-readable file (traj, xyz, POSCAR, etc.)
        element: List of atomic numbers present in the data.

    Returns:
        atom_energies as numpy array (indexed by element order).
    """
    print(f"Computing atom energies from: {data_path}")
    traj = list(tqdm(iread(data_path, index=":"), desc="Reading structures"))
    print(f"  Loaded {len(traj)} structures")

    # Target energies
    tgt_enr = np.array([atoms.get_potential_energy() for atoms in traj])

    # Element counts per structure
    uniq_element = {int(e): i for i, e in enumerate(element)}
    element_counts = {i: np.array([(atoms.numbers == e).sum() for atoms in traj])
                      for e, i in uniq_element.items()}
    c0 = np.array([element_counts[i] for i in range(len(element))])

    # Initial guess: average energy per atom
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


def evaluate_verbose(graphdef, params, test_files, config, log_path='eval_log.txt'):
    """Per-structure evaluation with detailed formatted logging.

    Outputs:
      1) Per-structure table: timestamp, predicted energy, MAE_E, MAE_F, exact energy
      2) Aggregate summary: Total Loss, MAE/RMSE for energy, force, stress

    Args:
        graphdef: Model graph definition
        params: Model parameters (from checkpoint)
        test_files: List of paths to test pkl files
        config: Model configuration dict
        log_path: Path to save the formatted log

    Returns:
        Dict with aggregate metrics
    """
    from datetime import datetime

    # JIT-compiled forward pass
    @jax.jit
    def predict(graphdef, params, data):
        model = nnx.merge(graphdef, params)
        return model(data)

    # Count parameters
    n_params = sum(x.size for x in jax.tree_util.tree_leaves(params))

    # Accumulators for per-structure metrics
    all_mae_e = []       # per-atom energy MAE
    all_mae_f = []       # force MAE
    all_se_e = []        # per-atom energy SE  (for RMSE)
    all_se_f = []        # force SE per component (for RMSE)
    all_ae_stress = []   # stress AE per component
    all_se_stress = []   # stress SE per component
    total_atoms = 0
    lines = []

    def _print_and_log(line):
        print(line)
        lines.append(line)

    # ── Per-structure table ──
    sep = "-" * 102
    _print_and_log(f"{'':30s}| {'PREDICT__________________________________________':42s}| {'EXACT____________':17s}")
    _print_and_log(f"{'MM/DD/YYYY HH/MM/SS':20s} {'DATA':7s} | {'ENERGY':16s} {'MAE_E':16s} {'MAE_F':17s}| {'ENERGY':17s}|")
    _print_and_log(sep)

    data_idx = 0
    for fname in test_files:
        with open(str(fname), 'rb') as f:
            graphs = pickle.load(f)

        for graph in graphs:
            energy, forces, stress = predict(graphdef, params, graph)

            pred_energy = float(energy[0])
            exact_energy = float(graph.globals['energy'][0])
            n_atoms = int(graph.n_node[0])
            total_atoms += n_atoms

            # Per-atom energy error
            e_diff = (pred_energy - exact_energy) / n_atoms
            mae_e = abs(e_diff)

            # Force error (component-wise)
            pred_forces = np.array(forces[:n_atoms])
            exact_forces = np.array(graph.nodes['forces'][:n_atoms])
            f_diff = pred_forces - exact_forces
            mae_f = float(np.mean(np.abs(f_diff)))

            # Stress error
            pred_stress = np.array(stress[0])       # (6,)
            exact_stress = np.array(graph.globals['stress'][0])  # (6,)
            s_diff = pred_stress - exact_stress

            # Accumulate
            all_mae_e.append(mae_e)
            all_mae_f.append(mae_f)
            all_se_e.append(e_diff ** 2)
            all_se_f.append(float(np.mean(f_diff ** 2)))
            all_ae_stress.append(float(np.mean(np.abs(s_diff))))
            all_se_stress.append(float(np.mean(s_diff ** 2)))

            timestamp = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
            _print_and_log(
                f"{timestamp}    {data_idx:<7d}"
                f"| {pred_energy:<16.7f} {mae_e:<16.11f} {mae_f:<17.11f}"
                f"| {exact_energy:<17.7f}|"
            )
            data_idx += 1

    n_structures = data_idx
    mean_mae_e = float(np.mean(all_mae_e)) if all_mae_e else 0.0
    mean_mae_f = float(np.mean(all_mae_f)) if all_mae_f else 0.0

    _print_and_log(sep)

    # ── Aggregate evaluation results ──
    energy_rmse = float(np.sqrt(np.mean(all_se_e))) if all_se_e else 0.0
    force_rmse = float(np.sqrt(np.mean(all_se_f))) if all_se_f else 0.0
    stress_mae = float(np.mean(all_ae_stress)) if all_ae_stress else 0.0
    stress_rmse = float(np.sqrt(np.mean(all_se_stress))) if all_se_stress else 0.0

    e_w = config.get('energy_weight', 1.0)
    f_w = config.get('force_weight', 1.0)
    s_w = config.get('stress_weight', 1.0)
    total_loss = e_w * mean_mae_e + f_w * mean_mae_f + s_w * stress_mae

    _print_and_log("")
    _print_and_log("=" * 70)
    _print_and_log("Evaluation Results")
    _print_and_log("=" * 70)
    _print_and_log(f"Total graphs: {n_structures}")
    _print_and_log(f"Total atoms:  {total_atoms}")
    _print_and_log("")
    _print_and_log(f"Total Loss:   {total_loss:.6f}")
    _print_and_log("")
    _print_and_log("Per-atom Energy (eV/atom):")
    _print_and_log(f"  MAE:  {mean_mae_e:.6f}")
    _print_and_log(f"  RMSE: {energy_rmse:.6f}")
    _print_and_log("")
    _print_and_log("Force (eV/Å):")
    _print_and_log(f"  MAE:  {mean_mae_f:.6f}")
    _print_and_log(f"  RMSE: {force_rmse:.6f}")
    _print_and_log("")
    _print_and_log("Stress (eV/Å³):")
    _print_and_log(f"  MAE:  {stress_mae:.6f}")
    _print_and_log(f"  RMSE: {stress_rmse:.6f}")
    _print_and_log("=" * 70)

    # ── Model info ──
    _print_and_log("")
    _print_and_log("* NUMBER OF PARAMETERS:")
    _print_and_log(f" - MODEL(TOTAL)   {n_params}")
    _print_and_log(f" --- HIDDEN.      {config['hidden_irreps']}")
    _print_and_log(f" --- N_LAYERS.    {config['n_layers']}")
    _print_and_log(f" --- RADI. BASIS. {config['radial_basis_size']}")

    # Save log file
    with open(log_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nLog saved to {log_path}")

    return {
        'total_loss': total_loss,
        'energy_mae': mean_mae_e,
        'energy_rmse': energy_rmse,
        'force_mae': mean_mae_f,
        'force_rmse': force_rmse,
        'stress_mae': stress_mae,
        'stress_rmse': stress_rmse,
        'n_structures': n_structures,
        'n_atoms': total_atoms,
        'n_params': n_params,
    }


def main():
    # ── Load config (all settings including eval paths are in JSON) ──
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('input_omat24.json')

    print(f"Loading config from {config_path}")
    with open(config_path) as f:
        config = json.load(f)

    ckpt_path = Path(config['ckpt'])
    test_path = Path(config['test'])

    # Load checkpoint early to check for embedded config
    print(f"Loading checkpoint from {ckpt_path}...")
    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    # Use config from checkpoint if available (includes atom_energies)
    if 'config' in ckpt:
        print("Using config from checkpoint")
        model_config = ckpt['config']
        # Override with eval-specific paths from input json
        model_config.update({k: v for k, v in config.items() if k in ('ckpt', 'test', 'data')})
        config = model_config
    else:
        # Fall back: compute atom energies from data file
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

    print("=" * 70)
    print("Model Evaluation")
    print("=" * 70)
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print(f"Config: {config_path}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Test data: {test_path}")
    print("=" * 70)

    # Create model (for graphdef)
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
        l_train=False,  # Evaluation mode
        rngs=nnx.Rngs(42)
    )
    graphdef, _ = nnx.split(model, nnx.Param)

    # Use EMA params if available, otherwise regular params
    if 'ema_params' in ckpt:
        print("Using EMA parameters")
        params = ckpt['ema_params']
    else:
        print("Using regular parameters")
        params = ckpt['params']

    print(f"Checkpoint step: {ckpt.get('step', 'N/A')}")
    if ckpt.get('best_val_loss') is not None:
        print(f"Checkpoint best_val_loss: {ckpt['best_val_loss']:.6f}")

    # Load test data - support pkl, traj, xyz, and other ASE-readable formats
    print(f"\nLoading test data from {test_path}...")
    PKL_SUFFIXES = {'.pkl', '.pickle'}

    if test_path.is_dir():
        # Directory of pkl files
        import re
        def natsorted(lst):
            def natural_key(s):
                return [int(t) if t.isdigit() else t.lower() for t in re.split(r'(\d+)', str(s))]
            return sorted(lst, key=natural_key)
        test_files = natsorted(list(test_path.glob('*.pkl')))
        print(f"Found {len(test_files)} test pkl files")

    elif test_path.suffix in PKL_SUFFIXES:
        # Single pkl file
        test_files = [test_path]
        print(f"Single test file: {test_path}")

    else:
        # ASE-readable format (traj, xyz, POSCAR, etc.) → convert to graph
        import tempfile
        from bam_omat24.data.data_nnx import atoms_to_graph

        print(f"  Converting {test_path.suffix} format to graph data...")
        traj = list(tqdm(iread(str(test_path), index=":"), desc="Reading test structures"))
        cutoff = config.get('cutoff', 6.0)

        graphs = []
        for atoms in tqdm(traj, desc="Converting to graphs"):
            g = atoms_to_graph(atoms, cutoff=cutoff)
            # Replace placeholder zeros with actual DFT values
            energy = np.array([atoms.get_potential_energy()], dtype=np.float32)
            forces = atoms.get_forces().astype(np.float32)
            try:
                stress = atoms.get_stress(voigt=True).reshape(1, 6).astype(np.float32)
            except Exception:
                stress = np.zeros((1, 6), dtype=np.float32)

            g = g._replace(
                nodes={**g.nodes, 'forces': forces},
                globals={**g.globals, 'energy': energy, 'stress': stress}
            )
            graphs.append(g)

        # Save as temporary pkl for evaluate_verbose()
        tmp_path = Path(tempfile.mktemp(suffix='.pkl'))
        with open(tmp_path, 'wb') as f:
            pickle.dump(graphs, f)
        test_files = [tmp_path]
        print(f"  Converted {len(graphs)} structures → {tmp_path}")

    # Run per-structure evaluation with formatted output
    print("\n" + "=" * 70)
    print("Running Evaluation")
    print("=" * 70 + "\n")

    metrics = evaluate_verbose(
        graphdef=graphdef,
        params=params,
        test_files=[str(f) for f in test_files],
        config=config,
        log_path='eval_log.txt',
    )

    # Save results to JSON
    results_path = Path('eval_results.json')
    with open(results_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
