"""Validate exported model against ASE Calculator reference.

Compares the trained RACE model (via ASE Calculator) with the BAMExporter
output to ensure numerical consistency before LAMMPS deployment.

Success criteria:
    - Energy difference: < 1e-5 eV
    - Max force difference: < 1e-4 eV/A

Usage:
    python -m bam.lammps.validate \
        --ckpt ckpt_best.pkl \
        --config input.json \
        --test-structure test.xyz

    # Or with a preprocessed pickle file:
    python -m bam.lammps.validate \
        --ckpt ckpt_best.pkl \
        --config input.json \
        --pkl data.pkl
"""

import argparse
import json
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from bam.lammps.exporter import BAMExporter
from bam.lammps.test_periodic_compat import build_ghost_atom_graph


def validate_with_ase_calculator(ckpt_path, config, atoms, cutoff):
    """Compare BAMExporter with ASE Calculator on the same structure.

    Args:
        ckpt_path: Path to checkpoint.
        config: Model config dict.
        atoms: ASE Atoms object.
        cutoff: Cutoff radius.

    Returns:
        Tuple of (passed, results_dict).
    """
    from bam.inference.calculator import RACECalculator
    from bam.data.data_nnx import atoms_to_graph

    print("=" * 60)
    print("ASE Calculator vs BAMExporter Validation")
    print("=" * 60)

    n_atoms = len(atoms)
    print(f"\nStructure: {n_atoms} atoms, {atoms.get_chemical_formula()}")

    # --- ASE Calculator (reference, periodic=True) ---
    print("\n[1/2] Computing with ASE Calculator (periodic=True)...")
    calc = RACECalculator(model_path=ckpt_path, cutoff=cutoff)
    atoms.calc = calc
    e_ase = atoms.get_potential_energy()
    f_ase = atoms.get_forces()
    s_ase = atoms.get_stress()  # Voigt 6-component
    print(f"  Energy: {e_ase:.8f} eV")
    print(f"  Max |force|: {np.max(np.abs(f_ase)):.6f} eV/A")

    # --- BAMExporter (periodic=False, ghost atoms) ---
    print("\n[2/2] Computing with BAMExporter (periodic=False + ghost atoms)...")
    exporter = BAMExporter(ckpt_path=ckpt_path, config=config, cutoff=cutoff)

    # Build graph with Sij, then convert to ghost atoms
    graph = atoms_to_graph(atoms, cutoff=cutoff, self_interaction=False)
    ghost_graph, n_real = build_ghost_atom_graph(graph, cutoff)

    ghost_pos = jnp.array(ghost_graph.nodes['positions'])
    ghost_species = jnp.array(ghost_graph.nodes['species'])
    ghost_senders = jnp.array(ghost_graph.senders)
    ghost_receivers = jnp.array(ghost_graph.receivers)

    # Energy
    node_energies = exporter.compute_node_energies(
        ghost_pos, ghost_species, ghost_senders, ghost_receivers
    )
    e_export = float(jnp.sum(node_energies[:n_real]))

    # Forces via jax.grad (same approach as LAMMPS deployment)
    def total_energy_fn(positions):
        data = exporter._build_graph(positions, ghost_species, ghost_senders, ghost_receivers)
        ne = exporter._model.node_energies(positions, None, None, data)
        ne = ne + jax.lax.stop_gradient(
            exporter._model.atom_energies[...][ghost_species, None]
        )
        return jnp.sum(ne[:n_real])

    grad = jax.grad(total_energy_fn)(ghost_pos)
    f_export = np.array(-grad[:n_real])

    print(f"  Energy: {e_export:.8f} eV")
    print(f"  Max |force|: {np.max(np.abs(f_export)):.6f} eV/A")

    # --- Compare ---
    print("\n--- Comparison ---")

    e_diff = abs(e_ase - e_export)
    f_diff = np.abs(f_ase - f_export)
    f_max_diff = np.max(f_diff)
    f_mean_diff = np.mean(f_diff)

    print(f"  |E_ase - E_export|: {e_diff:.2e} eV")
    print(f"  Max |F_ase - F_export|: {f_max_diff:.2e} eV/A")
    print(f"  Mean |F_ase - F_export|: {f_mean_diff:.2e} eV/A")

    e_pass = e_diff < 1e-5
    f_pass = f_max_diff < 1e-4

    print(f"\n  Energy: {'PASS' if e_pass else 'FAIL'} (threshold: 1e-5 eV)")
    print(f"  Forces: {'PASS' if f_pass else 'FAIL'} (threshold: 1e-4 eV/A)")

    passed = e_pass and f_pass
    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")

    results = {
        'e_ase': e_ase,
        'e_export': e_export,
        'e_diff': e_diff,
        'f_max_diff': f_max_diff,
        'f_mean_diff': f_mean_diff,
        'passed': passed,
    }

    return passed, results


def validate_with_pkl(ckpt_path, config, pkl_path, cutoff):
    """Validate using a preprocessed pickle graph (no ASE needed)."""
    print("=" * 60)
    print("Periodic vs Non-Periodic Validation (from pkl)")
    print("=" * 60)

    # Load graph
    with open(pkl_path, 'rb') as f:
        graphset = pickle.load(f)
    graph = graphset[0]
    n_real = int(graph.n_node[0])
    graph_cutoff = float(graph.globals['cutoff'][0])

    if cutoff is None:
        cutoff = graph_cutoff
    print(f"\nGraph: {n_real} atoms, cutoff={graph_cutoff}")

    # --- Reference: periodic=True ---
    print("\n[1/2] Computing with periodic=True model...")
    from bam.models.race_nnx import RACE
    from bam.data.atom_energies import ATOM_ENERGIES

    with open(ckpt_path, 'rb') as f:
        ckpt = pickle.load(f)

    atom_energies = ckpt.get('atom_energies', ATOM_ENERGIES)
    params = ckpt.get('ema_params', ckpt['params'])

    model_periodic = RACE(
        n_species=len(atom_energies),
        lmax=config.get('lmax', 3),
        hidden_irreps=config.get('hidden_irreps', '64x0e + 64x1o + 64x2e'),
        n_layers=config.get('n_layers', 5),
        radial_basis_size=config.get('radial_basis_size', 8),
        radial_mlp_size=config.get('radial_mlp_size', 64),
        radial_mlp_layers=config.get('radial_mlp_layers', 3),
        radial_polynomial_p=config.get('radial_polynomial_p', 2.0),
        mlp_init_scale=config.get('mlp_init_scale', 4.0),
        cutoff=cutoff, shift=0.0, scale=1.0,
        avg_n_neighbors=config.get('avg_n_neighbors', 25.0),
        atom_energies=atom_energies,
        l_train=False, periodic=True, rngs=nnx.Rngs(0),
    )
    graphdef, _ = nnx.split(model_periodic, nnx.Param)
    model_periodic = nnx.merge(graphdef, params)

    e_periodic, f_periodic, s_periodic = model_periodic(graph)
    e_ref = float(e_periodic[0])
    f_ref = np.array(f_periodic[:n_real])
    print(f"  Energy: {e_ref:.8f} eV")

    # --- BAMExporter: periodic=False ---
    print("\n[2/2] Computing with BAMExporter (periodic=False)...")
    exporter = BAMExporter(ckpt_path=ckpt_path, config=config, cutoff=cutoff)

    ghost_graph, _ = build_ghost_atom_graph(graph, cutoff)
    ghost_pos = jnp.array(ghost_graph.nodes['positions'])
    ghost_species = jnp.array(ghost_graph.nodes['species'])
    ghost_senders = jnp.array(ghost_graph.senders)
    ghost_receivers = jnp.array(ghost_graph.receivers)

    node_energies = exporter.compute_node_energies(
        ghost_pos, ghost_species, ghost_senders, ghost_receivers
    )
    e_export = float(jnp.sum(node_energies[:n_real]))

    # Forces
    def total_energy_fn(positions):
        data = exporter._build_graph(positions, ghost_species, ghost_senders, ghost_receivers)
        ne = exporter._model.node_energies(positions, None, None, data)
        ne = ne + jax.lax.stop_gradient(
            exporter._model.atom_energies[...][ghost_species, None]
        )
        return jnp.sum(ne[:n_real])

    grad = jax.grad(total_energy_fn)(ghost_pos)
    f_export = np.array(-grad[:n_real])
    print(f"  Energy: {e_export:.8f} eV")

    # --- Compare ---
    print("\n--- Comparison ---")
    e_diff = abs(e_ref - e_export)
    f_diff = np.abs(f_ref - f_export)
    f_max_diff = np.max(f_diff)
    f_mean_diff = np.mean(f_diff)

    print(f"  |E_periodic - E_export|: {e_diff:.2e} eV")
    print(f"  Max |F_periodic - F_export|: {f_max_diff:.2e} eV/A")
    print(f"  Mean |F_periodic - F_export|: {f_mean_diff:.2e} eV/A")

    e_pass = e_diff < 1e-4
    f_pass = f_max_diff < 1e-3

    print(f"\n  Energy: {'PASS' if e_pass else 'FAIL'} (threshold: 1e-4 eV)")
    print(f"  Forces: {'PASS' if f_pass else 'FAIL'} (threshold: 1e-3 eV/A)")

    passed = e_pass and f_pass
    print(f"\n  Overall: {'PASS' if passed else 'FAIL'}")
    return passed


def main():
    parser = argparse.ArgumentParser(
        description='Validate exported model against ASE Calculator'
    )
    parser.add_argument('--ckpt', required=True, help='Checkpoint path')
    parser.add_argument('--config', required=True, help='Config JSON path')
    parser.add_argument(
        '--test-structure', default=None,
        help='Test structure file (.xyz, .cif, etc.) for ASE Calculator comparison'
    )
    parser.add_argument(
        '--pkl', default=None,
        help='Preprocessed graph pickle file for periodic vs non-periodic comparison'
    )
    parser.add_argument('--cutoff', type=float, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    if 'atom_energies' not in config:
        from bam.data.atom_energies import ATOM_ENERGIES
        config['atom_energies'] = ATOM_ENERGIES.tolist()

    cutoff = args.cutoff or config.get('cutoff', 6.0)

    if args.test_structure:
        from ase.io import read
        atoms = read(args.test_structure)
        passed, _ = validate_with_ase_calculator(
            args.ckpt, config, atoms, cutoff
        )
    elif args.pkl:
        passed = validate_with_pkl(args.ckpt, config, args.pkl, cutoff)
    else:
        # Default: create a test structure
        try:
            from ase.build import bulk
            atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
            np.random.seed(42)
            atoms.positions += np.random.normal(0, 0.05, atoms.positions.shape)
            passed, _ = validate_with_ase_calculator(
                args.ckpt, config, atoms, cutoff
            )
        except ImportError:
            print("No test structure provided and ASE bulk not available.")
            print("Use --test-structure or --pkl to provide input data.")
            sys.exit(1)

    sys.exit(0 if passed else 1)


if __name__ == '__main__':
    main()
