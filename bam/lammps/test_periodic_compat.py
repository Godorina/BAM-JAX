"""Step 1: periodic=False Compatibility Verification.

Tests whether a model trained with periodic=True (Sij-based PBC) produces
identical results when run with periodic=False (direct displacement using
ghost atom coordinates).

The key insight:
    periodic=True:  r = pos[s] - (pos[r] + Sij @ cell)
    periodic=False: r = pos[s] - pos_expanded[r]   (where pos_expanded includes ghost images)

If ghost atom positions are constructed as pos + Sij @ cell, the two r vectors
are mathematically identical. The only other difference is cutoff_per_edge:
    periodic=True:  cutoff from data.globals['cutoff'] (per-graph, possibly adaptive)
    periodic=False: fixed to self.cutoff

For LAMMPS deployment, the cutoff is always fixed, so both paths should match
as long as the same cutoff value is used.

Usage:
    python -m bam.lammps.test_periodic_compat [--ckpt ckpt_best.pkl] [--pkl data.pkl]

If no checkpoint is provided, a random model is used to verify the math.
If no pkl is provided, a test structure is generated with ASE.
"""

import argparse
import pickle
import sys

import jax
import jax.numpy as jnp
import jraph
import numpy as np
from flax import nnx

from bam.models.race_nnx import RACE
from bam.data.atom_energies import ATOM_ENERGIES, ATOMIC_NUMBER_TO_INDEX


def build_ghost_atom_graph(graph, cutoff):
    """Convert a periodic graph (with Sij) into a non-periodic graph with ghost atoms.

    For each edge (i, j) with cell shift Sij != [0,0,0], we create a ghost copy
    of atom j at position pos[j] + Sij @ cell. The graph is then rebuilt with
    direct displacements (no Sij).

    Args:
        graph: jraph.GraphsTuple with periodic structure (has Sij in edges).
        cutoff: Cutoff radius used for the neighbor list.

    Returns:
        New jraph.GraphsTuple with ghost atoms and no Sij (periodic=False compatible).
        Also returns the number of real (non-ghost) atoms.
    """
    positions = np.array(graph.nodes['positions'])
    species = np.array(graph.nodes['species'])
    cell = np.array(graph.globals['cell'][0])  # (3, 3)
    senders = np.array(graph.senders)
    receivers = np.array(graph.receivers)
    Sij = np.array(graph.edges['Sij'])
    n_real = int(graph.n_node[0])

    # For each edge, compute the actual receiver position (with cell shift)
    # ghost_pos[r] = pos[r] + Sij @ cell
    # Then r = pos[s] - ghost_pos[r]
    #
    # Strategy: for unique (receiver, Sij) pairs where Sij != [0,0,0],
    # create ghost atoms. Remap receiver indices to ghost atom indices.

    # Build mapping: (receiver_idx, Sij_tuple) -> ghost_atom_idx
    ghost_map = {}
    ghost_positions = []
    ghost_species = []
    new_senders = []
    new_receivers = []

    for e in range(len(senders)):
        s, r = int(senders[e]), int(receivers[e])
        sij = tuple(Sij[e].tolist())

        new_senders.append(s)

        if sij == (0.0, 0.0, 0.0):
            # Same unit cell — receiver is a real atom
            new_receivers.append(r)
        else:
            # Different unit cell — need ghost atom
            key = (r, sij)
            if key not in ghost_map:
                ghost_idx = n_real + len(ghost_positions)
                ghost_map[key] = ghost_idx
                ghost_pos = positions[r] + np.array(sij) @ cell
                ghost_positions.append(ghost_pos)
                ghost_species.append(species[r])
            new_receivers.append(ghost_map[key])

    # Build expanded arrays
    if ghost_positions:
        all_positions = np.concatenate(
            [positions, np.array(ghost_positions)], axis=0
        ).astype(np.float32)
        all_species = np.concatenate(
            [species, np.array(ghost_species)]
        ).astype(np.int32)
    else:
        all_positions = positions.astype(np.float32)
        all_species = species.astype(np.int32)

    n_total = len(all_positions)
    n_edges = len(new_senders)

    new_graph = jraph.GraphsTuple(
        n_node=np.array([n_total], dtype=np.int32),
        n_edge=np.array([n_edges], dtype=np.int32),
        nodes={
            'species': all_species,
            'positions': all_positions,
            'forces': np.zeros((n_total, 3), dtype=np.float32),
        },
        edges={},  # No Sij needed
        senders=np.array(new_senders, dtype=np.int32),
        receivers=np.array(new_receivers, dtype=np.int32),
        globals={
            'energy': np.zeros(1, dtype=np.float32),
            'cell': np.zeros((1, 3, 3), dtype=np.float32),
            'volume': np.zeros(1, dtype=np.float32),
            'stress': np.zeros((1, 6), dtype=np.float32),
            'cutoff': np.array([cutoff], dtype=np.float32),
        },
    )

    return new_graph, n_real


def test_displacement_equivalence(graph, cutoff):
    """Verify that the two displacement computation paths produce identical r vectors.

    Path A (periodic=True):  r = pos[s] - (pos[r] + Sij @ cell)
    Path B (periodic=False): r = pos_expanded[s] - pos_expanded[r]  (with ghost atoms)
    """
    print("\n--- Test 1: Displacement Vector Equivalence ---")

    positions = np.array(graph.nodes['positions'])
    cell = np.array(graph.globals['cell'][0])  # (3, 3)
    senders = np.array(graph.senders)
    receivers = np.array(graph.receivers)
    Sij = np.array(graph.edges['Sij'])

    # Path A: periodic displacement
    Sij_cart = Sij @ cell  # (E, 3)
    r_periodic = positions[senders] - (positions[receivers] + Sij_cart)

    # Path B: ghost atom displacement
    ghost_graph, n_real = build_ghost_atom_graph(graph, cutoff)
    ghost_positions = np.array(ghost_graph.nodes['positions'])
    ghost_senders = np.array(ghost_graph.senders)
    ghost_receivers = np.array(ghost_graph.receivers)
    r_ghost = ghost_positions[ghost_senders] - ghost_positions[ghost_receivers]

    # Compare
    max_diff = np.max(np.abs(r_periodic - r_ghost))
    mean_diff = np.mean(np.abs(r_periodic - r_ghost))

    print(f"  Number of edges: {len(senders)}")
    print(f"  Number of ghost atoms: {len(ghost_positions) - n_real}")
    print(f"  Max |r_periodic - r_ghost|: {max_diff:.2e}")
    print(f"  Mean |r_periodic - r_ghost|: {mean_diff:.2e}")

    passed = max_diff < 1e-5
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_node_energies_equivalence(graph, model_periodic, model_nonperiodic, cutoff):
    """Compare node_energies from periodic=True vs periodic=False models.

    Both models share the same parameters. The periodic model uses the original
    graph (with Sij), while the non-periodic model uses the ghost atom graph.
    """
    print("\n--- Test 2: Node Energies Equivalence ---")

    # Path A: periodic=True
    positions = jnp.array(graph.nodes['positions'])
    cell = jnp.array(graph.globals['cell'])
    cutoff_arr = jnp.array(graph.globals['cutoff'])
    energies_periodic = model_periodic.node_energies(positions, cell, cutoff_arr, graph)
    n_real = int(graph.n_node[0])

    # Path B: periodic=False with ghost atoms
    ghost_graph, _ = build_ghost_atom_graph(graph, cutoff)
    ghost_positions = jnp.array(ghost_graph.nodes['positions'])
    energies_ghost_all = model_nonperiodic.node_energies(
        ghost_positions, None, None, ghost_graph
    )
    # Only compare real atoms (ghost atom energies don't matter)
    energies_ghost = energies_ghost_all[:n_real]

    # Compare
    diff = np.abs(np.array(energies_periodic) - np.array(energies_ghost))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)

    total_periodic = float(jnp.sum(energies_periodic))
    total_ghost = float(jnp.sum(energies_ghost[:n_real]))

    print(f"  Total energy (periodic=True):  {total_periodic:.8f}")
    print(f"  Total energy (periodic=False): {total_ghost:.8f}")
    print(f"  Total energy difference: {abs(total_periodic - total_ghost):.2e}")
    print(f"  Max per-atom |E_periodic - E_ghost|: {max_diff:.2e}")
    print(f"  Mean per-atom |E_periodic - E_ghost|: {mean_diff:.2e}")

    passed = max_diff < 1e-4
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def test_forces_equivalence(graph, model_periodic, model_nonperiodic, cutoff):
    """Compare forces from periodic=True vs periodic=False models.

    Forces for periodic=True come from grad w.r.t. original positions.
    Forces for periodic=False come from grad w.r.t. expanded positions (real atoms only).
    """
    print("\n--- Test 3: Forces Equivalence ---")

    # Path A: periodic=True — full forward pass with forces
    _, forces_periodic, _ = model_periodic(graph)
    n_real = int(graph.n_node[0])
    forces_periodic = forces_periodic[:n_real]

    # Path B: periodic=False with ghost atoms
    ghost_graph, _ = build_ghost_atom_graph(graph, cutoff)

    # For non-periodic model, we compute forces via jax.grad of node_energies
    # w.r.t. the expanded positions, then extract forces on real atoms
    def total_energy_fn(positions):
        ghost_graph_mod = ghost_graph._replace(
            nodes={**ghost_graph.nodes, 'positions': positions}
        )
        node_energies = model_nonperiodic.node_energies(
            positions, None, None, ghost_graph_mod
        )
        # Sum only real atom energies (LAMMPS would also sum only local atoms)
        return jnp.sum(node_energies[:n_real])

    ghost_positions = jnp.array(ghost_graph.nodes['positions'])
    grad = jax.grad(total_energy_fn)(ghost_positions)
    forces_ghost = -grad[:n_real]

    # Compare
    diff = np.abs(np.array(forces_periodic) - np.array(forces_ghost))
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    max_force = np.max(np.abs(np.array(forces_periodic)))

    print(f"  Max force magnitude (periodic): {max_force:.6f} eV/A")
    print(f"  Max |F_periodic - F_ghost|: {max_diff:.2e} eV/A")
    print(f"  Mean |F_periodic - F_ghost|: {mean_diff:.2e} eV/A")
    if max_force > 0:
        print(f"  Relative max error: {max_diff/max_force:.2e}")

    passed = max_diff < 1e-3
    print(f"  Result: {'PASS' if passed else 'FAIL'}")
    return passed


def create_test_structure():
    """Create a simple test periodic structure using ASE."""
    try:
        from ase.build import bulk
        from bam.data.data_nnx import atoms_to_graph
    except ImportError as e:
        print(f"Cannot create test structure: {e}")
        return None

    # Create a simple periodic structure (e.g., bulk silicon)
    atoms = bulk('Si', 'diamond', a=5.43) * (2, 2, 2)
    # Rattle positions slightly for non-trivial forces
    rng = np.random.default_rng(42)
    atoms.positions += rng.normal(0, 0.05, atoms.positions.shape)

    cutoff = 6.0
    graph = atoms_to_graph(atoms, cutoff=cutoff, self_interaction=False)
    return graph, cutoff


def load_test_graph_from_pkl(pkl_path):
    """Load a graph from a preprocessed pickle file."""
    with open(pkl_path, 'rb') as f:
        graphset = pickle.load(f)
    graph = graphset[0]
    cutoff = float(graph.globals['cutoff'][0])
    return graph, cutoff


def create_model_pair(config=None, ckpt_path=None, cutoff=6.0):
    """Create a pair of models (periodic=True, periodic=False) with shared params.

    If ckpt_path is provided, loads trained parameters. Otherwise uses random init.
    """
    if ckpt_path is not None:
        import json
        with open(ckpt_path, 'rb') as f:
            ckpt = pickle.load(f)

        config = ckpt.get('config', config)
        if config is None:
            raise ValueError("No config found in checkpoint and none provided")

        atom_energies = ckpt.get('atom_energies', ATOM_ENERGIES)
        params = ckpt.get('ema_params', ckpt['params'])
    else:
        atom_energies = ATOM_ENERGIES
        params = None

    if config is None:
        config = {
            'lmax': 3,
            'hidden_irreps': '64x0e + 64x1o + 64x2e',
            'n_layers': 3,
            'radial_basis_size': 8,
            'radial_mlp_size': 64,
            'radial_mlp_layers': 3,
            'radial_polynomial_p': 2.0,
            'mlp_init_scale': 4.0,
            'avg_n_neighbors': 25.0,
        }

    model_kwargs = dict(
        n_species=len(atom_energies),
        lmax=config.get('lmax', 3),
        hidden_irreps=config.get('hidden_irreps', '64x0e + 64x1o + 64x2e'),
        n_layers=config.get('n_layers', 3),
        radial_basis_size=config.get('radial_basis_size', 8),
        radial_mlp_size=config.get('radial_mlp_size', 64),
        radial_mlp_layers=config.get('radial_mlp_layers', 3),
        radial_polynomial_p=config.get('radial_polynomial_p', 2.0),
        mlp_init_scale=config.get('mlp_init_scale', 4.0),
        cutoff=cutoff,
        shift=0.0,
        scale=1.0,
        avg_n_neighbors=config.get('avg_n_neighbors', 25.0),
        atom_energies=atom_energies,
        l_train=False,
    )

    # Create periodic model
    model_periodic = RACE(**model_kwargs, periodic=True, rngs=nnx.Rngs(0))
    # Create non-periodic model (same random init seed)
    model_nonperiodic = RACE(**model_kwargs, periodic=False, rngs=nnx.Rngs(0))

    if params is not None:
        # Load trained params into both models
        graphdef_p, _ = nnx.split(model_periodic, nnx.Param)
        graphdef_np, _ = nnx.split(model_nonperiodic, nnx.Param)
        model_periodic = nnx.merge(graphdef_p, params)
        model_nonperiodic = nnx.merge(graphdef_np, params)

    return model_periodic, model_nonperiodic


def main():
    parser = argparse.ArgumentParser(
        description='Test periodic=False compatibility for LAMMPS deployment'
    )
    parser.add_argument('--ckpt', default=None, help='Checkpoint path (ckpt_best.pkl)')
    parser.add_argument('--pkl', default=None, help='Preprocessed graph pickle file')
    parser.add_argument('--config', default=None, help='Config JSON file')
    args = parser.parse_args()

    print("=" * 60)
    print("periodic=False Compatibility Test for LAMMPS Deployment")
    print("=" * 60)

    # Load or create test graph
    if args.pkl:
        print(f"\nLoading test graph from {args.pkl}...")
        graph, cutoff = load_test_graph_from_pkl(args.pkl)
    else:
        print("\nCreating test structure (bulk Si 2x2x2, rattled)...")
        result = create_test_structure()
        if result is None:
            print("Failed to create test structure. Provide --pkl instead.")
            sys.exit(1)
        graph, cutoff = result

    n_atoms = int(graph.n_node[0])
    n_edges = int(graph.n_edge[0])
    Sij = np.array(graph.edges['Sij'])
    n_ghost_edges = np.sum(np.any(Sij != 0, axis=-1))
    print(f"  Atoms: {n_atoms}, Edges: {n_edges}")
    print(f"  Edges with Sij != 0 (cross-cell): {n_ghost_edges}")
    print(f"  Cutoff: {cutoff} A")

    # Test 1: Pure displacement equivalence (no model needed)
    pass1 = test_displacement_equivalence(graph, cutoff)

    # Create model pair (shared params, different periodic flag)
    config = None
    if args.config:
        import json
        with open(args.config) as f:
            config = json.load(f)

    print(f"\nCreating model pair (periodic=True/False)...")
    if args.ckpt:
        print(f"  Loading trained params from {args.ckpt}")
    else:
        print("  Using random initialization (no checkpoint)")

    model_periodic, model_nonperiodic = create_model_pair(
        config=config, ckpt_path=args.ckpt, cutoff=cutoff
    )

    # Test 2: Node energies equivalence
    pass2 = test_node_energies_equivalence(
        graph, model_periodic, model_nonperiodic, cutoff
    )

    # Test 3: Forces equivalence
    pass3 = test_forces_equivalence(
        graph, model_periodic, model_nonperiodic, cutoff
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_passed = pass1 and pass2 and pass3
    results = [
        ("Displacement vectors", pass1),
        ("Node energies", pass2),
        ("Forces", pass3),
    ]
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")

    if all_passed:
        print("\nAll tests passed. periodic=False is compatible for LAMMPS deployment.")
        print("Proceed to Step 2: BAMExporter implementation using periodic=False.")
    else:
        print("\nSome tests failed. periodic=False is NOT directly compatible.")
        print("Proceed to Step 2 with a dedicated node_energies_lammps() function.")

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())
