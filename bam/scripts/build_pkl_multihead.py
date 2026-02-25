"""Convert .traj/.xyz files to .pkl graph files for multihead training.

Supports both large files (OMat24, chunked reading) and small files.
Uses bam.data.data_nnx.atoms_to_graph_with_targets() for graph construction.

For replay (OMat24): uses built-in ATOM_ENERGIES (no --fit-energies needed).
For target (new dataset): use --fit-energies to compute per-element E0 from data.

Usage:
    # OMat24 replay (built-in ATOM_ENERGIES)
    python build_pkl_multihead.py \
        --input data/omat24_sampled_10pct_train.traj \
        --output data/omat24_train_pkl \
        --prefix omat24_train \
        --chunk-size 100000

    # Target train (fit E0 from data, saves atom_energies.json)
    python build_pkl_multihead.py \
        --input /path/to/target_train.traj \
        --output data/lpsc_train_pkl \
        --prefix lpsc_train \
        --fit-energies

    # Target valid (use fitted E0)
    python build_pkl_multihead.py \
        --input /path/to/target_valid.traj \
        --output data/lpsc_valid_pkl \
        --prefix lpsc_valid \
        --atom-energies atom_energies.json
"""

import argparse
import gc
import json
import pickle
from pathlib import Path

import ase.io
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm

from bam.data.atom_energies import ATOM_ENERGIES, ATOMIC_NUMBER_TO_INDEX
from bam.data.data_nnx import atoms_to_graph_with_targets


def get_enr_avg_per_element(traj, element):
    """Fit per-element reference energies from dataset via least-squares.

    Solves: E_total ≈ Σ_i (n_i * e_i)
    where n_i is the count of element i and e_i is its reference energy.

    Args:
        traj: List of ASE Atoms objects with energies.
        element: Sorted list/array of unique atomic numbers in the dataset.

    Returns:
        enr_avg_per_element: Dict {index: fitted_energy}.
        uniq_element: Dict {atomic_number: index}.
    """
    tgt_enr = np.array([atoms.get_potential_energy() for atoms in traj])

    uniq_element = {int(e): i for i, e in enumerate(element)}
    element_counts = {i: np.array([(atoms.numbers == e).sum() for atoms in traj])
                      for e, i in uniq_element.items()}

    c0 = np.array([element_counts[i] for i in element_counts.keys()])
    m0 = tgt_enr.sum() / c0.sum()
    w0 = np.array([m0 for _ in element])

    def loss_fn(weight, count):
        prd_enr = np.einsum('i,ij->j', weight, count)
        diff = (tgt_enr - prd_enr)
        return (diff * diff).mean()

    results = minimize(loss_fn, x0=w0, args=(c0,), method='BFGS')
    w0 = results.x

    print(f"  Fitting converged: {results.success}, residual MSE: {results.fun:.6e}")

    enr_avg_per_element = {}
    for i, e in enumerate(element):
        enr_avg_per_element[i] = w0[i]

    return enr_avg_per_element, uniq_element


def build_pkl(
    input_path: str,
    output_dir: str,
    prefix: str,
    cutoff: float = 6.0,
    graphs_per_file: int = 50000,
    chunk_size: int = 0,
    atom_energies: np.ndarray = None,
    atom_indices: dict = None,
):
    """Convert trajectory/xyz to pkl files.

    Args:
        input_path: Path to .traj or .xyz file.
        output_dir: Directory to write pkl files.
        prefix: Prefix for pkl filenames (e.g., 'omat24_train').
        cutoff: Neighbor list cutoff radius.
        graphs_per_file: Max graphs per pkl file.
        chunk_size: If >0, read file in chunks of this size (for large files).
                    If 0, read entire file at once.
        atom_energies: Per-species reference energies array.
        atom_indices: Dict mapping atomic number -> species index.
    """
    if atom_energies is None:
        atom_energies = ATOM_ENERGIES
    if atom_indices is None:
        atom_indices = ATOMIC_NUMBER_TO_INDEX

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    graphs = []
    ipkl = 0
    skipped = 0
    total_processed = 0

    def save_graphs(graphs, ipkl):
        pkl_path = out / f'{prefix}_{ipkl:04d}.pkl'
        print(f"Saving {len(graphs)} graphs to {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump(graphs, f)
        return ipkl + 1

    if chunk_size > 0:
        # Chunked reading for large files
        start = 0
        while True:
            end = start + chunk_size
            print(f"\nReading frames {start}:{end} ...")
            atoms_list = ase.io.read(input_path, index=f'{start}:{end}')
            if len(atoms_list) == 0:
                break

            for atoms in tqdm(atoms_list, desc=f"Chunk {start}-{start+len(atoms_list)}"):
                g = atoms_to_graph_with_targets(
                    atoms, cutoff=cutoff,
                    atom_energies=atom_energies, atom_indices=atom_indices,
                )
                if g is not None:
                    graphs.append(g)
                else:
                    skipped += 1
                total_processed += 1

                if len(graphs) >= graphs_per_file:
                    ipkl = save_graphs(graphs, ipkl)
                    graphs = []
                    gc.collect()

            if len(atoms_list) < chunk_size:
                break
            start = end
            del atoms_list
            gc.collect()
    else:
        # Read entire file at once (small files)
        print(f"Reading {input_path} ...")
        atoms_list = ase.io.read(input_path, index=':')
        print(f"Read {len(atoms_list)} frames")

        for atoms in tqdm(atoms_list, desc="Converting"):
            g = atoms_to_graph_with_targets(
                atoms, cutoff=cutoff,
                atom_energies=atom_energies, atom_indices=atom_indices,
            )
            if g is not None:
                graphs.append(g)
            else:
                skipped += 1
            total_processed += 1

    # Save remaining graphs
    if len(graphs) > 0:
        ipkl = save_graphs(graphs, ipkl)

    print(f"\n{'='*50}")
    print(f"Completed!")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped: {skipped}")
    print(f"Total pkl files: {ipkl}")
    print(f"Output directory: {out}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .traj/.xyz to .pkl for multihead training")
    parser.add_argument("--input", required=True, help="Input trajectory/xyz file")
    parser.add_argument("--output", required=True, help="Output directory for pkl files")
    parser.add_argument("--prefix", required=True, help="Prefix for pkl filenames")
    parser.add_argument("--cutoff", type=float, default=6.0, help="Neighbor list cutoff (default: 6.0)")
    parser.add_argument("--graphs-per-file", type=int, default=50000, help="Max graphs per pkl file (default: 50000)")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Read file in chunks of this size (0=read all at once, default: 0)")
    parser.add_argument("--fit-energies", action="store_true",
                        help="Fit per-element reference energies from the input data "
                             "(saves to atom_energies.json)")
    parser.add_argument("--atom-energies", type=str, default=None,
                        help="Path to atom_energies.json (from a previous --fit-energies run). "
                             "Use this for validation/test sets to ensure consistency.")
    args = parser.parse_args()

    # Determine atom energies and index mapping
    atom_energies = ATOM_ENERGIES
    atom_indices = ATOMIC_NUMBER_TO_INDEX

    if args.atom_energies:
        # Load pre-computed atom energies (e.g., from training set fitting)
        print(f"Loading atom energies from {args.atom_energies}")
        with open(args.atom_energies) as f:
            ae_data = json.load(f)
        atom_energies = np.array(ae_data["atom_energies"])
        atom_indices = {int(k): v for k, v in ae_data["atomic_number_to_index"].items()}
        print(f"  Loaded {len(atom_energies)} species energies")

    elif args.fit_energies:
        # Fit atom energies from the input trajectory
        print("Fitting per-element reference energies from input data...")
        print(f"Reading {args.input} for fitting...")
        all_atoms = ase.io.read(args.input, index=':')
        print(f"  Read {len(all_atoms)} frames")

        # Find unique elements
        unique_z = sorted(set(z for atoms in all_atoms for z in atoms.get_atomic_numbers()))
        print(f"  Found {len(unique_z)} elements: {[int(z) for z in unique_z]}")

        # Fit using get_enr_avg_per_element
        enr_avg_per_element, uniq_element = get_enr_avg_per_element(all_atoms, unique_z)

        atom_indices = uniq_element  # {atomic_number: index}
        atom_energies = np.array([enr_avg_per_element[i] for i in range(len(unique_z))])

        # Print fitted values
        from ase.data import chemical_symbols
        print(f"\n  Fitted per-element energies:")
        for z, idx in sorted(atom_indices.items()):
            print(f"    {chemical_symbols[z]:>2s} (Z={z:3d}): {atom_energies[idx]:.6f} eV")

        # Save to JSON
        save_path = "atom_energies.json"
        ae_data = {
            "atom_energies": atom_energies.tolist(),
            "atomic_numbers": [int(z) for z in unique_z],
            "atomic_number_to_index": {str(z): i for z, i in atom_indices.items()},
        }
        with open(save_path, "w") as f:
            json.dump(ae_data, f, indent=2)
        print(f"\n  Saved to {save_path}")
        print(f"  Use --atom-energies {save_path} for validation/test sets")

        del all_atoms
        gc.collect()

    else:
        print("Using built-in ATOM_ENERGIES (OMat24)")

    build_pkl(
        input_path=args.input,
        output_dir=args.output,
        prefix=args.prefix,
        cutoff=args.cutoff,
        graphs_per_file=args.graphs_per_file,
        chunk_size=args.chunk_size,
        atom_energies=atom_energies,
        atom_indices=atom_indices,
    )
