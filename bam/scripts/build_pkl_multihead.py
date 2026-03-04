"""Convert .traj/.xyz files to .pkl graph files for training.

Unified script: replaces build_pkl.py and build_pkl_multi_input.py.
- Single or multiple input files (--input file1.traj file2.traj ...)
- Large file chunked reading (--chunk-size)
- Non-periodic molecular data support (fix_cell_for_nonperiodic)
- Per-element reference energy fitting (--fit-energies)

================================================================================
Usage
================================================================================

1) OMat24 replay (built-in ATOM_ENERGIES, no fitting needed):

    python -m bam.scripts.build_pkl_multihead \
        --input data/omat24_train.traj \
        --output data/omat24_train_pkl \
        --prefix replay_train \
        --chunk-size 100000

2) Single dataset training (fit E0 from data, saves atom_energies.json):

    python -m bam.scripts.build_pkl_multihead \
        --input data/train.traj \
        --output data/train_pkl \
        --prefix train \
        --fit-energies

3) Multiple files combined (fit E0 from all files):

    python -m bam.scripts.build_pkl_multihead \
        --input mptrj_train.traj salex_train.traj \
        --output data/combined_train_pkl \
        --prefix combined_train \
        --fit-energies \
        --chunk-size 100000

4) Validation/test set (use fitted E0 from training step):

    python -m bam.scripts.build_pkl_multihead \
        --input data/valid.traj \
        --output data/valid_pkl \
        --prefix valid \
        --atom-energies atom_energies.json

================================================================================
Arguments
================================================================================

    --input          One or more .traj/.xyz input files (required)
    --output         Output directory for .pkl files (required)
    --prefix         Prefix for .pkl filenames (required)
    --cutoff         Neighbor list cutoff radius (default: 6.0)
    --graphs-per-file  Max graphs per .pkl file (default: 50000)
    --chunk-size     Read in chunks of this size; 0 = read all at once (default: 0)
    --fit-energies   Fit per-element E0 via least-squares and save to atom_energies.json
    --atom-energies  Path to atom_energies.json from a previous --fit-energies run

================================================================================
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


def fix_cell_for_nonperiodic(atoms, cutoff):
    """Set a bounding-box cell for non-periodic systems so matscipy can compute neighbor lists."""
    if not any(atoms.pbc):
        # Save calculator results before modifying positions
        energy = atoms.calc.results.get('energy') if atoms.calc else None
        forces = atoms.calc.results.get('forces') if atoms.calc else None

        pos = atoms.positions
        max_range = np.ptp(pos, axis=0).max()
        L = max_range + 2 * cutoff + 2.0
        atoms.cell = [L, L, L]
        atoms.center()

        # Restore calculator after center() invalidates it
        if energy is not None or forces is not None:
            from ase.calculators.singlepoint import SinglePointCalculator
            calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
            atoms.calc = calc
    return atoms


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
    input_paths,
    output_dir: str,
    prefix: str,
    cutoff: float = 6.0,
    graphs_per_file: int = 50000,
    chunk_size: int = 0,
    atom_energies: np.ndarray = None,
    atom_indices: dict = None,
):
    """Convert trajectory/xyz file(s) to pkl files.

    Processes multiple input files sequentially, with continuous pkl numbering.

    Args:
        input_paths: Path (str) or list of paths to .traj or .xyz files.
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

    # Normalize to list
    if isinstance(input_paths, str):
        input_paths = [input_paths]

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

    for file_i, input_path in enumerate(input_paths):
        print(f"\n--- File {file_i+1}/{len(input_paths)}: {input_path} ---")

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
                    atoms = fix_cell_for_nonperiodic(atoms, cutoff)
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
                atoms = fix_cell_for_nonperiodic(atoms, cutoff)
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

    # Save remaining graphs
    if len(graphs) > 0:
        ipkl = save_graphs(graphs, ipkl)

    print(f"\n{'='*50}")
    print(f"Completed!")
    print(f"Input files: {len(input_paths)}")
    print(f"Total processed: {total_processed}")
    print(f"Total skipped: {skipped}")
    print(f"Total pkl files: {ipkl}")
    print(f"Output directory: {out}")
    print(f"{'='*50}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert .traj/.xyz to .pkl for training (single/multi-input supported)")
    parser.add_argument("--input", nargs='+', required=True,
                        help="Input trajectory/xyz file(s)")
    parser.add_argument("--output", required=True,
                        help="Output directory for pkl files")
    parser.add_argument("--prefix", required=True,
                        help="Prefix for pkl filenames")
    parser.add_argument("--cutoff", type=float, default=6.0,
                        help="Neighbor list cutoff (default: 6.0)")
    parser.add_argument("--graphs-per-file", type=int, default=50000,
                        help="Max graphs per pkl file (default: 50000)")
    parser.add_argument("--chunk-size", type=int, default=0,
                        help="Read file in chunks of this size (0=read all at once, default: 0)")
    parser.add_argument("--fit-energies", action="store_true",
                        help="Fit per-element reference energies from the input data "
                             "(saves to atom_energies.json by default)")
    parser.add_argument("--output-energies", type=str, default="atom_energies.json",
                        help="Output filename for fitted atom energies "
                             "(default: atom_energies.json)")
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
        # Fit atom energies from ALL input files combined (chunked reading for memory efficiency)
        fit_chunk = args.chunk_size if args.chunk_size > 0 else 100000
        print(f"Fitting per-element reference energies from input data (chunk_size={fit_chunk})...")

        if len(args.input) == 1 and args.chunk_size == 0:
            # Single small file: use simple path (read all at once + get_enr_avg_per_element)
            print(f"Reading {args.input[0]} for fitting...")
            all_atoms = ase.io.read(args.input[0], index=':')
            print(f"  Read {len(all_atoms)} frames")

            unique_z = sorted(set(z for atoms in all_atoms for z in atoms.get_atomic_numbers()))
            print(f"  Found {len(unique_z)} elements: {[int(z) for z in unique_z]}")

            enr_avg_per_element, uniq_element = get_enr_avg_per_element(all_atoms, unique_z)

            atom_indices = uniq_element
            atom_energies = np.array([enr_avg_per_element[i] for i in range(len(unique_z))])

            del all_atoms
            gc.collect()
        else:
            # Multiple files or large file: chunked fitting
            all_energies = []
            all_element_counts = []  # list of {Z: count} per structure
            unique_z_set = set()
            total_frames = 0

            for inp in args.input:
                print(f"Reading {inp} for fitting...")
                start = 0
                file_frames = 0
                while True:
                    end = start + fit_chunk
                    atoms_list = ase.io.read(inp, index=f'{start}:{end}')
                    if len(atoms_list) == 0:
                        break

                    for atoms in tqdm(atoms_list, desc=f"Fitting {start}-{start+len(atoms_list)}"):
                        all_energies.append(atoms.get_potential_energy())
                        nums = atoms.get_atomic_numbers()
                        unique_z_set.update(nums)
                        counts = {}
                        for z in nums:
                            counts[int(z)] = counts.get(int(z), 0) + 1
                        all_element_counts.append(counts)

                    file_frames += len(atoms_list)
                    if len(atoms_list) < fit_chunk:
                        break
                    start = end
                    del atoms_list
                    gc.collect()

                print(f"  Read {file_frames} frames")
                total_frames += file_frames

            print(f"  Total frames for fitting: {total_frames}")

            # Build arrays for fitting
            unique_z = sorted(unique_z_set)
            print(f"  Found {len(unique_z)} elements: {[int(z) for z in unique_z]}")

            uniq_element = {int(e): i for i, e in enumerate(unique_z)}
            tgt_enr = np.array(all_energies)
            c0 = np.zeros((len(unique_z), len(all_energies)))
            for j, counts in enumerate(all_element_counts):
                for z, cnt in counts.items():
                    c0[uniq_element[z], j] = cnt

            del all_energies, all_element_counts
            gc.collect()

            # Fit
            m0 = tgt_enr.sum() / c0.sum()
            w0 = np.array([m0 for _ in unique_z])

            def loss_fn(weight, count):
                prd_enr = np.einsum('i,ij->j', weight, count)
                diff = (tgt_enr - prd_enr)
                return (diff * diff).mean()

            results = minimize(loss_fn, x0=w0, args=(c0,), method='BFGS')
            print(f"  Fitting converged: {results.success}, residual MSE: {results.fun:.6e}")

            enr_avg_per_element = {i: results.x[i] for i in range(len(unique_z))}
            atom_indices = uniq_element
            atom_energies = np.array([enr_avg_per_element[i] for i in range(len(unique_z))])

            del tgt_enr, c0
            gc.collect()

        # Print fitted values
        from ase.data import chemical_symbols
        print(f"\n  Fitted per-element energies:")
        for z, idx in sorted(atom_indices.items()):
            print(f"    {chemical_symbols[z]:>2s} (Z={z:3d}): {atom_energies[idx]:.6f} eV")

        # Save to JSON
        save_path = args.output_energies
        ae_data = {
            "atom_energies": atom_energies.tolist(),
            "atomic_numbers": [int(z) for z in sorted(atom_indices.keys())],
            "atomic_number_to_index": {str(z): i for z, i in atom_indices.items()},
        }
        with open(save_path, "w") as f:
            json.dump(ae_data, f, indent=2)
        print(f"\n  Saved to {save_path}")
        print(f"  Use --atom-energies {save_path} for validation/test sets")

    else:
        print("Using built-in ATOM_ENERGIES (OMat24)")

    build_pkl(
        input_paths=args.input,
        output_dir=args.output,
        prefix=args.prefix,
        cutoff=args.cutoff,
        graphs_per_file=args.graphs_per_file,
        chunk_size=args.chunk_size,
        atom_energies=atom_energies,
        atom_indices=atom_indices,
    )
