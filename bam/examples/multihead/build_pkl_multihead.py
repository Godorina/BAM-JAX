"""Convert .traj/.xyz files to .pkl graph files for multihead training.

Supports both large files (OMat24, chunked reading) and small files (3BPA).
Uses bam.data.data_nnx.atoms_to_graph_with_targets() for graph construction.

Usage:
    # OMat24 train (large, chunked)
    python build_pkl_multihead.py \
        --input data/omat24_sampled_10pct_train.traj \
        --output data/omat24_train_pkl \
        --prefix omat24_train \
        --chunk-size 100000

    # 3BPA train (small, single pkl)
    python build_pkl_multihead.py \
        --input /path/to/train_300K.xyz \
        --output data/3bpa_train_pkl \
        --prefix 3bpa_train
"""

import argparse
import gc
import pickle
from pathlib import Path

import ase.io
import numpy as np
from tqdm import tqdm

from bam.data.atom_energies import ATOM_ENERGIES, ATOMIC_NUMBER_TO_INDEX
from bam.data.data_nnx import atoms_to_graph_with_targets


def count_frames(path: str) -> int:
    """Count total frames in a trajectory/xyz file."""
    # Try reading just the number of frames
    try:
        n = len(ase.io.read(path, index=':'))
        return n
    except MemoryError:
        # For very large files, estimate from file
        print("File too large to count frames in one pass, using chunk counting...")
        n = 0
        chunk = 10000
        while True:
            try:
                frames = ase.io.read(path, index=f'{n}:{n+chunk}')
                n += len(frames)
                if len(frames) < chunk:
                    break
            except Exception:
                break
        return n


def build_pkl(
    input_path: str,
    output_dir: str,
    prefix: str,
    cutoff: float = 6.0,
    graphs_per_file: int = 50000,
    chunk_size: int = 0,
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
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    atom_energies = ATOM_ENERGIES
    atom_indices = ATOMIC_NUMBER_TO_INDEX

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
        # Read entire file at once (small files like 3BPA)
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
    args = parser.parse_args()

    build_pkl(
        input_path=args.input,
        output_dir=args.output,
        prefix=args.prefix,
        cutoff=args.cutoff,
        graphs_per_file=args.graphs_per_file,
        chunk_size=args.chunk_size,
    )
