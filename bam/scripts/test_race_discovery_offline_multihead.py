"""Multihead RACE model test script for matbench-discovery evaluation.

Offline version for multi-head fine-tuned checkpoints.
Reads .extxyz files directly from local directory. No internet access required.

Selects a single head (head_idx) for inference.

Optimizations:
  - Fixed padding (64, 1024) -> 1 JIT compilation, no recompilation during relaxation
  - JIT warmup before main loop
  - Periodic checkpoint saving (every 500 structures) with resume support

Usage:
    python -m bam.scripts.test_race_discovery_offline_multihead   # single GPU
"""

import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import jax
from tqdm import tqdm

from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.io import read as ase_read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatviz.enums import Key

from matbench_discovery import today

from bam.inference.calculator_multihead import RACEMultiheadCalculator

jax.config.update("jax_enable_x64", False)


def as_dict_handler(obj):
    """JSON serialization handler for pymatgen objects."""
    if hasattr(obj, "as_dict"):
        return obj.as_dict()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


# %%  Configuration
ckpt_path = "/dataset/usr004/hgpark/jax_omat_data/checkpoints_weight_e100f1s1/ckpt_best.pkl"
wbm_data_dir = Path("/dataset/usr004/hgpark/jax_omat_data/test_wbm")
cutoff = 6.0
fmax = 0.05
max_steps = 500
model_name = "race"
save_every = 500  # checkpoint interval

# Multihead settings
num_heads = 2     # number of heads in the fine-tuned model
head_idx = 0      # which head to use (0: first head, 1: second head)

# Fixed padding: WBM data analysis shows 99% fit in (64, 1024)
FIXED_PAD_NODE = 64
FIXED_PAD_EDGE = 1024

# Runtime (set via env vars for multi-GPU splitting)
n_splits = int(os.getenv("RACE_N_SPLITS", "1"))
split_id = int(os.getenv("RACE_SPLIT_ID", "0"))


# %%  Paths
_SCRIPT_DIR = Path(__file__).resolve().parent
out_dir = _SCRIPT_DIR / f"{model_name}/{today}-wbm-IS2RE-FIRE-multihead-h{head_idx}"
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / (f"results-{split_id:>03}.json.gz" if n_splits > 1 else "results.json.gz")
ckpt_save_path = out_dir / (f"checkpoint-{split_id:>03}.json.gz" if n_splits > 1 else "checkpoint.json.gz")


# %%  Load data — read all extxyz files from local directory
print(f"Loading extxyz files from {wbm_data_dir} ...")
extxyz_files = sorted(wbm_data_dir.glob("*.extxyz"))
if not extxyz_files:
    raise FileNotFoundError(f"No .extxyz files found in {wbm_data_dir}")
print(f"Found {len(extxyz_files)} extxyz files")

atoms_list = []
for f in tqdm(extxyz_files, desc="Loading extxyz"):
    structures = ase_read(str(f), index=":")
    for atoms in structures:
        if Key.mat_id not in atoms.info and "material_id" not in atoms.info:
            atoms.info[str(Key.mat_id)] = f.stem
    atoms_list.extend(structures)
print(f"Loaded {len(atoms_list)} structures total")

if n_splits > 1:
    total = len(atoms_list)
    chunk = total // n_splits
    start = split_id * chunk
    end = total if split_id == n_splits - 1 else start + chunk
    atoms_list = atoms_list[start:end]
    print(f"Split {split_id}/{n_splits}: {len(atoms_list)} of {total}")

print(f"Processing {len(atoms_list)} structures")


# %%  Resume from checkpoint if available
relax_results = {}
if ckpt_save_path.exists():
    try:
        df_ckpt = pd.read_json(ckpt_save_path, orient="records", lines=True)
        for _, row in df_ckpt.iterrows():
            mat_id = row[str(Key.mat_id)]
            relax_results[mat_id] = {
                "structure": row[f"{model_name}_structure"],
                "energy": row[f"{model_name}_energy"],
            }
        print(f"Resumed from checkpoint: {len(relax_results)} structures already done")
    except Exception as e:
        print(f"Could not load checkpoint ({e}), starting fresh")
        relax_results = {}


# %%  Setup multihead calculator with fixed padding
race_calc = RACEMultiheadCalculator(
    model_path=ckpt_path,
    cutoff=cutoff,
    num_heads=num_heads,
    head_idx=head_idx,
    fixed_pad_node=FIXED_PAD_NODE,
    fixed_pad_edge=FIXED_PAD_EDGE,
)

# JIT warmup — compile once before the loop starts
race_calc.warmup(n_atoms=8, pad_node=FIXED_PAD_NODE, pad_edge=FIXED_PAD_EDGE)


# %%  Relaxation
def save_checkpoint(results, path):
    """Save intermediate results to disk."""
    if not results:
        return
    df = pd.DataFrame(results).T.add_prefix(f"{model_name}_")
    df.index.name = Key.mat_id
    df.reset_index().to_json(
        path, default_handler=as_dict_handler, orient="records", lines=True
    )


n_success = len(relax_results)
n_failed = 0
relax_times = []
total_start = time.time()

for atoms in tqdm(deepcopy(atoms_list), desc="Relaxing"):
    mat_id = atoms.info.get(Key.mat_id, atoms.info.get("material_id", "unknown"))
    if mat_id in relax_results:
        continue

    try:
        t0 = time.time()
        atoms.calc = race_calc
        FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=fmax, steps=max_steps)
        dt = time.time() - t0

        relax_results[mat_id] = {
            "structure": AseAtomsAdaptor.get_structure(atoms),
            "energy": atoms.get_potential_energy(),
        }
        relax_times.append(dt)
        n_success += 1

        if n_success % 100 == 0:
            avg_t = sum(relax_times[-500:]) / min(len(relax_times), 500)
            elapsed = time.time() - total_start
            remaining = avg_t * (len(atoms_list) - n_success - n_failed)
            print(f"  [{n_success}/{len(atoms_list)}] avg={avg_t:.2f}s/struct, "
                  f"elapsed={elapsed:.0f}s, remaining~{remaining:.0f}s")

        # Periodic checkpoint save
        if n_success % save_every == 0:
            save_checkpoint(relax_results, ckpt_save_path)
            print(f"  Checkpoint saved: {n_success} structures")

    except Exception as e:
        n_failed += 1
        if n_failed <= 3:
            import traceback
            traceback.print_exc()
        else:
            print(f"Failed {mat_id}: {e!r}")

total_time = time.time() - total_start

# %%  Save final results
df_out = pd.DataFrame(relax_results).T.add_prefix(f"{model_name}_")
df_out.index.name = Key.mat_id
df_out.reset_index().to_json(
    out_path, default_handler=as_dict_handler, orient="records", lines=True
)

# Clean up checkpoint after successful completion
if ckpt_save_path.exists():
    ckpt_save_path.unlink()
    print("Checkpoint removed (run complete)")

# Print timing summary
relax_times = np.array(relax_times) if relax_times else np.array([0.0])
print(f"\n{'='*60}")
print(f"Split {split_id} Complete (multihead head_idx={head_idx})")
print(f"{'='*60}")
print(f"  Structures: {n_success} success, {n_failed} failed")
print(f"  Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
print(f"  Per structure: {relax_times.mean():.3f}s avg, {relax_times.std():.3f}s std")
print(f"  Min/Max: {relax_times.min():.3f}s / {relax_times.max():.3f}s")
print(f"  Throughput: {n_success/total_time:.1f} structures/s")
print(f"  Fixed padding: ({FIXED_PAD_NODE}, {FIXED_PAD_EDGE})")
print(f"  Saved to: {out_path}")
print(f"{'='*60}")
