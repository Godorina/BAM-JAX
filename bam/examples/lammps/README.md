# LAMMPS Examples for BAM-NequIP

Si diamond 16 atoms examples using a BAM-RACE model exported via chemtrain-deploy.

## Prerequisites

- LAMMPS with ML-CHEMTRAIN plugin
- Exported model file: `model-lammps.ptb` (symlinked)
- NVIDIA GPU (CUDA 12)

## Files

- `test_structure.data` — Si diamond unit cell (16 atoms, triclinic)
- `model-lammps.ptb` — Symlink to exported MLIR model
- `run_lammps.sh` — Wrapper script (sets plugin path, GPU, runs LAMMPS)
- `single_point.in` — Single-point energy/force calculation (run 0)
- `nve.in` — NVE microcanonical MD, 100 steps at 300K
- `npt.in` — NpT ensemble MD, 100 steps, T=300K, P=0 bar

## Usage

```bash
bash run_lammps.sh single_point.in
bash run_lammps.sh nve.in
bash run_lammps.sh npt.in
```

## Notes

- `comm_modify cutoff 30.0` is required (5 interaction layers x 6.0 A cutoff)
- `pair_style chemtrain_deploy cuda12` — uses CUDA 12 backend
- `pair_coeff * * model-lammps.ptb 1.1 1.5` — cutoff_inner and cutoff_outer params
- First step is slow due to JAX compilation; subsequent steps are fast
