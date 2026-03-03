# Installation

![Python](https://img.shields.io/badge/Python-≥3.12-blue)
![JAX](https://img.shields.io/badge/JAX-≥0.8.0-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Prerequisites

- Python >= 3.12
- CUDA 12.x (for GPU support)
- conda (Miniconda or Miniforge recommended)

## Requirements

| Package | Version |
|---------|---------|
| JAX | >= 0.8.0 |
| Flax | >= 0.12.0 |
| e3nn-jax | >= 0.20.8 |
| jraph | >= 0.0.6 |
| optax | >= 0.2.5 |
| ASE | >= 3.27.0 |
| matscipy | >= 1.2.0 |

## Step 1: Create Conda Environment

```bash
conda create -n bam_jax python=3.12 -c conda-forge -y
conda activate bam_jax
```

## Step 2: Install JAX (GPU)

JAX requires a CUDA-specific installation. Choose the command matching your CUDA version.

### Check your CUDA version

```bash
nvidia-smi
```

Look for `CUDA Version: XX.X` in the top-right corner of the output.

### CUDA 12.x (recommended)

```bash
pip install "jax[cuda12]>=0.8.0"
```

### CPU only (no GPU)

```bash
pip install "jax>=0.8.0"
```

> **Note:** For other CUDA versions or advanced configurations, refer to the official JAX installation guide:
> https://jax.readthedocs.io/en/latest/installation.html

## Step 3: Install Dependencies

```bash
pip install "flax>=0.12.0" "e3nn-jax>=0.20.8" "jraph==0.0.6.dev0" "optax>=0.2.5" "ase>=3.27.0" "matscipy>=1.2.0"
```

> **Note:** `jraph` only provides dev releases on PyPI, so `0.0.6.dev0` is the correct version to install.

## Step 4: Install BAM

```bash
git clone https://github.com/Godorina/BAM-JAX.git
cd BAM-JAX
pip install -e .
```

## Step 5: Verify Installation

```bash
python -c "
import jax
import flax
import e3nn_jax
import jraph
import optax
import ase
import matscipy
import bam

print(f'JAX:       {jax.__version__}')
print(f'Flax:      {flax.__version__}')
print(f'e3nn-jax:  {e3nn_jax.__version__}')
print(f'jraph:     {jraph.__version__}')
print(f'optax:     {optax.__version__}')
print(f'ASE:       {ase.__version__}')
print(f'matscipy:  {matscipy.__version__}')
print(f'Backend:   {jax.default_backend()}')
print(f'Devices:   {jax.devices()}')
"
```

Expected output (example with 2 GPUs):

```
JAX:       0.9.0.1
Flax:      0.12.4
e3nn-jax:  0.20.8
jraph:     0.0.6.dev0
optax:     0.2.6
ASE:       3.27.0
matscipy:  1.2.0
Backend:   gpu
Devices:   [CudaDevice(id=0), CudaDevice(id=1)]
```

If `Backend: cpu` appears instead of `gpu`, your JAX CUDA installation may not match your driver. Re-check Step 2.

## Quick Start

### Training

1. Prepare your data and configuration file (`input.json`):

```bash
cd bam/examples/training
```

2. Edit `input.json` to set your data paths:

```json
{
    "train_traj": "../data/3BPA/train_300K_train.xyz",
    "valid_traj": "../data/3BPA/train_300K_val.xyz",
    "train_path": "data/train_pkl",
    "valid_path": "data/valid_pkl",
    ...
}
```

3. Run training:

```bash
export CUDA_VISIBLE_DEVICES=0          # select GPU
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95

# Build data
python -m bam.scripts.build_pkl --input <train.xyz> --output data/train_pkl --prefix train --fit-energies
python -m bam.scripts.build_pkl --input <valid.xyz> --output data/valid_pkl --prefix valid --atom-energies atom_energies.json

# Train
python -m bam.training.train_sharded input.json
```

Or simply use the provided script:

```bash
bash run.sh
```

### Evaluation

```bash
cd bam/examples/training/eval
bash run.sh
```

Set `"evaluate_tag": true` in the `"predict"` section of `input.json` to enable evaluation mode.

## Project Structure

```
BAM-JAX/
├── bam/
│   ├── configs/         # Default configuration templates
│   ├── data/            # Data loading and preprocessing
│   ├── examples/        # Training, evaluation, and LAMMPS examples
│   │   ├── training/    # Single-head training example
│   │   ├── multihead/   # Multi-head training example
│   │   ├── lammps/      # LAMMPS MD interface
│   │   └── data/        # Example datasets
│   ├── inference/       # ASE calculator and evaluation
│   ├── lammps/          # LAMMPS integration
│   ├── models/          # NequIP and RACE model architectures
│   ├── scripts/         # Data preprocessing scripts
│   └── training/        # Training loops (sharded, multi-head)
├── pyproject.toml
└── setup.py
```

## Troubleshooting

### JAX does not detect GPU

```bash
python -c "import jax; print(jax.devices())"
```

If this shows `[CpuDevice(id=0)]`:

1. Verify your NVIDIA driver: `nvidia-smi`
2. Reinstall JAX with CUDA support: `pip install "jax[cuda12]>=0.8.0" --force-reinstall`
3. Check for CUDA version mismatch between driver and JAX

### Out of Memory (OOM)

Reduce batch size in `input.json` or limit GPU memory:

```bash
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
```

### Multi-GPU Training

Use `CUDA_VISIBLE_DEVICES` to select specific GPUs:

```bash
export CUDA_VISIBLE_DEVICES=0,1    # use GPU 0 and 1
python -m bam.training.train_sharded input.json
```
~
