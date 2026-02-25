#!/bin/bash
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bam_jax_nequip

# Move to script directory
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_GPU_ALLOCATOR=cuda_malloc_async

# === Source data paths ===
OMAT24_TRAIN=/path/to/omat24_train.traj
OMAT24_VALID=/path/to/omat24_valid.traj

LPSC_TRAIN=/path/to/target_train.traj
LPSC_VALID=/path/to/target_valid.traj

# === Step 1: Build OMat24 replay pkl (built-in ATOM_ENERGIES) ===
if [ ! -d "data/omat24_train_pkl" ]; then
    echo "Building OMat24 train pkl (built-in ATOM_ENERGIES)..."
    python -m bam.scripts.build_pkl_multihead \
        --input $OMAT24_TRAIN \
        --output data/omat24_train_pkl \
        --prefix omat24_train \
        --chunk-size 100000
else
    echo "data/omat24_train_pkl already exists, skipping."
fi

if [ ! -d "data/omat24_valid_pkl" ]; then
    echo "Building OMat24 valid pkl (built-in ATOM_ENERGIES)..."
    python -m bam.scripts.build_pkl_multihead \
        --input $OMAT24_VALID \
        --output data/omat24_valid_pkl \
        --prefix omat24_valid \
        --chunk-size 100000
else
    echo "data/omat24_valid_pkl already exists, skipping."
fi

# === Step 2: Build LPSC target pkl (fit E0 from data) ===
if [ ! -d "data/lpsc_train_pkl" ]; then
    echo "Building LPSC train pkl (fitting atom energies)..."
    python -m bam.scripts.build_pkl_multihead \
        --input $LPSC_TRAIN \
        --output data/lpsc_train_pkl \
        --prefix lpsc_train \
        --fit-energies
else
    echo "data/lpsc_train_pkl already exists, skipping."
fi

if [ ! -d "data/lpsc_valid_pkl" ]; then
    echo "Building LPSC valid pkl (using fitted atom_energies.json)..."
    python -m bam.scripts.build_pkl_multihead \
        --input $LPSC_VALID \
        --output data/lpsc_valid_pkl \
        --prefix lpsc_valid \
        --atom-energies atom_energies.json
else
    echo "data/lpsc_valid_pkl already exists, skipping."
fi

# === Step 3: Train multihead ===
python -m bam.training.train_multihead_sharded input_multihead.json
