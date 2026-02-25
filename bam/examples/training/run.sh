#!/bin/bash
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bam_jax_nequip

export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_GPU_ALLOCATOR=cuda_malloc_async

CONFIG=${1:-input.json}

TRAIN_TRAJ=$(python -c "import json; print(json.load(open('$CONFIG'))['train_traj'])")
VALID_TRAJ=$(python -c "import json; print(json.load(open('$CONFIG'))['valid_traj'])")
TRAIN_PATH=$(python -c "import json; print(json.load(open('$CONFIG'))['train_path'])")
VALID_PATH=$(python -c "import json; print(json.load(open('$CONFIG'))['valid_path'])")

# Step 1: Fit atom energies + build train pkl (skip if already exists)
if [ ! -d "$TRAIN_PATH" ]; then
    python -m bam.scripts.build_pkl \
        --input $TRAIN_TRAJ \
        --output $TRAIN_PATH \
        --prefix train \
        --fit-energies
else
    echo "$TRAIN_PATH already exists, skipping build."
fi

# Step 2: Build valid pkl (skip if already exists)
if [ ! -d "$VALID_PATH" ]; then
    python -m bam.scripts.build_pkl \
        --input $VALID_TRAJ \
        --output $VALID_PATH \
        --prefix valid \
        --atom-energies atom_energies.json
else
    echo "$VALID_PATH already exists, skipping build."
fi

# Step 3: Train
python -m bam.training.train_sharded $CONFIG
