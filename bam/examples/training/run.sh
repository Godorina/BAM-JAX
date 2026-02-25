#!/bin/bash
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bam_jax_nequip

export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_GPU_ALLOCATOR=cuda_malloc_async

CONFIG=${1:-input.json}

eval $(python -c "
import json; c = json.load(open('$CONFIG'))
print(f\"TRAIN_TRAJ={c['train_traj']}\")
print(f\"VALID_TRAJ={c['valid_traj']}\")
")

# Step 1: Fit atom energies + build train pkl (skip if already exists)
if [ ! -d "data/train_pkl" ]; then
    python -m bam.scripts.build_pkl \
        --input $TRAIN_TRAJ \
        --output data/train_pkl \
        --prefix train \
        --fit-energies
else
    echo "data/train_pkl already exists, skipping build."
fi

# Step 2: Build valid pkl (skip if already exists)
if [ ! -d "data/valid_pkl" ]; then
    python -m bam.scripts.build_pkl \
        --input $VALID_TRAJ \
        --output data/valid_pkl \
        --prefix valid \
        --atom-energies atom_energies.json
else
    echo "data/valid_pkl already exists, skipping build."
fi

# Step 3: Train
python -m bam.training.train_sharded $CONFIG
