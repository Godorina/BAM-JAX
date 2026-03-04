#!/bin/bash
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bam_jax

# Move to script directory
cd "$(dirname "$0")"

export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_GPU_ALLOCATOR=cuda_malloc_async

CONFIG=${1:-input_multihead.json}

eval $(python -c "
import json; c = json.load(open('$CONFIG'))
for h in c['heads']:
    name = h['name'].upper()
    for key in ['train_traj', 'valid_traj']:
        val = h[key]
        envkey = f\"{name}_{key.upper()}\"
        if isinstance(val, list):
            print(f'{envkey}=\"{chr(32).join(val)}\"')
        else:
            print(f'{envkey}=\"{val}\"')
    print(f\"{name}_TRAIN_PATH={h['train_path']}\")
    print(f\"{name}_VALID_PATH={h['valid_path']}\")
")

# === Step 1: Build replay pkl (built-in ATOM_ENERGIES) ===
if [ ! -d "$REPLAY_TRAIN_PATH" ]; then
    echo "Building replay train pkl (built-in ATOM_ENERGIES)..."
    python -m bam.scripts.build_pkl_multihead \
        --input $REPLAY_TRAIN_TRAJ \
        --output $REPLAY_TRAIN_PATH \
        --prefix replay_train \
        --chunk-size 100000
else
    echo "$REPLAY_TRAIN_PATH already exists, skipping."
fi

if [ ! -d "$REPLAY_VALID_PATH" ]; then
    echo "Building replay valid pkl (built-in ATOM_ENERGIES)..."
    python -m bam.scripts.build_pkl_multihead \
        --input $REPLAY_VALID_TRAJ \
        --output $REPLAY_VALID_PATH \
        --prefix replay_valid \
        --chunk-size 100000
else
    echo "$REPLAY_VALID_PATH already exists, skipping."
fi

# === Step 2: Build target pkl (fit E0 from data, supports multiple traj files) ===
if [ ! -d "$TARGET_TRAIN_PATH" ]; then
    echo "Building target train pkl (fitting atom energies)..."
    python -m bam.scripts.build_pkl_multihead \
        --input $TARGET_TRAIN_TRAJ \
        --output $TARGET_TRAIN_PATH \
        --prefix lpsc_train \
        --chunk-size 100000 \
        --fit-energies
else
    echo "$TARGET_TRAIN_PATH already exists, skipping."
fi

if [ ! -d "$TARGET_VALID_PATH" ]; then
    echo "Building target valid pkl (using fitted atom_energies_target.json)..."
    python -m bam.scripts.build_pkl_multihead \
        --input $TARGET_VALID_TRAJ \
        --output $TARGET_VALID_PATH \
        --prefix lpsc_valid \
        --chunk-size 100000 \
        --atom-energies atom_energies.json
else
    echo "$TARGET_VALID_PATH already exists, skipping."
fi

# === Step 3: Train multihead ===
python -m bam.training.train_unified $CONFIG
