#!/bin/bash
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bam_jax_nequip

export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_GPU_ALLOCATOR=cuda_malloc_async

CONFIG=${1:-input.json}

python -m bam.inference.eval $CONFIG
