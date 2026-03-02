#!/bin/bash
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bam_jax_nequip

export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_GPU_ALLOCATOR=cuda_malloc_async
export PYTHONUNBUFFERED=1

CONFIG=${1:-input.json}

# 평가 실행
python -m bam.training.train_unified $CONFIG 2>&1 | tee eval.log
