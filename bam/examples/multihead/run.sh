#!/bin/bash
eval "$(conda shell.bash hook 2>/dev/null)"
conda activate bam_jax_nequip

export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export TF_GPU_ALLOCATOR=cuda_malloc_async

python -m bam.training.train_multihead_sharded input_multihead.json
