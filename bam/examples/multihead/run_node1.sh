#!/bin/bash
set -eo pipefail

########################################
# GPU 선택
########################################
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

########################################
# Python 출력 버퍼링 비활성화
########################################
export PYTHONUNBUFFERED=1

########################################
# JAX Distributed 설정 (Multi-Host)
# Node 1 + Node 2 + Node 3 = 3 hosts × 8 GPUs = 24 GPUs
########################################
export JAX_COORDINATOR_ADDRESS="192.169.0.2:29500"  # Node 1 IP (coordinator)
export JAX_NUM_PROCESSES=3
export JAX_PROCESS_INDEX=0    # Node 1 = coordinator = index 0

########################################
# JAX/XLA 설정
########################################
export JAX_LOG_COMPILES=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export JAX_COMPILATION_CACHE_DIR="/tmp/jax_cache"
mkdir -p $JAX_COMPILATION_CACHE_DIR

# Disable autotuning and Triton for multi-node stability
export XLA_FLAGS="${XLA_FLAGS:-} --xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"

########################################
# NCCL 설정 (Multi-Node with InfiniBand)
########################################
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=mlx5_0,mlx5_4,mlx5_5,mlx5_8
export NCCL_NET_GDR_LEVEL=2

export NCCL_SOCKET_IFNAME=bond-srv.1521
export NCCL_SOCKET_FAMILY=AF_INET

export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_ASYNC_ERROR_HANDLING=1

export NCCL_P2P_DISABLE=0
export NCCL_SHM_DISABLE=0

########################################
# CPU 스레드 (데이터로딩용)
########################################
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

########################################
# 환경 활성화 및 디렉토리 이동
########################################
source ~/envs/BAM-JAX-multihead/bin/activate
cd "$(dirname "$0")"

CONFIG=${1:-input_multihead.json}

########################################
# Step 1: Config에서 환경변수 추출
########################################
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

########################################
# Step 2: Build pkl (Node 1에서만 실행)
########################################
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

if [ ! -d "$TARGET_TRAIN_PATH" ]; then
    echo "Building target train pkl (fitting atom energies)..."
    python -m bam.scripts.build_pkl_multi_input \
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
    python -m bam.scripts.build_pkl_multi_input \
        --input $TARGET_VALID_TRAJ \
        --output $TARGET_VALID_PATH \
        --prefix lpsc_valid \
        --chunk-size 100000 \
        --atom-energies atom_energies_target.json
else
    echo "$TARGET_VALID_PATH already exists, skipping."
fi

########################################
# Step 3: Train multihead
########################################
echo "=========================================="
echo "JAX Unified Sharded Training - Node 1 (Coordinator)"
echo "Coordinator: $JAX_COORDINATOR_ADDRESS"
echo "Process: $JAX_PROCESS_INDEX / $JAX_NUM_PROCESSES"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "InfiniBand: ENABLED (mlx5_0,mlx5_4,mlx5_5,mlx5_8)"
echo "=========================================="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if [ -f training_node1.log ]; then
    mv training_node1.log training_node1.log.${TIMESTAMP}.bak
fi

echo "Starting training at $(date)"

python -m bam.training.train_unified $CONFIG 2>&1 | tee -a training_node1.log

EXIT_CODE=$?
echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "=========================================="
exit $EXIT_CODE
