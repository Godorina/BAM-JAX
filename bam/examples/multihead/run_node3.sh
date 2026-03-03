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
export JAX_PROCESS_INDEX=2    # Node 3 = index 2

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
# Train multihead (Worker node - pkl은 Node 1에서 빌드)
########################################
echo "=========================================="
echo "JAX Unified Sharded Training - Node 3 (Worker)"
echo "Coordinator: $JAX_COORDINATOR_ADDRESS"
echo "Process: $JAX_PROCESS_INDEX / $JAX_NUM_PROCESSES"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "InfiniBand: ENABLED (mlx5_0,mlx5_4,mlx5_5,mlx5_8)"
echo "=========================================="

TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Starting training at $(date)"

python -m bam.training.train_unified $CONFIG 2>&1 | tee -a training_node3.log

EXIT_CODE=$?
echo "=========================================="
echo "Training completed with exit code: $EXIT_CODE"
echo "Time: $(date)"
echo "=========================================="
exit $EXIT_CODE
