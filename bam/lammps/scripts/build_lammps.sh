#!/bin/bash
# Build LAMMPS with chemtrain-deploy plugin for BAM-JAX models.
#
# Prerequisites:
#   - CMake >= 3.18
#   - CUDA toolkit (for GPU support)
#   - protobuf (C++ runtime)
#   - Python 3.11 with JAX installed (pip install "jax[cuda12]")
#   - Bazel (installed automatically by build.py)
#
# Usage:
#   bash build_lammps.sh [--cpu-only]
#
# This script:
#   0. Clones chemtrain and LAMMPS into bam/lammps/
#   1. Builds libconnector.so via Bazel (build.py)
#   2. Builds LAMMPS plugin via CMake (links against libconnector.so)
#   3. Builds LAMMPS with PLUGIN package
#   4. Produces lmp binary at bam/lammps/lammps/build/lmp

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# --- Paths ---
CHEMTRAIN_DIR="$BASE_DIR/chemtrain"
CHEMTRAIN_DEPLOY_DIR="$CHEMTRAIN_DIR/chemtrain-deploy"
LAMMPS_DIR="$BASE_DIR/lammps"
LAMMPS_VERSION="${LAMMPS_VERSION:-stable_2Aug2023_update3}"
CPU_ONLY=false

for arg in "$@"; do
    case $arg in
        --cpu-only) CPU_ONLY=true ;;
    esac
done

echo "=== BAM-JAX LAMMPS Build ==="
echo "Base dir:         $BASE_DIR"
echo "chemtrain-deploy: $CHEMTRAIN_DEPLOY_DIR"
echo "LAMMPS:           $LAMMPS_DIR"
echo "CPU only:         $CPU_ONLY"
echo ""

# --- Step 0: Clone repositories if needed ---
if [ ! -d "$CHEMTRAIN_DIR" ]; then
    echo "=== Cloning chemtrain ==="
    git clone https://github.com/tummfm/chemtrain.git "$CHEMTRAIN_DIR"
fi

if [ ! -d "$LAMMPS_DIR" ]; then
    echo "=== Cloning LAMMPS ($LAMMPS_VERSION) ==="
    git clone -b "$LAMMPS_VERSION" --depth 1 \
        https://github.com/lammps/lammps.git "$LAMMPS_DIR"
fi

# --- Step 1: Build libconnector.so (Bazel) ---
echo ""
echo "=== Step 1: Building libconnector.so via Bazel ==="
cd "$CHEMTRAIN_DEPLOY_DIR"

python build.py

if [ "$CPU_ONLY" = true ]; then
    python build.py --load_cpu_pjrt_plugin
else
    python build.py --load_gpu_pjrt_plugin
fi

echo "  -> lib/libconnector.so"
echo "  -> lib/pjrt_plugin.xla_cuda12.so"

# --- Step 2: Build LAMMPS plugin (CMake) ---
echo ""
echo "=== Step 2: Building LAMMPS plugin ==="
cd "$CHEMTRAIN_DEPLOY_DIR"
mkdir -p build && cd build

cmake ../lammps_plugin \
    -DLAMMPS_HEADER_DIR="$LAMMPS_DIR/src"
make -j"$(nproc)"

echo "  -> build/chemtrain_deployplugin.so"

# --- Step 3: Build LAMMPS ---
echo ""
echo "=== Step 3: Building LAMMPS ==="
cd "$LAMMPS_DIR"
mkdir -p build && cd build

cmake ../cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DPKG_PLUGIN=on \
    -DBUILD_MPI=on

make -j"$(nproc)"

echo ""
echo "=== Build complete ==="
echo "LAMMPS binary: $LAMMPS_DIR/build/lmp"
echo ""
echo "Environment variables (add to ~/.bashrc):"
echo "  export PATH=$LAMMPS_DIR/build:\$PATH"
echo "  export LAMMPS_PLUGIN_PATH=$CHEMTRAIN_DEPLOY_DIR/build"
echo "  export JCN_PJRT_PATH=$CHEMTRAIN_DEPLOY_DIR/lib"
