#!/bin/bash
# BAM-NequIP LAMMPS NVT simulation runner
# LPSCl Li200S168P32Cl24
#
# Usage (new run):
#   bash run_lpscl.sh                  # default: 1x1x1 (424 atoms), 1 GPU
#   bash run_lpscl.sh 2                # 2x2x2
#   bash run_lpscl.sh 2 2 1            # 2x2x1
#   bash run_lpscl.sh 2 2 1 2          # 2x2x1, 2 GPUs
#
# Usage (restart from last checkpoint):
#   bash run_lpscl.sh --restart 2      # restart 2x2x2
#   bash run_lpscl.sh --restart 2 2 1  # restart 2x2x1
#
#   bash run_lpscl.sh [--restart] RX [RY] [RZ] [NP]

set -e

# ── Parse --restart flag ─────────────────────────────────────────
RESTART=0
if [ "${1:-}" = "--restart" ]; then
    RESTART=1
    shift
fi

# ── Arguments ─────────────────────────────────────────────────────
if [ $# -eq 0 ]; then
    RX=1; RY=1; RZ=1; NP=1
elif [ $# -eq 1 ]; then
    RX=$1; RY=$1; RZ=$1; NP=1
elif [ $# -eq 2 ]; then
    RX=$1; RY=$1; RZ=$1; NP=$2
elif [ $# -eq 3 ]; then
    RX=$1; RY=$2; RZ=$3; NP=1
elif [ $# -ge 4 ]; then
    RX=$1; RY=$2; RZ=$3; NP=$4
fi

N_ATOMS=$(( RX * RY * RZ * 424 ))
TAG="${RX}x${RY}x${RZ}"

# ── Configuration ─────────────────────────────────────────────────
LMP=/home/hgpark/main_git_251204/lammps/build_plugin/lmp
MPIRUN=/opt/intel/oneapi/mpi/latest/bin/mpirun

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BAM_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
export LAMMPS_PLUGIN_PATH="${BAM_ROOT}/lammps/chemtrain-deploy/build"
export JCN_PJRT_PATH="${BAM_ROOT}/lammps/chemtrain-deploy/lib"

# GPU setup
if [ "$NP" -eq 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
else
    export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NP - 1)))
fi

# XLA: no memory limit
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Intel MPI + MKL
source /opt/intel/oneapi/mpi/latest/env/vars.sh 2>/dev/null || true
source /opt/intel/oneapi/mkl/latest/env/vars.sh 2>/dev/null || true

# ── Output files ──────────────────────────────────────────────────
LOG="log_nvt_lpscl_${TAG}.lammps"
OUTPUT="output_nvt_lpscl_${TAG}.txt"
DUMP="dump_lpscl_${TAG}.lammpstrj"
TRAJ="nvt_lpscl_${TAG}.traj"

# ── Build LAMMPS args ────────────────────────────────────────────
LMP_ARGS="-in nvt_lpscl.in -var RX $RX -var RY $RY -var RZ $RZ -log $LOG"

if [ "$RESTART" -eq 1 ]; then
    # Find latest restart file
    RESTART_FILE=""
    if [ -f restart.b.bin ]; then
        RESTART_FILE=restart.b.bin
    elif [ -f restart.a.bin ]; then
        RESTART_FILE=restart.a.bin
    fi

    if [ -z "$RESTART_FILE" ]; then
        echo "ERROR: No restart file found (restart.a.bin / restart.b.bin)"
        exit 1
    fi

    LMP_ARGS="$LMP_ARGS -var RESTART 1 -var RESTART_FILE $RESTART_FILE"
    echo "================================================================"
    echo " RESTART NVT: LPSCl Li200S168P32Cl24"
    echo " ${TAG} = ${N_ATOMS} atoms, ${NP} GPU(s)"
    echo " Restart file: ${RESTART_FILE}"
    echo "================================================================"
else
    # Clean previous outputs for new run
    rm -f dump.lammpstrj "$LOG" "$OUTPUT" "$DUMP" "$TRAJ"
    rm -f restart.a.bin restart.b.bin

    echo "================================================================"
    echo " NVT 100ps: LPSCl Li200S168P32Cl24"
    echo " ${TAG} = ${N_ATOMS} atoms, ${NP} GPU(s)"
    echo " XLA_PYTHON_CLIENT_PREALLOCATE=false (no memory limit)"
    echo "================================================================"
fi

# ── Run ───────────────────────────────────────────────────────────
if [ "$NP" -eq 1 ]; then
    cd "$SCRIPT_DIR"
    "$LMP" $LMP_ARGS 2>&1 | tee "$OUTPUT"
else
    WRAPPER=$(mktemp /tmp/lmp_rank_wrapper.XXXXXX.sh)
    cat > "$WRAPPER" << 'WRAPPER_EOF'
#!/bin/bash
export PMI_LOCAL_RANK=${MPI_LOCALRANKID:-0}
export XLA_PYTHON_CLIENT_PREALLOCATE=false
exec "$@"
WRAPPER_EOF
    chmod +x "$WRAPPER"
    trap "rm -f '$WRAPPER'" EXIT

    cd "$SCRIPT_DIR"
    "$MPIRUN" -n "$NP" "$WRAPPER" \
        "$LMP" $LMP_ARGS 2>&1 | tee "$OUTPUT"
fi

# ── Post-processing ──────────────────────────────────────────────
if [ -f dump.lammpstrj ]; then
    mv dump.lammpstrj "$DUMP"
fi

if [ -f "$DUMP" ]; then
    echo ""
    echo "=== Converting to ASE trajectory ==="
    python "$SCRIPT_DIR/../lammpsout_to_traj.py" \
        --dump "$DUMP" --log "$LOG" --output "$TRAJ"
fi

echo ""
echo "=== DONE ==="
echo "  Log:        $LOG"
echo "  Dump:       $DUMP"
echo "  Trajectory: $TRAJ"
