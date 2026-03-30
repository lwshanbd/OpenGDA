#!/bin/bash
# Build script for GICC on MAPLE
# Usage: ./build.sh [clean]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# ============================================================================
# MAPLE environment setup
# ============================================================================
if ! type module &>/dev/null; then
    source /usr/share/lmod/lmod/init/bash 2>/dev/null || \
    source /etc/profile.d/lmod.sh 2>/dev/null || true
fi

module use /shared/data1/Projects/CSE_HPC/apps/modules/aarch64/maple
module load gcc/11.4.1
module load cuda/12.4.131 nvhpc/24.5.cuda_12.4 openmpi-gcc/5.0.5/cuda.12.4
module load cmake/3.29.6

echo "=== GICC Build (MAPLE) ==="
echo "nvcc:  $(which nvcc)"
echo "mpicxx: $(which mpicxx)"
echo ""

# ============================================================================
# Build
# ============================================================================
if [ "$1" == "clean" ]; then
    echo "Cleaning..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring..."
cmake \
    -DCMAKE_C_COMPILER=gcc \
    -DCMAKE_CXX_COMPILER=g++ \
    -DCMAKE_CUDA_COMPILER=$(which nvcc) \
    -DCMAKE_CUDA_ARCHITECTURES=90 \
    "$SCRIPT_DIR"

echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo ""
echo "Library: $BUILD_DIR/src/libgicc.so"
echo "Examples:"
echo "  srun -p maple --account=app -N 2 --ntasks-per-node=1 --gres=gpu:1 --mpi=pmix $BUILD_DIR/examples/gicc/gicc_pingpong_bench"
