#!/bin/bash
# Build script for GICC on LC — OFI/CXI backend with AMD GPUs
# Usage: ./build.sh [clean]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$SCRIPT_DIR/../.."
BUILD_DIR="$PROJECT_DIR/build_ofi"

echo "=== GICC Build (LC — OFI backend) ==="
echo "HIP:       $(which hipcc 2>/dev/null || echo 'not found')"
echo "ROCm:      ${ROCM_PATH:-/opt/rocm}"
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
    -DGICC_BACKEND=ofi \
    -Dhip_DIR=/opt/rocm-6.4.0/lib/cmake/hip \
    -DCMAKE_HIP_COMPILER=/opt/rocm-6.4.0/lib/llvm/bin/clang++ \
    "$PROJECT_DIR"

echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo ""
echo "Examples:"
echo "  FI_MR_CACHE_MAX_COUNT=0 srun -N 2 -n 2 --ntasks-per-node=1 -t 2 $BUILD_DIR/test_gicc"
echo "  FI_MR_CACHE_MAX_COUNT=0 srun -N 2 -n 2 --ntasks-per-node=1 -t 2 $BUILD_DIR/gda_benchmark_gicc"
echo "  FI_MR_CACHE_MAX_COUNT=0 srun -N 4 -n 4 --ntasks-per-node=1 -t 2 $BUILD_DIR/mm_gda_minimal_gicc"
