#!/bin/bash
# Build script for GICC
# Usage: ./build.sh [clean]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

export CUDA_PATH=/hrtc/apps/cuda/12.4.131/aarch64/rocky9
export CMAKE_PATH=/hrtc/apps/devtools/spack/MAPLE/linux-rocky9-neoverse_v2/gcc-12.2.0/cmake-3.29.6-yl7bm5nfishcx3d6dlkqfj7pc2muvowe/bin
export PATH=$CUDA_PATH/bin:$CMAKE_PATH:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

echo "=== GICC Build ==="
echo "CUDA: $CUDA_PATH"
echo "nvcc: $(which nvcc)"
echo ""

if [ "$1" == "clean" ]; then
    echo "Cleaning..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "Configuring..."
cmake \
    -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc \
    -DCUDAToolkit_ROOT=$CUDA_PATH \
    ..

echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "=== Build Complete ==="
echo ""
echo "Run with:"
echo "  mpirun -np 2 $BUILD_DIR/gicc_pingpong_bench"
