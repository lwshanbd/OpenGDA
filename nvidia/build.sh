#!/bin/bash
# Build script for OpenGDA NVIDIA + InfiniBand
# Usage: ./build.sh [clean]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/build"

# CUDA and tools paths
export CUDA_PATH=/hrtc/apps/cuda/12.4.131/aarch64/rocky9
export CMAKE_PATH=/hrtc/apps/devtools/spack/MAPLE/linux-rocky9-neoverse_v2/gcc-12.2.0/cmake-3.29.6-yl7bm5nfishcx3d6dlkqfj7pc2muvowe/bin

# Update PATH and LD_LIBRARY_PATH
export PATH=$CUDA_PATH/bin:$CMAKE_PATH:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

echo "=== OpenGDA NVIDIA Build ==="
echo "CUDA: $CUDA_PATH"
echo "CMake: $(which cmake)"
echo "nvcc: $(which nvcc)"
echo ""

# Clean build if requested
if [ "$1" == "clean" ]; then
    echo "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Configure
echo "Configuring..."
cmake \
    -DCMAKE_CUDA_COMPILER=$CUDA_PATH/bin/nvcc \
    -DCUDAToolkit_ROOT=$CUDA_PATH \
    ..

# Build
echo ""
echo "Building..."
make -j$(nproc)

echo ""
echo "=== Build Complete ==="
ls -la "$BUILD_DIR"/gda_*

echo ""
echo "Run with:"
echo "  mpirun -np 2 $BUILD_DIR/gda_pingpong"
echo "  mpirun -np 2 $BUILD_DIR/gda_am_test"
