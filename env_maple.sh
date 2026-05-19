#!/bin/bash
# Source this to set up the Maple build env for GICC MLX5 backend.
# Module files on Maple are currently misformatted (#%Module magic-cookie
# errors), so we set paths directly.

NVHPC_ROOT=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5
# Point at the inner cuda/12.4 dir; the outer cuda/ symlink package is
# missing a `targets/` symlink which breaks CMake's CUDAToolkit detection.
CUDA_HOME=$NVHPC_ROOT/cuda/12.4
CMAKE_BIN=/shared/data1/Users/l1057678/cmake-4.0.0-linux-aarch64/bin

export PATH=$CMAKE_BIN:$CUDA_HOME/bin:$NVHPC_ROOT/comm_libs/mpi/bin:$NVHPC_ROOT/compilers/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$NVHPC_ROOT/comm_libs/mpi/lib:$NVHPC_ROOT/math_libs/lib64:$LD_LIBRARY_PATH
export CUDACXX=$CUDA_HOME/bin/nvcc
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export MPI_HOME=$NVHPC_ROOT/comm_libs/mpi

# HPCx OpenMPI inside nvhpc was built with prefix /proj/nv/... which is
# absent on Maple. Point OMPI at the actual install location at runtime.
HPCX_OMPI=$NVHPC_ROOT/comm_libs/12.4/hpcx/hpcx-2.19/ompi
export OPAL_PREFIX=$HPCX_OMPI
export OPAL_LIBDIR=$HPCX_OMPI/lib
export OMPI_MCA_prefix=$HPCX_OMPI
export LD_LIBRARY_PATH=$HPCX_OMPI/lib:$LD_LIBRARY_PATH
