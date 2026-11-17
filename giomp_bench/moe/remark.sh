source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
L=/shared/data1/Users/l1065028/llvm-22.1.7; G=/shared/data1/Users/l1065028/new-gicc; T=/shared/data1/Users/l1065028/giomp_ib_test
export GICC_ROOT=$G GIOMP_BACKEND=ib GIOMP_CXX=$L/bin/clang++ GIOMP_MPI_ROOT=$MPI_ROOT GIOMP_CUDA_ROOT=$CUDA_HOME GIOMP_OFFLOAD_ARCH=sm_90 GIOMP_LIBDIR=$T/lib5 GIOMP_OMP_LIBDIR=$L/lib/aarch64-unknown-linux-gnu
GIOMP_EXTRA_FLAGS="-foffload-lto -DMOE_GIOMP=$1 -Rpass=openmp-opt -Rpass-missed=openmp-opt -Rpass-analysis=openmp-opt" bash $G/examples/omp/build_giomp_example.sh $PWD/moe.cpp /tmp/moe_remark_$1 2>&1
