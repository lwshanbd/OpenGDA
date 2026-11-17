source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
L=/shared/data1/Users/l1065028/llvm-22.1.7; G=/shared/data1/Users/l1065028/new-gicc; T=/shared/data1/Users/l1065028/giomp_ib_test
export GICC_ROOT=$G GIOMP_BACKEND=ib GIOMP_CXX=$L/bin/clang++ GIOMP_MPI_ROOT=$MPI_ROOT GIOMP_CUDA_ROOT=$CUDA_HOME GIOMP_OFFLOAD_ARCH=sm_90 GIOMP_LIBDIR=$T/lib8 GIOMP_OMP_LIBDIR=$L/lib/aarch64-unknown-linux-gnu
export GICC_PASSES_SO=$T/passes-dyn/libgicc-passes.so GICC_META_DIR=/shared/data1/Users/l1065028/giomp_bench/moe/meta4 GICC_DYNAMIC=${GICC_DYNAMIC:-1}
GIOMP_EXTRA_FLAGS="-DMOE_GIOMP=4 $EXTRA" bash $G/examples/omp/build_giomp_after.sh /shared/data1/Users/l1065028/giomp_bench/moe/moe.cpp /shared/data1/Users/l1065028/giomp_bench/moe/moe_giomp4${SUFFIX} 2>&1
