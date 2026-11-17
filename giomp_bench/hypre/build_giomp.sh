#!/bin/bash
# build_base.sh SRCDIR PREFIX: hypre with OpenMP target offload (sm_90) + CUDA-aware MPI.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
L=/shared/data1/Users/l1065028/llvm-22.1.7
S=$1; P=$2
OFF="${XOFF:-} -fopenmp --offload-arch=sm_90 --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version"
G=/shared/data1/Users/l1065028/new-gicc; T=/shared/data1/Users/l1065028/giomp_ib_test
GI="-DHYPRE_USING_GIOMP -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_MLX5 -DGICC_GPU_CUDA=1 -I$G -I$G/src -I$CUDA_HOME/include"
GL="-L$T/lib2 -Wl,-rpath,$T/lib2 -lgicc_omp -L$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib64 -lcudart -libverbs -lmlx5 -lpthread -lstdc++"
MPIF="-I$MPI_ROOT/include $GI"; MPIL="-L$MPI_ROOT/lib -Wl,-rpath,$MPI_ROOT/lib -lmpi $GL"
cd $S/src
make distclean >/dev/null 2>&1
./configure --prefix=$P --with-device-openmp --enable-gpu-aware-mpi --disable-fortran \
  CC=$L/bin/clang CXX=$L/bin/clang++ CUCC=$L/bin/clang++ \
  CFLAGS="-O3 $OFF $MPIF" CXXFLAGS="-O3 $OFF $MPIF" CUFLAGS="-O3 $OFF $MPIF" LDFLAGS="$OFF $MPIL -Wl,-rpath,$L/lib/aarch64-unknown-linux-gnu" LIBS="$MPIL" > $P.configure.log 2>&1 || { echo CONFIGURE FAILED; tail -20 $P.configure.log; exit 1; }
grep -E "DEVICE_OPENMP|GPU_AWARE|USING_GPU|UNIFIED" HYPRE_config.h
make -j32 > $P.make.log 2>&1 && make install > /dev/null 2>&1 && echo LIB OK || { echo MAKE FAILED; grep -m20 -E "error" $P.make.log; exit 1; }
cd test && make struct > $P.test.log 2>&1 && cp struct $P.struct && echo DRIVER OK || { grep -m20 error $P.test.log; }
