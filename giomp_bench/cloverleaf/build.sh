#!/bin/bash
# build.sh SRCDIR OUT [extra flags...]: CloverLeaf omp-target model + MPI, clang 22, sm_90.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
L=/shared/data1/Users/l1065028/llvm-22.1.7
S=$1; O=$2; shift 2
OFF="-fopenmp --offload-arch=sm_90 --cuda-path=$CUDA_HOME -Wno-unknown-cuda-version"
F="-O3 -std=c++17 $OFF -DOMP_TARGET -DDIFFUSE_OVERLOAD -I$S/driver -I$S/src/omp-target -I$MPI_ROOT/include $*"
mkdir -p $O.obj
for f in $S/driver/*.cpp $S/src/omp-target/*.cpp; do
  b=$(basename $(dirname $f))_$(basename $f .cpp)
  $L/bin/clang++ $F -c $f -o $O.obj/$b.o &
done; wait
$L/bin/clang++ $OFF $O.obj/*.o -o $O -L$MPI_ROOT/lib -Wl,-rpath,$MPI_ROOT/lib -lmpi $GIOMP_LIBS \
  -Wl,-rpath,$L/lib/aarch64-unknown-linux-gnu && echo BUILT $O
