#!/bin/bash
# build_giomp.sh OUT: CloverLeaf with the GiOMP halo exchange.
C=/shared/data1/Users/l1065028/giomp_bench/cloverleaf
G=/shared/data1/Users/l1065028/new-gicc; T=/shared/data1/Users/l1065028/giomp_ib_test
source $T/env_cudampi.sh >/dev/null 2>&1
export GIOMP_LIBS="-L$T/lib4 -Wl,-rpath,$T/lib4 -lgicc_omp -L$CUDA_HOME/lib64 -Wl,-rpath,$CUDA_HOME/lib64 -lcudart -libverbs -lmlx5 -lpthread -lstdc++"
bash $C/build.sh $C/src-giomp $1 -DCLOVER_GIOMP -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_MLX5 -DGICC_GPU_CUDA=1 -I$G -I$G/src -I$CUDA_HOME/include
