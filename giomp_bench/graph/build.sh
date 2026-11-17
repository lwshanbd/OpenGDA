#!/bin/bash
# build.sh: BFS and PageRank as GPU-aware MPI and as GiOMP, same kernels,
# -foffload-lto, no pass plugin, on maple-1.
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
T=/shared/data1/Users/l1065028/giomp_ib_test
cd /shared/data1/Users/l1065028/giomp_bench/graph
export LIBDIR=$T/lib5
for app in ${APPS:-bfs pagerank}; do for b in mpi giomp; do
  GIOMP_EXTRA_FLAGS="-foffload-lto -DGRAPH_BACKEND_${b^^}" PLAIN=1 bash $S $PWD/$app.cpp $PWD/${app}_$b | tail -5
done; done
