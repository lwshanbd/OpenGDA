#!/bin/bash
# Step 6 cross-check: the MPI build whose transpose is streamed from the same
# fftz_send kernel (FT_MPI=1 FT_STREAM=1), with and without C_giomp's choices
# on that kernel.
A=/shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/apps
D=/shared/data1/Users/l1065028/giomp_bench/tune2/ft
export FT_SOURCE=/shared/data1/Users/l1065028/new-gicc/examples/omp/npbft/ft_giomp.cpp
FT_MPI_DEFS=-DFT_STREAM=1 bash $A/ft_mpi/build.sh $D/xval_mpi_stream_default 2>&1 | grep -E "BUILT|FAIL" &
GICC_TUNE_IN=$D/xval_tune.json FT_MPI_DEFS=-DFT_STREAM=1 bash $A/ft_mpi/build.sh $D/xval_mpi_stream_cgiomp 2>&1 | grep -E "BUILT|FAIL|launch-shape|dist-chunk" &
wait
