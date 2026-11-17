#!/bin/bash
# Build both MPI layouts of FT and the GiOMP build, all default choices.
A=/shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/apps
export FT_SOURCE=/shared/data1/Users/l1065028/new-gicc/examples/omp/npbft/ft_giomp.cpp
D=/shared/data1/Users/l1065028/giomp_bench/tune2/ft
FT_MPI_DEFS= bash $A/ft_mpi/build.sh $D/ft_mpi_plain 2>&1 | grep -E "BUILT|FAIL|rror" &
FT_MPI_DEFS=-DFT_STREAM=1 bash $A/ft_mpi/build.sh $D/ft_mpi_stream 2>&1 | grep -E "BUILT|FAIL|rror" &
wait
