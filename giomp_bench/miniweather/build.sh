#!/bin/bash
# build.sh NX NZ SIM_TIME: the MPI baseline (LTO, no plugin) and the GiOMP
# port plain (LTO) and pipelined, on maple-1.
S=/shared/data1/Users/l1065028/giomp_ib_test/scen/build.sh
D="-D_NX=$1 -D_NZ=$2 -D_SIM_TIME=$3 -D_OUT_FREQ=-1 -D_DATA_SPEC=DATA_SPEC_THERMAL -DNO_INFORM"
cd /shared/data1/Users/l1065028/giomp_bench/miniweather
GIOMP_EXTRA_FLAGS="$D -foffload-lto" PLAIN=1 bash $S $PWD/miniWeather_mpi_omp.cpp $PWD/mw_mpi | tail -1
GIOMP_EXTRA_FLAGS="$D -foffload-lto" PLAIN=1 bash $S $PWD/miniWeather_giomp.cpp $PWD/mw_plain | tail -1
GIOMP_EXTRA_FLAGS="$D" bash $S $PWD/miniWeather_giomp.cpp $PWD/mw_pipe | grep -E "gicc-after|ILLEGAL|stays|rror|BUILT" | sort | uniq
