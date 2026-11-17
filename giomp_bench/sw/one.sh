#!/bin/bash
# one.sh APP NAME CHUNK CAP DEFS: one sweep build into sw/APP/NAME.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
T=/shared/data1/Users/l1065028/giomp_ib_test; G=/shared/data1/Users/l1065028/new-gicc; B=/shared/data1/Users/l1065028/giomp_bench
S=$T/scen/build.sh; P=$T/passes-build/libgicc-passes.so
export GICC_ROOT=$G GIOMP_LIBDIR=$T/lib2 GICC_PASSES_SO=$P
case $3 in d*) export GICC_DIST_CHUNK=${3#d};; p*) export GICC_X_PIPE_CHUNK=${3#p};; [0-9]*) [ $3 != 0 ] && export GICC_DIST_CHUNK=$3;; esac
[ "$4" != - ] && export GICC_X_CAP=$4
D=""; [ "$5" != - ] && D="$5"
O=$B/sw/$1/$2; mkdir -p $B/sw/$1; rm -f $O
case $1-${2%%_*} in
  jac-base) GIOMP_EXTRA_FLAGS="-foffload-lto -fpass-plugin=$P" PLAIN=1 bash $S $B/jacobi/jacobi_mpi_omp.cpp $O ;;
  jac-pipe) GIOMP_EXTRA_FLAGS="$D" bash $S $B/jacobi/jacobi_giomp.cpp $O ;;
  mm-base)  cd $B/mm; DIST_PLUGIN=$P bash build_mpi.sh $O ;;
  mm-pipe)  cd $B/mm; MM_DEFS="$D" PIPE=1 bash build_lto.sh $O ;;
  mw-base|mw-pipe)
    MW="-D_NX=4000 -D_NZ=2000 -D_SIM_TIME=10 -D_OUT_FREQ=-1 -D_DATA_SPEC=DATA_SPEC_THERMAL -DNO_INFORM"
    if [ ${2%%_*} = base ]; then GIOMP_EXTRA_FLAGS="$MW -foffload-lto -fpass-plugin=$P" PLAIN=1 bash $S $B/miniweather/miniWeather_mpi_omp.cpp $O
    else GIOMP_EXTRA_FLAGS="$MW $D" bash $S $B/miniweather/miniWeather_giomp.cpp $O; fi ;;
esac > $O.buildlog 2>&1
[ -x $O ] && echo "OK $1/$2" || echo "FAIL $1/$2"
