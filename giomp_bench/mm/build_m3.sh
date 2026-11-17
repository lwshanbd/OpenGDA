#!/bin/bash
# build_m3.sh: minimod plain / pipelined / control with GICC_HALO_OVERLAP=3
# (seven region kernels, faces put after the last), no dist-chunk.
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
T=/shared/data1/Users/l1065028/giomp_ib_test
G=/shared/data1/Users/l1065028/new-gicc
B=/shared/data1/Users/l1065028/giomp_bench
export GICC_ROOT=$G GIOMP_LIBDIR=$T/lib2 GICC_PASSES_SO=$T/passes-build/libgicc-passes.so
cd $B/mm
FORCE_LTO=1 PIPE=0 bash build_lto.sh $PWD/mm_plain3 2>&1 | grep -E " error|BUILT"
PIPE=1 bash build_lto.sh $PWD/mm_pipe3 2>&1 | grep -E " error|BUILT|gicc-after|gicc-chunk\]|ILLEGAL|LEGAL|stays|counted|rotated" | sed -E 's/__omp_offloading_[0-9a-f]+_[0-9a-f]+_//' | sort | uniq -c
H=$G/src/gicc/omp_pipeline_impl.h
cp $H $B/mm/omp_pipeline_impl.h.keep
sed -i 's/constexpr size_t kOmpxCountMinBytes = size_t(1) << 20;/constexpr size_t kOmpxCountMinBytes = ~size_t(0);/' $H
grep -q "kOmpxCountMinBytes = ~size_t(0)" $H && PIPE=1 bash build_lto.sh $PWD/mm_ctrl3 2>&1 | grep -E " error|BUILT"
cp $B/mm/omp_pipeline_impl.h.keep $H
(cd $G && git diff --quiet -- src/gicc/omp_pipeline_impl.h) && echo "header restored" || echo "HEADER NOT RESTORED"
