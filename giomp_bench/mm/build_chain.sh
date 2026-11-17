#!/bin/bash
# MiniMod GiOMP, mode 3 code, split on target_pml / target_inner (the LLM's
# best tuning file), built with the chain-capable plugin and runtime (lib8).
source /shared/data1/Users/l1065028/giomp_ib_test/env_cudampi.sh >/dev/null 2>&1
export CPATH=/hrtc/apps/devtools/nvidia-hpcsdk/aarch64/24.5-cuda12.4.131/Linux_aarch64/24.5/math_libs/12.4/targets/sbsa-linux/include
T=/shared/data1/Users/l1065028/giomp_ib_test
export GICC_ROOT=/shared/data1/Users/l1065028/new-gicc GIOMP_LIBDIR=$T/lib8
export GICC_PASSES_SO=$T/passes-dyn/libgicc-passes.so
export GICC_TUNE_IN=/shared/data1/Users/l1065028/giomp_bench/mm/split_tune.json
GICC_SPLIT_DUMP=${DUMP:-} PIPE=1 bash /shared/data1/Users/l1065028/minimod-giomp/targets/omp_gicc/build_maple_ib.sh /shared/data1/Users/l1065028/giomp_bench/mm/mm_chain 2>&1 | grep -E "gicc-split\].*(split|deferred)|rror|BUILT" 
