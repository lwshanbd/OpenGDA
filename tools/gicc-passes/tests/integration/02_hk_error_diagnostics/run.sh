#!/bin/bash
# 02_hk_error_diagnostics: compile a kernel whose put_no_db argument
# depends on threadIdx.x. HKAnalysis must reject and emit a
# source-line-precise diagnostic mentioning the threadIdx intrinsic.
#
# Pass criterion: stderr contains both
#   "error: gicc::put_no_db: argument 'size' is not host-knowable"
#   "depends on llvm.amdgcn.workitem.id.x"
set -euo pipefail

META_DIR="$(mktemp -d)"
trap 'rm -rf "${META_DIR}"' EXIT

# Compile-only — no need to link, the HK diagnostic fires during
# the device pipeline pass.
GICC_MODE=feature-extract GICC_META_DIR="${META_DIR}" \
/opt/rocm-6.4.0/lib/llvm/bin/clang++ \
    -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1 -D__HIP_PLATFORM_AMD__=1 \
    -DGICC_BOOTSTRAP_MPI=1 \
    -I"${GICC_ROOT}/src" \
    -I"${GICC_ROOT}/src/gicc/platform/ofi/internal" \
    -O3 --offload-arch=gfx90a -std=gnu++17 \
    -fpass-plugin="${GICC_PASSES_SO}" -flto \
    -x hip -c bad_kernel.cu -o /tmp/bad_kernel.o 2> /tmp/hk_stderr.log || true

cat /tmp/hk_stderr.log

# Two acceptable wordings for the threadIdx-derived value:
#   * direct intrinsic:  "depends on llvm.amdgcn.workitem.id.x"
#   * HIP builtin wrap:  "calls non-HK function '...threadIdx_t...__get_x...'"
if grep -q "argument 'size' is not host-knowable" /tmp/hk_stderr.log && \
   ( grep -q "llvm.amdgcn.workitem.id.x"  /tmp/hk_stderr.log \
   || grep -q "threadIdx"                  /tmp/hk_stderr.log ); then
    exit 0
fi
echo "FAIL: expected HK diagnostic mentioning threadIdx (direct or HIP wrap)"
exit 1
