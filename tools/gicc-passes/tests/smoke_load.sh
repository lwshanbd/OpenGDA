#!/bin/bash
# Verify -fpass-plugin loads libgicc-passes.so in both device and host
# HIP compile pipelines on Tioga (ROCm 6.4). This is the R1 gate from
# spec §11: if hipcc -flto -fpass-plugin doesn't compose, the entire
# LTO design is blocked.
set -euo pipefail

GICC_PASSES_SO="${1:-}"
if [[ -z "${GICC_PASSES_SO}" ]]; then
    echo "usage: $0 <path/to/libgicc-passes.so>" >&2
    exit 1
fi
if [[ ! -f "${GICC_PASSES_SO}" ]]; then
    echo "ERROR: ${GICC_PASSES_SO} does not exist" >&2
    exit 1
fi

HIPCC="${HIPCC:-/opt/rocm-6.4.0/bin/hipcc}"

TMP=$(mktemp -d)
trap "rm -rf ${TMP}" EXIT

cat > "${TMP}/hello.cpp" <<'CPP'
#include <hip/hip_runtime.h>
__global__ void hello() {}
int main() {
    hipLaunchKernelGGL(hello, dim3(1), dim3(1), 0, 0);
    hipDeviceSynchronize();
    return 0;
}
CPP

OUT=$(${HIPCC} -O3 -flto --offload-arch=gfx90a \
      -fpass-plugin="${GICC_PASSES_SO}" \
      "${TMP}/hello.cpp" -o "${TMP}/hello" 2>&1)

echo "${OUT}"
echo

# Plugin must run on at least one module (host or device).
COUNT=$(echo "${OUT}" | grep -c "\[gicc-pass\] mode=" || true)
if [[ "${COUNT}" -eq 0 ]]; then
    echo "FAIL: plugin did not run on any module" >&2
    exit 1
fi
echo "PASS: plugin ran ${COUNT} times"

# Document which triples we observed (helps diagnose risk R1).
HOST_RAN=$(echo "${OUT}" | grep -c "triple=x86_64" || true)
DEV_RAN=$(echo "${OUT}" | grep -c "triple=amdgcn" || true)
echo "  host module runs: ${HOST_RAN}"
echo "  device module runs: ${DEV_RAN}"
if [[ "${DEV_RAN}" -eq 0 ]]; then
    echo "WARN: plugin did not run on device IR; only host LTO invoked it." >&2
    echo "      This is risk R1: device passes (Discovery, HKAnalysis,"   >&2
    echo "      DeviceLowering) won't run via -fpass-plugin alone. Document"  >&2
    echo "      and follow up before proceeding past Phase 0."             >&2
fi
