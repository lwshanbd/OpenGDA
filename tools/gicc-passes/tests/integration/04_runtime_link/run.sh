#!/bin/bash
# 04_runtime_link: verify the runtime_helpers.cpp .o exports every
# extern "C" symbol the LTO-emitted IR calls. If even one is missing,
# linking would fail and minimod / examples would break silently —
# catch it independently of any srun cycle.
#
# Pass criterion: nm -j of build_ofi's runtime_helpers.cpp.o lists
# all 8 symbols below as defined (T = text section).
set -euo pipefail

OBJ_PATH="${GICC_ROOT}/build_ofi/examples/ofi/CMakeFiles/test_gicc.dir/__/__/src/gicc/platform/ofi/runtime_helpers.cpp.o"
if [[ ! -f "${OBJ_PATH}" ]]; then
    echo "ERROR: ${OBJ_PATH} missing — run 'make' under build_ofi first."
    exit 2
fi

EXPECTED=(
    gicc_runtime_peer_ipc_base
    gicc_runtime_local_buf_base
    gicc_runtime_ipc_stream
    gicc_runtime_dwq_enqueue
    gicc_runtime_dwq_enqueue_batched
    gicc_runtime_trigger_addr
    gicc_runtime_trigger_val
)

PRESENT=$(/opt/rocm-6.4.0/lib/llvm/bin/llvm-nm "${OBJ_PATH}")
echo "${PRESENT}" | grep gicc_runtime_

missing=0
for sym in "${EXPECTED[@]}"; do
    if ! echo "${PRESENT}" | grep -qE " T ${sym}$"; then
        echo "MISSING: ${sym}"
        missing=$((missing + 1))
    fi
done

if [[ "${missing}" -eq 0 ]]; then
    echo "PASS: all 7 runtime helpers exported"
    exit 0
fi
echo "FAIL: ${missing} runtime helper(s) missing from .o"
exit 1
