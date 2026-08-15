#!/bin/bash
# build_decider_e2e.sh — the real three-pass flow for decider_e2e.cpp:
#
#   1. discover        device pass writes per-kernel JSON, now including
#                      trip_count (ScalarEvolution) and compute_after
#                      (the static issue-to-first-use distance)
#   2. feature-extract host pass turns that into features.json
#   3. gicc_decider.py applies the calibrated rule -> hint.json
#   4. lower           dispatch lowering honours the hint per call site
#
# No hand-written hint.json anywhere. Prints the facts and the decision
# so the chain is auditable.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
PASSES_SO="${GICC_PASSES_SO:-${GICC_ROOT}/tools/gicc-passes/build/libgicc-passes.so}"
[[ -f "${PASSES_SO}" ]] || { echo "ERROR: build the pass plugin first: ${PASSES_SO}"; exit 1; }

OUT_DIR="${GICC_ROOT}/build_ofi/decider_e2e"
mkdir -p "${OUT_DIR}"
META_DIR="${OUT_DIR}/meta";     rm -rf "${META_DIR}"; mkdir -p "${META_DIR}"
OBJ_DIR="${OUT_DIR}/obj";       rm -rf "${OBJ_DIR}";  mkdir -p "${OBJ_DIR}"
FEATURES="${OUT_DIR}/features.json"
HINT="${OUT_DIR}/hint.json"

HIPCC=/opt/rocm-6.4.0/lib/llvm/bin/clang++
LIBFAB=/opt/cray/libfabric/2.1
MPI=/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0

CFLAGS=(
    -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1
    -DGICC_CPU_PROXY=1 -DUSE_PROF_API=1
    -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1
    -I"${GICC_ROOT}/src"
    -I"${GICC_ROOT}/src/gicc/platform/ofi/internal"
    -I"${LIBFAB}/include"
    -isystem "${MPI}/include"
    -O3 --offload-arch=gfx90a -std=gnu++17
    -fpass-plugin="${PASSES_SO}" -flto
)

SRC="${GICC_ROOT}/examples/proxy/decider_e2e.cpp"

echo "=== pass 1/4: discover (device analysis) ==="
GICC_MODE=discover GICC_META_DIR="${META_DIR}" GICC_PROXY_ENABLED=1 \
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c "${SRC}" -o "${OBJ_DIR}/discover.o"

echo
echo "=== pass 2/4: feature-extract (host) ==="
GICC_MODE=feature-extract GICC_META_DIR="${META_DIR}" \
    GICC_FEATURES_OUT="${FEATURES}" GICC_PROXY_ENABLED=1 \
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c "${SRC}" -o "${OBJ_DIR}/features.o"

echo
echo "--- what the compiler derived about each call site ---"
python3 - "${META_DIR}" <<'PY'
import json, sys, pathlib
for f in sorted(pathlib.Path(sys.argv[1]).glob("*.json")):
    d = json.loads(f.read_text())
    print(f"  kernel {d.get('kernel_simple')}")
    for op in d.get("ops", []):
        if op.get("kind") not in ("put_no_db", "get_no_db"): continue
        print(f"    {op.get('site_id')}")
        print(f"      hk_capable    = {op.get('hk_capable')}")
        print(f"      trip_count    = {op.get('trip_count', 'n/a')}"
              "   (ScalarEvolution)")
        print(f"      compute_after = {op.get('compute_after', 'n/a')}"
              "   (flops before the completion point)")
PY

echo
echo "=== pass 3/4: decider (calibrated rule, no hand-written hint) ==="
GICC_FEATURES_FILE="${FEATURES}" GICC_HINT_FILE="${HINT}" \
    python3 "${GICC_ROOT}/tools/gicc-passes/python/gicc_decider.py"

echo
echo "--- hint.json ---"
cat "${HINT}"

echo
echo "=== pass 4/4: lower (honour the hint per call site) ==="
GICC_MODE=lower GICC_META_DIR="${META_DIR}" GICC_HINT_IN="${HINT}" \
    GICC_PROXY_ENABLED=1 \
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c "${SRC}" -o "${OBJ_DIR}/decider_e2e.o"

for extra in "${GICC_ROOT}/src/gicc/platform/ofi/runtime_helpers.cpp" \
             "${GICC_ROOT}/src/gicc/platform/ofi/proxy/proxy_thread.cpp" \
             "${GICC_ROOT}/src/gicc/platform/ofi/proxy/proxy_libfabric.cpp"; do
    GICC_MODE=lower GICC_META_DIR="${META_DIR}" GICC_PROXY_ENABLED=1 \
        "${HIPCC}" "${CFLAGS[@]}" -x hip -c "${extra}" \
        -o "${OBJ_DIR}/$(basename "${extra}" .cpp).o"
done

"${HIPCC}" -O3 --offload-arch=gfx90a --hip-link -flto \
    --rtlib=compiler-rt -unwindlib=libgcc \
    "${OBJ_DIR}/decider_e2e.o" "${OBJ_DIR}/runtime_helpers.o" \
    "${OBJ_DIR}/proxy_thread.o" "${OBJ_DIR}/proxy_libfabric.o" \
    -o "${OUT_DIR}/decider_e2e" \
    -Wl,-rpath,"${LIBFAB}/lib64:${MPI}/lib" \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so \
    -lpthread /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 \
    "${MPI}/lib/libmpi_cray.so"

echo
echo "built: ${OUT_DIR}/decider_e2e"
echo "run:   srun -p pci -N 2 -n 2 --ntasks-per-node=1 -c 8 --gpu-bind=none -t 2 \\"
echo "         ${OUT_DIR}/decider_e2e"
