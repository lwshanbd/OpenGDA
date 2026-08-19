#!/bin/bash
# Build the disjoint compiler-path calibration workload through real HIP LTO.
#
#   facts     emit and validate compiler facts
#   freeze    freeze the calibration dossier and uniform action controls
#   controls  build proxy and trigger calibration binaries
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
MODE="${1:-controls}"
OUT="${GICC_ROOT}/build_ofi/compiler_lto_calibration"
META="${OUT}/meta"
OBJ="${OUT}/obj"
GENERATED="${OUT}/generated"
CONTROLS="${GENERATED}/controls"
PASSES="${GICC_PASSES_SO:-${GICC_ROOT}/tools/gicc-passes/build/libgicc-passes.so}"
SRC="${GICC_ROOT}/examples/proxy/compiler_lto_calibration.cpp"
PROFILE="${GICC_ROOT}/examples/proxy/compiler_lto_calibration_profile.json"
FEATURES="${GENERATED}/features.json"
DOSSIER="${GENERATED}/dossier.json"
PROMPT="${GENERATED}/prompt.txt"
FROZEN="${GICC_ROOT}/docs/experiments/compiler-lto-calibration/frozen-v1"
CAL_TOOL="${GICC_ROOT}/examples/proxy/compiler_lto_calibration.py"
BRIDGE="${GICC_ROOT}/tools/gicc-passes/python/gicc_llm_bridge.py"

case "${MODE}" in facts|freeze|controls) ;; *)
  echo "usage: $0 {facts|freeze|controls}" >&2
  exit 2
esac
test -f "${PASSES}" || {
  echo "ERROR: build the pass plugin first: ${PASSES}" >&2
  exit 1
}

mkdir -p "${META}" "${OBJ}" "${GENERATED}" "${CONTROLS}"
find "${META}" -mindepth 1 -maxdepth 1 -type f -delete
find "${OBJ}" -mindepth 1 -maxdepth 1 -type f -delete
find "${CONTROLS}" -mindepth 1 -maxdepth 1 -type f -delete

HIPCC=/opt/rocm-6.4.0/lib/llvm/bin/clang++
LIBFAB=/opt/cray/libfabric/2.1
MPI=/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0
CFLAGS=(
  -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1
  -DGICC_CPU_PROXY=1 -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1
  -D__HIP_ROCclr__=1 -I"${GICC_ROOT}/src"
  -I"${GICC_ROOT}/src/gicc/platform/ofi/internal" -I"${LIBFAB}/include"
  -isystem "${MPI}/include" -O3 --offload-arch=gfx90a -std=gnu++17
  -fpass-plugin="${PASSES}" -flto
)
# Keep device symbol identity independent of the policy arm/object name.  HIP
# otherwise hashes the full compiler command line into its default CUID.
CALIBRATION_CUID="gicc_compiler_lto_calibration_v1"

source_before=$(sha256sum "${SRC}")
source_before=${source_before%% *}

echo "=== discover calibration operations ==="
env GICC_MODE=discover GICC_META_DIR="${META}" GICC_PROXY_ENABLED=1 \
  "${HIPCC}" "${CFLAGS[@]}" -cuid="${CALIBRATION_CUID}" -x hip -c "${SRC}" \
  -o "${OBJ}/discover.o"

echo "=== extract calibration LTO facts ==="
env GICC_MODE=feature-extract GICC_META_DIR="${META}" \
  GICC_FEATURES_OUT="${FEATURES}" GICC_PROXY_ENABLED=1 \
  "${HIPCC}" "${CFLAGS[@]}" -cuid="${CALIBRATION_CUID}" -x hip -c "${SRC}" \
  -o "${OBJ}/features.o"

python3 "${BRIDGE}" emit --features "${FEATURES}" --platform "${PROFILE}" \
  --dossier "${DOSSIER}" --prompt "${PROMPT}"

cal_args=(
  --source "${SRC}" --features "${FEATURES}" --dossier "${DOSSIER}"
  --prompt "${PROMPT}" --profile "${PROFILE}" --controls "${CONTROLS}"
  --frozen "${FROZEN}"
)
python3 "${CAL_TOOL}" verify "${cal_args[@]}"

if [ "${MODE}" = freeze ]; then
  python3 "${CAL_TOOL}" freeze "${cal_args[@]}"
fi
if [ "${MODE}" = facts ] || [ "${MODE}" = freeze ]; then
  source_after=$(sha256sum "${SRC}")
  source_after=${source_after%% *}
  test "${source_before}" = "${source_after}"
  echo "SOURCE_SHA256=${source_after}"
  exit 0
fi

python3 "${CAL_TOOL}" check-frozen "${cal_args[@]}"

echo "=== lower unchanged calibration source ==="
for action in proxy trigger; do
  hint="${FROZEN}/controls/${action}-hint.json"
  env GICC_MODE=lower GICC_META_DIR="${META}" GICC_HINT_IN="${hint}" \
    GICC_PROXY_ENABLED=1 \
    "${HIPCC}" "${CFLAGS[@]}" -cuid="${CALIBRATION_CUID}" -x hip -c "${SRC}" \
    -o "${OBJ}/${action}.o"
done

for extra in runtime_helpers proxy_thread proxy_libfabric; do
  extra_src="${GICC_ROOT}/src/gicc/platform/ofi/"
  if [ "${extra}" = runtime_helpers ]; then
    extra_src+="runtime_helpers.cpp"
  else
    extra_src+="proxy/${extra}.cpp"
  fi
  env GICC_MODE=lower GICC_META_DIR="${META}" GICC_PROXY_ENABLED=1 \
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c "${extra_src}" \
    -o "${OBJ}/${extra}.o"
done

echo "=== link calibration controls ==="
for action in proxy trigger; do
  "${HIPCC}" -O3 --offload-arch=gfx90a --hip-link -flto \
    --rtlib=compiler-rt -unwindlib=libgcc \
    "${OBJ}/${action}.o" "${OBJ}/runtime_helpers.o" \
    "${OBJ}/proxy_thread.o" "${OBJ}/proxy_libfabric.o" \
    -o "${OUT}/compiler_lto_calibration_${action}" \
    -Wl,-rpath,"${LIBFAB}/lib64:${MPI}/lib" \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so -lpthread \
    /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 \
    "${MPI}/lib/libmpi_cray.so"
done

source_after=$(sha256sum "${SRC}")
source_after=${source_after%% *}
test "${source_before}" = "${source_after}"
echo "SOURCE_SHA256=${source_after}"
sha256sum "${OUT}"/compiler_lto_calibration_*
