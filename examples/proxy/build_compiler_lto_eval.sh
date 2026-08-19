#!/bin/bash
# Real HIP/LTO build for the frozen compiler-level decision benchmark.
#
#   facts                emit and validate compiler facts only
#   freeze               refresh the checked-in frozen dossier
#   controls             build default/proxy/trigger/hand-rule binaries
#   oracle               build all 16 proxy/trigger assignments for the
#                        four-site parallel completion group
#   candidate FILE NAME  validate one gicc-llm-decision-v1 response and build
#                        a binary from its pass hint
#
# Every mode hashes the source before and after the pipeline.  Model/provider
# invocation is intentionally absent: FILE must already contain a response to
# the compiler-fact-only prompt.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
MODE="${1:-controls}"
OUT="${GICC_ROOT}/build_ofi/compiler_lto_eval"
META="${OUT}/meta"
OBJ="${OUT}/obj"
GENERATED="${OUT}/generated"
CONTROLS="${GENERATED}/controls"
PASSES="${GICC_PASSES_SO:-${GICC_ROOT}/tools/gicc-passes/build/libgicc-passes.so}"
SRC="${GICC_ROOT}/examples/proxy/compiler_lto_eval.cpp"
PROFILE="${GICC_ROOT}/examples/proxy/compiler_lto_eval_profile.json"
FEATURES="${GENERATED}/features.json"
DOSSIER="${GENERATED}/dossier.json"
PROMPT="${GENERATED}/prompt.txt"
FROZEN="${GICC_ROOT}/docs/experiments/compiler-lto-eval/frozen-v1"
EVAL_TOOL="${GICC_ROOT}/examples/proxy/compiler_lto_eval.py"
BRIDGE="${GICC_ROOT}/tools/gicc-passes/python/gicc_llm_bridge.py"

test -f "${PASSES}" || {
  echo "ERROR: build the pass plugin first: ${PASSES}" >&2
  exit 1
}
case "${MODE}" in
  facts|freeze|controls|oracle) ;;
  candidate)
    test "$#" -eq 3 || {
      echo "usage: $0 candidate RESPONSE.json NAME" >&2
      exit 2
    }
    [[ "$3" =~ ^[a-zA-Z0-9_.-]+$ ]] || {
      echo "ERROR: candidate NAME must use only letters, digits, dot, dash, underscore" >&2
      exit 2
    }
    ;;
  *)
    echo "usage: $0 {facts|freeze|controls|oracle|candidate RESPONSE.json NAME}" >&2
    exit 2
    ;;
esac

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

source_before=$(sha256sum "${SRC}")
source_before=${source_before%% *}

echo "=== discover device operations ==="
env GICC_MODE=discover GICC_META_DIR="${META}" GICC_PROXY_ENABLED=1 \
  "${HIPCC}" "${CFLAGS[@]}" -x hip -c "${SRC}" -o "${OBJ}/discover.o"

echo "=== extract host-LTO facts ==="
env GICC_MODE=feature-extract GICC_META_DIR="${META}" \
  GICC_FEATURES_OUT="${FEATURES}" GICC_PROXY_ENABLED=1 \
  "${HIPCC}" "${CFLAGS[@]}" -x hip -c "${SRC}" -o "${OBJ}/features.o"

echo "=== emit source-free dossier and prompt ==="
python3 "${BRIDGE}" emit --features "${FEATURES}" --platform "${PROFILE}" \
  --dossier "${DOSSIER}" --prompt "${PROMPT}"
python3 "${EVAL_TOOL}" verify --features "${FEATURES}" \
  --dossier "${DOSSIER}" --prompt "${PROMPT}" --profile "${PROFILE}" \
  --source "${SRC}" --controls "${CONTROLS}"

if [ "${MODE}" = freeze ]; then
  python3 "${EVAL_TOOL}" freeze --features "${FEATURES}" \
    --dossier "${DOSSIER}" --prompt "${PROMPT}" --profile "${PROFILE}" \
    --source "${SRC}" --controls "${CONTROLS}" --frozen "${FROZEN}"
fi
if [ "${MODE}" = facts ] || [ "${MODE}" = freeze ]; then
  source_after=$(sha256sum "${SRC}")
  source_after=${source_after%% *}
  test "${source_before}" = "${source_after}"
  echo "SOURCE_SHA256=${source_after}"
  exit 0
fi

echo "=== verify frozen evaluation contract ==="
python3 "${EVAL_TOOL}" check-frozen --features "${FEATURES}" \
  --dossier "${DOSSIER}" --prompt "${PROMPT}" --profile "${PROFILE}" \
  --source "${SRC}" --controls "${CONTROLS}" --frozen "${FROZEN}"

declare -a NAMES=()
declare -a HINTS=()
if [ "${MODE}" = controls ]; then
  env GICC_FEATURES_FILE="${FEATURES}" \
      GICC_HINT_FILE="${GENERATED}/hand-hint.json" \
      python3 "${GICC_ROOT}/tools/gicc-passes/python/gicc_decider.py"
  NAMES=(default proxy trigger hand)
  HINTS=("${CONTROLS}/default-hint.json"
         "${CONTROLS}/proxy-hint.json"
         "${CONTROLS}/trigger-hint.json"
         "${GENERATED}/hand-hint.json")
elif [ "${MODE}" = oracle ]; then
  ORACLE="${GENERATED}/oracle"
  mkdir -p "${ORACLE}"
  find "${ORACLE}" -mindepth 1 -maxdepth 1 -type f -delete
  python3 "${EVAL_TOOL}" oracle --features "${FEATURES}" \
    --dossier "${DOSSIER}" --prompt "${PROMPT}" --profile "${PROFILE}" \
    --source "${SRC}" --controls "${ORACLE}" --frozen "${FROZEN}"
  for mask in $(seq 0 15); do
    printf -v suffix '%02d' "${mask}"
    NAMES+=("static4_mask${suffix}")
    HINTS+=("${ORACLE}/static4-mask${suffix}-hint.json")
  done
else
  RESPONSE=$(realpath "$2")
  NAME="$3"
  HINT="${GENERATED}/${NAME}-hint.json"
  python3 "${BRIDGE}" accept --strict --dossier "${DOSSIER}" \
    --response "${RESPONSE}" --hint "${HINT}"
  NAMES=("${NAME}")
  HINTS=("${HINT}")
fi

echo "=== lower unchanged source ==="
for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  hint=${HINTS[$i]}
  env GICC_MODE=lower GICC_META_DIR="${META}" GICC_HINT_IN="${hint}" \
    GICC_PROXY_ENABLED=1 \
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c "${SRC}" \
    -o "${OBJ}/${name}.o"
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

echo "=== link decision arms ==="
for name in "${NAMES[@]}"; do
  "${HIPCC}" -O3 --offload-arch=gfx90a --hip-link -flto \
    --rtlib=compiler-rt -unwindlib=libgcc \
    "${OBJ}/${name}.o" "${OBJ}/runtime_helpers.o" \
    "${OBJ}/proxy_thread.o" "${OBJ}/proxy_libfabric.o" \
    -o "${OUT}/compiler_lto_eval_${name}" \
    -Wl,-rpath,"${LIBFAB}/lib64:${MPI}/lib" \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so -lpthread \
    /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 \
    "${MPI}/lib/libmpi_cray.so"
done

source_after=$(sha256sum "${SRC}")
source_after=${source_after%% *}
test "${source_before}" = "${source_after}"
echo "SOURCE_SHA256=${source_after}"
for name in "${NAMES[@]}"; do
  sha256sum "${OUT}/compiler_lto_eval_${name}"
done
