#!/bin/bash
# Build a six-opportunity compiler-only communication-plan capacity sweep from
# the existing, unchanged compiler_lto_calibration.cpp workload.
#
# Set GICC_COMM_PLAN_VARIANT=placement (or use the placement wrapper) to build
# the four-action early/late trigger-placement graph in a disjoint directory.
#
#   facts                emit compiler graph/prompt only
#   controls             build uniform controls for the selected variant
#   candidate FILE NAME  validate candidate-ID-only response and build it
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
MODE="${1:-controls}"
VARIANT="${GICC_COMM_PLAN_VARIANT:-calibration}"
case "${VARIANT}" in
  calibration)
    OUT="${GICC_COMM_PLAN_OUT:-${GICC_ROOT}/build_ofi/compiler_comm_plan_calibration}"
    PROFILE="${GICC_ROOT}/examples/proxy/compiler_lto_calibration_profile.json"
    CONTROL_COMMAND=uniform-controls
    BINARY_STEM=compiler_comm_plan_calibration
    EVAL_CUID=gicc_compiler_comm_plan_calibration_v1
    ;;
  placement)
    OUT="${GICC_COMM_PLAN_OUT:-${GICC_ROOT}/build_ofi/compiler_comm_plan_placement}"
    PROFILE="${GICC_ROOT}/examples/proxy/compiler_comm_plan_placement_profile.json"
    CONTROL_COMMAND=placement-controls
    BINARY_STEM=compiler_comm_plan_placement
    EVAL_CUID=gicc_compiler_comm_plan_placement_v1
    ;;
  *)
    echo "ERROR: GICC_COMM_PLAN_VARIANT must be calibration or placement" >&2
    exit 2
    ;;
esac
META="${OUT}/meta"
OBJ="${OUT}/obj"
GENERATED="${OUT}/generated"
CONTROLS="${GENERATED}/controls"
IR="${GENERATED}/ir"
PASSES="${GICC_PASSES_SO:-${GICC_ROOT}/tools/gicc-passes/build/libgicc-passes.so}"
SRC="${GICC_ROOT}/examples/proxy/compiler_lto_calibration.cpp"
FEATURES="${GENERATED}/features.json"
DOSSIER="${GENERATED}/dossier.json"
GRAPH="${GENERATED}/opportunity-graph.json"
PROMPT="${GENERATED}/prompt.txt"
FACT_BRIDGE="${GICC_ROOT}/tools/gicc-passes/python/gicc_llm_bridge.py"
PLAN_BRIDGE="${GICC_ROOT}/tools/gicc-passes/python/gicc_comm_plan_bridge.py"
EVAL_TOOL="${GICC_ROOT}/examples/proxy/compiler_comm_plan_eval.py"
EXPECTED_SOURCE_SHA256=ac9a7f75da60dbac36f3871e1c0113ec8071b1f7d924ef3dd96e9fdb5888b37d

test -f "${PASSES}" || {
  echo "ERROR: build the pass plugin first: ${PASSES}" >&2
  exit 1
}
case "${MODE}" in
  facts|controls) ;;
  candidate)
    test "$#" -eq 3 || {
      echo "usage: $0 candidate RESPONSE.json NAME" >&2
      exit 2
    }
    [[ "$3" =~ ^llm_[a-zA-Z0-9_.-]+$ ]] || {
      echo "ERROR: candidate NAME must begin with llm_ and contain only " \
           "letters, digits, dot, underscore, or dash" >&2
      exit 2
    }
    ;;
  *)
    echo "usage: $0 {facts|controls|candidate RESPONSE.json NAME}" >&2
    exit 2
    ;;
esac

mkdir -p "${META}" "${OBJ}" "${GENERATED}" "${CONTROLS}" "${IR}"
find "${META}" -mindepth 1 -maxdepth 1 -type f -delete
find "${OBJ}" -mindepth 1 -maxdepth 1 -type f -delete
if [ "${MODE}" = controls ]; then
  find "${CONTROLS}" -mindepth 1 -maxdepth 1 -type f -delete
  find "${IR}" -mindepth 1 -maxdepth 1 -type f -delete
fi

HIPCC=/opt/rocm-6.4.0/lib/llvm/bin/clang++
LLVM_DIS=/opt/rocm-6.4.0/lib/llvm/bin/llvm-dis
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
test "${source_before}" = "${EXPECTED_SOURCE_SHA256}" || {
  echo "ERROR: calibration source hash changed: ${source_before}" >&2
  exit 1
}

echo "=== discover unchanged calibration operations ==="
env GICC_MODE=discover GICC_META_DIR="${META}" GICC_PROXY_ENABLED=1 \
  "${HIPCC}" "${CFLAGS[@]}" -cuid="${EVAL_CUID}" -x hip -c "${SRC}" \
  -o "${OBJ}/discover.o"

echo "=== extract calibration compiler/LTO facts ==="
env GICC_MODE=feature-extract GICC_META_DIR="${META}" \
  GICC_FEATURES_OUT="${FEATURES}" GICC_PROXY_ENABLED=1 \
  "${HIPCC}" "${CFLAGS[@]}" -cuid="${EVAL_CUID}" -x hip -c "${SRC}" \
  -o "${OBJ}/features.o"

python3 "${FACT_BRIDGE}" emit --features "${FEATURES}" \
  --platform "${PROFILE}" --dossier "${DOSSIER}" \
  --prompt "${GENERATED}/site-prompt-unused.txt"
python3 "${PLAN_BRIDGE}" emit --dossier "${DOSSIER}" \
  --graph "${GRAPH}" --prompt "${PROMPT}"

if [ "${MODE}" = facts ]; then
  source_after=$(sha256sum "${SRC}")
  source_after=${source_after%% *}
  test "${source_before}" = "${source_after}"
  echo "SOURCE_SHA256=${source_after}"
  sha256sum "${DOSSIER}" "${GRAPH}" "${PROMPT}"
  exit 0
fi

declare -a NAMES=()
declare -a HINTS=()
if [ "${MODE}" = controls ]; then
  python3 "${EVAL_TOOL}" "${CONTROL_COMMAND}" --graph "${GRAPH}" \
    --out "${CONTROLS}" --source-sha256 "${source_before}"
  python3 "${EVAL_TOOL}" verify --graph "${GRAPH}" \
    --manifest "${CONTROLS}/manifest.json"
  mapfile -t NAMES < <(python3 -c \
    'import json,sys; print("\n".join(x["name"] for x in json.load(open(sys.argv[1]))["arms"]))' \
    "${CONTROLS}/manifest.json")
  for name in "${NAMES[@]}"; do
    HINTS+=("${CONTROLS}/${name}-hint.json")
  done
else
  RESPONSE=$(realpath "$2")
  NAME="$3"
  HINT="${GENERATED}/${NAME}-hint.json"
  python3 "${PLAN_BRIDGE}" accept --strict --graph "${GRAPH}" \
    --response "${RESPONSE}" --hint "${HINT}"
  NAMES=("${NAME}")
  HINTS=("${HINT}")
fi

echo "=== materialize calibration plans from unchanged source ==="
for i in "${!NAMES[@]}"; do
  name=${NAMES[$i]}
  hint=${HINTS[$i]}
  env GICC_MODE=lower GICC_META_DIR="${META}" GICC_HINT_IN="${hint}" \
    GICC_PROXY_ENABLED=1 \
    "${HIPCC}" "${CFLAGS[@]}" -cuid="${EVAL_CUID}" -x hip -c "${SRC}" \
    -o "${OBJ}/${name}.o"
  "${LLVM_DIS}" "${OBJ}/${name}.o" -o "${IR}/${name}.ll"
  if [ "${VARIANT}" = placement ]; then
    env GICC_MODE=lower GICC_META_DIR="${META}" GICC_HINT_IN="${hint}" \
      GICC_PROXY_ENABLED=1 \
      "${HIPCC}" "${CFLAGS[@]}" -cuid="${EVAL_CUID}" -x hip \
      --cuda-device-only -S -emit-llvm "${SRC}" \
      -o "${IR}/${name}.device.ll"
  fi
done

if [ "${MODE}" = controls ]; then
  python3 "${EVAL_TOOL}" verify-ir --graph "${GRAPH}" \
    --manifest "${CONTROLS}/manifest.json" --ir "${IR}"
  if [ "${VARIANT}" = placement ]; then
    python3 "${EVAL_TOOL}" verify-placement-device-ir --graph "${GRAPH}" \
      --manifest "${CONTROLS}/manifest.json" --ir "${IR}"
  fi
fi

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

echo "=== link calibration plan arms ==="
for name in "${NAMES[@]}"; do
  "${HIPCC}" -O3 --offload-arch=gfx90a --hip-link -flto \
    --rtlib=compiler-rt -unwindlib=libgcc \
    "${OBJ}/${name}.o" "${OBJ}/runtime_helpers.o" \
    "${OBJ}/proxy_thread.o" "${OBJ}/proxy_libfabric.o" \
    -o "${OUT}/${BINARY_STEM}_${name}" \
    -Wl,-rpath,"${LIBFAB}/lib64:${MPI}/lib" \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so -lpthread \
    /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 \
    "${MPI}/lib/libmpi_cray.so"
done

source_after=$(sha256sum "${SRC}")
source_after=${source_after%% *}
test "${source_before}" = "${source_after}"
test "${source_after}" = "${EXPECTED_SOURCE_SHA256}"
echo "SOURCE_SHA256=${source_after}"
for name in "${NAMES[@]}"; do
  sha256sum "${OUT}/${BINARY_STEM}_${name}" "${IR}/${name}.ll"
  if [ "${VARIANT}" = placement ]; then
    sha256sum "${IR}/${name}.device.ll"
  fi
done
