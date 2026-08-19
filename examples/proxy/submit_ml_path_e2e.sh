#!/bin/bash
# Submit a same-source default/GBT pair.  Each job is independently bounded;
# the GBT leg starts only after a successful default leg.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
QUEUE="${GICC_ML_QUEUE:-pci}"
LIMIT="${GICC_ML_TIME_LIMIT:-2m}"
TAG="${1:-manual}"
OUT="${GICC_ROOT}/docs/experiments/compiler-ml-path"
HINT="${GICC_ROOT}/build_ofi/ml_path_e2e/gbt-hint.json"

case "${LIMIT}" in
  *m|*s) ;;
  *) echo "GICC_ML_TIME_LIMIT must be a bounded minutes/seconds value" >&2; exit 2 ;;
esac
test -x "${GICC_ROOT}/build_ofi/ml_path_e2e/ml_path_default"
test -x "${GICC_ROOT}/build_ofi/ml_path_e2e/ml_path_gbt"
test -f "${HINT}"
mkdir -p "${OUT}"

# This process-global choice is part of the learned action.  Apply it to both
# binaries so the comparison isolates compiler lowering rather than CPU-worker
# count.  Remove an inherited Python path because Tioga's system NumPy/sklearn
# installation is for its system interpreter.
ML_THREADS=$(env -u PYTHONPATH python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["ml_metadata"]["runtime_global"]["GICC_NUM_PROXY_THREADS"])' \
  "${HINT}")

COMMON=(
  -q "${QUEUE}" -N2 -n2 -c8 -g1 --time-limit="${LIMIT}"
  --cwd="${GICC_ROOT}"
)

DEFAULT_ID=$(env -u GICC_SKIP_DWQ_INIT -u MPICH_GPU_SUPPORT_ENABLED \
  GICC_NUM_PROXY_THREADS="${ML_THREADS}" GICC_PROXY_ENABLED=1 \
  flux submit "${COMMON[@]}" --job-name="gicc-ml-default-${TAG}" \
  --output="${OUT}/ml-default-${TAG}.out" \
  --error="${OUT}/ml-default-${TAG}.err" \
  ./build_ofi/ml_path_e2e/ml_path_default --warmup=10 --runs=100)

GBT_ID=$(env -u GICC_SKIP_DWQ_INIT -u MPICH_GPU_SUPPORT_ENABLED \
  GICC_NUM_PROXY_THREADS="${ML_THREADS}" GICC_PROXY_ENABLED=1 \
  flux submit "${COMMON[@]}" --dependency="afterok:${DEFAULT_ID}" \
  --job-name="gicc-ml-gbt-${TAG}" \
  --output="${OUT}/ml-gbt-${TAG}.out" \
  --error="${OUT}/ml-gbt-${TAG}.err" \
  ./build_ofi/ml_path_e2e/ml_path_gbt --warmup=10 --runs=100)

printf 'default=%s\ngbt=%s\n' "${DEFAULT_ID}" "${GBT_ID}"
