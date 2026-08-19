#!/bin/bash
# Submit one bounded allocation containing the complete static4 oracle.
set -euo pipefail

ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
TAG="${1:-$(date -u +%Y%m%dT%H%M%SZ)}"
QUEUE="${GICC_LTO_EVAL_QUEUE:-pci}"
LIMIT="${GICC_LTO_EVAL_ORACLE_LIMIT:-4m}"
OUT="${ROOT}/docs/experiments/compiler-lto-eval/runs/${TAG}"
SCRIPT="${ROOT}/examples/proxy/run_compiler_lto_oracle_allocation.sh"

case "${LIMIT}" in *m|*s) ;; *)
  echo "GICC_LTO_EVAL_ORACLE_LIMIT must be bounded in minutes or seconds" >&2
  exit 2
esac
for mask in $(seq 0 15); do
  printf -v suffix '%02d' "${mask}"
  test -x "${ROOT}/build_ofi/compiler_lto_eval/compiler_lto_eval_static4_mask${suffix}"
done
mkdir -p "${OUT}/jobs" "${OUT}/raw"
job_id=$(flux batch -q "${QUEUE}" -x -N2 -n2 -c8 -g1 \
  --time-limit="${LIMIT}" --cwd="${ROOT}" \
  --job-name="gicc-lto-static4-oracle" \
  --output="${OUT}/jobs/oracle.out" --error="${OUT}/jobs/oracle.err" \
  "${SCRIPT}" "${OUT}/raw")
test -n "${job_id}"
printf 'job_id\tqueue\n%s\t%s\n' "${job_id}" "${QUEUE}" > "${OUT}/jobs.tsv"
printf 'job=%s\nmanifest=%s\n' "${job_id}" "${OUT}/jobs.tsv"
