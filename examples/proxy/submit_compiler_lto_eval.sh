#!/bin/bash
# Submit bounded, allocation-paired control or model-policy runs.
set -euo pipefail

ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
MODE="${1:-smoke}"
TAG="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
QUEUE="${GICC_LTO_EVAL_QUEUE:-pci}"
LIMIT="${GICC_LTO_EVAL_TIME_LIMIT:-4m}"
OUT="${ROOT}/docs/experiments/compiler-lto-eval/runs/${TAG}"
SCRIPT="${ROOT}/examples/proxy/run_compiler_lto_eval_allocation.sh"

case "${LIMIT}" in *m|*s) ;; *)
  echo "GICC_LTO_EVAL_TIME_LIMIT must be bounded in minutes or seconds" >&2
  exit 2
esac
case "${MODE}" in
  smoke)
    REPS="1"
    ARMS=(default proxy trigger hand)
    ;;
  paired)
    REPS="1 2 3 4"
    ARMS=(default proxy trigger hand)
    ;;
  gbt)
    REPS="1 2 3 4"
    ARMS=(default hand gbt-history measured-oracle)
    ;;
  *) echo "usage: $0 {smoke|paired|gbt} [TAG]" >&2; exit 2 ;;
esac

for arm in "${ARMS[@]}"; do
  test -x "${ROOT}/build_ofi/compiler_lto_eval/compiler_lto_eval_${arm}"
done
mkdir -p "${OUT}/jobs" "${OUT}/raw"
MANIFEST="${OUT}/jobs.tsv"
printf 'job_id\trep\torder\tqueue\n' > "${MANIFEST}"

order_for() {
  local a=${ARMS[0]} b=${ARMS[1]} c=${ARMS[2]} d=${ARMS[3]}
  case "$1" in
    1) echo "${a},${b},${c},${d}" ;;
    2) echo "${b},${d},${a},${c}" ;;
    3) echo "${d},${c},${b},${a}" ;;
    4) echo "${c},${a},${d},${b}" ;;
  esac
}

previous=""
for rep in ${REPS}; do
  order=$(order_for "${rep}")
  args=(-q "${QUEUE}" -x -N2 -n2 -c8 -g1 --time-limit="${LIMIT}"
        --cwd="${ROOT}" --job-name="gicc-lto-eval-r${rep}"
        --output="${OUT}/jobs/rep${rep}.out"
        --error="${OUT}/jobs/rep${rep}.err")
  if test -n "${previous}"; then
    args+=(--dependency="afterany:${previous}")
  fi
  job_id=$(flux batch "${args[@]}" "${SCRIPT}" "${rep}" "${order}" \
    "${OUT}/raw")
  test -n "${job_id}"
  printf '%s\t%s\t%s\t%s\n' "${job_id}" "${rep}" "${order}" "${QUEUE}" \
    >> "${MANIFEST}"
  previous="${job_id}"
done

printf 'manifest=%s\n' "${MANIFEST}"
