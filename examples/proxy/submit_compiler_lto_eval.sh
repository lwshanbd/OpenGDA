#!/bin/bash
# Submit bounded, allocation-paired control runs for compiler_lto_eval.
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
  smoke) REPS="1" ;;
  paired) REPS="1 2 3 4" ;;
  *) echo "usage: $0 {smoke|paired} [TAG]" >&2; exit 2 ;;
esac

for arm in default proxy trigger hand; do
  test -x "${ROOT}/build_ofi/compiler_lto_eval/compiler_lto_eval_${arm}"
done
mkdir -p "${OUT}/jobs" "${OUT}/raw"
MANIFEST="${OUT}/jobs.tsv"
printf 'job_id\trep\torder\tqueue\n' > "${MANIFEST}"

order_for() {
  case "$1" in
    1) echo default,proxy,trigger,hand ;;
    2) echo proxy,hand,default,trigger ;;
    3) echo hand,trigger,proxy,default ;;
    4) echo trigger,default,hand,proxy ;;
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
