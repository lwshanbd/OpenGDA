#!/bin/bash
# Submit bounded paired compiler-path calibration controls.
set -euo pipefail

ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
MODE="${1:-smoke}"
TAG="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
QUEUE="${GICC_LTO_CAL_QUEUE:-pci}"
LIMIT="${GICC_LTO_CAL_TIME_LIMIT:-4m}"
OUT="${ROOT}/docs/experiments/compiler-lto-calibration/runs/${TAG}"
SCRIPT="${ROOT}/examples/proxy/run_compiler_lto_calibration_allocation.sh"

case "${LIMIT}" in *m|*s) ;; *)
  echo "GICC_LTO_CAL_TIME_LIMIT must be bounded in minutes or seconds" >&2
  exit 2
esac
case "${MODE}" in
  smoke) REPS="0" ;;
  paired) REPS="1 2 3 4" ;;
  *) echo "usage: $0 {smoke|paired} [TAG]" >&2; exit 2 ;;
esac
for arm in proxy trigger; do
  test -x "${ROOT}/build_ofi/compiler_lto_calibration/compiler_lto_calibration_${arm}"
done

mkdir -p "${OUT}/jobs" "${OUT}/raw"
MANIFEST="${OUT}/jobs.tsv"
printf 'job_id\trep\torder\tqueue\n' > "${MANIFEST}"
previous=""
for rep in ${REPS}; do
  if (( rep % 2 == 0 )); then order=proxy,trigger; else order=trigger,proxy; fi
  args=(-q "${QUEUE}" -x -N2 -n2 -c8 -g1 --time-limit="${LIMIT}"
        --cwd="${ROOT}" --job-name="gicc-lto-cal-r${rep}"
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
