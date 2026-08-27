#!/bin/bash
# Submit exactly one pdebug replicate of the six-opportunity controls.  Invoke
# again only after the earlier replicate is inactive and validated.
set -euo pipefail

if test "$#" -lt 1 || test "$#" -gt 2; then
  echo "usage: $0 REP [TAG]" >&2
  exit 2
fi

ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
REP="$1"
TAG="${2:-compiler-comm-plan-calibration-v1}"
[[ "${REP}" =~ ^[1-6]$ ]] || {
  echo "REP must be in 1..6" >&2
  exit 2
}
[[ "${TAG}" =~ ^[a-zA-Z0-9_.-]+$ ]] || {
  echo "TAG contains unsupported characters" >&2
  exit 2
}

QUEUE=pdebug
LIMIT="${GICC_COMM_CAL_TIME_LIMIT:-8m}"
case "${LIMIT}" in *m|*s) ;; *)
  echo "GICC_COMM_CAL_TIME_LIMIT must be bounded in minutes or seconds" >&2
  exit 2
esac

BUILD="${ROOT}/build_ofi/compiler_comm_plan_calibration"
CONTROL_MANIFEST="${BUILD}/generated/controls/manifest.json"
OUT="${ROOT}/docs/experiments/compiler-comm-plan-capacity/runs/${TAG}"
SCRIPT="${ROOT}/examples/proxy/run_compiler_comm_plan_calibration_allocation.sh"
test -f "${CONTROL_MANIFEST}"
mapfile -t ARMS < <(python3 -c \
  'import json,sys; print("\n".join(x["name"] for x in json.load(open(sys.argv[1]))["arms"]))' \
  "${CONTROL_MANIFEST}")
test "${#ARMS[@]}" -eq 3
test "${ARMS[*]}" = "uniform_p uniform_t uniform_c"
for arm in "${ARMS[@]}"; do
  test -x "${BUILD}/compiler_comm_plan_calibration_${arm}"
done

# Check only job names owned by this experiment thread.  Unrelated jobs under
# the user's account are neither blockers nor targets for any operation here.
active_owned=$(flux jobs -f pending,running -n -o '{id.f58} {name}' | \
  awk '$2 ~ /^gicc-lto-eval-/ || $2 ~ /^gicc-comm-plan-/ || \
       $2 ~ /^gicc-comm-cal-/ { print }')
if test -n "${active_owned}"; then
  printf 'refusing parallel submission; owned job still active:\n%s\n' \
    "${active_owned}" >&2
  exit 3
fi

# Three cyclic rows followed by their reverses balance position and immediate
# carryover over six manually submitted replicates.
case "${REP}" in
  1) indices=(0 1 2) ;;
  2) indices=(1 2 0) ;;
  3) indices=(2 0 1) ;;
  4) indices=(2 1 0) ;;
  5) indices=(0 2 1) ;;
  6) indices=(1 0 2) ;;
esac
order=""
for index in "${indices[@]}"; do
  if test -n "${order}"; then order+=,; fi
  order+="${ARMS[$index]}"
done

mkdir -p "${OUT}/jobs" "${OUT}/raw"
JOB_MANIFEST="${OUT}/jobs.tsv"
if test ! -f "${JOB_MANIFEST}"; then
  printf 'job_id\trep\torder\tqueue\n' > "${JOB_MANIFEST}"
fi
if awk -F '\t' -v rep="${REP}" \
    'NR > 1 && $2 == rep { found=1 } END { exit !found }' \
    "${JOB_MANIFEST}"; then
  echo "rep=${REP} already exists in ${JOB_MANIFEST}" >&2
  exit 2
fi

args=(-q "${QUEUE}" -x -N2 -n2 -c8 -g1 --time-limit="${LIMIT}"
      --cwd="${ROOT}" --job-name="gicc-comm-cal-r${REP}"
      --output="${OUT}/jobs/rep${REP}.out"
      --error="${OUT}/jobs/rep${REP}.err")
job_id=$(flux batch "${args[@]}" "${SCRIPT}" "${REP}" "${order}" \
  "${OUT}/raw")
test -n "${job_id}"
printf '%s\t%s\t%s\t%s\n' "${job_id}" "${REP}" "${order}" "${QUEUE}" \
  >> "${JOB_MANIFEST}"
printf 'submitted one job: id=%s rep=%s queue=%s order=%s\n' \
  "${job_id}" "${REP}" "${QUEUE}" "${order}"
