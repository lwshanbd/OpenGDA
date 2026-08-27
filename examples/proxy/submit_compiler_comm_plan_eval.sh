#!/bin/bash
# Submit exactly one pdebug replicate.  Invoke again only after that job is
# inactive and validated; this script deliberately has no submission loop.
set -euo pipefail

if test "$#" -lt 1 || test "$#" -gt 2; then
  echo "usage: $0 REP [TAG]" >&2
  exit 2
fi

ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
REP="$1"
TAG="${2:-compiler-comm-plan-controls-v1}"
[[ "${REP}" =~ ^([1-9]|1[0-8])$ ]] || {
  echo "REP must be in 1..18" >&2
  exit 2
}
[[ "${TAG}" =~ ^[a-zA-Z0-9_.-]+$ ]] || {
  echo "TAG contains unsupported characters" >&2
  exit 2
}

QUEUE=pdebug
LIMIT="${GICC_COMM_PLAN_TIME_LIMIT:-12m}"
case "${LIMIT}" in *m|*s) ;; *)
  echo "GICC_COMM_PLAN_TIME_LIMIT must be bounded in minutes or seconds" >&2
  exit 2
esac

BUILD="${ROOT}/build_ofi/compiler_comm_plan_eval"
CONTROL_MANIFEST="${BUILD}/generated/controls/manifest.json"
OUT="${ROOT}/docs/experiments/compiler-comm-plan-capacity/runs/${TAG}"
SCRIPT="${ROOT}/examples/proxy/run_compiler_comm_plan_allocation.sh"
test -f "${CONTROL_MANIFEST}"
mapfile -t ARMS < <(python3 -c \
  'import json,sys; print("\n".join(x["name"] for x in json.load(open(sys.argv[1]))["arms"]))' \
  "${CONTROL_MANIFEST}")
test "${#ARMS[@]}" -eq 9
for arm in "${ARMS[@]}"; do
  test -x "${BUILD}/compiler_comm_plan_eval_${arm}"
done

# Protect the user's one-job-at-a-time rule without treating unrelated agents'
# jobs as ours.  The older compiler-LTO series and this structural-plan series
# are the only names owned by this experiment thread.
active_owned=$(flux jobs -f pending,running -n -o '{id.f58} {name}' | \
  awk '$2 ~ /^gicc-lto-eval-/ || $2 ~ /^gicc-comm-plan-/ { print }')
if test -n "${active_owned}"; then
  printf 'refusing parallel submission; owned job still active:\n%s\n' \
    "${active_owned}" >&2
  exit 3
fi

# Odd-order Williams-style base sequence.  Rows 10..18 reverse rows 1..9,
# balancing both position and first-order carryover when all rows are run.
base=(0 1 8 2 7 3 6 4 5)
row=${REP}
reverse=0
if (( row > 9 )); then
  row=$((row - 9))
  reverse=1
fi
rotation=$((row - 1))
indices=()
for index in "${base[@]}"; do
  indices+=( $(( (index + rotation) % 9 )) )
done
order=""
if (( reverse )); then
  for (( index=8; index>=0; --index )); do
    position=${indices[$index]}
    if test -n "${order}"; then order+=,; fi
    order+="${ARMS[$position]}"
  done
else
  for position in "${indices[@]}"; do
    if test -n "${order}"; then order+=,; fi
    order+="${ARMS[$position]}"
  done
fi

mkdir -p "${OUT}/jobs" "${OUT}/raw"
JOB_MANIFEST="${OUT}/jobs.tsv"
if test ! -f "${JOB_MANIFEST}"; then
  printf 'job_id\trep\torder\tqueue\n' > "${JOB_MANIFEST}"
fi
if awk -F '\t' -v rep="${REP}" 'NR > 1 && $2 == rep { found=1 } END { exit !found }' \
    "${JOB_MANIFEST}"; then
  echo "rep=${REP} already exists in ${JOB_MANIFEST}" >&2
  exit 2
fi

args=(-q "${QUEUE}" -x -N2 -n2 -c8 -g1 --time-limit="${LIMIT}"
      --cwd="${ROOT}" --job-name="gicc-comm-plan-r${REP}"
      --output="${OUT}/jobs/rep${REP}.out"
      --error="${OUT}/jobs/rep${REP}.err")
job_id=$(flux batch "${args[@]}" "${SCRIPT}" "${REP}" "${order}" \
  "${OUT}/raw")
test -n "${job_id}"
printf '%s\t%s\t%s\t%s\n' "${job_id}" "${REP}" "${order}" "${QUEUE}" \
  >> "${JOB_MANIFEST}"
printf 'submitted one job: id=%s rep=%s queue=%s order=%s\n' \
  "${job_id}" "${REP}" "${QUEUE}" "${order}"
