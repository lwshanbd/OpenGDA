#!/bin/bash
# Submit exactly one pdebug trigger-placement replicate.  Never queues a
# second job; invoke again only after validating the prior replicate.
set -euo pipefail

if test "$#" -lt 1 || test "$#" -gt 2; then
  echo "usage: $0 REP [TAG]" >&2
  exit 2
fi

ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
REP="$1"
TAG="${2:-compiler-comm-plan-placement-v1}"
[[ "${REP}" =~ ^[1-8]$ ]] || {
  echo "REP must be in 1..8" >&2
  exit 2
}
[[ "${TAG}" =~ ^[a-zA-Z0-9_.-]+$ ]] || {
  echo "TAG contains unsupported characters" >&2
  exit 2
}

QUEUE=pdebug
LIMIT="${GICC_COMM_PLACE_TIME_LIMIT:-10m}"
case "${LIMIT}" in *m|*s) ;; *)
  echo "GICC_COMM_PLACE_TIME_LIMIT must be bounded in minutes or seconds" >&2
  exit 2
esac

BUILD="${ROOT}/build_ofi/compiler_comm_plan_placement"
CONTROL_MANIFEST="${BUILD}/generated/controls/manifest.json"
PROTOCOL="${ROOT}/docs/experiments/compiler-comm-plan-capacity/protocol-placement-v1.json"
OUT="${ROOT}/docs/experiments/compiler-comm-plan-capacity/runs/${TAG}"
SCRIPT="${ROOT}/examples/proxy/run_compiler_comm_plan_placement_allocation.sh"
test -f "${CONTROL_MANIFEST}"
python3 "${ROOT}/examples/proxy/verify_compiler_comm_plan_placement.py" \
  --root "${ROOT}" --build "${BUILD}" --protocol "${PROTOCOL}"
mapfile -t ARMS < <(python3 -c \
  'import json,sys; print("\n".join(x["name"] for x in json.load(open(sys.argv[1]))["arms"]))' \
  "${CONTROL_MANIFEST}")
test "${#ARMS[@]}" -eq 4
test "${ARMS[*]}" = "uniform_p uniform_t uniform_c uniform_e"
for arm in "${ARMS[@]}"; do
  test -x "${BUILD}/compiler_comm_plan_placement_${arm}"
done

active_owned=$(flux jobs -f pending,running -n -o '{id.f58} {name}' | \
  awk '$2 ~ /^gicc-lto-eval-/ || $2 ~ /^gicc-comm-plan-/ || \
       $2 ~ /^gicc-comm-cal-/ || $2 ~ /^gicc-comm-place-/ { print }')
if test -n "${active_owned}"; then
  printf 'refusing parallel submission; owned job still active:\n%s\n' \
    "${active_owned}" >&2
  exit 3
fi

# Four-row Williams order; rows 5..8 are the reverses if later replication is
# needed.  The preregistered first four balance every arm across positions.
case "${REP}" in
  1) indices=(0 1 3 2) ;;
  2) indices=(1 2 0 3) ;;
  3) indices=(2 3 1 0) ;;
  4) indices=(3 0 2 1) ;;
  5) indices=(0 1 3 2) ;;
  6) indices=(1 2 0 3) ;;
  7) indices=(2 3 1 0) ;;
  8) indices=(3 0 2 1) ;;
esac
if (( REP > 4 )); then
  reversed=()
  for ((i=3; i>=0; --i)); do reversed+=("${indices[$i]}"); done
  indices=("${reversed[@]}")
fi
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
      --cwd="${ROOT}" --job-name="gicc-comm-place-r${REP}"
      --output="${OUT}/jobs/rep${REP}.out"
      --error="${OUT}/jobs/rep${REP}.err")
job_id=$(flux batch "${args[@]}" "${SCRIPT}" "${REP}" "${order}" \
  "${OUT}/raw")
test -n "${job_id}"
printf '%s\t%s\t%s\t%s\n' "${job_id}" "${REP}" "${order}" "${QUEUE}" \
  >> "${JOB_MANIFEST}"
printf 'submitted one job: id=%s rep=%s queue=%s order=%s\n' \
  "${job_id}" "${REP}" "${QUEUE}" "${order}"
