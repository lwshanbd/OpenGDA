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
  calibrated-balanced)
    # Five Williams-style rows followed by their reversals.  For an odd
    # number of arms this balances both position and ordered carryover.
    REPS="1 2 3 4 5 6 7 8 9 10"
    ARMS=(default hand gbt-history gbt-calibrated measured-oracle)
    ;;
  llm-balanced)
    test "$#" -eq 3 || {
      echo "usage: $0 llm-balanced TAG CANDIDATE_ARM" >&2
      exit 2
    }
    CANDIDATE_ARM="$3"
    [[ "${CANDIDATE_ARM}" =~ ^llm-policy[0-9][0-9]$ ]] || {
      echo "candidate arm must match llm-policyNN" >&2
      exit 2
    }
    REPS="1 2 3 4 5 6 7 8 9 10"
    ARMS=(default hand gbt-history gbt-calibrated "${CANDIDATE_ARM}")
    ;;
  *) echo "usage: $0 {smoke|paired|gbt|calibrated-balanced|llm-balanced} [TAG] [CANDIDATE_ARM]" >&2; exit 2 ;;
esac

for arm in "${ARMS[@]}"; do
  test -x "${ROOT}/build_ofi/compiler_lto_eval/compiler_lto_eval_${arm}"
done
mkdir -p "${OUT}/jobs" "${OUT}/raw"
MANIFEST="${OUT}/jobs.tsv"
printf 'job_id\trep\torder\tqueue\n' > "${MANIFEST}"

order_for() {
  if [ "${MODE}" = calibrated-balanced ] || [ "${MODE}" = llm-balanced ]; then
    local rep=$1 rotation reverse=0
    if (( rep > 5 )); then
      rep=$((rep - 5))
      reverse=1
    fi
    rotation=$((rep - 1))
    local base=(0 1 4 2 3) indices=() index position order=""
    for index in "${base[@]}"; do
      indices+=( $(( (index + rotation) % 5 )) )
    done
    if (( reverse )); then
      for (( index=4; index>=0; --index )); do
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
    echo "${order}"
    return
  fi
  if test "${#ARMS[@]}" -eq 4; then
    local a=${ARMS[0]} b=${ARMS[1]} c=${ARMS[2]} d=${ARMS[3]}
    case "$1" in
      1) echo "${a},${b},${c},${d}" ;;
      2) echo "${b},${d},${a},${c}" ;;
      3) echo "${d},${c},${b},${a}" ;;
      4) echo "${c},${a},${d},${b}" ;;
    esac
    return
  fi
  local rep=$1 index order=""
  for (( index=0; index<${#ARMS[@]}; ++index )); do
    local position=$(( (index + rep - 1) % ${#ARMS[@]} ))
    if test -n "${order}"; then order+=,; fi
    order+="${ARMS[${position}]}"
  done
  echo "${order}"
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
