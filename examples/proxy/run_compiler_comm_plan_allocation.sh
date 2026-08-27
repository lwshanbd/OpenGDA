#!/bin/bash
# Run all nine structural-plan controls sequentially inside one paired-node
# allocation.  The enclosing allocation is one scheduler job; arm order is
# supplied by the one-job-at-a-time submitter and rotated across replicates.
set -euo pipefail

if test "$#" -ne 3; then
  echo "usage: $0 REP ORDER OUTDIR" >&2
  exit 2
fi

REP="$1"
ORDER="$2"
OUTDIR="$3"
ROOT="${GICC_ROOT:-$PWD}"
BIN_DIR="${ROOT}/build_ofi/compiler_comm_plan_eval"
WARMUP="${GICC_COMM_PLAN_WARMUP:-10}"
RUNS="${GICC_COMM_PLAN_RUNS:-100}"
RUN_TIMEOUT="${GICC_COMM_PLAN_RUN_TIMEOUT:-120}"

mkdir -p "${OUTDIR}"
IFS=, read -r -a ARMS <<< "${ORDER}"
test "${#ARMS[@]}" -eq 9

printf 'ALLOC rep=%s order=%s warmup=%s runs=%s job=%s\n' \
  "${REP}" "${ORDER}" "${WARMUP}" "${RUNS}" "${FLUX_JOB_ID:-unknown}"

declare -A SEEN_ARMS=()
for arm in "${ARMS[@]}"; do
  [[ "${arm}" =~ ^plan_[ptc][ptc]$ ]] || {
    echo "invalid arm=${arm}" >&2
    exit 2
  }
  if [[ -n "${SEEN_ARMS[${arm}]:-}" ]]; then
    echo "duplicate arm=${arm}" >&2
    exit 2
  fi
  SEEN_ARMS["${arm}"]=1
  binary="${BIN_DIR}/compiler_comm_plan_eval_${arm}"
  test -x "${binary}"
  log="${OUTDIR}/rep${REP}-${arm}.log"
  printf 'RUN rep=%s arm=%s binary=%s sha256=%s\n' \
    "${REP}" "${arm}" "${binary}" \
    "$(sha256sum "${binary}" | cut -d' ' -f1)" > "${log}"
  printf 'BEGIN rep=%s arm=%s log=%s time=%s\n' \
    "${REP}" "${arm}" "${log}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"

  env -u GICC_SKIP_DWQ_INIT -u MPICH_GPU_SUPPORT_ENABLED \
    FI_MR_CACHE_MAX_COUNT=0 GICC_PROXY_ENABLED=1 \
    GICC_NUM_PROXY_THREADS=8 \
    timeout "${RUN_TIMEOUT}s" flux run -N2 -n2 -c8 -g1 \
      "${binary}" --warmup="${WARMUP}" --runs="${RUNS}" \
      < /dev/null >> "${log}" 2>&1

  test "$(grep -c '^COMPILER_LTO_EVAL ' "${log}")" -eq 7
  test "$(grep -c '^COMPILER_LTO_EVAL .* data=OK$' "${log}")" -eq 7
  printf 'END rep=%s arm=%s time=%s\n' \
    "${REP}" "${arm}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

printf 'ALLOC_DONE rep=%s\n' "${REP}"
