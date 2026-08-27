#!/bin/bash
# Run all four uniform early/late placement controls sequentially inside one
# paired-node allocation.  The submitter rotates the order across manually
# submitted replicates.
set -euo pipefail

if test "$#" -ne 3; then
  echo "usage: $0 REP ORDER OUTDIR" >&2
  exit 2
fi

REP="$1"
ORDER="$2"
OUTDIR="$3"
ROOT="${GICC_ROOT:-$PWD}"
BIN_DIR="${ROOT}/build_ofi/compiler_comm_plan_placement"
WARMUP="${GICC_COMM_PLACE_WARMUP:-10}"
RUNS="${GICC_COMM_PLACE_RUNS:-100}"
RUN_TIMEOUT="${GICC_COMM_PLACE_RUN_TIMEOUT:-180}"

mkdir -p "${OUTDIR}"
IFS=, read -r -a ARMS <<< "${ORDER}"
test "${#ARMS[@]}" -eq 4

printf 'ALLOC rep=%s order=%s warmup=%s runs=%s job=%s\n' \
  "${REP}" "${ORDER}" "${WARMUP}" "${RUNS}" "${FLUX_JOB_ID:-unknown}"

declare -A SEEN_ARMS=()
for arm in "${ARMS[@]}"; do
  case "${arm}" in uniform_p|uniform_t|uniform_c|uniform_e) ;; *)
    echo "invalid placement arm=${arm}" >&2
    exit 2
  esac
  if [[ -n "${SEEN_ARMS[${arm}]:-}" ]]; then
    echo "duplicate placement arm=${arm}" >&2
    exit 2
  fi
  SEEN_ARMS["${arm}"]=1
  binary="${BIN_DIR}/compiler_comm_plan_placement_${arm}"
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

  test "$(grep -c '^COMPILER_LTO_CALIBRATION ' "${log}")" -eq 18
  test "$(grep -c '^COMPILER_LTO_CALIBRATION .* data=OK$' "${log}")" -eq 18
  printf 'END rep=%s arm=%s time=%s\n' \
    "${REP}" "${arm}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done

test "${#SEEN_ARMS[@]}" -eq 4
printf 'ALLOC_DONE rep=%s\n' "${REP}"
