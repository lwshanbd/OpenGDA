#!/bin/bash
# Run the 16 exact mixed-action configurations for the static4 scenario.
set -euo pipefail

if test "$#" -ne 1; then
  echo "usage: $0 OUTDIR" >&2
  exit 2
fi

OUTDIR="$1"
ROOT="${GICC_ROOT:-$PWD}"
BIN_DIR="${ROOT}/build_ofi/compiler_lto_eval"
WARMUP="${GICC_LTO_EVAL_WARMUP:-10}"
RUNS="${GICC_LTO_EVAL_RUNS:-100}"
RUN_TIMEOUT="${GICC_LTO_EVAL_RUN_TIMEOUT:-90}"
# Alternate extremes so a monotonic time drift cannot masquerade as a
# dependence on the number of proxy sites.
ORDER=(0 15 1 14 2 13 3 12 4 11 5 10 6 9 7 8)

mkdir -p "${OUTDIR}"
printf 'ORACLE_ALLOC warmup=%s runs=%s job=%s\n' \
  "${WARMUP}" "${RUNS}" "${FLUX_JOB_ID:-unknown}"
for mask in "${ORDER[@]}"; do
  printf -v suffix '%02d' "${mask}"
  binary="${BIN_DIR}/compiler_lto_eval_static4_mask${suffix}"
  log="${OUTDIR}/static4-mask${suffix}.log"
  test -x "${binary}"
  printf 'RUN mask=%s binary=%s sha256=%s\n' \
    "${mask}" "${binary}" "$(sha256sum "${binary}" | cut -d' ' -f1)" \
    > "${log}"
  printf 'BEGIN mask=%s time=%s\n' \
    "${mask}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  env -u GICC_SKIP_DWQ_INIT -u MPICH_GPU_SUPPORT_ENABLED \
    FI_MR_CACHE_MAX_COUNT=0 GICC_PROXY_ENABLED=1 \
    GICC_NUM_PROXY_THREADS=8 \
    timeout "${RUN_TIMEOUT}s" flux run -N2 -n2 -c8 -g1 \
      "${binary}" --scenario=static4-k4-grid4 \
      --warmup="${WARMUP}" --runs="${RUNS}" \
      < /dev/null >> "${log}" 2>&1
  test "$(grep -c '^COMPILER_LTO_EVAL .* data=OK$' "${log}")" -eq 1
  printf 'END mask=%s time=%s\n' \
    "${mask}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
done
printf 'ORACLE_DONE\n'
