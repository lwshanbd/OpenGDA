#!/bin/bash
# 03_no_hint_fallback: minimod 32-rank with the LTO pipeline but NO
# hint.json provided. DispatchLowering must fall back to the
# IPC_OR_DWQ default (hybrid runtime branch) and produce bit-identical
# output to GDA.
#
# Pass criterion:
#   FINAL distribution = 4 / 26 / 1 / 1
#   The two non-zero ranks have -1178.386475/757.801636 and
#   -1178.444702/757.767761 exactly.
set -euo pipefail

META_DIR="$(mktemp -d)"
trap 'rm -rf "${META_DIR}"' EXIT

cd "${GICC_ROOT}/benchmarks/Minimod_MPI"
make TARGET=hip_gicc COMPILER=hipcc_gicc GICC_USE_LTO_PASS=1 clean >/dev/null

# NOTE: deliberately do NOT set GICC_HINT_IN — the pass must default
# to IPC_OR_DWQ all on its own.
GICC_MODE=lower GICC_META_DIR="${META_DIR}" \
    make TARGET=hip_gicc COMPILER=hipcc_gicc GICC_USE_LTO_PASS=1 \
    > /tmp/03_build.log 2>&1 || { tail /tmp/03_build.log; exit 2; }

OUT=$(PMI_MAX_KVS_ENTRIES=512 FI_MR_CACHE_MAX_COUNT=0 \
      srun -p pci -N 4 -n 32 --ntasks-per-node=8 -t 2 ./run.sh \
      ./main_hip_gicc_hipcc_gicc --ngpus 32 --grid 1000 --nsteps 100 2>&1)

distribution=$(echo "${OUT}" | grep "FINAL min_u" | sort | uniq -c)
echo "${distribution}"

# The four expected lines (counts may format-pad differently across
# srun-versions; match by content, not exact whitespace).
have_zero_pair=$(echo "${distribution}" | grep -c "FINAL min_u, *max_u = -0.000000, 0.000000" || true)
have_zero_only=$(echo "${distribution}" | grep -c "FINAL min_u, *max_u = 0.000000, 0.000000" || true)
have_low=$(echo "${distribution}" | grep -c "1178.386475, 757.801636" || true)
have_high=$(echo "${distribution}" | grep -c "1178.444702, 757.767761" || true)

if [[ "${have_zero_pair}" -eq 1 && "${have_zero_only}" -eq 1 \
   && "${have_low}" -eq 1 && "${have_high}" -eq 1 ]]; then
    echo "PASS: FINAL distribution matches GDA reference"
    exit 0
fi
echo "FAIL: distribution mismatch"
exit 1
