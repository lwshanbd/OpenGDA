#!/bin/bash
# tools/gicc-passes/tests/run_perf.sh — minimod 32-rank perf regression
# check. Builds the LTO pipeline binary, runs minimod 3 times, computes
# median, compares against the recorded baseline at
# tests/perf_baseline/tioga_minimod_4x8_grid1000_nsteps100.json.
#
# Pass criterion: new median ≤ baseline_median × 1.05 (5% tolerance).
# Use this nightly / before tagging a release. Not in CI — needs the
# Tioga compute node + 4-node allocation.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../../.." && pwd)}"
BASELINE_JSON="${GICC_ROOT}/tests/perf_baseline/tioga_minimod_4x8_grid1000_nsteps100.json"

if [[ ! -f "${BASELINE_JSON}" ]]; then
    echo "ERROR: baseline ${BASELINE_JSON} not found"
    exit 2
fi

BASELINE_MEDIAN=$(python3 -c "
import json, sys
b = json.load(open('${BASELINE_JSON}'))
print(b['results']['hip_gicc_lto']['median_ms'])
")
echo "Baseline LTO median: ${BASELINE_MEDIAN} ms"

cd "${GICC_ROOT}/benchmarks/Minimod_MPI"

META_DIR="${GICC_ROOT}/build_ofi/integration_scratch/perf_meta"
rm -rf "${META_DIR}"; mkdir -p "${META_DIR}"

make TARGET=hip_gicc COMPILER=hipcc_gicc GICC_USE_LTO_PASS=1 clean >/dev/null
GICC_MODE=lower GICC_META_DIR="${META_DIR}" \
    make TARGET=hip_gicc COMPILER=hipcc_gicc GICC_USE_LTO_PASS=1 >/dev/null 2>&1

declare -a RUNS=()
for i in 1 2 3; do
    out=$(PMI_MAX_KVS_ENTRIES=512 FI_MR_CACHE_MAX_COUNT=0 \
          srun -p pci -N 4 -n 32 --ntasks-per-node=8 -t 2 ./run.sh \
          ./main_hip_gicc_hipcc_gicc --ngpus 32 --grid 1000 --nsteps 100 2>&1)
    ms=$(echo "${out}" | grep "rank 0 Time comm" \
                       | awk '{print $NF*1000}')
    echo "  run ${i}: ${ms} ms"
    RUNS+=("${ms}")
done

NEW_MEDIAN=$(python3 -c "
runs = sorted([float(x) for x in '${RUNS[*]}'.split()])
print(runs[len(runs)//2])
")
echo "New LTO median: ${NEW_MEDIAN} ms"

python3 -c "
b = float('${BASELINE_MEDIAN}')
n = float('${NEW_MEDIAN}')
ratio = n / b
import sys
if ratio <= 1.05:
    print(f'PASS: {n:.2f} ms / {b:.2f} ms = {ratio:.3f}x  (<= 1.05x)')
    sys.exit(0)
else:
    print(f'FAIL: {n:.2f} ms / {b:.2f} ms = {ratio:.3f}x  (> 1.05x — perf regression)')
    sys.exit(1)
"
