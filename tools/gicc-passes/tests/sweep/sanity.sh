#!/bin/bash
# tools/gicc-passes/tests/sweep/sanity.sh
# Pre-sweep sanity: verify (a) minimod baseline still ~86 ms and
# (b) barrier_bench W=8 vs W=4 produce different numbers.
# Exit non-zero on either failure.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
cd "${GICC_ROOT}"

echo "=== Sanity gate 1: minimod baseline ==="
META_DIR="${GICC_ROOT}/build_ofi/integration_scratch/sanity_meta"
rm -rf "${META_DIR}"; mkdir -p "${META_DIR}"
cd benchmarks/Minimod_MPI
make TARGET=hip_gicc COMPILER=hipcc_gicc GICC_USE_LTO_PASS=1 clean >/dev/null
GICC_MODE=lower GICC_META_DIR="${META_DIR}" \
    make TARGET=hip_gicc COMPILER=hipcc_gicc GICC_USE_LTO_PASS=1 >/dev/null 2>&1
ms=$(PMI_MAX_KVS_ENTRIES=512 FI_MR_CACHE_MAX_COUNT=0 \
     srun -p pci -N 4 -n 32 --ntasks-per-node=8 -t 2 ./run.sh \
     ./main_hip_gicc_hipcc_gicc --ngpus 32 --grid 1000 --nsteps 100 \
     | grep "rank 0 Time comm" | awk '{print $(NF-1)*1000}')
echo "minimod median: ${ms} ms (expect ~86)"
python3 -c "
m = float('${ms}')
if m > 95 or m < 70:
    print('FAIL: out of expected range'); exit(1)
print('PASS')
"

echo "=== Sanity gate 2: W=8 vs W=4 barrier_bench differ ==="
cd "${GICC_ROOT}/build_ofi"
make -j barrier_bench >/dev/null 2>&1
# Output format: "barrier_bench --gicc: 16 ranks, 100 iters ... DONE  avg=64.78 us  min=...  max=..."
W8=$(FI_MR_CACHE_MAX_COUNT=0 GICC_WINDOW=8 srun -p pci -N 2 -n 16 --ntasks-per-node=8 -t 2 \
     ./examples/ofi/barrier_bench --gicc 1000 | grep "avg=" | sed 's/.*avg=\([0-9.]*\).*/\1/')
W4=$(FI_MR_CACHE_MAX_COUNT=0 GICC_WINDOW=4 srun -p pci -N 2 -n 16 --ntasks-per-node=8 -t 2 \
     ./examples/ofi/barrier_bench --gicc 1000 | grep "avg=" | sed 's/.*avg=\([0-9.]*\).*/\1/')
echo "W=8: ${W8} us;  W=4: ${W4} us"
python3 -c "
import sys
a, b = float('${W8}'), float('${W4}')
diff = abs(a-b) / max(a,b)
if diff < 0.02:
    print(f'WARN: W=8 and W=4 within 2% ({diff*100:.1f}%) — W may have no signal')
    sys.exit(2)
print(f'PASS: W=8 and W=4 differ by {diff*100:.1f}%')
"
