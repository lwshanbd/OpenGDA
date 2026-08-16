#!/bin/bash
# run_grid.sh - sweep the joint configuration space for offline decider work.
#
# The proxy lane count is process-wide (GICC_NUM_PROXY_THREADS is read in the
# Runtime constructor), so it needs one binary run per value; everything else
# is swept inside a single run. Output is one concatenated GRID CSV.
#
#   ./examples/proxy/run_grid.sh [outfile]
#
# 2 nodes, 1 rank each. Ask for enough cores that an 8-lane proxy fleet is not
# time-sliced onto one core -- that alone reads as a regression.
set -u

BIN=${BIN:-./build_ofi/examples/proxy/ctx_bench}
OUT=${1:-docs/experiments/grid/grid.csv}
SAMPLES=${SAMPLES:-9}
WARMUP=${WARMUP:-3}
LANES=${LANES:-"1 2 4 8"}
# Rank layout. ctx_bench pairs rank r with rank r + n/2, so with PERNODE
# ranks on each of 2 nodes every pair crosses the node boundary and PERNODE
# pairs push concurrently -- that is the knob for the contention regime.
PERNODE=${PERNODE:-1}
NRANKS=$((PERNODE * 2))
SRUN="srun -p pci -N 2 -n $NRANKS --ntasks-per-node=$PERNODE -c 16 --gpu-bind=none -t 20"
ARGS="--exp=grid --samples=$SAMPLES --warmup=$WARMUP $EXTRA_ARGS"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

echo "### trigger path"
$SRUN "$BIN" --path=trigger $ARGS 2>&1 | tee /dev/stderr | grep '^GRID,' >> "$OUT"

for L in $LANES; do
    echo "### proxy path, $L lane(s)"
    GICC_SKIP_DWQ_INIT=1 GICC_PROXY_ENABLED=1 GICC_NUM_PROXY_THREADS=$L \
        $SRUN "$BIN" --path=proxy $ARGS 2>&1 | tee /dev/stderr | grep '^GRID,' >> "$OUT"
done

echo
echo "wrote $(grep -c '^GRID,' "$OUT") rows to $OUT"
