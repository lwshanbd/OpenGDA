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
# NODES=1 puts both ends of every pair on one node, which is a different
# physical regime (no wire between them) rather than a smaller version of
# the same one -- useful for asking whether a decider transfers.
NODES=${NODES:-2}
PERNODE=${PERNODE:-1}
NRANKS=$((PERNODE * NODES))
SRUN="srun -p pci -N $NODES -n $NRANKS --ntasks-per-node=$PERNODE -c 16 --gpu-bind=none -t 20"
ARGS="--exp=grid --samples=$SAMPLES --warmup=$WARMUP $EXTRA_ARGS"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"

# Cap each run. A sweep that wedges on one cell otherwise takes the whole
# session with it, and the rows collected before the wedge are still useful.
RUN_TIMEOUT=${RUN_TIMEOUT:-900}

echo "### trigger path"
timeout "$RUN_TIMEOUT" $SRUN "$BIN" --path=trigger $ARGS 2>&1 \
    | tee /dev/stderr | grep '^GRID,' >> "$OUT"

for L in $LANES; do
    echo "### proxy path, $L lane(s)"
    GICC_SKIP_DWQ_INIT=1 GICC_PROXY_ENABLED=1 GICC_NUM_PROXY_THREADS=$L \
        timeout "$RUN_TIMEOUT" $SRUN "$BIN" --path=proxy $ARGS 2>&1 \
        | tee /dev/stderr | grep '^GRID,' >> "$OUT"
    echo "###   -> $(grep -c '^GRID,' "$OUT") rows so far"
done

echo
echo "wrote $(grep -c '^GRID,' "$OUT") rows to $OUT"
