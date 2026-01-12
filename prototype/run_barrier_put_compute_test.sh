#!/bin/bash
#
# Run barrier + put + compute verification test
# Usage: ./run_barrier_put_compute_test.sh [num_nodes] [num_iters] [window_size]
#
# Output written to: ../build/tmp/prototype_barrier_put/
#

NODES=${1:-4}
ITERS=${2:-500}
WINDOW=${3:-16}

# Output directory
OUTPUT_DIR="../build/tmp/prototype_barrier_put"
mkdir -p "$OUTPUT_DIR"

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_LOG="$OUTPUT_DIR/run_${TIMESTAMP}.log"

echo "========================================="
echo "Barrier + Put + Compute Verification Test"
echo "========================================="
echo "Nodes:       $NODES"
echo "Iterations:  $ITERS"
echo "Window size: $WINDOW"
echo "Output dir:  $OUTPUT_DIR"
echo "Log file:    $RUN_LOG"
echo "========================================="

# Build if needed
if [ ! -f barrier_put_compute_verify ]; then
    echo "Building barrier_put_compute_verify..."
    make barrier_put_compute_verify
    if [ $? -ne 0 ]; then
        echo "ERROR: Build failed"
        exit 1
    fi
fi

# Run the test
echo "Running test..."
srun -N $NODES -n $NODES --ntasks-per-node=1 \
    ./barrier_put_compute_verify $ITERS $WINDOW \
    2>&1 | tee "$RUN_LOG"

# Check results
echo ""
echo "========================================="
echo "Results Summary"
echo "========================================="

# Parse JSON files for pass/fail status
PASS_COUNT=0
FAIL_COUNT=0

for i in $(seq 0 $((NODES-1))); do
    JSON_FILE="$OUTPUT_DIR/stats_rank${i}.json"
    if [ -f "$JSON_FILE" ]; then
        STATUS=$(grep '"status"' "$JSON_FILE" | cut -d'"' -f4)
        JITTER=$(grep '"jitter_ratio"' "$JSON_FILE" | grep -o '[0-9.]*')

        if [ "$STATUS" == "PASS" ]; then
            ((PASS_COUNT++))
            echo "Rank $i: PASS (jitter ratio: $JITTER)"
        else
            ((FAIL_COUNT++))
            echo "Rank $i: FAIL"
        fi
    else
        echo "Rank $i: JSON file not found"
    fi
done

echo ""
echo "Total: $PASS_COUNT PASS, $FAIL_COUNT FAIL"
echo "Output files in: $OUTPUT_DIR"
echo "========================================="

if [ $FAIL_COUNT -eq 0 ]; then
    echo "ALL RANKS PASSED!"
    exit 0
else
    echo "SOME RANKS FAILED!"
    exit 1
fi
