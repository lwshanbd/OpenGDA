#!/bin/bash
#
# Stress test: Run barrier+put+compute verification multiple times
# and generate a summary report
#

NODES=${1:-4}
ITERS=${2:-500}
WINDOW=${3:-16}
RUNS=${4:-10}

OUTPUT_DIR="../build/tmp/prototype_barrier_put"
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="$OUTPUT_DIR/stress_test_summary_${TIMESTAMP}.txt"

echo "========================================="
echo "Stress Test: $RUNS runs"
echo "========================================="
echo "Nodes:       $NODES"
echo "Iterations:  $ITERS"
echo "Window size: $WINDOW"
echo "Runs:        $RUNS"
echo "========================================="
echo ""

# Arrays to collect per-run data
declare -a WALL_TIMES
declare -a JITTER_RATIOS
declare -a PASS_STATUS

for run in $(seq 1 $RUNS); do
    echo "--- Run $run/$RUNS ---"

    # Clear previous stats files
    rm -f "$OUTPUT_DIR"/stats_rank*.json

    # Run the test
    OUTPUT=$(srun -N $NODES -n $NODES --ntasks-per-node=1 \
        ./barrier_put_compute_verify $ITERS $WINDOW 2>&1)

    # Parse results
    ALL_PASS=1
    MAX_JITTER=0
    WALL_TIME=0

    for r in $(seq 0 $((NODES-1))); do
        JSON_FILE="$OUTPUT_DIR/stats_rank${r}.json"
        if [ -f "$JSON_FILE" ]; then
            STATUS=$(grep '"status"' "$JSON_FILE" | cut -d'"' -f4)
            JITTER=$(grep '"jitter_ratio"' "$JSON_FILE" | grep -oE '[0-9]+\.[0-9]+')
            WTIME=$(grep '"wall_time_ms"' "$JSON_FILE" | grep -oE '[0-9]+\.[0-9]+')

            if [ "$STATUS" != "PASS" ]; then
                ALL_PASS=0
            fi

            # Track max jitter
            if (( $(echo "$JITTER > $MAX_JITTER" | bc -l) )); then
                MAX_JITTER=$JITTER
            fi

            # Track wall time from rank 0
            if [ $r -eq 0 ]; then
                WALL_TIME=$WTIME
            fi
        else
            ALL_PASS=0
        fi
    done

    if [ $ALL_PASS -eq 1 ]; then
        echo "  Run $run: PASS, wall_time=${WALL_TIME}ms, max_jitter=${MAX_JITTER}"
        PASS_STATUS+=("PASS")
    else
        echo "  Run $run: FAIL"
        PASS_STATUS+=("FAIL")
    fi

    WALL_TIMES+=($WALL_TIME)
    JITTER_RATIOS+=($MAX_JITTER)

    # Brief pause between runs
    sleep 1
done

echo ""
echo "========================================="
echo "Summary"
echo "========================================="

# Count passes
PASS_COUNT=0
for status in "${PASS_STATUS[@]}"; do
    if [ "$status" == "PASS" ]; then
        ((PASS_COUNT++))
    fi
done

echo "Total: $PASS_COUNT/$RUNS PASS"

# Calculate wall time stats
if [ ${#WALL_TIMES[@]} -gt 0 ]; then
    echo ""
    echo "Wall times (ms):"
    printf "  Values: "
    for wt in "${WALL_TIMES[@]}"; do
        printf "%.1f " $wt
    done
    echo ""

    # Calculate min, max, mean using awk
    echo "${WALL_TIMES[@]}" | tr ' ' '\n' | awk '
    BEGIN { min=1e9; max=0; sum=0; n=0 }
    {
        if ($1+0 > 0) {
            if ($1 < min) min=$1
            if ($1 > max) max=$1
            sum += $1
            n++
        }
    }
    END {
        if (n > 0) {
            printf "  Min: %.1f, Max: %.1f, Mean: %.1f\n", min, max, sum/n
        }
    }'
fi

# Calculate jitter ratio stats
if [ ${#JITTER_RATIOS[@]} -gt 0 ]; then
    echo ""
    echo "Max jitter ratios:"
    printf "  Values: "
    for jr in "${JITTER_RATIOS[@]}"; do
        printf "%.2f " $jr
    done
    echo ""

    echo "${JITTER_RATIOS[@]}" | tr ' ' '\n' | awk '
    BEGIN { min=1e9; max=0; sum=0; n=0 }
    {
        if ($1+0 > 0) {
            if ($1 < min) min=$1
            if ($1 > max) max=$1
            sum += $1
            n++
        }
    }
    END {
        if (n > 0) {
            printf "  Min: %.2f, Max: %.2f, Mean: %.2f\n", min, max, sum/n
        }
    }'
fi

echo ""
echo "========================================="

# Write summary to file
{
    echo "Stress Test Summary: $(date)"
    echo "Nodes: $NODES, Iterations: $ITERS, Window: $WINDOW, Runs: $RUNS"
    echo "Pass Rate: $PASS_COUNT/$RUNS"
    echo ""
    echo "Wall times (ms): ${WALL_TIMES[@]}"
    echo "Jitter ratios: ${JITTER_RATIOS[@]}"
    echo "Status: ${PASS_STATUS[@]}"
} > "$SUMMARY_FILE"

echo "Summary written to: $SUMMARY_FILE"

if [ $PASS_COUNT -eq $RUNS ]; then
    echo "ALL RUNS PASSED!"
    exit 0
else
    echo "SOME RUNS FAILED!"
    exit 1
fi
