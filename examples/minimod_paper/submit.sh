#!/bin/bash
# Submit bounded Minimod paper experiments.  Full repetitions are separate
# allocations; arms within an allocation are paired and randomized.
set -euo pipefail

ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
SCRIPT="$ROOT/examples/minimod_paper"
OUT_BASE="${MINIMOD_PAPER_RESULTS:-$ROOT/docs/experiments/minimod-paper}"
MODE="${1:-smoke}"
TAG="${2:-$(date -u +%Y%m%dT%H%M%SZ)}"
OUT="$OUT_BASE/$TAG"
mkdir -p "$OUT/raw" "$OUT/jobs"
MANIFEST="$OUT/jobs.tsv"
printf 'job_id\tkind\tnodes\trpn\trep\tgrid\tsteps\torder\tqueue\n' > "$MANIFEST"

submit_one() {
    local kind="$1" nodes="$2" rpn="$3" rep="$4" grid="$5" steps="$6"
    local queue="$7" limit="$8" dependency="${9:-}"
    local ranks=$((nodes * rpn)) seed order jobid
    seed="${TAG}:${kind}:${nodes}:${rpn}:${rep}:${grid}:${steps}"
    order=$(python3 "$SCRIPT/order.py" "$kind" "$seed")
    local args=(-q "$queue" -x -N "$nodes" -n "$ranks" -c 8 -g 1
                --time-limit="$limit" --cwd="$ROOT"
                --job-name="gicc-mm-${kind}-n${nodes}r${rpn}p${rep}"
                --output="$OUT/jobs/${kind}-n${nodes}-rpn${rpn}-rep${rep}.out"
                --error="$OUT/jobs/${kind}-n${nodes}-rpn${rpn}-rep${rep}.err")
    if test -n "$dependency"; then
        args+=(--dependency="afterany:$dependency")
    fi
    if ! jobid=$(flux batch "${args[@]}" "$SCRIPT/run_allocation.sh" \
        "$kind" "$nodes" "$rpn" "$rep" "$grid" "$steps" "$order" \
        "$OUT/raw"); then
        echo "submission failed: kind=$kind nodes=$nodes rpn=$rpn rep=$rep queue=$queue" >&2
        return 1
    fi
    if test -z "$jobid"; then
        echo "submission returned no job id: kind=$kind nodes=$nodes rpn=$rpn rep=$rep queue=$queue" >&2
        return 1
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$jobid" "$kind" "$nodes" "$rpn" "$rep" "$grid" "$steps" \
        "$order" "$queue" >> "$MANIFEST"
    printf '%s\n' "$jobid"
}

case "$MODE" in
    smoke)
        # One short two-node allocation exercises all standard paths.
        submit_one standard 2 1 0 100 10 pci 2m
        ;;
    o5-smoke)
        submit_one o5 2 1 0 100 10 pci 2m
        ;;
    o5)
        prev=""
        o5_grid="${MINIMOD_O5_GRID:-100}"
        o5_steps="${MINIMOD_O5_STEPS:-1000}"
        for rep in 1 2 3 4 5; do
            prev=$(submit_one o5 2 1 "$rep" "$o5_grid" "$o5_steps" \
                pci 2m "$prev")
        done
        ;;
    matrix)
        # PCI has four dedicated nodes; serialize its 30 allocation-level
        # replicates.  Eight-node jobs use a second independent chain on the
        # configured larger queue, so no allocation waits while holding nodes.
        prev_pci=""
        for nodes in 1 2 4; do
            for rpn in 1 8; do
                for rep in 1 2 3 4 5; do
                    prev_pci=$(submit_one standard "$nodes" "$rpn" "$rep" \
                        800 100 pci 3m "$prev_pci")
                done
            done
        done
        prev_pllm=""
        queue8="${MINIMOD_8_QUEUE:-pdebug}"
        for rpn in 1 8; do
            for rep in 1 2 3 4 5; do
                prev_pllm=$(submit_one standard 8 "$rpn" "$rep" \
                    800 100 "$queue8" 3m "$prev_pllm")
            done
        done
        ;;
    matrix-pci)
        # PCI-only robustness matrix.  Environment controls make it possible
        # to repeat the complete 1/2/4-node design at another problem size
        # without also creating an impossible duplicate 8-node chain.
        prev=""
        matrix_grid="${MINIMOD_STANDARD_GRID:-800}"
        matrix_steps="${MINIMOD_STANDARD_STEPS:-100}"
        for nodes in 1 2 4; do
            for rpn in 1 8; do
                for rep in 1 2 3 4 5; do
                    prev=$(submit_one standard "$nodes" "$rpn" "$rep" \
                        "$matrix_grid" "$matrix_steps" pci 3m "$prev")
                done
            done
        done
        ;;
    matrix8)
        prev=""
        queue8="${MINIMOD_8_QUEUE:-pdebug}"
        for rpn in 1 8; do
            for rep in 1 2 3 4 5; do
                prev=$(submit_one standard 8 "$rpn" "$rep" \
                    800 100 "$queue8" 3m "$prev")
            done
        done
        ;;
    matrix4)
        # Re-submit only the four-node cells on a user-selected queue.  This
        # is useful when the four-node PCI partition is temporarily occupied.
        prev=""
        queue4="${MINIMOD_4_QUEUE:-pdebug}"
        for rpn in 1 8; do
            if test "$rpn" -eq 1; then
                reps="${MINIMOD_4_RPN1_REPS-1 2 3 4 5}"
            else
                reps="${MINIMOD_4_RPN8_REPS-1 2 3 4 5}"
            fi
            for rep in $reps; do
                case "$rep" in
                    1|2|3|4|5) ;;
                    *) echo "invalid four-node rep: $rep" >&2; exit 2 ;;
                esac
                prev=$(submit_one standard 4 "$rpn" "$rep" \
                    800 100 "$queue4" 3m "$prev")
            done
        done
        ;;
    *)
        echo "usage: $0 {smoke|o5-smoke|o5|matrix|matrix-pci|matrix4|matrix8} [tag]" >&2
        exit 2
        ;;
esac

printf 'manifest=%s\n' "$MANIFEST"
