#!/usr/bin/env bash
# Run one same-allocation, balanced array-batch/scalar-reuse loop scout.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 BASELINE_BINARY REUSED_BINARY OUTPUT_DIR" >&2
    exit 2
fi

baseline=$1
reused=$2
output_dir=$3
for binary in "$baseline" "$reused"; do
    if [[ ! -x $binary ]]; then
        echo "missing executable: $binary" >&2
        exit 2
    fi
done
if [[ -e $output_dir && ! -d $output_dir ]]; then
    echo "scout output is not a directory: $output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir"
for replicate in 1 2 3 4 5 6; do
    if [[ -e $output_dir/rep$replicate ]]; then
        echo "refusing preexisting scout replicate: $output_dir/rep$replicate" >&2
        exit 2
    fi
done

batches=(4 64)

run_arm() {
    local replicate=$1
    local batch=$2
    local arm=$3
    local binary
    case $arm in
        baseline) binary=$baseline ;;
        reused) binary=$reused ;;
        *) echo "invalid arm: $arm" >&2; exit 2 ;;
    esac
    local run_dir="$output_dir/rep${replicate}/batch${batch}"
    mkdir -p "$run_dir"
    printf 'REUSED_DESCRIPTOR_RUN_START replicate=%s batch=%s arm=%s\n' \
        "$replicate" "$batch" "$arm"
    env -u GICC_SKIP_DWQ_INIT -u MPICH_GPU_SUPPORT_ENABLED \
        FI_MR_CACHE_MAX_COUNT=0 FI_MR_CACHE_MONITOR=kdreg2 \
        HSA_ENABLE_IPC_MODE_LEGACY=1 GICC_PROXY_ENABLED=1 \
        flux run -N2 -n2 -c64 -g1 -t 5m -u \
        "$binary" "--batch=$batch" \
        >"$run_dir/$arm.out" 2>"$run_dir/$arm.err"
    printf 'REUSED_DESCRIPTOR_RUN_DONE replicate=%s batch=%s arm=%s\n' \
        "$replicate" "$batch" "$arm"
}

printf 'REUSED_DESCRIPTOR_SCOUT_CONFIG nodes=2 ranks=2 ppn=1 '
printf 'batches=4,64 sizes=canonical16 orders=AB,BA,BA,AB,AB,BA\n'
for replicate in 1 2 3 4 5 6; do
    case $replicate in
        1|4|5) order=(baseline reused) ;;
        2|3|6) order=(reused baseline) ;;
    esac
    for batch in "${batches[@]}"; do
        for arm in "${order[@]}"; do
            run_arm "$replicate" "$batch" "$arm"
        done
    done
done
printf 'REUSED_DESCRIPTOR_SCOUT_DONE pairs=12 runs=24\n'
