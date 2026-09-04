#!/usr/bin/env bash
# Run one same-allocation, balanced baseline/guarded mm_minimal scout.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 BASELINE_BINARY GUARDED_BINARY OUTPUT_DIR" >&2
    exit 2
fi

baseline=$1
guarded=$2
output_dir=$3
for binary in "$baseline" "$guarded"; do
    if [[ ! -x $binary ]]; then
        echo "missing executable: $binary" >&2
        exit 2
    fi
done

sizes=(4096 8192)

run_arm() {
    local replicate=$1
    local size=$2
    local arm=$3
    local binary
    case $arm in
        baseline) binary=$baseline ;;
        guarded)  binary=$guarded ;;
        *) echo "invalid arm: $arm" >&2; exit 2 ;;
    esac
    local run_dir="$output_dir/rep${replicate}/size${size}"
    mkdir -p "$run_dir"
    printf 'GUARDED_EARLY_RUN_START replicate=%s size=%s arm=%s\n' \
        "$replicate" "$size" "$arm"
    env -u GICC_SKIP_DWQ_INIT -u MPICH_GPU_SUPPORT_ENABLED \
        FI_MR_CACHE_MAX_COUNT=0 FI_MR_CACHE_MONITOR=kdreg2 \
        HSA_ENABLE_IPC_MODE_LEGACY=1 GICC_PROXY_ENABLED=1 \
        flux run -N2 -n16 -c8 -g1 -t 8m -u \
        "$binary" "$size" \
        >"$run_dir/$arm.out" 2>"$run_dir/$arm.err"
    printf 'GUARDED_EARLY_RUN_DONE replicate=%s size=%s arm=%s\n' \
        "$replicate" "$size" "$arm"
}

printf 'GUARDED_EARLY_SCOUT_CONFIG nodes=2 ranks=16 ppn=8 '
printf 'sizes=4096,8192 app_runs=10 app_warmup=2 orders=AB,BA,BA,AB\n'
for replicate in 1 2 3 4; do
    case $replicate in
        1|4) order=(baseline guarded) ;;
        2|3) order=(guarded baseline) ;;
    esac
    for size in "${sizes[@]}"; do
        for arm in "${order[@]}"; do
            run_arm "$replicate" "$size" "$arm"
        done
    done
done
printf 'GUARDED_EARLY_SCOUT_DONE pairs=8 runs=16\n'
