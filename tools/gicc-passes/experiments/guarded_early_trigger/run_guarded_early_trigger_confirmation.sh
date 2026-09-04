#!/usr/bin/env bash
# Run one independent, internally order-balanced mm_minimal confirmation.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 BASELINE_BINARY GUARDED_BINARY OUTPUT_DIR ALLOCATION" >&2
    exit 2
fi

baseline=$1
guarded=$2
output_dir=$3
allocation=$4
if [[ ! $allocation =~ ^[123]$ ]]; then
    echo "allocation must be 1, 2, or 3" >&2
    exit 2
fi
for binary in "$baseline" "$guarded"; do
    if [[ ! -x $binary ]]; then
        echo "missing executable: $binary" >&2
        exit 2
    fi
done

allocation_dir="$output_dir/allocation$allocation"
exec >"$allocation_dir/driver.out" 2>"$allocation_dir/driver.err"
printf 'GUARDED_EARLY_CONFIRM_CONFIG allocation=%s nodes=2 ranks=16 ppn=8 cores=8 sizes=4096,8192 app_runs=10 app_warmup=2 blocks=AB,BA\n' \
    "$allocation"

run_arm() {
    local block=$1
    local size=$2
    local arm=$3
    local binary
    case $arm in
        baseline) binary=$baseline ;;
        guarded) binary=$guarded ;;
        *) echo "invalid arm: $arm" >&2; exit 2 ;;
    esac
    local run_dir="$allocation_dir/block${block}/size${size}"
    printf 'GUARDED_EARLY_CONFIRM_RUN_START allocation=%s block=%s size=%s arm=%s\n' \
        "$allocation" "$block" "$size" "$arm"
    env -u GICC_SKIP_DWQ_INIT -u MPICH_GPU_SUPPORT_ENABLED \
        FI_MR_CACHE_MAX_COUNT=0 FI_MR_CACHE_MONITOR=kdreg2 \
        HSA_ENABLE_IPC_MODE_LEGACY=1 GICC_PROXY_ENABLED=1 \
        flux run -N2 -n16 -c8 -g1 -t 8m -u \
        "$binary" "$size" \
        >"$run_dir/$arm.out" 2>"$run_dir/$arm.err"
    printf 'GUARDED_EARLY_CONFIRM_RUN_DONE allocation=%s block=%s size=%s arm=%s\n' \
        "$allocation" "$block" "$size" "$arm"
}

for block in 1 2; do
    case $block in
        1) order=(baseline guarded) ;;
        2) order=(guarded baseline) ;;
    esac
    printf 'GUARDED_EARLY_CONFIRM_BLOCK_START allocation=%s block=%s order=%s\n' \
        "$allocation" "$block" "${order[*]}"
    for size in 4096 8192; do
        for arm in "${order[@]}"; do
            run_arm "$block" "$size" "$arm"
        done
    done
    printf 'GUARDED_EARLY_CONFIRM_BLOCK_DONE allocation=%s block=%s\n' \
        "$allocation" "$block"
done
printf 'GUARDED_EARLY_CONFIRM_DONE allocation=%s pairs=4 runs=8\n' "$allocation"
