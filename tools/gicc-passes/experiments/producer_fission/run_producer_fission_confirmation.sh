#!/usr/bin/env bash
# Run one independent, internally order-balanced Jacobi confirmation allocation.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 BASELINE_BINARY FISSION_BINARY OUTPUT_DIR ALLOCATION" >&2
    exit 2
fi

baseline=$1
fission=$2
output_dir=$3
allocation=$4
if [[ ! $allocation =~ ^[123]$ ]]; then
    echo "allocation must be 1, 2, or 3" >&2
    exit 2
fi
for binary in "$baseline" "$fission"; do
    if [[ ! -x $binary ]]; then
        echo "missing executable: $binary" >&2
        exit 2
    fi
done

allocation_dir="$output_dir/allocation$allocation"
exec >"$allocation_dir/driver.out" 2>"$allocation_dir/driver.err"
printf 'PFISSION_CONFIRM_CONFIG allocation=%s nodes=2 ranks=16 ppn=8 cores=8 sizes=1024,4096 iterations=200 nccheck=10 blocks=AB,BA\n' \
    "$allocation"

run_arm() {
    local block=$1
    local size=$2
    local arm=$3
    local binary
    case $arm in
        baseline) binary=$baseline ;;
        fission) binary=$fission ;;
        *) echo "invalid arm: $arm" >&2; exit 2 ;;
    esac
    local run_dir="$allocation_dir/block${block}/size${size}"
    printf 'PFISSION_CONFIRM_RUN_START allocation=%s block=%s size=%s arm=%s\n' \
        "$allocation" "$block" "$size" "$arm"
    env HSA_ENABLE_IPC_MODE_LEGACY=1 FI_MR_CACHE_MONITOR=kdreg2 \
        flux run -N2 -n16 -c8 -g1 -t 5m -u \
        "$binary" -nx "$size" -ny "$size" -niter 200 -nccheck 10 \
        >"$run_dir/$arm.out" 2>"$run_dir/$arm.err"
    printf 'PFISSION_CONFIRM_RUN_DONE allocation=%s block=%s size=%s arm=%s\n' \
        "$allocation" "$block" "$size" "$arm"
}

for block in 1 2; do
    case $block in
        1) order=(baseline fission) ;;
        2) order=(fission baseline) ;;
    esac
    printf 'PFISSION_CONFIRM_BLOCK_START allocation=%s block=%s order=%s\n' \
        "$allocation" "$block" "${order[*]}"
    for size in 1024 4096; do
        for arm in "${order[@]}"; do
            run_arm "$block" "$size" "$arm"
        done
    done
    printf 'PFISSION_CONFIRM_BLOCK_DONE allocation=%s block=%s\n' \
        "$allocation" "$block"
done
printf 'PFISSION_CONFIRM_DONE allocation=%s pairs=4 runs=8\n' "$allocation"
