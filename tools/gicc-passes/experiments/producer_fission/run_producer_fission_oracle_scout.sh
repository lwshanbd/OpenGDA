#!/usr/bin/env bash
# Run a balanced fused-vs-fission Jacobi scout inside one Flux allocation.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 BASELINE_BINARY FISSION_BINARY OUTPUT_DIR" >&2
    exit 2
fi

baseline=$1
fission=$2
output_dir=$3
for binary in "$baseline" "$fission"; do
    if [[ ! -x $binary ]]; then
        echo "missing executable: $binary" >&2
        exit 2
    fi
done

sizes=(1024 4096)

run_arm() {
    local replicate=$1
    local size=$2
    local arm=$3
    local binary
    case $arm in
        baseline) binary=$baseline ;;
        fission)  binary=$fission ;;
        *) echo "invalid arm: $arm" >&2; exit 2 ;;
    esac
    local run_dir="$output_dir/rep${replicate}/size${size}"
    mkdir -p "$run_dir"
    printf 'PFISSION_RUN_START replicate=%s size=%s arm=%s\n' \
        "$replicate" "$size" "$arm"
    env HSA_ENABLE_IPC_MODE_LEGACY=1 FI_MR_CACHE_MONITOR=kdreg2 \
        flux run -N2 -n16 -c8 -g1 -t 5m -u \
        "$binary" -nx "$size" -ny "$size" -niter 200 -nccheck 10 \
        >"$run_dir/$arm.out" 2>"$run_dir/$arm.err"
    printf 'PFISSION_RUN_DONE replicate=%s size=%s arm=%s\n' \
        "$replicate" "$size" "$arm"
}

printf 'PFISSION_SCOUT_CONFIG nodes=2 ranks=16 ppn=8 cores=8 '
printf 'sizes=1024,4096 iterations=200 nccheck=10 orders=AB,BA,BA,AB\n'
for replicate in 1 2 3 4; do
    case $replicate in
        1|4) order=(baseline fission) ;;
        2|3) order=(fission baseline) ;;
    esac
    for size in "${sizes[@]}"; do
        for arm in "${order[@]}"; do
            run_arm "$replicate" "$size" "$arm"
        done
    done
done
printf 'PFISSION_SCOUT_DONE pairs=8 runs=16\n'
