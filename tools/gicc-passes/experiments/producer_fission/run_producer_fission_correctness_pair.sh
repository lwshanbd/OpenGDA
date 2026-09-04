#!/usr/bin/env bash
# Run one baseline/fission correctness pair serially inside one allocation.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 BASELINE FISSION OUTPUT_DIR" >&2
    exit 2
fi

baseline=$1
fission=$2
output_dir=$3
for path in "$baseline" "$fission"; do
    if [[ ! -x $path ]]; then
        echo "missing executable: $path" >&2
        exit 2
    fi
done
if [[ ! -d $output_dir ]]; then
    echo "missing output directory: $output_dir" >&2
    exit 2
fi

for arm in baseline fission; do
    case $arm in
        baseline) binary=$baseline ;;
        fission) binary=$fission ;;
    esac
    printf 'PRODUCER_FISSION_TRIAGE_START arm=%s\n' "$arm"
    env HSA_ENABLE_IPC_MODE_LEGACY=1 FI_MR_CACHE_MONITOR=kdreg2 \
        flux run -N2 -n16 -c8 -g1 -t 5m -u "$binary" \
        -nx 1024 -ny 1024 -niter 200 -nccheck 10 \
        >"$output_dir/$arm.out" 2>"$output_dir/$arm.err"
    printf 'PRODUCER_FISSION_TRIAGE_DONE arm=%s\n' "$arm"
done
