#!/usr/bin/env bash
# Run one confirmatory replicate block inside a Flux batch allocation. The
# batch starts one child Flux instance; this script launches all already-built
# compiler controls sequentially on its exact same two nodes.
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 BUNDLE_DIR OUTPUT_DIR REPLICATE ARM [ARM ...]" >&2
    exit 2
fi

bundle_dir=$1
output_dir=$2
replicate=$3
shift 3
sizes=1024,4096,8192,65536,262144,1048576,4194304,8388608,16777216

if [[ ! "$replicate" =~ ^[1-9][0-9]*$ ]]; then
    echo "invalid replicate: $replicate" >&2
    exit 2
fi
exec >"$output_dir/driver.out" 2>"$output_dir/driver.err"
printf 'REPLICATE_CONFIG replicate=%s arms=%s\n' "$replicate" "$*"

for name in "$@"; do
    if [[ ! "$name" =~ ^[a-z0-9_]+$ ]]; then
        echo "invalid compiler-control name: $name" >&2
        exit 2
    fi
    binary="$bundle_dir/binaries/$name/compiler_collective_eval"
    if [[ ! -x "$binary" ]]; then
        echo "missing compiler-control binary: $binary" >&2
        exit 2
    fi
    printf 'REPLICATE_ARM_START replicate=%s arm=%s\n' \
        "$replicate" "$name"
    env GICC_COLL_SIZES="$sizes" GICC_COLLECTIVE_PLAN_LABEL="$name" \
        HSA_ENABLE_IPC_MODE_LEGACY=1 FI_MR_CACHE_MONITOR=kdreg2 \
        flux run -N2 -n16 -c8 -g1 -t 5m -u "$binary" 7 2 \
        >>"$output_dir/$name.out" 2>>"$output_dir/$name.err"
    printf 'REPLICATE_ARM_DONE replicate=%s arm=%s\n' \
        "$replicate" "$name"
done

printf 'REPLICATE_DONE replicate=%s\n' "$replicate"
