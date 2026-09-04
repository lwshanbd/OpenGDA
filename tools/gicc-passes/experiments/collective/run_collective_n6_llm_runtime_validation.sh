#!/usr/bin/env bash
# Run one rotated block of materialized compiler policies on an N6 allocation.
set -euo pipefail

if [[ $# -lt 4 ]]; then
    echo "usage: $0 PLAN_DIR OUTPUT_DIR REPLICATE POLICY..." >&2
    exit 2
fi

plan_dir=$1
output_dir=$2
replicate=$3
shift 3
sizes=1024,4096,8192,65536,262144,1048576,4194304,8388608,16777216
if [[ ! $replicate =~ ^[123]$ ]]; then
    echo "replicate must be 1, 2, or 3" >&2
    exit 2
fi

exec >"$output_dir/driver.out" 2>"$output_dir/driver.err"
printf 'COLLECTIVE_N6_LLM_RUNTIME_CONFIG replicate=%s policies=%s\n' \
    "$replicate" "$*"
for name in "$@"; do
    if [[ ! $name =~ ^policy[0-9][0-9]$ ]]; then
        echo "invalid runtime policy name: $name" >&2
        exit 2
    fi
    binary="$plan_dir/policies/$name/build/compiler_collective_eval"
    if [[ ! -x $binary ]]; then
        echo "missing materialized runtime binary: $binary" >&2
        exit 2
    fi
    printf 'COLLECTIVE_N6_LLM_RUNTIME_START replicate=%s policy=%s\n' \
        "$replicate" "$name"
    env GICC_COLL_SIZES="$sizes" GICC_COLLECTIVE_PLAN_LABEL="$name" \
        HSA_ENABLE_IPC_MODE_LEGACY=1 FI_MR_CACHE_MONITOR=kdreg2 \
        flux run -N6 -n48 -c8 -g1 -t 7m -u "$binary" 7 2 \
        >>"$output_dir/$name.out" 2>>"$output_dir/$name.err"
    printf 'COLLECTIVE_N6_LLM_RUNTIME_DONE replicate=%s policy=%s\n' \
        "$replicate" "$name"
done
printf 'COLLECTIVE_N6_LLM_RUNTIME_COMPLETE replicate=%s\n' "$replicate"
