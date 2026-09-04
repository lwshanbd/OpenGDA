#!/usr/bin/env bash
# Run one preregistered N6 confirmation block in one independent allocation.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 BINARY_DIR OUTPUT_DIR REPLICATE" >&2
    exit 2
fi

binary_dir=$1
output_dir=$2
replicate=$3
sizes=1024,4096,8192,65536,262144,1048576,4194304,8388608,16777216

case "$replicate" in
    1)
        arms=(
            derived_bin_policy
            scout_best_uniform
            frozen_structural_heuristic
        )
        ;;
    2)
        arms=(
            scout_best_uniform
            frozen_structural_heuristic
            derived_bin_policy
        )
        ;;
    3)
        arms=(
            frozen_structural_heuristic
            derived_bin_policy
            scout_best_uniform
        )
        ;;
    *)
        echo "replicate must be 1, 2, or 3" >&2
        exit 2
        ;;
esac

rep_dir="$output_dir/rep$replicate"
exec >"$rep_dir/driver.out" 2>"$rep_dir/driver.err"
printf 'COLLECTIVE_N6_CONFIRM_CONFIG replicate=%s arms=%s\n' \
    "$replicate" "${arms[*]}"
for name in "${arms[@]}"; do
    binary="$binary_dir/$name/compiler_collective_eval"
    if [[ ! -x "$binary" ]]; then
        echo "missing compiler-control binary: $binary" >&2
        exit 2
    fi
    printf 'COLLECTIVE_N6_CONFIRM_ARM_START replicate=%s arm=%s\n' \
        "$replicate" "$name"
    env GICC_COLL_SIZES="$sizes" GICC_COLLECTIVE_PLAN_LABEL="$name" \
        HSA_ENABLE_IPC_MODE_LEGACY=1 FI_MR_CACHE_MONITOR=kdreg2 \
        flux run -N6 -n48 -c8 -g1 -t 7m -u "$binary" 7 2 \
        >>"$rep_dir/$name.out" 2>>"$rep_dir/$name.err"
    printf 'COLLECTIVE_N6_CONFIRM_ARM_DONE replicate=%s arm=%s\n' \
        "$replicate" "$name"
done
printf 'COLLECTIVE_N6_CONFIRM_DONE replicate=%s\n' "$replicate"
