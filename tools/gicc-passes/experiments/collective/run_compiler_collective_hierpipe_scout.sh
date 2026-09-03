#!/usr/bin/env bash
# Run three topology-matched compiler catalog arms in one 4-node allocation.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 V4_BUNDLE_DIR OUTPUT_DIR" >&2
    exit 2
fi

bundle_dir=$1
output_dir=$2
sizes=1024,4096,8192,65536,262144,1048576,4194304,8388608,16777216
arms=(
    hierarchical_double_tree
    hierarchical_double_tree_pipe4
    hierarchical_double_tree_pipe8
)

exec >"$output_dir/driver.out" 2>"$output_dir/driver.err"
printf 'HIERPIPE_CONFIG nodes=4 ranks=32 ppn=8 runs=3 warmup=1 arms=%s\n' \
    "${arms[*]}"
for name in "${arms[@]}"; do
    binary="$bundle_dir/binaries/$name/compiler_collective_eval"
    if [[ ! -x "$binary" ]]; then
        echo "missing compiler-control binary: $binary" >&2
        exit 2
    fi
    printf 'HIERPIPE_ARM_START arm=%s\n' "$name"
    env GICC_COLL_SIZES="$sizes" GICC_COLLECTIVE_PLAN_LABEL="$name" \
        HSA_ENABLE_IPC_MODE_LEGACY=1 FI_MR_CACHE_MONITOR=kdreg2 \
        flux run -N4 -n32 -c8 -g1 -t 7m -u "$binary" 3 1 \
        >>"$output_dir/$name.out" 2>>"$output_dir/$name.err"
    printf 'HIERPIPE_ARM_DONE arm=%s\n' "$name"
done
printf 'HIERPIPE_DONE\n'
