#!/usr/bin/env bash
# Run three rotated hierarchy-pipeline blocks in one eight-node allocation.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 N8_BUNDLE_DIR OUTPUT_DIR" >&2
    exit 2
fi

bundle_dir=$1
output_dir=$2
sizes=1024,4096,8192,65536,262144,1048576,4194304,8388608,16777216

run_block() {
    local replicate=$1
    shift
    local rep_dir="$output_dir/rep$replicate"
    (
        exec >"$rep_dir/driver.out" 2>"$rep_dir/driver.err"
        printf 'HIERPIPE_N8_CONFIG replicate=%s arms=%s\n' \
            "$replicate" "$*"
        for name in "$@"; do
            if [[ ! "$name" =~ ^[a-z0-9_]+$ ]]; then
                echo "invalid compiler-control name: $name" >&2
                exit 2
            fi
            local binary="$bundle_dir/binaries/$name/compiler_collective_eval"
            if [[ ! -x "$binary" ]]; then
                echo "missing compiler-control binary: $binary" >&2
                exit 2
            fi
            printf 'HIERPIPE_N8_ARM_START replicate=%s arm=%s\n' \
                "$replicate" "$name"
            env GICC_COLL_SIZES="$sizes" GICC_COLLECTIVE_PLAN_LABEL="$name" \
                HSA_ENABLE_IPC_MODE_LEGACY=1 FI_MR_CACHE_MONITOR=kdreg2 \
                flux run -N8 -n64 -c8 -g1 -t 7m -u "$binary" 7 2 \
                >>"$rep_dir/$name.out" 2>>"$rep_dir/$name.err"
            printf 'HIERPIPE_N8_ARM_DONE replicate=%s arm=%s\n' \
                "$replicate" "$name"
        done
        printf 'HIERPIPE_N8_DONE replicate=%s\n' "$replicate"
    )
}

run_block 1 \
    hierarchical_double_tree \
    hierarchical_double_tree_pipe4 \
    hierarchical_double_tree_pipe8
run_block 2 \
    hierarchical_double_tree_pipe4 \
    hierarchical_double_tree_pipe8 \
    hierarchical_double_tree
run_block 3 \
    hierarchical_double_tree_pipe8 \
    hierarchical_double_tree \
    hierarchical_double_tree_pipe4
