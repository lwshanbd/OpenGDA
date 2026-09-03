#!/usr/bin/env bash
# Submit and autonomously analyze exactly one eight-node pdebug scout.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 N8_BUNDLE_DIR OUTPUT_DIR" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
bundle_dir=$(cd -- "$1" && pwd)
if [[ $2 = /* ]]; then
    output_dir=$2
else
    output_dir="$repo_root/$2"
fi
runner="$script_dir/run_compiler_collective_hierpipe_n8_scout.sh"
controller="$script_dir/continue_compiler_collective_hierpipe_n8_scout.sh"
monitor="$script_dir/monitor_compiler_collective_replicate.py"
analyzer="$script_dir/analyze_compiler_collective_hierpipe_n8_scout.py"
evaluator="$script_dir/compiler_collective_eval.py"
protocol="$script_dir/HIERPIPE_N8_PROTOCOL.md"
state_path="$output_dir.state"
events_path="$output_dir.events"
lock_path="$output_dir.lock"
arms=(
    hierarchical_double_tree
    hierarchical_double_tree_pipe4
    hierarchical_double_tree_pipe8
)

mkdir -p "$(dirname -- "$output_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another n8 hierarchy-pipeline scout owns this output" >&2
    exit 2
fi

set_state() {
    local state=$1
    local detail=${2:-}
    local temporary="$state_path.tmp.$$"
    printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$state" "$detail" >"$temporary"
    mv -- "$temporary" "$state_path"
    printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$state" "$detail" >>"$events_path"
}

on_exit() {
    local status=$?
    if [[ $status -ne 0 ]]; then
        set_state failed "controller_exit=$status"
    fi
}
trap on_exit EXIT

python3 "$evaluator" verify-offline-freeze \
    --manifest "$bundle_dir/FROZEN_V3_MANIFEST.json" \
    --repo-root "$repo_root"
python3 -c \
    'import json,sys; x=json.load(open(sys.argv[1])); assert x["manifest_id"] == "sha256:4ed11ecb6a6049b07e18ccf4e88cc7f54f7c1247c119ed1d37a133d6b017f361"; assert x["graph"]["graph_id"] == "sha256:ce569e2575cfdc924004631dcf96a2d81077520c3198c8f78edde3a4405cf806"' \
    "$bundle_dir/FROZEN_V3_MANIFEST.json"
if [[ -e "$output_dir" ]]; then
    echo "refusing existing n8 hierarchy-pipeline output: $output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir"
for replicate in 1 2 3; do
    rep_dir="$output_dir/rep$replicate"
    mkdir -p "$rep_dir"
    for name in "${arms[@]}"; do
        : >"$rep_dir/$name.out"
        : >"$rep_dir/$name.err"
    done
done

set_state submitting "one eight-node pdebug allocation with three rotated blocks"
job_id=$(flux batch -q pdebug -N8 -n64 -c8 -g1 -t 20m -u \
    --job-name=coll-n8-hier-scout --cwd="$output_dir" \
    "$runner" "$bundle_dir" "$output_dir")
if [[ -z "$job_id" || "$job_id" == *$'\n'* ]]; then
    echo "Flux returned an invalid n8 scout job ID: $job_id" >&2
    exit 2
fi
printf '%s\n' "$job_id" >"$output_dir/job-id"

artifacts=(
    "$bundle_dir/FROZEN_V3_MANIFEST.json"
    "$bundle_dir/discovery/graph.json"
    "$bundle_dir/inputs/platform.json"
    "$protocol"
    "$runner"
    "$controller"
    "$monitor"
    "$analyzer"
)
for name in "${arms[@]}"; do
    artifacts+=(
        "$bundle_dir/binaries/$name/compiler_collective_eval"
        "$bundle_dir/binaries/$name/build-provenance.json"
        "$bundle_dir/controls/$name-hint.json"
    )
done
artifact_args=()
for artifact in "${artifacts[@]}"; do
    digest=$(sha256sum "$artifact")
    digest=${digest%% *}
    artifact_args+=(--artifact "$artifact=$digest")
done

set_state monitoring "$job_id"
monitor_paths=()
for replicate in 1 2 3; do
    rep_dir="$output_dir/rep$replicate"
    status="$rep_dir/monitor.json"
    monitor_paths+=("$status")
    args=(
        --job-id "$job_id"
        --driver-stdout "$rep_dir/driver.out"
        --driver-stderr "$rep_dir/driver.err"
        --status "$status"
        --replicate "$replicate"
        --expected-nodes 8
        --expected-ranks 64
        --expected-ppn 8
        --expected-runs 7
        --expected-warmup 2
    )
    for size in 1024 4096 8192 65536 262144 1048576 4194304 8388608 16777216; do
        args+=(--expected-size "$size")
    done
    for name in "${arms[@]}"; do
        args+=(
            --benchmark "$name=$rep_dir/$name.out"
            --benchmark-stderr "$name=$rep_dir/$name.err"
        )
    done
    python3 "$monitor" "${args[@]}" "${artifact_args[@]}"
done

set_state analyzing "$output_dir/analysis.json"
analysis_args=(--out "$output_dir/analysis.json")
for status in "${monitor_paths[@]}"; do
    analysis_args+=(--monitor "$status")
done
python3 "$analyzer" "${analysis_args[@]}"
promising=$(python3 -c \
    'import json,sys; print("yes" if json.load(open(sys.argv[1]))["n8_capacity_gate"]["passed"] else "no")' \
    "$output_dir/analysis.json")
if [[ $promising == yes ]]; then
    set_state promising "n8 hierarchy pipeline has rotated compiler headroom"
else
    set_state negative "n8 hierarchy pipeline lacks model-worthy headroom"
fi
trap - EXIT
