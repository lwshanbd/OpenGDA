#!/usr/bin/env bash
# Submit one pdebug allocation, monitor three rotated blocks, and analyze.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 V4_BUNDLE_DIR SCOUT_DIR OUTPUT_DIR" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
bundle_dir=$(cd -- "$1" && pwd)
scout_dir=$(cd -- "$2" && pwd)
if [[ $3 = /* ]]; then
    output_dir=$3
else
    output_dir="$repo_root/$3"
fi
runner="$script_dir/run_compiler_collective_hierpipe_confirm.sh"
monitor="$script_dir/monitor_compiler_collective_replicate.py"
analyzer="$script_dir/analyze_compiler_collective_hierpipe_confirm.py"
evaluator="$script_dir/compiler_collective_eval.py"
protocol="$script_dir/PROTOCOL_DRAFT.md"
controller="$script_dir/continue_compiler_collective_hierpipe_confirm.sh"
scout_analysis="$scout_dir/analysis.json"
scout_monitor="$scout_dir/monitor.json"
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
    echo "another hierarchy-pipeline confirmation owns this output" >&2
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
scout_passed=$(python3 -c \
    'import json,sys; x=json.load(open(sys.argv[1])); print("yes" if x["hierarchical_pipeline_gate"]["passed"] else "no")' \
    "$scout_analysis")
if [[ $scout_passed != yes || ! -f $scout_monitor ]]; then
    echo "confirmation requires the complete passed hierarchy-pipeline scout" >&2
    exit 2
fi
if [[ -e "$output_dir" ]]; then
    echo "refusing existing hierarchy-pipeline confirmation: $output_dir" >&2
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

set_state submitting "one pdebug allocation containing three rotated blocks"
job_id=$(flux batch -q pdebug -N4 -n32 -c8 -g1 -t 30m -u \
    --job-name=coll-n4-hier-confirm --cwd="$output_dir" \
    "$runner" "$bundle_dir" "$output_dir")
if [[ -z "$job_id" || "$job_id" == *$'\n'* ]]; then
    echo "Flux returned an invalid confirmation job ID: $job_id" >&2
    exit 2
fi
printf '%s\n' "$job_id" >"$output_dir/job-id"

artifacts=(
    "$bundle_dir/FROZEN_V3_MANIFEST.json"
    "$bundle_dir/discovery/graph.json"
    "$bundle_dir/inputs/platform.json"
    "$scout_analysis"
    "$scout_monitor"
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
        --expected-nodes 4
        --expected-ranks 32
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
analysis_args=(--scout-analysis "$scout_analysis" --out "$output_dir/analysis.json")
for status in "${monitor_paths[@]}"; do
    analysis_args+=(--monitor "$status")
done
python3 "$analyzer" "${analysis_args[@]}"
confirmed=$(python3 -c \
    'import json,sys; print("yes" if json.load(open(sys.argv[1]))["confirmation_gate"]["passed"] else "no")' \
    "$output_dir/analysis.json")
if [[ $confirmed == yes ]]; then
    set_state confirmed "stable nonuniform compiler headroom"
else
    set_state negative "scout headroom did not survive rotated confirmation"
fi
trap - EXIT
