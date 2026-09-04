#!/usr/bin/env bash
# Submit and audit one post-archive N6 pdebug full-catalog control job.
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 N6_BUNDLE REQUEST_DIR ARCHIVE_DIR CONFIRMATION OUTPUT_DIR" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
bundle_dir=$(cd -- "$1" && pwd)
request_dir=$(cd -- "$2" && pwd)
archive_dir=$(cd -- "$3" && pwd)
confirmation=$(cd -- "$(dirname -- "$4")" && pwd)/$(basename -- "$4")
case $5 in /*) output_dir=$5 ;; *) output_dir="$repo_root/$5" ;; esac
graph="$bundle_dir/discovery/graph.json"
runner="$script_dir/run_compiler_collective_n6_llm_controls.sh"
controller="$script_dir/continue_compiler_collective_n6_llm_controls.sh"
monitor="$script_dir/monitor_compiler_collective_replicate.py"
adapter="$script_dir/prepare_collective_n6_llm_policy_screen.py"
state_path="$output_dir.state"
events_path="$output_dir.events"
lock_path="$output_dir.lock"
arms=(
    baseline_auto
    flat_double_tree
    flat_double_tree_pipe4
    flat_double_tree_pipe8
    hierarchical_double_tree
    hierarchical_double_tree_pipe4
    hierarchical_double_tree_pipe8
    locality_ring
)

mkdir -p "$(dirname -- "$output_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another N6 LLM-control campaign owns this output" >&2
    exit 2
fi

set_state() {
    local state=$1 detail=${2:-} temporary="$state_path.tmp.$$"
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

set_state preflight "provider=disabled scheduler=not_invoked"
python3 "$adapter" preflight \
    --graph "$graph" --bundle "$bundle_dir" \
    --request-dir "$request_dir" --archive-dir "$archive_dir" \
    --confirmation "$confirmation" --repo-root "$repo_root"

if [[ -e $output_dir ]]; then
    echo "refusing existing N6 LLM-control output: $output_dir" >&2
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

while true; do
    active_jobs=$(flux jobs --filter=active -n -o '{id.f58} {name}')
    if [[ -z $active_jobs ]]; then
        break
    fi
    set_state waiting_scheduler_idle "active_user_jobs=$(printf '%s\n' "$active_jobs" | awk 'NF {n++} END {print n+0}')"
    sleep 5
done

set_state submitting "one six-node pdebug allocation; all arms sequential"
job_id=$(flux batch -q pdebug -N6 -n48 -c8 -g1 -t 55m -u \
    --job-name=coll-n6-llm-controls --cwd="$output_dir" \
    "$runner" "$bundle_dir" "$output_dir")
if [[ -z $job_id || $job_id == *$'\n'* ]]; then
    echo "Flux returned an invalid N6 control job ID: $job_id" >&2
    exit 2
fi
printf '%s\n' "$job_id" >"$output_dir/job-id"

artifacts=(
    "$bundle_dir/FROZEN_V3_MANIFEST.json"
    "$bundle_dir/discovery/graph.json"
    "$bundle_dir/controls/manifest.json"
    "$runner" "$controller" "$monitor" "$adapter"
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
    artifact_args+=(--artifact "$artifact=${digest%% *}")
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
        --expected-nodes 6
        --expected-ranks 48
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

set_state analyzing "$output_dir/policy-screen.json"
emit_args=(
    emit
    --graph "$graph"
    --bundle "$bundle_dir"
    --request-dir "$request_dir"
    --archive-dir "$archive_dir"
    --confirmation "$confirmation"
    --repo-root "$repo_root"
    --out "$output_dir/policy-screen.json"
)
for status in "${monitor_paths[@]}"; do
    emit_args+=(--monitor "$status")
done
python3 "$adapter" "${emit_args[@]}"
set_state complete "$output_dir/policy-screen.json"
trap - EXIT
