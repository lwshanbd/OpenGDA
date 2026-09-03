#!/usr/bin/env bash
# Submit, monitor, and analyze exactly one exploratory 4-node pdebug scout.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 V3_BUNDLE_DIR OUTPUT_DIR" >&2
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
runner="$script_dir/run_compiler_collective_topology_scout.sh"
monitor="$script_dir/monitor_compiler_collective_replicate.py"
analyzer="$script_dir/analyze_compiler_collective_topology_scout.py"
state_path="$output_dir.state"
events_path="$output_dir.events"
lock_path="$output_dir.lock"

mkdir -p "$(dirname -- "$output_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another topology scout controller owns this output" >&2
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

if [[ -e "$output_dir" ]]; then
    echo "refusing existing topology scout output: $output_dir" >&2
    exit 2
fi
mkdir -p "$output_dir"
for name in baseline_auto hierarchical_double_tree; do
    : >"$output_dir/$name.out"
    : >"$output_dir/$name.err"
done

set_state submitting "one exploratory 4-node pdebug allocation"
job_id=$(flux batch -q pdebug -N4 -n32 -c8 -g1 -t 15m -u \
    --job-name=coll-v4-scout --cwd="$output_dir" \
    "$runner" "$bundle_dir" "$output_dir")
if [[ -z "$job_id" || "$job_id" == *$'\n'* ]]; then
    echo "Flux returned an invalid topology scout job ID: $job_id" >&2
    exit 2
fi
printf '%s\n' "$job_id" >"$output_dir/job-id"

args=(
    --job-id "$job_id"
    --driver-stdout "$output_dir/driver.out"
    --driver-stderr "$output_dir/driver.err"
    --status "$output_dir/monitor.json"
    --replicate 1
    --expected-nodes 4
    --expected-ranks 32
    --expected-ppn 8
    --expected-runs 3
    --expected-warmup 1
)
for size in 1024 4096 8192 65536 262144 1048576 4194304 8388608 16777216; do
    args+=(--expected-size "$size")
done
for name in baseline_auto hierarchical_double_tree; do
    args+=(
        --benchmark "$name=$output_dir/$name.out"
        --benchmark-stderr "$name=$output_dir/$name.err"
    )
done
artifacts=(
    "$bundle_dir/FROZEN_V3_MANIFEST.json"
    "$runner"
    "$monitor"
    "$analyzer"
)
for name in baseline_auto hierarchical_double_tree; do
    artifacts+=(
        "$bundle_dir/binaries/$name/compiler_collective_eval"
        "$bundle_dir/binaries/$name/build-provenance.json"
        "$bundle_dir/controls/$name-hint.json"
    )
done
for artifact in "${artifacts[@]}"; do
    digest=$(sha256sum "$artifact")
    digest=${digest%% *}
    args+=(--artifact "$artifact=$digest")
done

set_state monitoring "$job_id"
python3 "$monitor" "${args[@]}"
set_state analyzing "$output_dir/monitor.json"
python3 "$analyzer" --monitor "$output_dir/monitor.json" \
    --out "$output_dir/analysis.json"
promising=$(python3 -c \
    'import json,sys; print("yes" if json.load(open(sys.argv[1]))["v4_topology_hypothesis"]["promising"] else "no")' \
    "$output_dir/analysis.json")
if [[ $promising == yes ]]; then
    set_state promising "topology-matched v4 protocol warranted"
else
    set_state negative "do not build a 4-node v4 graph"
fi
trap - EXIT
