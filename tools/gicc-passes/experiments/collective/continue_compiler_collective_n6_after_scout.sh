#!/usr/bin/env bash
# Wait for the N6 scout and conditionally run its preregistered confirmation.
set -euo pipefail

if [[ $# -ne 6 ]]; then
    echo "usage: $0 SCOUT_STATE SCOUT_DIR BUNDLE_DIR HEURISTIC_DECISION TRANSITION_DIR OUTPUT_DIR" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
case $1 in /*) scout_state=$1 ;; *) scout_state="$repo_root/$1" ;; esac
case $2 in /*) scout_dir=$2 ;; *) scout_dir="$repo_root/$2" ;; esac
bundle_dir=$(cd -- "$3" && pwd)
heuristic_decision=$(cd -- "$(dirname -- "$4")" && pwd)/$(basename -- "$4")
case $5 in /*) transition_dir=$5 ;; *) transition_dir="$repo_root/$5" ;; esac
case $6 in /*) output_dir=$6 ;; *) output_dir="$repo_root/$6" ;; esac

preparer="$script_dir/prepare_compiler_collective_n6_confirmation.py"
preparer_base="$script_dir/prepare_compiler_collective_n8_confirmation.py"
protocol="$script_dir/HIERPIPE_N6_CONFIRMATION_TRANSITION.md"
runner="$script_dir/run_compiler_collective_n6_confirmation.sh"
controller="$script_dir/continue_compiler_collective_n6_confirmation.sh"
successor="$script_dir/continue_compiler_collective_n6_after_scout.sh"
monitor="$script_dir/monitor_compiler_collective_replicate.py"
analyzer="$script_dir/analyze_compiler_collective_n6_confirmation.py"
analyzer_base="$script_dir/analyze_compiler_collective_n8_confirmation.py"
build_script="$script_dir/build_compiler_collective_eval.sh"
evaluator="$script_dir/compiler_collective_eval.py"
graph="$bundle_dir/discovery/graph.json"
state_path="$output_dir.chain.state"
events_path="$output_dir.chain.events"
lock_path="$output_dir.chain.lock"

mkdir -p "$(dirname -- "$output_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another N6 confirmation successor is active" >&2
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
        set_state failed "successor_exit=$status"
    fi
}
trap on_exit EXIT

artifacts=(
    "$bundle_dir/FROZEN_V3_MANIFEST.json" "$graph" "$heuristic_decision"
    "$preparer" "$preparer_base" "$protocol" "$runner" "$controller"
    "$successor" "$monitor" "$analyzer" "$analyzer_base" "$build_script"
    "$evaluator"
)
hashes=()
for artifact in "${artifacts[@]}"; do
    if [[ ! -f $artifact ]]; then
        echo "missing frozen N6 confirmation artifact: $artifact" >&2
        exit 2
    fi
    digest=$(sha256sum "$artifact")
    hashes+=("${digest%% *}")
done
verify_artifacts() {
    local index actual
    for index in "${!artifacts[@]}"; do
        actual=$(sha256sum "${artifacts[$index]}")
        actual=${actual%% *}
        if [[ $actual != "${hashes[$index]}" ]]; then
            echo "N6 confirmation artifact changed: ${artifacts[$index]}" >&2
            exit 2
        fi
    done
}

for output in "$transition_dir" "$output_dir"; do
    if [[ -e $output ]]; then
        echo "refusing pre-existing N6 confirmation output: $output" >&2
        exit 2
    fi
done

set_state waiting_scout "$scout_state"
phase=
while [[ -z $phase ]]; do
    if [[ ! -f $scout_state ]]; then
        sleep 5
        continue
    fi
    phase=$(awk 'NR == 1 {print $2}' "$scout_state")
    case $phase in
        promising|negative|failed) ;;
        *) phase=; sleep 5 ;;
    esac
done
verify_artifacts
if [[ $phase != promising ]]; then
    set_state skipped "scout_state=$phase"
    trap - EXIT
    exit 0
fi

set_state preparing_transition "$transition_dir"
python3 "$preparer" prepare \
    --graph "$graph" \
    --scout "$scout_dir/analysis.json" \
    --heuristic-decision "$heuristic_decision" \
    --output-dir "$transition_dir"
verify_artifacts
transition_status=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
    "$transition_dir/transition.json")
if [[ $transition_status != confirmation_plan_ready ]]; then
    set_state skipped_no_incremental_policy \
        "transition_status=$transition_status"
    trap - EXIT
    exit 0
fi

while true; do
    active_jobs=$(flux jobs --filter=active -n -o '{id.f58} {name}')
    if [[ -z $active_jobs ]]; then
        break
    fi
    active_count=$(printf '%s\n' "$active_jobs" \
        | awk 'NF {count++} END {print count+0}')
    set_state waiting_scheduler_idle "active_user_jobs=$active_count"
    sleep 5
done
verify_artifacts

set_state running_confirmation "$output_dir"
bash "$controller" "$bundle_dir" "$scout_dir" "$heuristic_decision" \
    "$transition_dir" "$output_dir"
confirmation_phase=$(awk 'NR == 1 {print $2}' "$output_dir.state")
case $confirmation_phase in
    confirmed) set_state confirmed "$output_dir/analysis.json" ;;
    negative) set_state negative "$output_dir/analysis.json" ;;
    *) echo "unexpected N6 confirmation state: $confirmation_phase" >&2; exit 2 ;;
esac
trap - EXIT
