#!/usr/bin/env bash
# Continue the pending compiler-headroom campaign without invoking any model.
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
collective="$script_dir/collective"
producer="$script_dir/producer_fission"

n8_bundle="$repo_root/build_ofi/compiler_collective_capacity_n8_hierpipe_v1_20260903"
n8_scout="$repo_root/build_ofi/compiler_collective_hierpipe_n8_scout_20260903"
n8_scout_state="$n8_scout.state"
n8_heuristic="$repo_root/build_ofi/compiler_decision_suite_20260904/controls/collective-n8-heuristic-decision.json"
n8_transition="$repo_root/build_ofi/compiler_collective_n8_confirmation_transition_20260904"
n8_confirmation="$repo_root/build_ofi/compiler_collective_n8_confirmation_20260904"

producer_binary="$repo_root/build_ofi/producer_fission_oracle_90b9123"
producer_scout="$repo_root/build_ofi/producer_fission_oracle_scout_90b9123_20260904"
producer_scout_state="$producer_scout.state"
coverage_report="$repo_root/build_ofi/compiler_fact_coverage_20260904/coverage-report.json"
producer_transition="$repo_root/build_ofi/producer_fission_confirmation_transition_20260904.json"
producer_confirmation="$repo_root/build_ofi/producer_fission_confirmation_20260904"

campaign="$repo_root/build_ofi/compiler_headroom_confirmations_20260904"
state_path="$campaign.state"
events_path="$campaign.events"
lock_path="$campaign.lock"

exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another compiler-headroom successor is active" >&2
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
        set_state failed "successor_exit=$status"
    fi
}
trap on_exit EXIT

terminal_phase() {
    local path=$1
    local phase
    if [[ -f $path ]]; then
        phase=$(awk 'NR == 1 {print $2}' "$path")
        case $phase in
            promising|negative|failed|confirmed) printf '%s\n' "$phase" ;;
        esac
    fi
    return 0
}

set_state waiting_producer_scout "$producer_scout_state"
producer_phase=
while [[ -z $producer_phase ]]; do
    producer_phase=$(terminal_phase "$producer_scout_state")
    if [[ -z $producer_phase ]]; then
        sleep 5
    fi
done

# The N8 scout is the producer scout's predecessor, so it is terminal before
# this point.  Jacobi runs first because its N2 confirmation can make progress
# even if pdebug still cannot provide a full N8 allocation.
if [[ $producer_phase == promising ]]; then
    set_state preparing_producer_confirmation "$producer_transition"
    if [[ -e $producer_transition ]]; then
        python3 "$producer/prepare_producer_fission_confirmation.py" \
            verify-contained --report "$producer_transition"
    else
        python3 "$producer/prepare_producer_fission_confirmation.py" prepare \
            --monitor "$producer_scout/monitor.json" \
            --analysis "$producer_scout/analysis.json" \
            --coverage-report "$coverage_report" \
            --binary-dir "$producer_binary" \
            --out "$producer_transition"
    fi
    set_state running_producer_confirmation "$producer_confirmation"
    bash "$producer/continue_producer_fission_confirmation.sh" \
        "$producer_transition" "$producer_binary" "$producer_confirmation"
else
    set_state skipped_producer_confirmation "scout_state=$producer_phase"
fi

n8_phase=$(terminal_phase "$n8_scout_state")
if [[ -z $n8_phase ]]; then
    echo "N8 predecessor state is not terminal: $n8_scout_state" >&2
    exit 2
fi
if [[ $n8_phase == promising ]]; then
    set_state preparing_n8_confirmation "$n8_transition"
    if [[ -e $n8_transition ]]; then
        python3 "$collective/prepare_compiler_collective_n8_confirmation.py" \
            verify \
            --graph "$n8_bundle/discovery/graph.json" \
            --scout "$n8_scout/analysis.json" \
            --heuristic-decision "$n8_heuristic" \
            --output-dir "$n8_transition"
    else
        python3 "$collective/prepare_compiler_collective_n8_confirmation.py" \
            prepare \
            --graph "$n8_bundle/discovery/graph.json" \
            --scout "$n8_scout/analysis.json" \
            --heuristic-decision "$n8_heuristic" \
            --output-dir "$n8_transition"
    fi
    n8_transition_phase=$(python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["status"])' \
        "$n8_transition/transition.json")
    if [[ $n8_transition_phase == confirmation_plan_ready ]]; then
        set_state running_n8_confirmation "$n8_confirmation"
        bash "$collective/continue_compiler_collective_n8_confirmation.sh" \
            "$n8_bundle" "$n8_scout" "$n8_heuristic" \
            "$n8_transition" "$n8_confirmation"
    else
        set_state skipped_n8_confirmation \
            "transition_state=$n8_transition_phase"
    fi
else
    set_state skipped_n8_confirmation "scout_state=$n8_phase"
fi

set_state complete "all eligible compiler-only confirmations consumed"
trap - EXIT
