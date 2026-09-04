#!/usr/bin/env bash
# Recover fail-closed compiler-headroom scouts, strictly one pdebug job at a time.
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
producer="$script_dir/producer_fission"
guarded="$script_dir/guarded_early_trigger"
collective="$script_dir/collective"

reused_scout="$repo_root/build_ofi/reused_loop_descriptor_scout_aff76f9_20260904"
reused_chain_state="$repo_root/build_ofi/reused_loop_descriptor_confirmation_aff76f9_20260904.chain.state"

producer_binary="$repo_root/build_ofi/producer_fission_oracle_7687377"
producer_scout="$repo_root/build_ofi/producer_fission_oracle_scout_7687377_20260904_r2"
producer_transition="$repo_root/build_ofi/producer_fission_confirmation_transition_7687377_20260904_r2.json"
producer_confirmation="$repo_root/build_ofi/producer_fission_confirmation_7687377_20260904_r2"
producer_done_state="$repo_root/build_ofi/compiler_headroom_recovery_producer_done_20260904.state"
coverage_report="$repo_root/build_ofi/compiler_fact_coverage_20260904/coverage-report.json"

guarded_binary="$repo_root/build_ofi/guarded_early_trigger_oracle_77897d9"
guarded_scout="$repo_root/build_ofi/guarded_early_trigger_scout_77897d9_20260904_r2"
guarded_transition="$repo_root/build_ofi/guarded_early_trigger_confirmation_transition_77897d9_20260904_r2.json"
guarded_confirmation="$repo_root/build_ofi/guarded_early_trigger_confirmation_77897d9_20260904_r2"

n6_bundle="$repo_root/build_ofi/compiler_collective_capacity_n6_hierpipe_v1_20260904"
n6_scout="$repo_root/build_ofi/compiler_collective_hierpipe_n6_scout_20260904"

state_path="$repo_root/build_ofi/compiler_headroom_recovery_20260904.state"
events_path="$repo_root/build_ofi/compiler_headroom_recovery_20260904.events"
lock_path="$repo_root/build_ofi/compiler_headroom_recovery_20260904.lock"

exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another compiler-headroom recovery controller is active" >&2
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
        set_state failed "recovery_exit=$status"
    fi
}
trap on_exit EXIT

phase_of() {
    local path=$1
    if [[ -f $path ]]; then
        awk 'NR == 1 {print $2}' "$path"
    fi
}

wait_for_phase() {
    local path=$1
    shift
    local phase= wanted
    while true; do
        phase=$(phase_of "$path")
        for wanted in "$@"; do
            if [[ $phase == "$wanted" ]]; then
                printf '%s\n' "$phase"
                return 0
            fi
        done
        sleep 5
    done
}

wait_scheduler_idle() {
    local next_stage=$1 active_jobs active_count
    while true; do
        active_jobs=$(flux jobs --filter=active -n -o '{id.f58} {name}')
        if [[ -z $active_jobs ]]; then
            return 0
        fi
        active_count=$(printf '%s\n' "$active_jobs" \
            | awk 'NF {count++} END {print count+0}')
        set_state waiting_scheduler_idle \
            "next=$next_stage active_user_jobs=$active_count"
        sleep 5
    done
}

self_sha=$(sha256sum "$0")
self_sha=${self_sha%% *}
verify_self() {
    local actual
    actual=$(sha256sum "$0")
    actual=${actual%% *}
    if [[ $actual != "$self_sha" ]]; then
        echo "recovery controller changed while active" >&2
        exit 2
    fi
}

for required in \
    "$reused_scout/job-id" "$reused_chain_state" \
    "$producer_binary/baseline/jacobi" "$producer_binary/fission/jacobi" \
    "$guarded_binary/baseline/mm_minimal" "$guarded_binary/guarded/mm_minimal" \
    "$coverage_report" "$n6_bundle/FROZEN_V3_MANIFEST.json"; do
    if [[ ! -f $required ]]; then
        echo "missing recovery input: $required" >&2
        exit 2
    fi
done
for output in "$producer_scout" "$producer_transition" \
    "$producer_confirmation" "$producer_done_state" "$guarded_scout" \
    "$guarded_transition" "$guarded_confirmation" "$n6_scout"; do
    if [[ -e $output ]]; then
        echo "refusing pre-existing recovery output: $output" >&2
        exit 2
    fi
done

set_state waiting_reused_chain "$reused_chain_state"
reused_phase=$(wait_for_phase \
    "$reused_chain_state" confirmed negative skipped failed)
verify_self
wait_scheduler_idle producer_scout
verify_self

reused_job=$(sed -n '1p' "$reused_scout/job-id")
set_state running_producer_scout "$producer_scout"
bash "$producer/continue_producer_fission_oracle_scout.sh" \
    "$reused_job" "$producer_binary" "$producer_scout" \
    "$reused_scout.state"
producer_phase=$(wait_for_phase \
    "$producer_scout.state" promising negative failed)
verify_self

if [[ $producer_phase == promising ]]; then
    set_state preparing_producer_confirmation "$producer_transition"
    python3 "$producer/prepare_producer_fission_confirmation.py" prepare \
        --monitor "$producer_scout/monitor.json" \
        --analysis "$producer_scout/analysis.json" \
        --coverage-report "$coverage_report" \
        --binary-dir "$producer_binary" \
        --out "$producer_transition"
    wait_scheduler_idle producer_confirmation
    verify_self
    set_state running_producer_confirmation "$producer_confirmation"
    bash "$producer/continue_producer_fission_confirmation.sh" \
        "$producer_transition" "$producer_binary" "$producer_confirmation"
    producer_confirmation_phase=$(wait_for_phase \
        "$producer_confirmation.state" confirmed negative failed)
else
    producer_confirmation_phase=skipped
fi
printf '%s\tcomplete\tscout=%s confirmation=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$producer_phase" \
    "$producer_confirmation_phase" >"$producer_done_state"

wait_scheduler_idle guarded_scout
verify_self
set_state running_guarded_scout "$guarded_scout"
bash "$guarded/continue_guarded_early_trigger_scout.sh" \
    "$producer_done_state" "$guarded_binary" "$guarded_scout"
guarded_phase=$(wait_for_phase \
    "$guarded_scout.state" promising negative failed)
verify_self

set_state running_guarded_successor "scout=$guarded_phase"
bash "$guarded/continue_guarded_early_trigger_after_scout.sh" \
    "$guarded_scout.state" "$guarded_scout" "$guarded_binary" \
    "$guarded_transition" "$guarded_confirmation"
guarded_confirmation_phase=$(wait_for_phase \
    "$guarded_confirmation.chain.state" \
    confirmed negative skipped failed)

wait_scheduler_idle n6_collective_scout
verify_self
set_state running_n6_collective_scout "$n6_scout"
bash "$collective/continue_compiler_collective_hierpipe_n6_scout.sh" \
    "$n6_bundle" "$n6_scout"
n6_phase=$(wait_for_phase "$n6_scout.state" promising negative failed)

set_state complete \
    "reused=$reused_phase producer=$producer_phase/$producer_confirmation_phase guarded=$guarded_phase/$guarded_confirmation_phase n6=$n6_phase"
trap - EXIT
