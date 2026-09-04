#!/usr/bin/env bash
# Submit, monitor, and analyze three descriptor confirmations serially.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 TRANSITION_REPORT BINARY_DIR OUTPUT_DIR" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
transition=$(cd -- "$(dirname -- "$1")" && pwd)/$(basename -- "$1")
binary_dir=$(cd -- "$2" && pwd)
if [[ $3 = /* ]]; then output_dir=$3; else output_dir="$repo_root/$3"; fi
baseline="$binary_dir/baseline/bench_pingpong_lto"
reused="$binary_dir/reused/bench_pingpong_lto"
provenance="$binary_dir/BUILD_PROVENANCE.txt"
protocol="$script_dir/REUSED_LOOP_DESCRIPTOR_CONFIRMATION_TRANSITION.md"
preparer="$script_dir/prepare_reused_loop_descriptor_confirmation.py"
runner="$script_dir/run_reused_loop_descriptor_confirmation.sh"
controller="$script_dir/continue_reused_loop_descriptor_confirmation.sh"
monitor="$script_dir/monitor_reused_loop_descriptor_confirmation.py"
analyzer="$script_dir/analyze_reused_loop_descriptor_confirmation.py"
state_path="$output_dir.state"
events_path="$output_dir.events"
lock_path="$output_dir.lock"

mkdir -p "$(dirname -- "$output_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another reused-loop confirmation owns this output" >&2
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
    [[ $status -eq 0 ]] || set_state failed "controller_exit=$status"
}
trap on_exit EXIT

python3 "$preparer" verify-contained --report "$transition"
python3 -c \
    'import json,pathlib,sys; x=json.load(open(sys.argv[1])); root=pathlib.Path(sys.argv[2]); p={r["role"]:(pathlib.Path(r["path"]) if pathlib.Path(r["path"]).is_absolute() else root/pathlib.Path(r["path"])).resolve() for r in x["files"]}; assert p["frozen_baseline_binary"] == pathlib.Path(sys.argv[3]).resolve(); assert p["frozen_reused_binary"] == pathlib.Path(sys.argv[4]).resolve(); assert p["frozen_build_provenance"] == pathlib.Path(sys.argv[5]).resolve()' \
    "$transition" "$repo_root" "$baseline" "$reused" "$provenance"
if [[ -e $output_dir ]]; then
    echo "refusing existing reused-loop confirmation: $output_dir" >&2
    exit 2
fi

artifacts=(
    "$transition" "$baseline" "$reused" "$provenance" "$protocol"
    "$preparer" "$runner" "$controller" "$monitor" "$analyzer"
)
artifact_hashes=()
artifact_args=()
for artifact in "${artifacts[@]}"; do
    digest=$(sha256sum "$artifact"); digest=${digest%% *}
    artifact_hashes+=("$digest")
    artifact_args+=(--artifact "$artifact=$digest")
done
verify_frozen_artifacts() {
    local index actual
    for index in "${!artifacts[@]}"; do
        actual=$(sha256sum "${artifacts[$index]}"); actual=${actual%% *}
        if [[ $actual != "${artifact_hashes[$index]}" ]]; then
            echo "confirmation artifact changed: ${artifacts[$index]}" >&2
            exit 2
        fi
    done
}
wait_scheduler_idle() {
    local active_jobs active_count
    while true; do
        active_jobs=$(flux jobs --filter=active -n -o '{id.f58} {name}')
        if [[ -z $active_jobs ]]; then
            return
        fi
        active_count=$(printf '%s\n' "$active_jobs" \
            | awk 'NF {count++} END {print count+0}')
        set_state waiting_scheduler_idle \
            "allocation=$1/3 active_user_jobs=$active_count"
        sleep 5
    done
}

mkdir -p "$output_dir"
monitor_paths=()
for allocation in 1 2 3; do
    allocation_dir="$output_dir/allocation$allocation"
    mkdir -p "$allocation_dir"
    : >"$allocation_dir/driver.out"
    : >"$allocation_dir/driver.err"
    for block in 1 2; do
        for batch in 4 64; do
            run_dir="$allocation_dir/block$block/batch$batch"
            mkdir -p "$run_dir"
            : >"$run_dir/baseline.out"; : >"$run_dir/baseline.err"
            : >"$run_dir/reused.out"; : >"$run_dir/reused.err"
        done
    done
    wait_scheduler_idle "$allocation"
    verify_frozen_artifacts
    set_state submitting \
        "allocation=$allocation/3 queue=pdebug max_active_or_queued=1"
    job_id=$(flux batch -q pdebug -N2 -n2 -c64 -g1 -t 20m -u \
        --job-name="reused-loop-confirm-a$allocation" --cwd="$output_dir" \
        "$runner" "$baseline" "$reused" "$output_dir" "$allocation")
    if [[ -z $job_id || $job_id == *$'\n'* ]]; then
        echo "Flux returned an invalid confirmation job ID: $job_id" >&2
        exit 2
    fi
    printf '%s\n' "$job_id" >"$allocation_dir/job-id"
    status="$allocation_dir/monitor.json"
    monitor_paths+=("$status")
    set_state monitoring "allocation=$allocation/3 job=$job_id"
    python3 "$monitor" --job-id "$job_id" --output-dir "$output_dir" \
        --status "$status" --allocation "$allocation" "${artifact_args[@]}"
done

set_state analyzing "$output_dir/analysis.json"
analysis_args=(--transition "$transition" --out "$output_dir/analysis.json")
for status in "${monitor_paths[@]}"; do
    analysis_args+=(--monitor "$status")
done
python3 "$analyzer" "${analysis_args[@]}"
confirmed=$(python3 -c \
    'import json,sys; print("yes" if json.load(open(sys.argv[1]))["confirmation_gate"]["passed"] else "no")' \
    "$output_dir/analysis.json")
if [[ $confirmed == yes ]]; then
    set_state confirmed "reused-loop-descriptor compiler headroom confirmed"
else
    set_state negative "reused-loop-descriptor confirmation gate failed"
fi
trap - EXIT
