#!/usr/bin/env bash
# Submit, monitor, and analyze three producer-fission allocations serially.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 TRANSITION_REPORT BINARY_DIR OUTPUT_DIR" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
transition=$(cd -- "$(dirname -- "$1")" && pwd)/$(basename -- "$1")
binary_dir=$(cd -- "$2" && pwd)
if [[ $3 = /* ]]; then
    output_dir=$3
else
    output_dir="$repo_root/$3"
fi

baseline="$binary_dir/baseline/jacobi"
fission="$binary_dir/fission/jacobi"
provenance="$binary_dir/BUILD_PROVENANCE.txt"
protocol="$script_dir/PRODUCER_FISSION_CONFIRMATION_TRANSITION.md"
preparer="$script_dir/prepare_producer_fission_confirmation.py"
runner="$script_dir/run_producer_fission_confirmation.sh"
controller="$script_dir/continue_producer_fission_confirmation.sh"
monitor="$script_dir/monitor_producer_fission_confirmation.py"
analyzer="$script_dir/analyze_producer_fission_confirmation.py"
state_path="$output_dir.state"
events_path="$output_dir.events"
lock_path="$output_dir.lock"

mkdir -p "$(dirname -- "$output_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another producer-fission confirmation owns this output" >&2
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

python3 "$preparer" verify-contained --report "$transition"
python3 -c \
    'import json,pathlib,sys; x=json.load(open(sys.argv[1])); root=pathlib.Path(sys.argv[2]); paths={r["role"]:(pathlib.Path(r["path"]) if pathlib.Path(r["path"]).is_absolute() else root/pathlib.Path(r["path"])).resolve() for r in x["files"]}; assert paths["frozen_baseline_binary"] == pathlib.Path(sys.argv[3]).resolve(); assert paths["frozen_fission_binary"] == pathlib.Path(sys.argv[4]).resolve(); assert paths["frozen_build_provenance"] == pathlib.Path(sys.argv[5]).resolve()' \
    "$transition" "$repo_root" "$baseline" "$fission" "$provenance"
if [[ -e "$output_dir" ]]; then
    echo "refusing existing producer-fission confirmation: $output_dir" >&2
    exit 2
fi

artifacts=(
    "$transition" "$baseline" "$fission" "$provenance"
    "$protocol" "$preparer" "$runner" "$controller" "$monitor" "$analyzer"
)
artifact_args=()
for artifact in "${artifacts[@]}"; do
    digest=$(sha256sum "$artifact")
    digest=${digest%% *}
    artifact_args+=(--artifact "$artifact=$digest")
done

mkdir -p "$output_dir"
monitor_paths=()
for allocation in 1 2 3; do
    allocation_dir="$output_dir/allocation$allocation"
    mkdir -p "$allocation_dir"
    : >"$allocation_dir/driver.out"
    : >"$allocation_dir/driver.err"
    for block in 1 2; do
        for size in 1024 4096; do
            run_dir="$allocation_dir/block$block/size$size"
            mkdir -p "$run_dir"
            : >"$run_dir/baseline.out"
            : >"$run_dir/baseline.err"
            : >"$run_dir/fission.out"
            : >"$run_dir/fission.err"
        done
    done
    set_state submitting \
        "allocation=$allocation/3 queue=pdebug max_active_or_queued=1"
    job_id=$(flux batch -q pdebug -N2 -n16 -c8 -g1 -t 20m -u \
        --job-name="pfission-confirm-a$allocation" --cwd="$output_dir" \
        "$runner" "$baseline" "$fission" "$output_dir" "$allocation")
    if [[ -z $job_id || $job_id == *$'\n'* ]]; then
        echo "Flux returned an invalid confirmation job ID: $job_id" >&2
        exit 2
    fi
    printf '%s\n' "$job_id" >"$allocation_dir/job-id"
    status="$allocation_dir/monitor.json"
    monitor_paths+=("$status")
    set_state monitoring "allocation=$allocation/3 job=$job_id"
    python3 "$monitor" \
        --job-id "$job_id" \
        --output-dir "$output_dir" \
        --status "$status" \
        --allocation "$allocation" \
        "${artifact_args[@]}"
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
    set_state confirmed "producer-fission compiler schedule headroom confirmed"
else
    set_state negative "producer-fission confirmation gate failed"
fi
trap - EXIT
