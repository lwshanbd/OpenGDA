#!/usr/bin/env bash
# Wait for a named campaign state, then submit/monitor one pdebug scout.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 PREDECESSOR_STATE BINARY_DIR OUTPUT_DIR" >&2
    exit 2
fi

predecessor_state=$1
binary_dir=$(cd -- "$2" && pwd)
if [[ $3 = /* ]]; then
    output_dir=$3
else
    output_dir=$(pwd)/$3
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
runner="$script_dir/run_guarded_early_trigger_scout.sh"
monitor="$script_dir/monitor_guarded_early_trigger_scout.py"
analyzer="$script_dir/analyze_guarded_early_trigger_scout.py"
auditor="$script_dir/audit_guarded_early_trigger_ir.py"
protocol="$script_dir/ORACLE_SCOUT_PROTOCOL.md"
builder="$script_dir/build_guarded_early_trigger_oracle.sh"
checksum_source="$script_dir/checksum_hip_memcpy.cpp"
baseline_hint="$script_dir/hint_baseline_dwq.json"
guarded_hint="$script_dir/hint_guarded_early_dwq.json"
baseline="$binary_dir/baseline/mm_minimal"
guarded="$binary_dir/guarded/mm_minimal"
source_file="$repo_root/examples/ofi/mm_minimal.cpp"
helper_source="$repo_root/src/gicc/platform/ofi/runtime_helpers.cpp"
plugin="$repo_root/tools/gicc-passes/build/libgicc-passes.so"
provenance="$binary_dir/BUILD_PROVENANCE.txt"
state_path="$output_dir.state"
events_path="$output_dir.events"
lock_path="$output_dir.lock"

mkdir -p "$(dirname -- "$output_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another guarded-early-trigger controller owns this output" >&2
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

artifacts=(
    "$baseline" "$guarded" "$source_file" "$helper_source" "$plugin"
    "$baseline_hint" "$guarded_hint" "$checksum_source" "$builder" "$runner"
    "$monitor" "$analyzer" "$auditor" "$protocol" "$provenance"
    "$binary_dir/baseline/meta/features.json"
    "$binary_dir/guarded/meta/features.json"
    "$binary_dir/guarded/ir-audit/device-final.ll"
    "$binary_dir/guarded/ir-audit/host-final.ll"
    "$binary_dir/guarded/ir-audit/audit.json"
)
for metadata in "$binary_dir/baseline/meta/"*.json \
                "$binary_dir/guarded/meta/"*.json; do
    case $metadata in
        */features.json) ;;
        *) artifacts+=("$metadata") ;;
    esac
done
for path in "${artifacts[@]}" "$predecessor_state"; do
    if [[ ! -f $path ]]; then
        echo "missing frozen artifact: $path" >&2
        exit 2
    fi
done
if [[ -e $output_dir ]]; then
    echo "refusing existing guarded-early-trigger output: $output_dir" >&2
    exit 2
fi

artifact_hashes=()
artifact_args=()
for artifact in "${artifacts[@]}"; do
    digest=$(sha256sum "$artifact")
    digest=${digest%% *}
    artifact_hashes+=("$digest")
    artifact_args+=(--artifact "$artifact=$digest")
done

verify_frozen_artifacts() {
    local index actual
    for index in "${!artifacts[@]}"; do
        actual=$(sha256sum "${artifacts[$index]}")
        actual=${actual%% *}
        if [[ $actual != "${artifact_hashes[$index]}" ]]; then
            echo "artifact changed while waiting: ${artifacts[$index]}" >&2
            exit 2
        fi
    done
}

set_state waiting_predecessor "$predecessor_state"
predecessor_phase=
while [[ -z $predecessor_phase ]]; do
    predecessor_phase=$(awk 'NR == 1 {print $2}' "$predecessor_state")
    case $predecessor_phase in
        complete|negative|failed) ;;
        *) predecessor_phase=; sleep 5 ;;
    esac
done
verify_frozen_artifacts

mkdir -p "$output_dir"
printf '%s\n' "$predecessor_state" >"$output_dir/predecessor-state-path"
sed -n '1p' "$predecessor_state" >"$output_dir/predecessor-state-final"
for replicate in 1 2 3 4; do
    for size in 4096 8192; do
        mkdir -p "$output_dir/rep${replicate}/size${size}"
    done
done

set_state submitting "one N2 pdebug allocation after predecessor=$predecessor_phase"
job_id=$(flux batch -q pdebug -N2 -n16 -c8 -g1 -t 30m -u \
    --job-name=guarded-early-mm-ab --cwd="$output_dir" \
    "$runner" "$baseline" "$guarded" "$output_dir")
if [[ -z $job_id || $job_id == *$'\n'* ]]; then
    echo "Flux returned an invalid guarded-early-trigger job ID: $job_id" >&2
    exit 2
fi
printf '%s\n' "$job_id" >"$output_dir/job-id"

set_state monitoring "$job_id"
python3 "$monitor" --job-id "$job_id" --output-dir "$output_dir" \
    --status "$output_dir/monitor.json" "${artifact_args[@]}"
set_state analyzing "$output_dir/analysis.json"
python3 "$analyzer" --monitor "$output_dir/monitor.json" \
    --out "$output_dir/analysis.json"
promising=$(python3 -c \
    'import json,sys; print("yes" if json.load(open(sys.argv[1]))["oracle_headroom_gate"]["passed"] else "no")' \
    "$output_dir/analysis.json")
if [[ $promising == yes ]]; then
    set_state promising "compiler guarded early trigger has scout-level headroom"
else
    set_state negative "compiler guarded early trigger lacks scout-level headroom"
fi
trap - EXIT
