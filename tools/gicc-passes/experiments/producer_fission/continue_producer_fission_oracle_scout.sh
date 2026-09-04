#!/usr/bin/env bash
# Wait for one named predecessor, then submit/monitor/analyze one pdebug scout.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 PREDECESSOR_JOB_ID BINARY_DIR OUTPUT_DIR PREDECESSOR_STATE" >&2
    exit 2
fi

predecessor_job=$1
binary_dir=$(cd -- "$2" && pwd)
if [[ $3 = /* ]]; then
    output_dir=$3
else
    output_dir=$(pwd)/$3
fi
predecessor_state=$4
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
runner="$script_dir/run_producer_fission_oracle_scout.sh"
monitor="$script_dir/monitor_producer_fission_oracle_scout.py"
analyzer="$script_dir/analyze_producer_fission_oracle_scout.py"
protocol="$script_dir/ORACLE_SCOUT_PROTOCOL.md"
builder="$script_dir/build_producer_fission_oracle.sh"
baseline="$binary_dir/baseline/jacobi"
fission="$binary_dir/fission/jacobi"
source_file="$repo_root/examples/ofi/jacobi.cpp"
helper_source="$repo_root/src/gicc/platform/ofi/runtime_helpers.cpp"
hint="$repo_root/tools/gicc-passes/tests/lit/Inputs/hint_fission_dwq.json"
provenance="$binary_dir/BUILD_PROVENANCE.txt"
state_path="$output_dir.state"
events_path="$output_dir.events"
lock_path="$output_dir.lock"

mkdir -p "$(dirname -- "$output_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another producer-fission controller owns this output" >&2
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

for path in "$baseline" "$fission" "$source_file" "$helper_source" "$hint" \
            "$runner" "$monitor" "$analyzer" "$protocol" "$builder" \
            "$provenance"; do
    if [[ ! -f $path ]]; then
        echo "missing frozen artifact: $path" >&2
        exit 2
    fi
done
if [[ -e $output_dir ]]; then
    echo "refusing existing producer-fission output: $output_dir" >&2
    exit 2
fi

artifacts=(
    "$baseline" "$fission" "$source_file" "$helper_source" "$hint"
    "$runner" "$monitor" "$analyzer" "$protocol" "$builder" "$provenance"
    "$binary_dir/baseline/meta/features.json"
    "$binary_dir/fission/meta/features.json"
)
for metadata in "$binary_dir/baseline/meta/"*.json \
                "$binary_dir/fission/meta/"*.json; do
    case $metadata in
        */features.json) ;;
        *) artifacts+=("$metadata") ;;
    esac
done
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

set_state waiting_predecessor "$predecessor_job"
flux job wait-event "$predecessor_job" clean >/dev/null
for ((attempt = 0; attempt < 120; ++attempt)); do
    if [[ -f $predecessor_state ]]; then
        predecessor_phase=$(awk 'NR == 1 {print $2}' "$predecessor_state")
        case $predecessor_phase in
            promising|negative|failed) break ;;
        esac
    fi
    sleep 5
done
if [[ ${predecessor_phase:-} != promising && \
      ${predecessor_phase:-} != negative && \
      ${predecessor_phase:-} != failed ]]; then
    echo "predecessor controller did not reach a terminal state" >&2
    exit 2
fi

mkdir -p "$output_dir"
printf '%s\n' "$predecessor_job" >"$output_dir/predecessor-job-id"
printf '%s\n' "$predecessor_phase" >"$output_dir/predecessor-state"
for replicate in 1 2 3 4; do
    for size in 1024 4096; do
        mkdir -p "$output_dir/rep${replicate}/size${size}"
    done
done

verify_frozen_artifacts
set_state submitting "one two-node pdebug allocation after $predecessor_job"
job_id=$(flux batch -q pdebug -N2 -n16 -c8 -g1 -t 20m -u \
    --job-name=pfission-oracle-ab --cwd="$output_dir" \
    "$runner" "$baseline" "$fission" "$output_dir")
if [[ -z $job_id || $job_id == *$'\n'* ]]; then
    echo "Flux returned an invalid producer-fission job ID: $job_id" >&2
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
    set_state promising "compiler fission has scout-level oracle headroom"
else
    set_state negative "compiler fission lacks scout-level oracle headroom"
fi
trap - EXIT
