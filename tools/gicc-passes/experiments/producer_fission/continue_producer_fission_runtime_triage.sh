#!/usr/bin/env bash
# Submit and audit one fail-closed pdebug correctness triage allocation.
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 FAILED_MONITOR FAILED_STDERR BINARY_DIR OUTPUT_DIR" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
case $1 in /*) failed_monitor=$1 ;; *) failed_monitor="$repo_root/$1" ;; esac
case $2 in /*) failed_stderr=$2 ;; *) failed_stderr="$repo_root/$2" ;; esac
binary_dir=$(cd -- "$3" && pwd)
case $4 in /*) output_dir=$4 ;; *) output_dir="$repo_root/$4" ;; esac

runner="$script_dir/run_producer_fission_correctness_pair.sh"
auditor="$script_dir/audit_producer_fission_runtime_triage.py"
protocol="$script_dir/RUNTIME_TRIAGE_PROTOCOL.md"
baseline="$binary_dir/baseline/jacobi"
fission="$binary_dir/fission/jacobi"
provenance="$binary_dir/BUILD_PROVENANCE.txt"
application_source="$repo_root/examples/ofi/jacobi.cpp"
pass_source="$repo_root/tools/gicc-passes/src/GICCProducerFission.cpp"
lit_test="$repo_root/tools/gicc-passes/tests/lit/producer_fission/host_preinline_stub.ll"
state_path="$output_dir.state"
events_path="$output_dir.events"
lock_path="$output_dir.lock"
job_name=producer-fission-triage

mkdir -p "$(dirname -- "$output_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another producer-fission triage owns this output" >&2
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

artifacts=(
    "$failed_monitor" "$failed_stderr" "$baseline" "$fission" "$provenance"
    "$application_source" "$pass_source" "$lit_test" "$runner" "$auditor"
    "$protocol"
)
hashes=()
for artifact in "${artifacts[@]}"; do
    if [[ ! -f $artifact ]]; then
        echo "missing triage artifact: $artifact" >&2
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
            echo "triage artifact changed: ${artifacts[$index]}" >&2
            exit 2
        fi
    done
}

if [[ -e $output_dir ]]; then
    echo "refusing existing triage output: $output_dir" >&2
    exit 2
fi
active_jobs=$(flux jobs --filter=active -n -o '{id.f58} {name}')
if [[ -n $active_jobs ]]; then
    echo "refusing to submit while another user job is active" >&2
    exit 2
fi

mkdir -p "$output_dir"
set_state submitting "one N2 pdebug baseline/fission pair"
job_id=$(flux batch -q pdebug -N2 -n16 -c8 -g1 -t 8m -u \
    --job-name="$job_name" --cwd="$output_dir" \
    "$runner" "$baseline" "$fission" "$output_dir")
if [[ -z $job_id || $job_id == *$'\n'* ]]; then
    echo "Flux returned an invalid triage job ID: $job_id" >&2
    exit 2
fi
printf '%s\n' "$job_id" >"$output_dir/job-id"

set_state monitoring "$job_id"
python3 "$auditor" \
    --failed-monitor "$failed_monitor" --failed-stderr "$failed_stderr" \
    --fixed-provenance "$provenance" --fixed-binary-dir "$binary_dir" \
    --paired-baseline "$output_dir/baseline.out" \
    --paired-fission "$output_dir/fission.out" --paired-runner "$runner" \
    --paired-job-id "$job_id" --paired-job-name "$job_name" \
    --application-source "$application_source" --pass-source "$pass_source" \
    --lit-test "$lit_test" --out "$output_dir/report.json"
verify_artifacts
set_state negative "correctness gate failed; candidate remains model-invisible"
trap - EXIT
