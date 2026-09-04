#!/usr/bin/env bash
# Wait for the existing serial chain, then submit/monitor one pdebug scout.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "usage: $0 PREDECESSOR_STATE ARTIFACT_DIR OUTPUT_DIR" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
case $1 in /*) predecessor_state=$1 ;; *) predecessor_state="$repo_root/$1" ;; esac
artifact_dir=$(cd -- "$2" && pwd)
case $3 in /*) output_dir=$3 ;; *) output_dir="$repo_root/$3" ;; esac

baseline="$artifact_dir/baseline/bench_pingpong_lto"
reused="$artifact_dir/reused/bench_pingpong_lto"
source_file="$repo_root/examples/proxy/bench_pingpong_lto.cpp"
provenance="$artifact_dir/BUILD_PROVENANCE.txt"
ir_audit="$artifact_dir/ir-audit/audit.json"
baseline_hint="$script_dir/hint_baseline_dwq.json"
reused_hint="$script_dir/hint_reused_dwq.json"
builder="$script_dir/build_reused_loop_descriptor_oracle.sh"
auditor="$script_dir/audit_reused_loop_descriptor_ir.py"
protocol="$script_dir/ORACLE_SCOUT_PROTOCOL.md"
runner="$script_dir/run_reused_loop_descriptor_scout.sh"
controller="$script_dir/continue_reused_loop_descriptor_scout.sh"
monitor="$script_dir/monitor_reused_loop_descriptor_scout.py"
analyzer="$script_dir/analyze_reused_loop_descriptor_scout.py"
state_path="$output_dir.state"
events_path="$output_dir.events"
lock_path="$output_dir.lock"

mkdir -p "$(dirname -- "$output_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another reused-loop-descriptor controller owns this output" >&2
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

if [[ -e $output_dir ]]; then
    echo "refusing existing reused-loop-descriptor output: $output_dir" >&2
    exit 2
fi
for path in "$predecessor_state" "$baseline" "$reused" "$source_file" \
        "$provenance" "$ir_audit" "$baseline_hint" "$reused_hint" \
        "$builder" "$auditor" "$protocol" "$runner" "$controller" \
        "$monitor" "$analyzer"; do
    if [[ ! -f $path ]]; then
        echo "missing frozen artifact: $path" >&2
        exit 2
    fi
done
for binary in "$baseline" "$reused"; do
    if [[ ! -x $binary ]]; then
        echo "frozen artifact is not executable: $binary" >&2
        exit 2
    fi
done

verify_exact_hash() {
    local path=$1 expected=$2 actual
    actual=$(sha256sum "$path"); actual=${actual%% *}
    if [[ $actual != "$expected" ]]; then
        echo "protocol hash mismatch: $path" >&2
        exit 2
    fi
}
verify_exact_hash "$source_file" \
    a116f2148923d5678c865e134c3c09e69e8b0645c80db1b160e1fde6aea2a143
verify_exact_hash "$baseline" \
    2be1526aa6fb64c6f567d11fcdd4c15d227f969a7bc4db36cfe413d01a814559
verify_exact_hash "$reused" \
    6797a87646f22d83c3e19e63610739766b2557fae21e327b6b6482582d307dfc
verify_exact_hash "$ir_audit" \
    5a696217891099c672dbf559b303a7a88ed2fb2ae98c271b8e589e95d212912f

artifacts=(
    "$baseline" "$reused" "$source_file" "$provenance" "$ir_audit"
    "$baseline_hint" "$reused_hint" "$builder" "$auditor" "$protocol"
    "$runner" "$controller" "$monitor" "$analyzer"
)
for metadata in "$artifact_dir/baseline/meta/"*.json \
                "$artifact_dir/reused/meta/"*.json; do
    artifacts+=("$metadata")
done
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
        confirmed|negative|skipped|failed) ;;
        *) predecessor_phase=; sleep 5 ;;
    esac
done
verify_frozen_artifacts

# Respect other agents sharing this account: wait for an idle scheduler but
# never inspect, alter, or cancel their individual jobs.
active_jobs=
while true; do
    active_jobs=$(flux jobs --filter=active -n -o '{id.f58} {name}')
    if [[ -z $active_jobs ]]; then
        break
    fi
    active_count=$(printf '%s\n' "$active_jobs" | awk 'NF {count++} END {print count+0}')
    set_state waiting_scheduler_idle "active_user_jobs=$active_count"
    sleep 5
done
verify_frozen_artifacts

mkdir -p "$output_dir"
printf '%s\n' "$predecessor_state" >"$output_dir/predecessor-state-path"
sed -n '1p' "$predecessor_state" >"$output_dir/predecessor-state-final"
set_state submitting \
    "one N2 pdebug allocation after predecessor=$predecessor_phase"
job_id=$(flux batch -q pdebug -N2 -n2 -c64 -g1 -t 30m -u \
    --job-name=reused-descriptor-scout --cwd="$output_dir" \
    "$runner" "$baseline" "$reused" "$output_dir")
if [[ -z $job_id || $job_id == *$'\n'* ]]; then
    echo "Flux returned an invalid reused-descriptor job ID: $job_id" >&2
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
    set_state promising "reused descriptor has scout-level compiler headroom"
else
    set_state negative "reused descriptor lacks scout-level compiler headroom"
fi
trap - EXIT
