#!/usr/bin/env bash
# Deterministically continue one collective experiment through Gate C.
#
# The controller waits for an already-submitted Gate-A monitor.  On a pass it
# prepares the offline v3 bundle, then submits exactly one pdebug Gate-B arm at
# a time and synchronously monitors it.  Any failure stops the chain.  No model
# or provider is invoked, and this script stops after computing Gate C.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 GATE_A_MONITOR V3_BUNDLE_DIR [CALIBRATION_JSON]" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
gate_a_monitor=$1
bundle_dir=$2
calibration=${3:-$repo_root/build_ofi/compiler_lto_calibration/generated/dossier.json}
evaluator="$script_dir/compiler_collective_eval.py"
preparer="$script_dir/prepare_compiler_collective_v3.sh"
monitor="$script_dir/monitor_compiler_collective_job.py"
poll_seconds=${GICC_CONTROLLER_POLL_SECONDS:-60}
state_path="$bundle_dir.controller.state"
controller_log="$bundle_dir.controller.events"
mkdir -p "$(dirname -- "$bundle_dir")"
exec 9>"$bundle_dir.controller.lock"
if ! flock -n 9; then
    echo "another controller already owns this v3 bundle" >&2
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
        "$state" "$detail" >>"$controller_log"
}

on_exit() {
    local status=$?
    if [[ $status -ne 0 ]]; then
        set_state failed "controller_exit=$status"
    fi
}
trap on_exit EXIT

if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "GICC_CONTROLLER_POLL_SECONDS must be a positive integer" >&2
    exit 2
fi

set_state waiting_gate_a "$gate_a_monitor"
while true; do
    gate_state=$(python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1])).get("state", "invalid"))' \
        "$gate_a_monitor")
    case "$gate_state" in
        passed)
            python3 "$evaluator" verify-gate-a-monitor \
                --monitor "$gate_a_monitor"
            break
            ;;
        failed)
            echo "Gate A failed; refusing v3 preparation" >&2
            exit 1
            ;;
        monitoring)
            sleep "$poll_seconds"
            ;;
        *)
            echo "invalid Gate-A monitor state: $gate_state" >&2
            exit 2
            ;;
    esac
done

if [[ -f "$bundle_dir/FROZEN_V3_MANIFEST.json" ]]; then
    set_state verifying_existing_v3 "$bundle_dir"
    python3 "$evaluator" verify-offline-freeze \
        --manifest "$bundle_dir/FROZEN_V3_MANIFEST.json" \
        --repo-root "$repo_root"
elif [[ -e "$bundle_dir" ]]; then
    echo "partial v3 directory exists without a frozen manifest: $bundle_dir" >&2
    exit 2
else
    set_state preparing_v3 "$bundle_dir"
    bash "$preparer" "$bundle_dir" "$gate_a_monitor" "$calibration"
fi

bundle_dir=$(cd -- "$bundle_dir" && pwd)
manifest="$bundle_dir/controls/manifest.json"
graph="$bundle_dir/discovery/graph.json"
gate_b_dir="$bundle_dir/gate-b"
mkdir -p "$gate_b_dir"
sizes=1024,4096,8192,65536,262144,1048576,4194304,8388608,16777216
expected_sizes=(1024 4096 8192 65536 262144 1048576 4194304 8388608 16777216)
mapfile -t names < <(python3 -c \
    'import json,sys; print("\n".join(a["name"] for a in json.load(open(sys.argv[1]))["arms"]))' \
    "$manifest")
if [[ ${#names[@]} -ne 8 ]]; then
    echo "Gate B requires exactly eight uniform compiler controls" >&2
    exit 2
fi

logs=()
for name in "${names[@]}"; do
    arm_dir="$bundle_dir/binaries/$name"
    binary="$arm_dir/compiler_collective_eval"
    stdout="$gate_b_dir/$name.out"
    stderr="$gate_b_dir/$name.err"
    status="$gate_b_dir/$name.monitor.json"
    job_file="$gate_b_dir/$name.job-id"
    logs+=("$stdout")

    if [[ -f "$status" ]]; then
        existing_state=$(python3 -c \
            'import json,sys; print(json.load(open(sys.argv[1])).get("state", "invalid"))' \
            "$status")
        if [[ "$existing_state" == passed ]]; then
            continue
        fi
        echo "refusing to replace non-passed Gate-B monitor: $status" >&2
        exit 2
    fi
    for unexpected in "$stdout" "$stderr" "$job_file"; do
        if [[ -e "$unexpected" ]]; then
            echo "refusing stale Gate-B artifact: $unexpected" >&2
            exit 2
        fi
    done

    python3 "$evaluator" verify-build-provenance \
        --manifest "$arm_dir/build-provenance.json" \
        --repo-root "$repo_root"
    set_state submitting_gate_b "$name"
    job_id=$(flux submit -q pdebug -N2 -n16 -c8 -g1 -t 10m -u \
        --job-name="coll-b-$name" \
        --cwd="$repo_root" \
        --output="$stdout" --error="$stderr" \
        --env="GICC_COLL_SIZES=$sizes" \
        --env="GICC_COLLECTIVE_PLAN_LABEL=$name" \
        --env=HSA_ENABLE_IPC_MODE_LEGACY=1 \
        --env=FI_MR_CACHE_MONITOR=kdreg2 \
        "$binary" 3 1)
    if [[ -z "$job_id" || "$job_id" == *$'\n'* ]]; then
        echo "Flux returned an invalid job ID: $job_id" >&2
        exit 2
    fi
    printf '%s\n' "$job_id" >"$job_file"

    monitor_args=(
        --job-id "$job_id"
        --stdout "$stdout"
        --stderr "$stderr"
        --status "$status"
        --expected-label "$name"
        --expected-nodes 2
        --expected-ranks 16
        --expected-ppn 8
        --expected-runs 3
        --expected-warmup 1
    )
    for size in "${expected_sizes[@]}"; do
        monitor_args+=(--expected-size "$size")
    done
    artifacts=(
        "$binary"
        "$arm_dir/build-provenance.json"
        "$arm_dir/materialized.ll"
        "$arm_dir/materialized-device.ll"
        "$bundle_dir/controls/$name-hint.json"
    )
    for artifact in "${artifacts[@]}"; do
        digest=$(sha256sum "$artifact")
        digest=${digest%% *}
        monitor_args+=(--artifact "$artifact=$digest")
    done
    set_state monitoring_gate_b "$name:$job_id"
    python3 "$monitor" "${monitor_args[@]}"
    set_state passed_gate_b_arm "$name:$job_id"
done

set_state qualifying_gate_b "${#logs[@]} arms"
python3 "$evaluator" qualify \
    --manifest "$manifest" --gate b \
    --out "$gate_b_dir/qualification.json" \
    "${logs[@]}"

set_state analyzing_gate_c "$gate_b_dir/analysis.json"
python3 "$evaluator" analyze \
    --graph "$graph" --manifest "$manifest" \
    --out "$gate_b_dir/analysis.json" \
    "${logs[@]}"
gate_c_passed=$(python3 -c \
    'import json,sys; print("yes" if json.load(open(sys.argv[1]))["gate_c"]["passed"] else "no")' \
    "$gate_b_dir/analysis.json")
if [[ "$gate_c_passed" == yes ]]; then
    set_state gate_c_passed "ready_for_confirmatory_controls"
else
    set_state gate_c_failed "negative_capacity_result"
fi
trap - EXIT
