#!/usr/bin/env bash
# Wait for Gate C and, only on a pass, run three same-allocation Gate-D blocks.
# One pdebug batch allocation is active at a time. Every allocation executes
# all compiler controls sequentially on the same nodes, with rotated arm order.
# This controller stops after confirmatory analysis and never invokes a model.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 V3_BUNDLE_DIR" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
if [[ $1 = /* ]]; then
    bundle_dir=$1
else
    bundle_dir="$repo_root/$1"
fi
main_state="$bundle_dir.controller.state"
state_path="$bundle_dir.gate-d.state"
events_path="$bundle_dir.gate-d.events"
lock_path="$bundle_dir.gate-d.lock"
poll_seconds=${GICC_CONTROLLER_POLL_SECONDS:-60}
evaluator="$script_dir/compiler_collective_eval.py"
monitor="$script_dir/monitor_compiler_collective_replicate.py"
runner="$script_dir/run_compiler_collective_replicate.sh"

mkdir -p "$(dirname -- "$bundle_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another Gate-D controller already owns this bundle" >&2
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

if [[ ! "$poll_seconds" =~ ^[1-9][0-9]*$ ]]; then
    echo "GICC_CONTROLLER_POLL_SECONDS must be a positive integer" >&2
    exit 2
fi

set_state waiting_gate_c "$main_state"
while true; do
    if [[ ! -f "$main_state" ]]; then
        sleep "$poll_seconds"
        continue
    fi
    IFS=$'\t' read -r _ main_status main_detail <"$main_state"
    case "$main_status" in
        gate_c_passed)
            break
            ;;
        gate_c_failed)
            set_state not_run_gate_c_failed "$main_detail"
            trap - EXIT
            exit 0
            ;;
        failed)
            echo "Gate-A/B/C controller failed: $main_detail" >&2
            exit 1
            ;;
        *)
            sleep "$poll_seconds"
            ;;
    esac
done

freeze="$bundle_dir/FROZEN_V3_MANIFEST.json"
manifest="$bundle_dir/controls/manifest.json"
graph="$bundle_dir/discovery/graph.json"
screen="$bundle_dir/gate-b/analysis.json"
python3 "$evaluator" verify-offline-freeze \
    --manifest "$freeze" --repo-root "$repo_root"
screen_passed=$(python3 -c \
    'import json,sys; print("yes" if json.load(open(sys.argv[1]))["gate_c"]["passed"] else "no")' \
    "$screen")
if [[ "$screen_passed" != yes ]]; then
    echo "Gate-D controller received a non-passing Gate-C analysis" >&2
    exit 2
fi

mapfile -t names < <(python3 -c \
    'import json,sys; print("\n".join(a["name"] for a in json.load(open(sys.argv[1]))["arms"]))' \
    "$manifest")
if [[ ${#names[@]} -ne 8 ]]; then
    echo "Gate D requires exactly eight qualified compiler controls" >&2
    exit 2
fi

gate_d_dir="$bundle_dir/gate-d"
mkdir -p "$gate_d_dir"
expected_sizes=(1024 4096 8192 65536 262144 1048576 4194304 8388608 16777216)
monitor_paths=()
for replicate in 1 2 3; do
    shift_count=$((replicate - 1))
    order=("${names[@]:shift_count}" "${names[@]:0:shift_count}")
    rep_dir="$gate_d_dir/rep$replicate"
    status="$rep_dir/monitor.json"
    job_file="$rep_dir/job-id"
    monitor_paths+=("$status")

    if [[ -f "$status" ]]; then
        existing_state=$(python3 -c \
            'import json,sys; print(json.load(open(sys.argv[1])).get("state", "invalid"))' \
            "$status")
        if [[ "$existing_state" == passed ]]; then
            continue
        fi
        echo "refusing to replace non-passed Gate-D monitor: $status" >&2
        exit 2
    fi
    if [[ -e "$rep_dir" ]]; then
        echo "refusing partial Gate-D replicate directory: $rep_dir" >&2
        exit 2
    fi
    mkdir -p "$rep_dir"
    for name in "${names[@]}"; do
        : >"$rep_dir/$name.out"
        : >"$rep_dir/$name.err"
    done

    set_state submitting_gate_d "replicate=$replicate"
    job_id=$(flux batch -q pdebug -N2 -n16 -c8 -g1 -t 45m -u \
        --job-name="coll-d-rep$replicate" --cwd="$rep_dir" \
        "$runner" "$bundle_dir" "$rep_dir" "$replicate" "${order[@]}")
    if [[ -z "$job_id" || "$job_id" == *$'\n'* ]]; then
        echo "Flux returned an invalid Gate-D job ID: $job_id" >&2
        exit 2
    fi
    printf '%s\n' "$job_id" >"$job_file"

    monitor_args=(
        --job-id "$job_id"
        --driver-stdout "$rep_dir/driver.out"
        --driver-stderr "$rep_dir/driver.err"
        --status "$status"
        --replicate "$replicate"
        --expected-nodes 2
        --expected-ranks 16
        --expected-ppn 8
        --expected-runs 7
        --expected-warmup 2
    )
    for size in "${expected_sizes[@]}"; do
        monitor_args+=(--expected-size "$size")
    done
    for name in "${names[@]}"; do
        monitor_args+=(
            --benchmark "$name=$rep_dir/$name.out"
            --benchmark-stderr "$name=$rep_dir/$name.err"
        )
    done
    artifacts=("$freeze" "$manifest" "$graph" "$screen" "$runner" "$monitor")
    for name in "${names[@]}"; do
        artifacts+=(
            "$bundle_dir/binaries/$name/compiler_collective_eval"
            "$bundle_dir/binaries/$name/build-provenance.json"
            "$bundle_dir/controls/$name-hint.json"
        )
    done
    for artifact in "${artifacts[@]}"; do
        digest=$(sha256sum "$artifact")
        digest=${digest%% *}
        monitor_args+=(--artifact "$artifact=$digest")
    done

    set_state monitoring_gate_d "replicate=$replicate job=$job_id"
    python3 "$monitor" "${monitor_args[@]}"
    set_state passed_gate_d "replicate=$replicate job=$job_id"
done

set_state analyzing_gate_d "$gate_d_dir/analysis.json"
confirm_args=(
    confirm
    --graph "$graph"
    --manifest "$manifest"
    --screen-analysis "$screen"
    --out "$gate_d_dir/analysis.json"
)
for path in "${monitor_paths[@]}"; do
    confirm_args+=(--replicate-monitor "$path")
done
python3 "$evaluator" "${confirm_args[@]}"
set_state gate_d_complete "$gate_d_dir/analysis.json"
trap - EXIT
