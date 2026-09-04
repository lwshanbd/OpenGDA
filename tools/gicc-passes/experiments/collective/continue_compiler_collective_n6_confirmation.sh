#!/usr/bin/env bash
# Build compiler policies, then submit/monitor three N6 allocations serially.
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 N6_BUNDLE_DIR SCOUT_DIR HEURISTIC_DECISION TRANSITION_DIR OUTPUT_DIR" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
bundle_dir=$(cd -- "$1" && pwd)
scout_dir=$(cd -- "$2" && pwd)
heuristic_decision=$(cd -- "$(dirname -- "$3")" && pwd)/$(basename -- "$3")
transition_dir=$(cd -- "$4" && pwd)
if [[ $5 = /* ]]; then
    output_dir=$5
else
    output_dir="$repo_root/$5"
fi

graph="$bundle_dir/discovery/graph.json"
scout_analysis="$scout_dir/analysis.json"
transition="$transition_dir/transition.json"
protocol="$script_dir/HIERPIPE_N6_CONFIRMATION_TRANSITION.md"
preparer="$script_dir/prepare_compiler_collective_n6_confirmation.py"
preparer_base="$script_dir/prepare_compiler_collective_n8_confirmation.py"
build_script="$script_dir/build_compiler_collective_eval.sh"
evaluator="$script_dir/compiler_collective_eval.py"
runner="$script_dir/run_compiler_collective_n6_confirmation.sh"
controller="$script_dir/continue_compiler_collective_n6_confirmation.sh"
monitor="$script_dir/monitor_compiler_collective_replicate.py"
analyzer="$script_dir/analyze_compiler_collective_n6_confirmation.py"
analyzer_base="$script_dir/analyze_compiler_collective_n8_confirmation.py"
state_path="$output_dir.state"
events_path="$output_dir.events"
lock_path="$output_dir.lock"
binary_root="$output_dir/binaries"
arms=(
    derived_bin_policy
    scout_best_uniform
    frozen_structural_heuristic
)
hints=(
    "$transition_dir/derived-bin-policy-hint.json"
    "$transition_dir/best-uniform-hint.json"
    "$transition_dir/frozen-structural-heuristic-hint.json"
)

mkdir -p "$(dirname -- "$output_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another N6 confirmation controller owns this output" >&2
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

python3 "$evaluator" verify-offline-freeze \
    --manifest "$bundle_dir/FROZEN_V3_MANIFEST.json" \
    --repo-root "$repo_root"
python3 "$preparer" verify \
    --graph "$graph" \
    --scout "$scout_analysis" \
    --heuristic-decision "$heuristic_decision" \
    --output-dir "$transition_dir"
python3 -c \
    'import json,sys; x=json.load(open(sys.argv[1])); c=x["confirmation_contract"]; assert x["status"] == "confirmation_plan_ready"; assert c["queue"] == "pdebug"; assert c["nodes"] == 6; assert c["ranks"] == 48; assert c["independent_allocations"] == 3; assert c["maximum_active_or_queued_jobs"] == 1' \
    "$transition"
if [[ -e "$output_dir" ]]; then
    echo "refusing existing N6 confirmation output: $output_dir" >&2
    exit 2
fi

mkdir -p "$binary_root"
set_state building "three same-source compiler/LTO policies"
for index in "${!arms[@]}"; do
    arm=${arms[$index]}
    hint=${hints[$index]}
    "$build_script" lower "$binary_root/$arm" "$hint"
    python3 "$evaluator" verify-build-provenance \
        --manifest "$binary_root/$arm/build-provenance.json" \
        --repo-root "$repo_root"
done

artifacts=(
    "$bundle_dir/FROZEN_V3_MANIFEST.json"
    "$graph"
    "$scout_analysis"
    "$heuristic_decision"
    "$protocol"
    "$preparer"
    "$preparer_base"
    "$transition_dir/derived-bin-policy-decision.json"
    "$transition_dir/derived-bin-policy-hint.json"
    "$transition_dir/best-uniform-decision.json"
    "$transition_dir/best-uniform-hint.json"
    "$transition_dir/frozen-structural-heuristic-hint.json"
    "$transition"
    "$build_script"
    "$evaluator"
    "$runner"
    "$controller"
    "$monitor"
    "$analyzer"
    "$analyzer_base"
)
for arm in "${arms[@]}"; do
    artifacts+=(
        "$binary_root/$arm/compiler_collective_eval"
        "$binary_root/$arm/build-provenance.json"
    )
done
artifact_args=()
for artifact in "${artifacts[@]}"; do
    digest=$(sha256sum "$artifact")
    digest=${digest%% *}
    artifact_args+=(--artifact "$artifact=$digest")
done

monitor_paths=()
for replicate in 1 2 3; do
    rep_dir="$output_dir/rep$replicate"
    mkdir -p "$rep_dir"
    : >"$rep_dir/driver.out"
    : >"$rep_dir/driver.err"
    for arm in "${arms[@]}"; do
        : >"$rep_dir/$arm.out"
        : >"$rep_dir/$arm.err"
    done
    set_state submitting \
        "allocation=$replicate/3 queue=pdebug max_active_or_queued=1"
    job_id=$(flux batch -q pdebug -N6 -n48 -c8 -g1 -t 20m -u \
        --job-name="coll-n6-confirm-r$replicate" --cwd="$output_dir" \
        "$runner" "$binary_root" "$output_dir" "$replicate")
    if [[ -z "$job_id" || "$job_id" == *$'\n'* ]]; then
        echo "Flux returned an invalid confirmation job ID: $job_id" >&2
        exit 2
    fi
    printf '%s\n' "$job_id" >"$rep_dir/job-id"
    status="$rep_dir/monitor.json"
    monitor_paths+=("$status")
    monitor_args=(
        --job-id "$job_id"
        --driver-stdout "$rep_dir/driver.out"
        --driver-stderr "$rep_dir/driver.err"
        --status "$status"
        --replicate "$replicate"
        --expected-nodes 6
        --expected-ranks 48
        --expected-ppn 8
        --expected-runs 7
        --expected-warmup 2
    )
    for size in 1024 4096 8192 65536 262144 1048576 4194304 8388608 16777216; do
        monitor_args+=(--expected-size "$size")
    done
    for arm in "${arms[@]}"; do
        monitor_args+=(
            --benchmark "$arm=$rep_dir/$arm.out"
            --benchmark-stderr "$arm=$rep_dir/$arm.err"
        )
    done
    set_state monitoring "allocation=$replicate/3 job=$job_id"
    python3 "$monitor" "${monitor_args[@]}" "${artifact_args[@]}"
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
    set_state confirmed "derived compiler policy beats both comparators"
else
    set_state negative "derived compiler policy failed a co-primary gate"
fi
trap - EXIT
