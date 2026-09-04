#!/usr/bin/env bash
# Submit and audit three independent N6 compiler-policy allocations serially.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 PLAN_DIR OUTPUT_DIR" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
plan_dir=$(cd -- "$1" && pwd)
case $2 in /*) output_dir=$2 ;; *) output_dir="$repo_root/$2" ;; esac
preparer="$script_dir/prepare_collective_n6_llm_runtime_validation.py"
build_script="$script_dir/build_collective_n6_llm_runtime_validation.sh"
compiler_builder="$script_dir/build_compiler_collective_eval.sh"
evaluator="$script_dir/compiler_collective_eval.py"
runner="$script_dir/run_collective_n6_llm_runtime_validation.sh"
controller="$script_dir/continue_collective_n6_llm_runtime_validation.sh"
monitor="$script_dir/monitor_compiler_collective_replicate.py"
analyzer="$script_dir/analyze_collective_n6_llm_runtime_validation.py"
state_path="$output_dir.state"
events_path="$output_dir.events"
lock_path="$output_dir.lock"

mkdir -p "$(dirname -- "$output_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another N6 LLM runtime controller owns this output" >&2
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

set_state preflight "compiler_only=true scheduler=not_invoked"
python3 "$preparer" verify-built \
    --plan-dir "$plan_dir" --repo-root "$repo_root"
if [[ -e $output_dir ]]; then
    echo "refusing existing N6 LLM runtime output: $output_dir" >&2
    exit 2
fi

mapfile -t policies < <(python3 -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print(*[x["name"] for x in p["policies"]], sep="\n")' \
    "$plan_dir/plan.json")
if [[ ${#policies[@]} -lt 1 || ${#policies[@]} -gt 9 ]]; then
    echo "runtime plan must contain 1..9 deduplicated policies" >&2
    exit 2
fi

artifacts=(
    "$plan_dir/plan.json" "$plan_dir/manifest.json"
    "$preparer" "$build_script" "$compiler_builder" "$evaluator"
    "$runner" "$controller" "$monitor" "$analyzer"
)
for name in "${policies[@]}"; do
    artifacts+=(
        "$plan_dir/policies/$name/hint.json"
        "$plan_dir/policies/$name/build/compiler_collective_eval"
        "$plan_dir/policies/$name/build/build-provenance.json"
    )
done
artifact_args=()
for artifact in "${artifacts[@]}"; do
    digest=$(sha256sum "$artifact")
    artifact_args+=(--artifact "$artifact=${digest%% *}")
done

mkdir -p "$output_dir"
monitor_paths=()
for replicate in 1 2 3; do
    mapfile -t order < <(python3 -c \
        'import json,sys; p=json.load(open(sys.argv[1])); print(*p["runtime_contract"]["policy_order_by_allocation"][sys.argv[2]], sep="\n")' \
        "$plan_dir/plan.json" "$replicate")
    if [[ ${#order[@]} -ne ${#policies[@]} ]]; then
        echo "allocation $replicate has an incomplete frozen policy order" >&2
        exit 2
    fi
    rep_dir="$output_dir/rep$replicate"
    mkdir -p "$rep_dir"
    : >"$rep_dir/driver.out"
    : >"$rep_dir/driver.err"
    for name in "${policies[@]}"; do
        : >"$rep_dir/$name.out"
        : >"$rep_dir/$name.err"
    done

    while true; do
        active_jobs=$(flux jobs --filter=active -n -o '{id.f58} {name}')
        if [[ -z $active_jobs ]]; then
            break
        fi
        count=$(printf '%s\n' "$active_jobs" | awk 'NF {n++} END {print n+0}')
        set_state waiting_scheduler_idle \
            "allocation=$replicate/3 active_user_jobs=$count"
        sleep 5
    done

    set_state submitting \
        "allocation=$replicate/3 queue=pdebug max_active_or_queued=1"
    job_id=$(flux batch -q pdebug -N6 -n48 -c8 -g1 -t 59m -u \
        --job-name="coll-n6-llm-runtime-r$replicate" --cwd="$rep_dir" \
        "$runner" "$plan_dir" "$rep_dir" "$replicate" "${order[@]}")
    if [[ -z $job_id || $job_id == *$'\n'* ]]; then
        echo "Flux returned an invalid N6 runtime job ID: $job_id" >&2
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
    for name in "${policies[@]}"; do
        monitor_args+=(
            --benchmark "$name=$rep_dir/$name.out"
            --benchmark-stderr "$name=$rep_dir/$name.err"
        )
    done
    set_state monitoring "allocation=$replicate/3 job=$job_id"
    python3 "$monitor" "${monitor_args[@]}" "${artifact_args[@]}"
done

set_state analyzing "$output_dir/analysis.json"
analysis_args=(
    --plan-dir "$plan_dir" --repo-root "$repo_root"
    --out "$output_dir/analysis.json"
)
for status in "${monitor_paths[@]}"; do
    analysis_args+=(--monitor "$status")
done
python3 "$analyzer" "${analysis_args[@]}"
stable=$(python3 -c \
    'import json,sys; print("yes" if json.load(open(sys.argv[1]))["claim_gates"]["stable_relational_modal_runtime_improvement_supported"] else "no")' \
    "$output_dir/analysis.json")
if [[ $stable == yes ]]; then
    set_state supported "stable relational modal beats both compiler controls"
else
    set_state negative "stable relational modal failed a preregistered gate"
fi
trap - EXIT
