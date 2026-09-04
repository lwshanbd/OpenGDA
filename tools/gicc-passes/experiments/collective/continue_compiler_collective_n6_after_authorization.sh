#!/usr/bin/env bash
# Complete the authorized N6 LLM-to-LTO pipeline, one call/job at a time.
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 PRIORITY_REQUEST_ROOT AUTHORIZATION N6_BUNDLE N6_CONFIRMATION OUTPUT_ROOT" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
experiment_dir=$(cd -- "$script_dir/.." && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
priority_root=$(cd -- "$1" && pwd)
authorization=$(cd -- "$(dirname -- "$2")" && pwd)/$(basename -- "$2")
n6_bundle=$(cd -- "$3" && pwd)
n6_confirmation=$(cd -- "$(dirname -- "$4")" && pwd)/$(basename -- "$4")
case $5 in /*) output_root=$5 ;; *) output_root="$repo_root/$5" ;; esac
selection="$priority_root/selection.json"
request_dir="$priority_root/request"
request_json="$request_dir/request.json"
authorization_preflight="$output_root/authorization-preflight.json"
archive_dir="$output_root/archive"
control_dir="$output_root/hidden-controls"
analysis_path="$output_root/capability-analysis.json"
plan_dir="$output_root/runtime-plan"
runtime_dir="$output_root/paired-runtime"
paper_audit="$output_root/paper-claim-audit.json"
state_path="$output_root.chain.state"
events_path="$output_root.chain.events"
lock_path="$output_root.chain.lock"
trial_runner="$experiment_dir/run_compiler_llm_capability_trials.py"
preflight="$experiment_dir/preflight_compiler_llm_capability_authorization.py"
capability_analyzer="$experiment_dir/analyze_compiler_llm_capability_trials.py"
selector="$experiment_dir/freeze_compiler_llm_priority_request.py"
screen_controller="$script_dir/continue_compiler_collective_n6_llm_controls.sh"
screen_adapter="$script_dir/prepare_collective_n6_llm_policy_screen.py"
runtime_preparer="$script_dir/prepare_collective_n6_llm_runtime_validation.py"
runtime_builder="$script_dir/build_collective_n6_llm_runtime_validation.sh"
runtime_controller="$script_dir/continue_collective_n6_llm_runtime_validation.sh"
runtime_analyzer="$script_dir/analyze_collective_n6_llm_runtime_validation.py"
paper_auditor="$experiment_dir/audit_collective_n6_llm_paper_claims.py"
successor="$script_dir/continue_compiler_collective_n6_after_authorization.sh"

mkdir -p "$(dirname -- "$output_root")"
exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another authorized N6 LLM pipeline owns this output" >&2
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
        set_state failed "successor_exit=$status"
    fi
}
trap on_exit EXIT

for required in "$selection" "$request_json" "$authorization" \
        "$n6_confirmation"; do
    if [[ ! -f $required ]]; then
        echo "missing authorized-pipeline input: $required" >&2
        exit 2
    fi
done

input_lines=$(python3 -c '
import hashlib,json,pathlib,sys
sys.path.insert(0, sys.argv[1])
import freeze_compiler_llm_priority_request as freeze
repo = pathlib.Path(sys.argv[2]).resolve()
selection_path = pathlib.Path(sys.argv[3])
request_path = pathlib.Path(sys.argv[4])
selected = freeze.verify_selection(json.loads(selection_path.read_text()))
entry = selected.get("selected_entry")
if not isinstance(entry, dict) or entry.get("label") != "collective_n6":
    raise SystemExit("priority selection is not collective_n6")
request = json.loads(request_path.read_text())
if request.get("request_id") != selected.get("request_id"):
    raise SystemExit("priority selection and request ID differ")
if request.get("label") != "collective_n6":
    raise SystemExit("request label is not collective_n6")
for role in ("suite", "readiness", "input_separation", "sampling_null", "capability_protocol", "source_graph"):
    record = request.get("evidence", {}).get(role, {})
    path = pathlib.Path(record.get("path", ""))
    path = path if path.is_absolute() else repo / path
    path = path.resolve()
    if (not path.is_file()
            or path.stat().st_size != record.get("bytes")
            or hashlib.sha256(path.read_bytes()).hexdigest() != record.get("sha256")):
        raise SystemExit(f"request evidence changed: {role}")
    print(path)
' "$experiment_dir" "$repo_root" "$selection" "$request_json")
mapfile -t inputs <<<"$input_lines"
if [[ ${#inputs[@]} -ne 6 ]]; then
    echo "priority request did not resolve six exact evidence inputs" >&2
    exit 2
fi
suite=${inputs[0]}
readiness=${inputs[1]}
input_separation=${inputs[2]}
sampling_null=${inputs[3]}
capability_protocol=${inputs[4]}
graph=${inputs[5]}
prompt_dir="$(dirname -- "$suite")/prompts"

artifacts=(
    "$successor" "$selector" "$selection" "$request_json" "$authorization"
    "$suite" "$readiness" "$input_separation" "$sampling_null"
    "$capability_protocol" "$graph" "$n6_confirmation"
    "$n6_bundle/FROZEN_V3_MANIFEST.json"
    "$n6_bundle/discovery/graph.json"
    "$n6_bundle/controls/manifest.json"
    "$preflight" "$trial_runner" "$capability_analyzer"
    "$screen_controller" "$screen_adapter"
    "$runtime_preparer" "$runtime_builder"
    "$runtime_controller" "$runtime_analyzer" "$paper_auditor"
)
hashes=()
for artifact in "${artifacts[@]}"; do
    if [[ ! -f $artifact ]]; then
        echo "missing frozen authorized-pipeline artifact: $artifact" >&2
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
            echo "authorized-pipeline artifact changed: ${artifacts[$index]}" >&2
            exit 2
        fi
    done
}

common_args=(
    --suite "$suite" --prompt-dir "$prompt_dir"
    --readiness "$readiness" --input-separation "$input_separation"
    --sampling-null "$sampling_null"
    --capability-protocol "$capability_protocol"
    --label collective_n6 --graph "$graph"
    --request-dir "$request_dir" --authorization "$authorization"
)

preflight_args=("${common_args[@]}" --out "$authorization_preflight")
if [[ -f $authorization_preflight ]]; then
    set_state verifying_exact_authorization "$authorization_preflight"
    python3 "$preflight" verify "${preflight_args[@]}"
else
    set_state preflighting_exact_authorization \
        "request, delivery, boundary, provider version; inference=false"
    python3 "$preflight" emit "${preflight_args[@]}"
fi
digest=$(sha256sum "$authorization_preflight")
artifacts+=("$authorization_preflight")
hashes+=("${digest%% *}")
verify_artifacts

set_state running_model_archive \
    "request_id=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["request_id"])' "$request_json") calls=strictly_serial"
verify_artifacts
python3 "$trial_runner" "${common_args[@]}" --output-dir "$archive_dir"
verify_artifacts

if [[ -f $control_dir/policy-screen.json ]]; then
    set_state reusing_hidden_controls "$control_dir/policy-screen.json"
elif [[ -e $control_dir ]]; then
    echo "refusing partial hidden-control output: $control_dir" >&2
    exit 2
else
    set_state running_hidden_controls \
        "one N6 pdebug allocation; scheduler serialization delegated"
    "$screen_controller" "$n6_bundle" "$request_dir" "$archive_dir" \
        "$n6_confirmation" "$control_dir"
fi
verify_artifacts

analysis_args=(
    "${common_args[@]}" --archive-dir "$archive_dir"
    --policy-screen "$control_dir/policy-screen.json"
    --repo-root "$repo_root"
)
if [[ -f $analysis_path ]]; then
    set_state verifying_capability_analysis "$analysis_path"
    python3 "$capability_analyzer" verify "${analysis_args[@]}" \
        --report "$analysis_path"
else
    set_state analyzing_capability_archive "$analysis_path"
    python3 "$capability_analyzer" emit "${analysis_args[@]}" \
        --out "$analysis_path"
fi
verify_artifacts

plan_args=(
    "${analysis_args[@]}" --analysis "$analysis_path"
)
if [[ -f $plan_dir/manifest.json ]]; then
    set_state verifying_lto_policies "$plan_dir/manifest.json"
    python3 "$runtime_preparer" verify-built \
        --plan-dir "$plan_dir" --repo-root "$repo_root"
elif [[ -e $plan_dir ]]; then
    echo "refusing partial LTO policy bundle: $plan_dir" >&2
    exit 2
else
    set_state preparing_lto_policies \
        "deduplicate then materialize graph-bound compiler IDs"
    python3 "$runtime_preparer" prepare "${plan_args[@]}" \
        --output-dir "$plan_dir"
    "$runtime_builder" "$plan_dir"
fi
verify_artifacts

if [[ -f $runtime_dir/analysis.json ]]; then
    set_state verifying_paired_runtime "$runtime_dir/analysis.json"
    runtime_verify_args=(
        --plan-dir "$plan_dir" --repo-root "$repo_root"
        --out "$runtime_dir/analysis.json" --verify
    )
    for replicate in 1 2 3; do
        runtime_verify_args+=(--monitor "$runtime_dir/rep$replicate/monitor.json")
    done
    python3 "$runtime_analyzer" "${runtime_verify_args[@]}"
elif [[ -e $runtime_dir ]]; then
    echo "refusing partial paired-runtime output: $runtime_dir" >&2
    exit 2
else
    set_state running_paired_runtime \
        "three independent N6 pdebug allocations; maximum queued=1"
    "$runtime_controller" "$plan_dir" "$runtime_dir"
fi
verify_artifacts

paper_args=(
    --capability-analysis "$analysis_path"
    --runtime-analysis "$runtime_dir/analysis.json"
    --repo-root "$repo_root" --out "$paper_audit"
)
if [[ -f $paper_audit ]]; then
    set_state verifying_paper_claims "$paper_audit"
    python3 "$paper_auditor" verify "${paper_args[@]}"
else
    set_state auditing_paper_claims "$paper_audit"
    python3 "$paper_auditor" emit "${paper_args[@]}"
fi
verify_artifacts

detail=$(python3 -c '
import json,sys
x=json.load(open(sys.argv[1]))
g=x["claim_matrix"]
print("status="+x["status"]+" stable="+str(g["stable_relational_modal_runtime_improvement_supported"]).lower()+" context="+str(g["relational_context_runtime_effect_supported"]).lower()+" ceiling="+str(g["posthoc_capability_ceiling_runtime_potential_observed"]).lower()+" generalization=false")
' "$paper_audit")
set_state complete "$detail"
trap - EXIT
