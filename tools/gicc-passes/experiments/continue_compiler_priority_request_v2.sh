#!/usr/bin/env bash
# Freeze at most one exact compiler-only request after the v2 final audit.
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
audit_dir="$repo_root/build_ofi/compiler_final_suite_audit_v2_20260904"
audit_state="$audit_dir.chain.state"
output_dir="$repo_root/build_ofi/compiler_llm_priority_request_v2_20260904"
state_path="$output_dir.chain.state"
events_path="$output_dir.chain.events"
lock_path="$output_dir.chain.lock"
selector="$script_dir/freeze_compiler_llm_priority_request.py"
successor="$script_dir/continue_compiler_priority_request_v2.sh"
portfolio="$repo_root/build_ofi/compiler_fact_coverage_20260904/portfolio"

exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another v2 priority-request successor is active" >&2
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

phase_of() {
    if [[ -f $1 ]]; then
        awk 'NR == 1 {print $2}' "$1"
    fi
}

artifacts=(
    "$successor" "$selector" "$script_dir/FINAL_SUITE_V2_PROTOCOL.md"
    "$script_dir/prepare_compiler_llm_capability_request.py"
    "$script_dir/run_compiler_llm_capability_trials.py"
    "$script_dir/analyze_compiler_llm_capability_trials.py"
    "$script_dir/audit_compiler_llm_capability_protocol.py"
    "$script_dir/audit_compiler_llm_readiness.py"
    "$script_dir/audit_compiler_llm_readiness_terminal.py"
    "$script_dir/audit_compiler_llm_mainline_claims.py"
    "$script_dir/audit_compiler_terminal_negatives.py"
    "$script_dir/audit_compiler_input_separation.py"
    "$script_dir/audit_compiler_action_authority.py"
    "$script_dir/audit_compiler_action_authority_terminal.py"
    "$script_dir/audit_llm_sampling_null.py"
    "$repo_root/tools/gicc-passes/python/gicc_compiler_policy_bridge.py"
    "$repo_root/tools/gicc-passes/python/gicc_compiler_decision_suite.py"
    "$repo_root/tools/gicc-passes/python/gicc_llm_bridge.py"
)
hashes=()
for artifact in "${artifacts[@]}"; do
    if [[ ! -f $artifact ]]; then
        echo "missing frozen v2 request artifact: $artifact" >&2
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
            echo "v2 priority-request artifact changed: ${artifacts[$index]}" >&2
            exit 2
        fi
    done
}

if [[ -e $output_dir ]]; then
    echo "refusing pre-existing v2 priority-request output: $output_dir" >&2
    exit 2
fi
set_state waiting_final_audit "$audit_state"
audit_phase=
while [[ -z $audit_phase ]]; do
    audit_phase=$(phase_of "$audit_state")
    case $audit_phase in
        complete|failed) ;;
        *) audit_phase=; sleep 5 ;;
    esac
done
if [[ $audit_phase != complete ]]; then
    echo "v2 final suite audit failed" >&2
    exit 2
fi
verify_artifacts

suite=$(python3 -c \
    'import json,sys,pathlib; x=json.load(open(sys.argv[1])); p=x["evidence"]["suite"]["path"]; root=pathlib.Path(sys.argv[2]); print(pathlib.Path(p) if pathlib.Path(p).is_absolute() else root/p)' \
    "$audit_dir/readiness.json" "$repo_root")
prompt_dir="$(dirname -- "$suite")/prompts"

resolve_graph() {
    local label=$1 expected candidate digest matches=0 selected=
    shift
    expected=$(python3 -c \
        'import json,sys; x=json.load(open(sys.argv[1])); e={i["label"]:i for i in x["entries"]}; print(e[sys.argv[2]]["graph_file_sha256"])' \
        "$suite" "$label")
    for candidate in "$@"; do
        if [[ -f $candidate ]]; then
            digest=$(sha256sum "$candidate")
            digest=${digest%% *}
            if [[ $digest == "$expected" ]]; then
                matches=$((matches + 1))
                selected=$candidate
            fi
        fi
    done
    if [[ $matches -ne 1 ]]; then
        echo "$label graph resolution has $matches exact matches" >&2
        return 2
    fi
    printf '%s\n' "$selected"
}

jacobi=$(resolve_graph jacobi "$portfolio/jacobi/group-graph.json")
mm=$(resolve_graph mm_minimal "$portfolio/mm_minimal/group-graph.json")
loop=$(resolve_graph loop_lto "$portfolio/loop_lto/group-graph.json")
minimod=$(resolve_graph minimod "$portfolio/minimod/group-graph.json")
mixed=$(resolve_graph mixed_lto "$portfolio/mixed_lto/group-graph.json")
structural=$(resolve_graph coalescing_placement \
    "$repo_root/build_ofi/compiler_comm_plan_placement_current_20260904/generated/opportunity-graph.json")
collective_label=$(python3 -c \
    'import json,sys; labels={x["label"] for x in json.load(open(sys.argv[1]))["entries"]}; print("collective_n6" if "collective_n6" in labels else "collective_n8")' \
    "$suite")
collective=$(resolve_graph "$collective_label" \
    "$repo_root/build_ofi/compiler_collective_capacity_n8_hierpipe_v1_20260903/discovery/graph.json" \
    "$repo_root/build_ofi/compiler_collective_capacity_n6_hierpipe_v2_20260904/discovery/graph.json")
verify_artifacts

set_state freezing_one_request "suite=$suite collective=$collective_label"
python3 "$selector" freeze \
    --suite "$suite" --prompt-dir "$prompt_dir" \
    --readiness "$audit_dir/readiness.json" \
    --input-separation "$audit_dir/input-separation.json" \
    --sampling-null "$audit_dir/sampling-null.json" \
    --capability-protocol "$audit_dir/capability-protocol.json" \
    --graph "jacobi=$jacobi" --graph "mm_minimal=$mm" \
    --graph "loop_lto=$loop" --graph "minimod=$minimod" \
    --graph "mixed_lto=$mixed" --graph "coalescing_placement=$structural" \
    --graph "$collective_label=$collective" \
    --output-dir "$output_dir"
python3 "$selector" verify \
    --suite "$suite" --prompt-dir "$prompt_dir" \
    --readiness "$audit_dir/readiness.json" \
    --input-separation "$audit_dir/input-separation.json" \
    --sampling-null "$audit_dir/sampling-null.json" \
    --capability-protocol "$audit_dir/capability-protocol.json" \
    --graph "jacobi=$jacobi" --graph "mm_minimal=$mm" \
    --graph "loop_lto=$loop" --graph "minimod=$minimod" \
    --graph "mixed_lto=$mixed" --graph "coalescing_placement=$structural" \
    --graph "$collective_label=$collective" \
    --request-root "$output_dir"
verify_artifacts

detail=$(python3 -c \
    'import json,sys; x=json.load(open(sys.argv[1])); s=x["selected_entry"]; print("selected="+(s["label"] if s else "none")+" selection_id="+x["selection_id"]+" request_id="+(x["request_id"] or "none"))' \
    "$output_dir/selection.json")
set_state complete "$detail"
trap - EXIT
