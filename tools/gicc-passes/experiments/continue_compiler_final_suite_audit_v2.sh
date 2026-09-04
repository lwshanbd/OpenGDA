#!/usr/bin/env bash
# Refreeze a confirmed N6 suite and run the terminal compiler-only audits.
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
collective="$script_dir/collective"
portfolio="$repo_root/build_ofi/compiler_fact_coverage_20260904/portfolio"
metadata="$repo_root/build_ofi/compiler_fact_coverage_20260904"

original_suite="$repo_root/build_ofi/compiler_decision_suite_20260904/suite.json"
original_prompts="$repo_root/build_ofi/compiler_decision_suite_20260904/prompts"
n8_graph="$repo_root/build_ofi/compiler_collective_capacity_n8_hierpipe_v1_20260903/discovery/graph.json"
n6_graph="$repo_root/build_ofi/compiler_collective_capacity_n6_hierpipe_v2_20260904/discovery/graph.json"
structural_graph="$repo_root/build_ofi/compiler_comm_plan_placement_current_20260904/generated/opportunity-graph.json"

n6_chain_state="$repo_root/build_ofi/compiler_collective_n6_confirmation_v2_20260904.chain.state"
n6_scout_state="$repo_root/build_ofi/compiler_collective_hierpipe_n6_scout_v2_20260904.state"
n6_scout_analysis="$repo_root/build_ofi/compiler_collective_hierpipe_n6_scout_v2_20260904/analysis.json"
n6_confirmation="$repo_root/build_ofi/compiler_collective_n6_confirmation_v2_20260904"
n6_confirmation_state="$n6_confirmation.state"
n6_confirmation_analysis="$n6_confirmation/analysis.json"

producer_report="$repo_root/build_ofi/producer_fission_runtime_triage_9528fed_20260904/report.json"
producer_terminal_state="$repo_root/build_ofi/producer_fission_runtime_triage_9528fed_20260904.state"
producer_scout="$repo_root/build_ofi/producer_fission_oracle_scout_7687377_20260904_r2"
guarded_report="$repo_root/build_ofi/guarded_early_trigger_partial_negative_1323564_20260904/analysis.json"
guarded_scout="$repo_root/build_ofi/guarded_early_trigger_scout_1323564_20260904"
reused_scout="$repo_root/build_ofi/reused_loop_descriptor_scout_aff76f9_20260904"
reused_confirmation="$repo_root/build_ofi/reused_loop_descriptor_confirmation_aff76f9_20260904"

terminal_dir="$repo_root/build_ofi/compiler_terminal_negatives_v2_20260904"
terminal_report="$terminal_dir/report.json"
final_n6="$repo_root/build_ofi/compiler_final_n6_suite_refreeze_v2_20260904"
audit_dir="$repo_root/build_ofi/compiler_final_suite_audit_v2_20260904"
state_path="$audit_dir.chain.state"
events_path="$audit_dir.chain.events"
lock_path="$audit_dir.chain.lock"
successor="$script_dir/continue_compiler_final_suite_audit_v2.sh"

n6_refreezer="$collective/prepare_collective_n6_suite_refreeze.py"
terminal_auditor="$script_dir/audit_compiler_terminal_negatives.py"
readiness_auditor="$script_dir/audit_compiler_llm_readiness_terminal.py"
input_auditor="$script_dir/audit_compiler_input_separation.py"
authority_base="$script_dir/audit_compiler_action_authority.py"
authority_auditor="$script_dir/audit_compiler_action_authority_terminal.py"
null_auditor="$script_dir/audit_llm_sampling_null.py"
protocol_auditor="$script_dir/audit_compiler_llm_capability_protocol.py"
claim_auditor="$script_dir/audit_compiler_llm_mainline_claims.py"

# These paths intentionally remain absent: terminal-negative candidates are
# never materialized as graph expansions or suite refreezes.
producer_expansion="$repo_root/build_ofi/compiler_final_producer_graph_expansion_v2_20260904"
producer_refreeze="$repo_root/build_ofi/compiler_final_producer_suite_refreeze_v2_20260904"
guarded_expansion="$repo_root/build_ofi/compiler_final_guarded_graph_expansion_v2_20260904"
guarded_refreeze="$repo_root/build_ofi/compiler_final_guarded_suite_refreeze_v2_20260904"
reused_expansion="$repo_root/build_ofi/compiler_final_reused_graph_expansion_v2_20260904"
reused_refreeze="$repo_root/build_ofi/compiler_final_reused_suite_refreeze_v2_20260904"

exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another v2 final-suite audit successor is active" >&2
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
    local path=$1
    if [[ -f $path ]]; then
        awk 'NR == 1 {print $2}' "$path"
    fi
}

wait_for_phase() {
    local path=$1
    shift
    local phase= wanted
    while true; do
        phase=$(phase_of "$path")
        for wanted in "$@"; do
            if [[ $phase == "$wanted" ]]; then
                printf '%s\n' "$phase"
                return 0
            fi
        done
        sleep 5
    done
}

artifacts=(
    "$successor" "$script_dir/FINAL_SUITE_V2_PROTOCOL.md"
    "$terminal_auditor" "$readiness_auditor" "$n6_refreezer"
    "$script_dir/audit_compiler_llm_readiness.py"
    "$input_auditor" "$authority_base" "$authority_auditor"
    "$null_auditor" "$protocol_auditor" "$claim_auditor"
    "$script_dir/audit_compiler_action_frontier.py"
    "$script_dir/producer_fission/audit_producer_fission_runtime_triage.py"
    "$script_dir/guarded_early_trigger/audit_partial_negative_scout.py"
    "$script_dir/reused_loop_descriptor/analyze_reused_loop_descriptor_scout.py"
    "$original_suite" "$n8_graph" "$n6_graph" "$structural_graph"
    "$producer_report" "$producer_terminal_state" "$producer_scout.state"
    "$guarded_report" "$guarded_scout.state"
    "$reused_scout/analysis.json" "$reused_scout.state"
    "$repo_root/build_ofi/compiler_lto_eval/generated/gbt-history-report.json"
    "$repo_root/build_ofi/compiler_action_frontier_20260904/report.json"
    "$repo_root/build_ofi/historical_compiler_llm_ceiling_20260904/report.json"
    "$repo_root/docs/experiments/compiler-comm-plan-capacity/runs/compiler-comm-plan-placement-v1/summary.json"
    "$repo_root/build_ofi/compiler_comm_plan_placement/generated/opportunity-graph.json"
    "$portfolio/jacobi/group-graph.json"
    "$portfolio/mm_minimal/group-graph.json"
    "$portfolio/loop_lto/group-graph.json"
    "$portfolio/minimod/group-graph.json"
    "$portfolio/mixed_lto/group-graph.json"
    "$metadata/jacobi_disjoint/meta/_Z18jacobi_step_kernelILi32ELi32EEvPN4gicc9DeviceCtxEPfPKfS3_iiibiiiiimmmmm.json"
    "$metadata/mm_minimal_disjoint/meta/_Z18matmul_step_kernelPN4gicc9DeviceCtxEPKfS3_Pfiiiiiim.json"
    "$metadata/bench_pingpong_lto/meta/_Z15dwq_loop_kernelPN4gicc9DeviceCtxEiimi.json"
    "$repo_root/tools/gicc-passes/python/gicc_compiler_decision_suite.py"
    "$repo_root/tools/gicc-passes/python/gicc_collective_plan_bridge.py"
    "$repo_root/tools/gicc-passes/python/gicc_comm_group_plan_bridge.py"
    "$repo_root/tools/gicc-passes/python/gicc_comm_plan_bridge.py"
    "$repo_root/tools/gicc-passes/python/gicc_llm_bridge.py"
)
while IFS= read -r -d '' prompt; do
    artifacts+=("$prompt")
done < <(find "$original_prompts" -type f -print0 | sort -z)

hashes=()
for artifact in "${artifacts[@]}"; do
    if [[ ! -f $artifact ]]; then
        echo "missing frozen v2 final-suite artifact: $artifact" >&2
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
            echo "v2 final-suite artifact changed: ${artifacts[$index]}" >&2
            exit 2
        fi
    done
}

for output in "$terminal_dir" "$final_n6" "$audit_dir" \
        "$producer_expansion" "$producer_refreeze" \
        "$guarded_expansion" "$guarded_refreeze" \
        "$reused_expansion" "$reused_refreeze"; do
    if [[ -e $output ]]; then
        echo "refusing pre-existing v2 final-suite output: $output" >&2
        exit 2
    fi
done

set_state waiting_n6_confirmation "$n6_chain_state"
n6_phase=$(wait_for_phase \
    "$n6_chain_state" confirmed negative skipped \
    skipped_no_incremental_policy failed)
verify_artifacts

set_state auditing_terminal_negatives "$terminal_report"
python3 "$terminal_auditor" emit \
    --producer-report "$producer_report" \
    --producer-state "$producer_terminal_state" \
    --guarded-report "$guarded_report" \
    --guarded-state "$guarded_scout.state" \
    --guarded-output-dir "$guarded_scout" \
    --reused-analysis "$reused_scout/analysis.json" \
    --reused-state "$reused_scout.state" \
    --out "$terminal_report"
python3 "$terminal_auditor" verify-contained --report "$terminal_report"
verify_artifacts

current_suite=$original_suite
current_prompts=$original_prompts
collective_label=collective_n8
collective_graph=$n8_graph
collective_state="$repo_root/build_ofi/compiler_collective_hierpipe_n8_scout_20260903.state"
collective_analysis="$repo_root/build_ofi/compiler_collective_hierpipe_n8_scout_20260903/analysis.json"
collective_confirmation_state="$repo_root/build_ofi/compiler_collective_n8_confirmation_20260904.state"
collective_confirmation_analysis="$repo_root/build_ofi/compiler_collective_n8_confirmation_20260904/analysis.json"
collective_refreeze_manifest="$repo_root/build_ofi/compiler_collective_n8_suite_refreeze_20260904/manifest.json"

if [[ $n6_phase == confirmed ]]; then
    set_state replacing_collective_n6 "$final_n6"
    python3 "$n6_refreezer" prepare \
        --current-suite "$current_suite" \
        --current-prompt-dir "$current_prompts" \
        --confirmation-analysis "$n6_confirmation_analysis" \
        --n8-graph "$n8_graph" --n6-graph "$n6_graph" \
        --communication "jacobi=$portfolio/jacobi/group-graph.json" \
        --communication "loop_lto=$portfolio/loop_lto/group-graph.json" \
        --communication "minimod=$portfolio/minimod/group-graph.json" \
        --communication "mixed_lto=$portfolio/mixed_lto/group-graph.json" \
        --communication "mm_minimal=$portfolio/mm_minimal/group-graph.json" \
        --structural "coalescing_placement=$structural_graph" \
        --output-dir "$final_n6"
    python3 "$n6_refreezer" verify-contained \
        --manifest "$final_n6/manifest.json"
    current_suite="$final_n6/suite.json"
    current_prompts="$final_n6/prompts"
    collective_label=collective_n6
    collective_graph=$n6_graph
    collective_state=$n6_scout_state
    collective_analysis=$n6_scout_analysis
    collective_confirmation_state=$n6_confirmation_state
    collective_confirmation_analysis=$n6_confirmation_analysis
    collective_refreeze_manifest="$final_n6/manifest.json"
fi
verify_artifacts

mkdir "$audit_dir"
log_path="$audit_dir/audit.log"
set_state auditing "suite=$current_suite collective=$collective_label n6_chain=$n6_phase"

python3 "$input_auditor" emit \
    --suite "$current_suite" --prompt-dir "$current_prompts" \
    --gbt-report "$repo_root/build_ofi/compiler_lto_eval/generated/gbt-history-report.json" \
    --out "$audit_dir/input-separation.json" >>"$log_path" 2>&1

readiness_args=(
    --collective-label "$collective_label"
    --terminal-negatives "$terminal_report"
    --suite "$current_suite" --prompt-dir "$current_prompts"
    --graph-refreeze-finalizer "$successor"
    --placement-summary "$repo_root/docs/experiments/compiler-comm-plan-capacity/runs/compiler-comm-plan-placement-v1/summary.json"
    --placement-historical-graph "$repo_root/build_ofi/compiler_comm_plan_placement/generated/opportunity-graph.json"
    --placement-current-graph "$structural_graph"
    --collective-state "$collective_state"
    --collective-analysis "$collective_analysis"
    --collective-confirmation-state "$collective_confirmation_state"
    --collective-confirmation-analysis "$collective_confirmation_analysis"
    --collective-refreeze-manifest "$collective_refreeze_manifest"
    --producer-state "$producer_scout.state"
    --producer-analysis "$producer_scout/analysis.json"
    --producer-confirmation-state "$repo_root/build_ofi/producer_fission_confirmation_7687377_20260904_r2.state"
    --producer-confirmation-analysis "$repo_root/build_ofi/producer_fission_confirmation_7687377_20260904_r2/analysis.json"
    --producer-expansion-manifest "$producer_expansion/manifest.json"
    --producer-refreeze-manifest "$producer_refreeze/manifest.json"
    --guarded-state "$guarded_scout.state"
    --guarded-analysis "$guarded_scout/analysis.json"
    --guarded-confirmation-state "$repo_root/build_ofi/guarded_early_trigger_confirmation_1323564_20260904.state"
    --guarded-confirmation-analysis "$repo_root/build_ofi/guarded_early_trigger_confirmation_1323564_20260904/analysis.json"
    --guarded-expansion-manifest "$guarded_expansion/manifest.json"
    --guarded-refreeze-manifest "$guarded_refreeze/manifest.json"
    --reused-state "$reused_scout.state"
    --reused-analysis "$reused_scout/analysis.json"
    --reused-confirmation-state "$reused_confirmation.chain.state"
    --reused-confirmation-analysis "$reused_confirmation/analysis.json"
    --reused-expansion-manifest "$reused_expansion/manifest.json"
    --reused-refreeze-manifest "$reused_refreeze/manifest.json"
)
python3 "$readiness_auditor" emit "${readiness_args[@]}" \
    --out "$audit_dir/readiness.json" >>"$log_path" 2>&1

authority_args=(
    --collective-label "$collective_label"
    --terminal-negatives "$terminal_report"
    --suite "$current_suite" --prompt-dir "$current_prompts" \
    --gbt-report "$repo_root/build_ofi/compiler_lto_eval/generated/gbt-history-report.json"
    --input-separation "$audit_dir/input-separation.json"
    --communication "jacobi=$portfolio/jacobi/group-graph.json"
    --communication "loop_lto=$portfolio/loop_lto/group-graph.json"
    --communication "minimod=$portfolio/minimod/group-graph.json"
    --communication "mixed_lto=$portfolio/mixed_lto/group-graph.json"
    --communication "mm_minimal=$portfolio/mm_minimal/group-graph.json"
    --structural-graph "$structural_graph"
    --collective-graph "$collective_graph"
    --conditional-frontier "$repo_root/build_ofi/compiler_action_frontier_20260904/report.json"
)
python3 "$authority_auditor" emit "${authority_args[@]}" \
    --out "$audit_dir/action-authority.json" >>"$log_path" 2>&1

python3 "$null_auditor" \
    --suite "$current_suite" --prompt-dir "$current_prompts" \
    --action-authority "$audit_dir/action-authority.json" \
    --conditional-frontier "$repo_root/build_ofi/compiler_action_frontier_20260904/report.json" \
    --historical "$repo_root/build_ofi/historical_compiler_llm_ceiling_20260904/report.json" \
    --out "$audit_dir/sampling-null.json" >>"$log_path" 2>&1

python3 "$protocol_auditor" emit \
    --suite "$current_suite" --prompt-dir "$current_prompts" \
    --readiness "$audit_dir/readiness.json" \
    --input-separation "$audit_dir/input-separation.json" \
    --sampling-null "$audit_dir/sampling-null.json" \
    --out "$audit_dir/capability-protocol.json" >>"$log_path" 2>&1

claim_args=(
    --suite "$current_suite" --prompt-dir "$current_prompts"
    --terminal-negatives "$terminal_report"
    --readiness "$audit_dir/readiness.json"
    --action-authority "$audit_dir/action-authority.json"
    --input-separation "$audit_dir/input-separation.json"
    --sampling-null "$audit_dir/sampling-null.json"
    --capability-protocol "$audit_dir/capability-protocol.json"
    --n6-chain-state "$n6_chain_state"
)
python3 "$claim_auditor" emit "${claim_args[@]}" \
    --out "$audit_dir/paper-mainline.json" >>"$log_path" 2>&1

python3 "$input_auditor" verify \
    --suite "$current_suite" --prompt-dir "$current_prompts" \
    --gbt-report "$repo_root/build_ofi/compiler_lto_eval/generated/gbt-history-report.json" \
    --report "$audit_dir/input-separation.json" >>"$log_path" 2>&1
python3 "$readiness_auditor" verify "${readiness_args[@]}" \
    --report "$audit_dir/readiness.json" >>"$log_path" 2>&1
python3 "$authority_auditor" verify "${authority_args[@]}" \
    --report "$audit_dir/action-authority.json" >>"$log_path" 2>&1
python3 "$protocol_auditor" verify \
    --suite "$current_suite" --prompt-dir "$current_prompts" \
    --readiness "$audit_dir/readiness.json" \
    --input-separation "$audit_dir/input-separation.json" \
    --sampling-null "$audit_dir/sampling-null.json" \
    --report "$audit_dir/capability-protocol.json" >>"$log_path" 2>&1
python3 "$claim_auditor" verify "${claim_args[@]}" \
    --report "$audit_dir/paper-mainline.json" >>"$log_path" 2>&1
python3 "$terminal_auditor" verify-contained \
    --report "$terminal_report" >>"$log_path" 2>&1
verify_artifacts

ids=$(python3 -c \
    'import json,sys; specs=(("terminal_id",sys.argv[1],"terminal_negatives_id"),("readiness_id",sys.argv[2],"readiness_id"),("authority_id",sys.argv[3],"authority_id"),("null_id",sys.argv[4],"null_id"),("protocol_id",sys.argv[5],"protocol_id"),("claim_id",sys.argv[6],"claim_audit_id")); print(" ".join(name+"="+json.load(open(path))[key] for name,path,key in specs))' \
    "$terminal_report" "$audit_dir/readiness.json" \
    "$audit_dir/action-authority.json" "$audit_dir/sampling-null.json" \
    "$audit_dir/capability-protocol.json" "$audit_dir/paper-mainline.json")
set_state complete \
    "collective=$collective_label n6_chain=$n6_phase suite=$current_suite $ids"
trap - EXIT
