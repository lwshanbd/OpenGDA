#!/usr/bin/env bash
# Merge recovered compiler headroom, refreeze N6 when confirmed, and audit it.
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../.." && pwd)
producer="$script_dir/producer_fission"
guarded="$script_dir/guarded_early_trigger"
reused="$script_dir/reused_loop_descriptor"
collective="$script_dir/collective"

original_suite="$repo_root/build_ofi/compiler_decision_suite_20260904/suite.json"
original_prompts="$repo_root/build_ofi/compiler_decision_suite_20260904/prompts"
portfolio="$repo_root/build_ofi/compiler_fact_coverage_20260904/portfolio"
metadata="$repo_root/build_ofi/compiler_fact_coverage_20260904"
n8_graph="$repo_root/build_ofi/compiler_collective_capacity_n8_hierpipe_v1_20260903/discovery/graph.json"
n6_graph="$repo_root/build_ofi/compiler_collective_capacity_n6_hierpipe_v1_20260904/discovery/graph.json"
structural_graph="$repo_root/build_ofi/compiler_comm_plan_placement_current_20260904/generated/opportunity-graph.json"

recovery_state="$repo_root/build_ofi/compiler_headroom_recovery_20260904.state"
n6_chain_state="$repo_root/build_ofi/compiler_collective_n6_confirmation_20260904.chain.state"
n6_confirmation="$repo_root/build_ofi/compiler_collective_n6_confirmation_20260904"

producer_scout="$repo_root/build_ofi/producer_fission_oracle_scout_7687377_20260904_r2"
producer_confirmation="$repo_root/build_ofi/producer_fission_confirmation_7687377_20260904_r2"
producer_expansion="$repo_root/build_ofi/compiler_final_producer_graph_expansion_20260904"
producer_refreeze="$repo_root/build_ofi/compiler_final_producer_suite_refreeze_20260904"

guarded_scout="$repo_root/build_ofi/guarded_early_trigger_scout_77897d9_20260904_r2"
guarded_confirmation="$repo_root/build_ofi/guarded_early_trigger_confirmation_77897d9_20260904_r2"
guarded_expansion="$repo_root/build_ofi/compiler_final_guarded_graph_expansion_20260904"
guarded_refreeze="$repo_root/build_ofi/compiler_final_guarded_suite_refreeze_20260904"

reused_scout="$repo_root/build_ofi/reused_loop_descriptor_scout_aff76f9_20260904"
reused_confirmation="$repo_root/build_ofi/reused_loop_descriptor_confirmation_aff76f9_20260904"
reused_expansion="$repo_root/build_ofi/compiler_final_reused_graph_expansion_20260904"
reused_refreeze="$repo_root/build_ofi/compiler_final_reused_suite_refreeze_20260904"

final_n6="$repo_root/build_ofi/compiler_final_n6_suite_refreeze_20260904"
audit_dir="$repo_root/build_ofi/compiler_final_suite_audit_20260904"
state_path="$audit_dir.chain.state"
events_path="$audit_dir.chain.events"
lock_path="$audit_dir.chain.lock"
successor="$script_dir/continue_compiler_final_suite_audit.sh"

exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another final-suite audit successor is active" >&2
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

producer_expander="$producer/prepare_confirmed_producer_fission_graph.py"
producer_refreezer="$producer/prepare_producer_fission_suite_refreeze.py"
guarded_expander="$guarded/prepare_confirmed_guarded_early_graph.py"
guarded_refreezer="$guarded/prepare_guarded_early_suite_refreeze.py"
reused_expander="$reused/prepare_confirmed_reused_loop_descriptor_graph.py"
reused_refreezer="$reused/prepare_reused_loop_descriptor_suite_refreeze.py"
n6_refreezer="$collective/prepare_collective_n6_suite_refreeze.py"
readiness_base="$script_dir/audit_compiler_llm_readiness.py"
readiness_n6="$script_dir/audit_compiler_llm_readiness_n6.py"
input_auditor="$script_dir/audit_compiler_input_separation.py"
authority_base="$script_dir/audit_compiler_action_authority.py"
authority_n6="$script_dir/audit_compiler_action_authority_n6.py"
null_auditor="$script_dir/audit_llm_sampling_null.py"
protocol_auditor="$script_dir/audit_compiler_llm_capability_protocol.py"

artifacts=(
    "$successor" "$producer_expander" "$producer_refreezer"
    "$guarded_expander" "$guarded_refreezer"
    "$reused_expander" "$reused_refreezer" "$n6_refreezer"
    "$readiness_base" "$readiness_n6" "$input_auditor"
    "$authority_base" "$authority_n6" "$null_auditor" "$protocol_auditor"
    "$script_dir/audit_compiler_action_frontier.py"
    "$original_suite" "$n8_graph" "$n6_graph" "$structural_graph"
    "$repo_root/build_ofi/compiler_lto_eval/generated/gbt-history-report.json"
    "$repo_root/build_ofi/compiler_action_frontier_20260904/report.json"
    "$repo_root/build_ofi/historical_compiler_llm_ceiling_20260904/report.json"
    "$repo_root/docs/experiments/compiler-comm-plan-capacity/runs/compiler-comm-plan-placement-v1/summary.json"
    "$repo_root/build_ofi/compiler_comm_plan_placement/generated/opportunity-graph.json"
    "$portfolio/jacobi/dossier.json" "$portfolio/jacobi/group-graph.json"
    "$portfolio/mm_minimal/dossier.json" "$portfolio/mm_minimal/group-graph.json"
    "$portfolio/loop_lto/dossier.json" "$portfolio/loop_lto/group-graph.json"
    "$portfolio/minimod/group-graph.json" "$portfolio/mixed_lto/group-graph.json"
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
        echo "missing frozen final-suite artifact: $artifact" >&2
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
            echo "final-suite artifact changed: ${artifacts[$index]}" >&2
            exit 2
        fi
    done
}

for output in "$producer_expansion" "$producer_refreeze" \
        "$guarded_expansion" "$guarded_refreeze" \
        "$reused_expansion" "$reused_refreeze" "$final_n6" "$audit_dir"; do
    if [[ -e $output ]]; then
        echo "refusing pre-existing final-suite output: $output" >&2
        exit 2
    fi
done

set_state waiting_runtime_recovery "$recovery_state"
recovery_phase=$(wait_for_phase "$recovery_state" complete failed)
if [[ $recovery_phase != complete ]]; then
    echo "runtime recovery did not complete" >&2
    exit 2
fi
set_state waiting_n6_confirmation "$n6_chain_state"
n6_phase=$(wait_for_phase \
    "$n6_chain_state" confirmed negative skipped \
    skipped_no_incremental_policy failed)
verify_artifacts

current_suite=$original_suite
current_prompts=$original_prompts
jacobi_graph="$portfolio/jacobi/group-graph.json"
mm_graph="$portfolio/mm_minimal/group-graph.json"
loop_graph="$portfolio/loop_lto/group-graph.json"
producer_phase=$(phase_of "$producer_confirmation.state")
guarded_phase=$(phase_of "$guarded_confirmation.chain.state")
reused_phase=$(phase_of "$reused_confirmation.chain.state")

if [[ $producer_phase == confirmed ]]; then
    set_state merging_producer "$producer_refreeze"
    python3 "$producer_expander" prepare \
        --confirmation-analysis "$producer_confirmation/analysis.json" \
        --dossier "$portfolio/jacobi/dossier.json" \
        --template "$metadata/jacobi_disjoint/meta/_Z18jacobi_step_kernelILi32ELi32EEvPN4gicc9DeviceCtxEPfPKfS3_iiibiiiiimmmmm.json" \
        --graph "$jacobi_graph" --output-dir "$producer_expansion"
    python3 "$producer_expander" verify-contained \
        --manifest "$producer_expansion/manifest.json"
    python3 "$producer_refreezer" prepare \
        --current-suite "$current_suite" --current-prompt-dir "$current_prompts" \
        --expansion-manifest "$producer_expansion/manifest.json" \
        --communication "minimod=$portfolio/minimod/group-graph.json" \
        --communication "mm_minimal=$mm_graph" \
        --communication "mixed_lto=$portfolio/mixed_lto/group-graph.json" \
        --communication "loop_lto=$loop_graph" \
        --collective "collective_n8=$n8_graph" \
        --structural "coalescing_placement=$structural_graph" \
        --output-dir "$producer_refreeze"
    python3 "$producer_refreezer" verify-contained \
        --manifest "$producer_refreeze/manifest.json"
    current_suite="$producer_refreeze/suite.json"
    current_prompts="$producer_refreeze/prompts"
    jacobi_graph="$producer_expansion/expanded-group-graph.json"
fi
verify_artifacts

if [[ $guarded_phase == confirmed ]]; then
    set_state merging_guarded "$guarded_refreeze"
    python3 "$guarded_expander" prepare \
        --confirmation-analysis "$guarded_confirmation/analysis.json" \
        --dossier "$portfolio/mm_minimal/dossier.json" \
        --template "$metadata/mm_minimal_disjoint/meta/_Z18matmul_step_kernelPN4gicc9DeviceCtxEPKfS3_Pfiiiiiim.json" \
        --graph "$mm_graph" --output-dir "$guarded_expansion"
    python3 "$guarded_expander" verify-contained \
        --manifest "$guarded_expansion/manifest.json"
    python3 "$guarded_refreezer" prepare \
        --current-suite "$current_suite" --current-prompt-dir "$current_prompts" \
        --expansion-manifest "$guarded_expansion/manifest.json" \
        --communication "jacobi=$jacobi_graph" \
        --communication "minimod=$portfolio/minimod/group-graph.json" \
        --communication "mixed_lto=$portfolio/mixed_lto/group-graph.json" \
        --communication "loop_lto=$loop_graph" \
        --collective "collective_n8=$n8_graph" \
        --structural "coalescing_placement=$structural_graph" \
        --output-dir "$guarded_refreeze"
    python3 "$guarded_refreezer" verify-contained \
        --manifest "$guarded_refreeze/manifest.json"
    current_suite="$guarded_refreeze/suite.json"
    current_prompts="$guarded_refreeze/prompts"
    mm_graph="$guarded_expansion/expanded-group-graph.json"
fi
verify_artifacts

if [[ $reused_phase == confirmed ]]; then
    set_state merging_reused "$reused_refreeze"
    python3 "$reused_expander" prepare \
        --confirmation-analysis "$reused_confirmation/analysis.json" \
        --dossier "$portfolio/loop_lto/dossier.json" \
        --template "$metadata/bench_pingpong_lto/meta/_Z15dwq_loop_kernelPN4gicc9DeviceCtxEiimi.json" \
        --graph "$loop_graph" --output-dir "$reused_expansion"
    python3 "$reused_expander" verify-contained \
        --manifest "$reused_expansion/manifest.json"
    python3 "$reused_refreezer" prepare \
        --current-suite "$current_suite" --current-prompt-dir "$current_prompts" \
        --expansion-manifest "$reused_expansion/manifest.json" \
        --communication "jacobi=$jacobi_graph" \
        --communication "minimod=$portfolio/minimod/group-graph.json" \
        --communication "mm_minimal=$mm_graph" \
        --communication "mixed_lto=$portfolio/mixed_lto/group-graph.json" \
        --collective "collective_n8=$n8_graph" \
        --structural "coalescing_placement=$structural_graph" \
        --output-dir "$reused_refreeze"
    python3 "$reused_refreezer" verify-contained \
        --manifest "$reused_refreeze/manifest.json"
    current_suite="$reused_refreeze/suite.json"
    current_prompts="$reused_refreeze/prompts"
    loop_graph="$reused_expansion/expanded-group-graph.json"
fi
verify_artifacts

collective_label=collective_n8
collective_graph=$n8_graph
readiness_script=$readiness_base
collective_state="$repo_root/build_ofi/compiler_collective_hierpipe_n8_scout_20260903.state"
collective_analysis="$repo_root/build_ofi/compiler_collective_hierpipe_n8_scout_20260903/analysis.json"
collective_confirmation_state="$repo_root/build_ofi/compiler_collective_n8_confirmation_20260904.state"
collective_confirmation_analysis="$repo_root/build_ofi/compiler_collective_n8_confirmation_20260904/analysis.json"
collective_refreeze_manifest="$repo_root/build_ofi/compiler_collective_n8_suite_refreeze_20260904/manifest.json"
authority_script=$authority_base
if [[ $n6_phase == confirmed ]]; then
    set_state replacing_collective_n6 "$final_n6"
    python3 "$n6_refreezer" prepare \
        --current-suite "$current_suite" --current-prompt-dir "$current_prompts" \
        --confirmation-analysis "$n6_confirmation/analysis.json" \
        --n8-graph "$n8_graph" --n6-graph "$n6_graph" \
        --communication "jacobi=$jacobi_graph" \
        --communication "loop_lto=$loop_graph" \
        --communication "minimod=$portfolio/minimod/group-graph.json" \
        --communication "mixed_lto=$portfolio/mixed_lto/group-graph.json" \
        --communication "mm_minimal=$mm_graph" \
        --structural "coalescing_placement=$structural_graph" \
        --output-dir "$final_n6"
    python3 "$n6_refreezer" verify-contained \
        --manifest "$final_n6/manifest.json"
    current_suite="$final_n6/suite.json"
    current_prompts="$final_n6/prompts"
    collective_label=collective_n6
    collective_graph=$n6_graph
    readiness_script=$readiness_n6
    collective_state="$repo_root/build_ofi/compiler_collective_hierpipe_n6_scout_20260904.state"
    collective_analysis="$repo_root/build_ofi/compiler_collective_hierpipe_n6_scout_20260904/analysis.json"
    collective_confirmation_state="$n6_confirmation.state"
    collective_confirmation_analysis="$n6_confirmation/analysis.json"
    collective_refreeze_manifest="$final_n6/manifest.json"
    authority_script=$authority_n6
fi
verify_artifacts

mkdir "$audit_dir"
log_path="$audit_dir/audit.log"
set_state auditing "suite=$current_suite collective=$collective_label"
python3 "$input_auditor" emit \
    --suite "$current_suite" --prompt-dir "$current_prompts" \
    --gbt-report "$repo_root/build_ofi/compiler_lto_eval/generated/gbt-history-report.json" \
    --out "$audit_dir/input-separation.json" >>"$log_path" 2>&1

python3 "$readiness_script" emit \
    --suite "$current_suite" --prompt-dir "$current_prompts" \
    --graph-refreeze-finalizer "$successor" \
    --placement-summary "$repo_root/docs/experiments/compiler-comm-plan-capacity/runs/compiler-comm-plan-placement-v1/summary.json" \
    --placement-historical-graph "$repo_root/build_ofi/compiler_comm_plan_placement/generated/opportunity-graph.json" \
    --placement-current-graph "$structural_graph" \
    --collective-state "$collective_state" \
    --collective-analysis "$collective_analysis" \
    --collective-confirmation-state "$collective_confirmation_state" \
    --collective-confirmation-analysis "$collective_confirmation_analysis" \
    --collective-refreeze-manifest "$collective_refreeze_manifest" \
    --producer-state "$producer_scout.state" \
    --producer-analysis "$producer_scout/analysis.json" \
    --producer-confirmation-state "$producer_confirmation.state" \
    --producer-confirmation-analysis "$producer_confirmation/analysis.json" \
    --producer-expansion-manifest "$producer_expansion/manifest.json" \
    --producer-refreeze-manifest "$producer_refreeze/manifest.json" \
    --guarded-state "$guarded_scout.state" \
    --guarded-analysis "$guarded_scout/analysis.json" \
    --guarded-confirmation-state "$guarded_confirmation.state" \
    --guarded-confirmation-analysis "$guarded_confirmation/analysis.json" \
    --guarded-expansion-manifest "$guarded_expansion/manifest.json" \
    --guarded-refreeze-manifest "$guarded_refreeze/manifest.json" \
    --reused-state "$reused_scout.state" \
    --reused-analysis "$reused_scout/analysis.json" \
    --reused-confirmation-state "$reused_confirmation.state" \
    --reused-confirmation-analysis "$reused_confirmation/analysis.json" \
    --reused-expansion-manifest "$reused_expansion/manifest.json" \
    --reused-refreeze-manifest "$reused_refreeze/manifest.json" \
    --out "$audit_dir/readiness.json" >>"$log_path" 2>&1

python3 "$authority_script" \
    --suite "$current_suite" --prompt-dir "$current_prompts" \
    --gbt-report "$repo_root/build_ofi/compiler_lto_eval/generated/gbt-history-report.json" \
    --input-separation "$audit_dir/input-separation.json" \
    --communication "jacobi=$jacobi_graph" \
    --communication "loop_lto=$loop_graph" \
    --communication "minimod=$portfolio/minimod/group-graph.json" \
    --communication "mixed_lto=$portfolio/mixed_lto/group-graph.json" \
    --communication "mm_minimal=$mm_graph" \
    --structural-graph "$structural_graph" \
    --collective-graph "$collective_graph" \
    --conditional-frontier "$repo_root/build_ofi/compiler_action_frontier_20260904/report.json" \
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

python3 "$input_auditor" verify \
    --suite "$current_suite" --prompt-dir "$current_prompts" \
    --gbt-report "$repo_root/build_ofi/compiler_lto_eval/generated/gbt-history-report.json" \
    --report "$audit_dir/input-separation.json" >>"$log_path" 2>&1
python3 "$readiness_script" verify \
    --suite "$current_suite" --prompt-dir "$current_prompts" \
    --graph-refreeze-finalizer "$successor" \
    --placement-summary "$repo_root/docs/experiments/compiler-comm-plan-capacity/runs/compiler-comm-plan-placement-v1/summary.json" \
    --placement-historical-graph "$repo_root/build_ofi/compiler_comm_plan_placement/generated/opportunity-graph.json" \
    --placement-current-graph "$structural_graph" \
    --collective-state "$collective_state" --collective-analysis "$collective_analysis" \
    --collective-confirmation-state "$collective_confirmation_state" \
    --collective-confirmation-analysis "$collective_confirmation_analysis" \
    --collective-refreeze-manifest "$collective_refreeze_manifest" \
    --producer-state "$producer_scout.state" --producer-analysis "$producer_scout/analysis.json" \
    --producer-confirmation-state "$producer_confirmation.state" \
    --producer-confirmation-analysis "$producer_confirmation/analysis.json" \
    --producer-expansion-manifest "$producer_expansion/manifest.json" \
    --producer-refreeze-manifest "$producer_refreeze/manifest.json" \
    --guarded-state "$guarded_scout.state" --guarded-analysis "$guarded_scout/analysis.json" \
    --guarded-confirmation-state "$guarded_confirmation.state" \
    --guarded-confirmation-analysis "$guarded_confirmation/analysis.json" \
    --guarded-expansion-manifest "$guarded_expansion/manifest.json" \
    --guarded-refreeze-manifest "$guarded_refreeze/manifest.json" \
    --reused-state "$reused_scout.state" --reused-analysis "$reused_scout/analysis.json" \
    --reused-confirmation-state "$reused_confirmation.state" \
    --reused-confirmation-analysis "$reused_confirmation/analysis.json" \
    --reused-expansion-manifest "$reused_expansion/manifest.json" \
    --reused-refreeze-manifest "$reused_refreeze/manifest.json" \
    --report "$audit_dir/readiness.json" >>"$log_path" 2>&1
python3 "$protocol_auditor" verify \
    --suite "$current_suite" --prompt-dir "$current_prompts" \
    --readiness "$audit_dir/readiness.json" \
    --input-separation "$audit_dir/input-separation.json" \
    --sampling-null "$audit_dir/sampling-null.json" \
    --report "$audit_dir/capability-protocol.json" >>"$log_path" 2>&1
verify_artifacts

ids=$(python3 -c \
    'import json,sys; specs=(("readiness_id",sys.argv[1]),("authority_id",sys.argv[2]),("null_id",sys.argv[3]),("protocol_id",sys.argv[4])); print(" ".join(k+"="+json.load(open(p))[k] for k,p in specs))' \
    "$audit_dir/readiness.json" "$audit_dir/action-authority.json" \
    "$audit_dir/sampling-null.json" "$audit_dir/capability-protocol.json")
set_state complete \
    "collective=$collective_label suite=$current_suite $ids"
trap - EXIT
