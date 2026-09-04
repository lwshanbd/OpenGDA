#!/usr/bin/env bash
# Deterministically expand/refreeze confirmed compiler-only candidates.
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

original_suite="$repo_root/build_ofi/compiler_decision_suite_20260904/suite.json"
original_prompts="$repo_root/build_ofi/compiler_decision_suite_20260904/prompts"
portfolio="$repo_root/build_ofi/compiler_fact_coverage_20260904/portfolio"
metadata="$repo_root/build_ofi/compiler_fact_coverage_20260904"
collective_graph="$repo_root/build_ofi/compiler_collective_capacity_n8_hierpipe_v1_20260903/discovery/graph.json"
structural_graph="$repo_root/build_ofi/compiler_comm_plan_placement_current_20260904/generated/opportunity-graph.json"

producer_scout_state="$repo_root/build_ofi/producer_fission_oracle_scout_7687377_20260904.state"
producer_confirmation="$repo_root/build_ofi/producer_fission_confirmation_7687377_20260904"
producer_confirmation_state="$producer_confirmation.state"
producer_expansion_dir="$repo_root/build_ofi/producer_fission_graph_expansion_20260904"
producer_refreeze_dir="$repo_root/build_ofi/producer_fission_suite_refreeze_20260904"
producer_expander="$producer/prepare_confirmed_producer_fission_graph.py"
producer_refreezer="$producer/prepare_producer_fission_suite_refreeze.py"

guarded_chain_state="$repo_root/build_ofi/guarded_early_trigger_confirmation_77897d9_20260904.chain.state"
guarded_confirmation="$repo_root/build_ofi/guarded_early_trigger_confirmation_77897d9_20260904"
guarded_expansion_dir="$repo_root/build_ofi/guarded_early_trigger_graph_expansion_20260904"
guarded_refreeze_dir="$repo_root/build_ofi/guarded_early_trigger_suite_refreeze_20260904"
guarded_expander="$guarded/prepare_confirmed_guarded_early_graph.py"
guarded_refreezer="$guarded/prepare_guarded_early_suite_refreeze.py"

reused_chain_state="$repo_root/build_ofi/reused_loop_descriptor_confirmation_aff76f9_20260904.chain.state"
reused_confirmation="$repo_root/build_ofi/reused_loop_descriptor_confirmation_aff76f9_20260904"
reused_expansion_dir="$repo_root/build_ofi/reused_loop_descriptor_graph_expansion_20260904"
reused_refreeze_dir="$repo_root/build_ofi/reused_loop_descriptor_suite_refreeze_20260904"
reused_expander="$reused/prepare_confirmed_reused_loop_descriptor_graph.py"
reused_refreezer="$reused/prepare_reused_loop_descriptor_suite_refreeze.py"

campaign_state="$repo_root/build_ofi/compiler_headroom_confirmations_preinline_20260904.state"
state_path="$repo_root/build_ofi/compiler_headroom_graph_refreezes_20260904.state"
events_path="$repo_root/build_ofi/compiler_headroom_graph_refreezes_20260904.events"
lock_path="$repo_root/build_ofi/compiler_headroom_graph_refreezes_20260904.lock"
finalizer="$script_dir/continue_compiler_headroom_graph_refreezes.sh"

exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another compiler graph-refreeze finalizer is active" >&2
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
    [[ $status -eq 0 ]] || set_state failed "finalizer_exit=$status"
}
trap on_exit EXIT

artifacts=(
    "$finalizer" "$producer_expander" "$producer_refreezer"
    "$guarded_expander" "$guarded_refreezer"
    "$reused_expander" "$reused_refreezer"
    "$script_dir/../python/gicc_llm_bridge.py"
    "$script_dir/../python/gicc_comm_group_plan_bridge.py"
    "$script_dir/../python/gicc_compiler_decision_suite.py"
    "$original_suite" "$collective_graph" "$structural_graph"
    "$portfolio/jacobi/dossier.json" "$portfolio/jacobi/group-graph.json"
    "$portfolio/mm_minimal/dossier.json"
    "$portfolio/mm_minimal/group-graph.json"
    "$portfolio/loop_lto/dossier.json" "$portfolio/loop_lto/group-graph.json"
    "$portfolio/minimod/group-graph.json"
    "$portfolio/mixed_lto/group-graph.json"
    "$metadata/jacobi_disjoint/meta/_Z18jacobi_step_kernelILi32ELi32EEvPN4gicc9DeviceCtxEPfPKfS3_iiibiiiiimmmmm.json"
    "$metadata/mm_minimal_disjoint/meta/_Z18matmul_step_kernelPN4gicc9DeviceCtxEPKfS3_Pfiiiiiim.json"
    "$metadata/bench_pingpong_lto/meta/_Z15dwq_loop_kernelPN4gicc9DeviceCtxEiimi.json"
)
while IFS= read -r -d '' prompt; do
    artifacts+=("$prompt")
done < <(find "$original_prompts" -type f -print0 | sort -z)
hashes=()
for artifact in "${artifacts[@]}"; do
    if [[ ! -f $artifact ]]; then
        echo "missing frozen graph-refreeze artifact: $artifact" >&2
        exit 2
    fi
    digest=$(sha256sum "$artifact"); hashes+=("${digest%% *}")
done
verify_artifacts() {
    local index actual
    for index in "${!artifacts[@]}"; do
        actual=$(sha256sum "${artifacts[$index]}"); actual=${actual%% *}
        if [[ $actual != "${hashes[$index]}" ]]; then
            echo "graph-refreeze artifact changed: ${artifacts[$index]}" >&2
            exit 2
        fi
    done
}
phase_of() {
    local path=$1
    if [[ -f $path ]]; then
        awk 'NR == 1 {print $2}' "$path"
    fi
}

current_suite=$original_suite
current_prompts=$original_prompts
jacobi_graph="$portfolio/jacobi/group-graph.json"
mm_graph="$portfolio/mm_minimal/group-graph.json"

set_state waiting_producer_scout "$producer_scout_state"
producer_scout_phase=
while [[ -z $producer_scout_phase ]]; do
    producer_scout_phase=$(phase_of "$producer_scout_state")
    case $producer_scout_phase in
        promising|negative|failed) ;;
        *) producer_scout_phase=; sleep 5 ;;
    esac
done
if [[ $producer_scout_phase == promising ]]; then
    set_state waiting_producer_confirmation "$producer_confirmation_state"
    producer_phase=
    while [[ -z $producer_phase ]]; do
        producer_phase=$(phase_of "$producer_confirmation_state")
        case $producer_phase in
            confirmed|negative|failed) ;;
            *)
                if [[ $(phase_of "$campaign_state") == failed ]]; then
                    echo "compiler confirmation campaign failed before producer result" >&2
                    exit 2
                fi
                producer_phase=; sleep 5
                ;;
        esac
    done
else
    producer_phase=$producer_scout_phase
fi
verify_artifacts
if [[ $producer_phase == confirmed ]]; then
    set_state preparing_producer_expansion "$producer_expansion_dir"
    if [[ -e $producer_expansion_dir ]]; then
        python3 "$producer_expander" verify-contained \
            --manifest "$producer_expansion_dir/manifest.json"
    else
        python3 "$producer_expander" prepare \
            --confirmation-analysis "$producer_confirmation/analysis.json" \
            --dossier "$portfolio/jacobi/dossier.json" \
            --template "$metadata/jacobi_disjoint/meta/_Z18jacobi_step_kernelILi32ELi32EEvPN4gicc9DeviceCtxEPfPKfS3_iiibiiiiimmmmm.json" \
            --graph "$portfolio/jacobi/group-graph.json" \
            --output-dir "$producer_expansion_dir"
    fi
    verify_artifacts
    set_state preparing_producer_refreeze "$producer_refreeze_dir"
    if [[ -e $producer_refreeze_dir ]]; then
        python3 "$producer_refreezer" verify-contained \
            --manifest "$producer_refreeze_dir/manifest.json"
    else
        python3 "$producer_refreezer" prepare \
            --current-suite "$current_suite" \
            --current-prompt-dir "$current_prompts" \
            --expansion-manifest "$producer_expansion_dir/manifest.json" \
            --communication "minimod=$portfolio/minimod/group-graph.json" \
            --communication "mm_minimal=$portfolio/mm_minimal/group-graph.json" \
            --communication "mixed_lto=$portfolio/mixed_lto/group-graph.json" \
            --communication "loop_lto=$portfolio/loop_lto/group-graph.json" \
            --collective "collective_n8=$collective_graph" \
            --structural "coalescing_placement=$structural_graph" \
            --output-dir "$producer_refreeze_dir"
    fi
    current_suite="$producer_refreeze_dir/suite.json"
    current_prompts="$producer_refreeze_dir/prompts"
    jacobi_graph="$producer_expansion_dir/expanded-group-graph.json"
else
    set_state producer_skipped "confirmation_state=$producer_phase"
fi

set_state waiting_guarded_confirmation "$guarded_chain_state"
guarded_phase=
while [[ -z $guarded_phase ]]; do
    guarded_phase=$(phase_of "$guarded_chain_state")
    case $guarded_phase in
        confirmed|negative|skipped|failed) ;;
        *) guarded_phase=; sleep 5 ;;
    esac
done
verify_artifacts
if [[ $guarded_phase == confirmed ]]; then
    set_state preparing_guarded_expansion "$guarded_expansion_dir"
    if [[ -e $guarded_expansion_dir ]]; then
        python3 "$guarded_expander" verify-contained \
            --manifest "$guarded_expansion_dir/manifest.json"
    else
        python3 "$guarded_expander" prepare \
            --confirmation-analysis "$guarded_confirmation/analysis.json" \
            --dossier "$portfolio/mm_minimal/dossier.json" \
            --template "$metadata/mm_minimal_disjoint/meta/_Z18matmul_step_kernelPN4gicc9DeviceCtxEPKfS3_Pfiiiiiim.json" \
            --graph "$portfolio/mm_minimal/group-graph.json" \
            --output-dir "$guarded_expansion_dir"
    fi
    verify_artifacts
    set_state preparing_guarded_refreeze "$guarded_refreeze_dir"
    if [[ -e $guarded_refreeze_dir ]]; then
        python3 "$guarded_refreezer" verify-contained \
            --manifest "$guarded_refreeze_dir/manifest.json"
    else
        python3 "$guarded_refreezer" prepare \
            --current-suite "$current_suite" \
            --current-prompt-dir "$current_prompts" \
            --expansion-manifest "$guarded_expansion_dir/manifest.json" \
            --communication "jacobi=$jacobi_graph" \
            --communication "minimod=$portfolio/minimod/group-graph.json" \
            --communication "mixed_lto=$portfolio/mixed_lto/group-graph.json" \
            --communication "loop_lto=$portfolio/loop_lto/group-graph.json" \
            --collective "collective_n8=$collective_graph" \
            --structural "coalescing_placement=$structural_graph" \
            --output-dir "$guarded_refreeze_dir"
    fi
    current_suite="$guarded_refreeze_dir/suite.json"
    current_prompts="$guarded_refreeze_dir/prompts"
    mm_graph="$guarded_expansion_dir/expanded-group-graph.json"
else
    set_state guarded_skipped "confirmation_state=$guarded_phase"
fi

set_state waiting_reused_confirmation "$reused_chain_state"
reused_phase=
while [[ -z $reused_phase ]]; do
    reused_phase=$(phase_of "$reused_chain_state")
    case $reused_phase in
        confirmed|negative|skipped|failed) ;;
        *) reused_phase=; sleep 5 ;;
    esac
done
verify_artifacts
if [[ $reused_phase == confirmed ]]; then
    set_state preparing_reused_expansion "$reused_expansion_dir"
    if [[ -e $reused_expansion_dir ]]; then
        python3 "$reused_expander" verify-contained \
            --manifest "$reused_expansion_dir/manifest.json"
    else
        python3 "$reused_expander" prepare \
            --confirmation-analysis "$reused_confirmation/analysis.json" \
            --dossier "$portfolio/loop_lto/dossier.json" \
            --template "$metadata/bench_pingpong_lto/meta/_Z15dwq_loop_kernelPN4gicc9DeviceCtxEiimi.json" \
            --graph "$portfolio/loop_lto/group-graph.json" \
            --output-dir "$reused_expansion_dir"
    fi
    verify_artifacts
    set_state preparing_reused_refreeze "$reused_refreeze_dir"
    if [[ -e $reused_refreeze_dir ]]; then
        python3 "$reused_refreezer" verify-contained \
            --manifest "$reused_refreeze_dir/manifest.json"
    else
        python3 "$reused_refreezer" prepare \
            --current-suite "$current_suite" \
            --current-prompt-dir "$current_prompts" \
            --expansion-manifest "$reused_expansion_dir/manifest.json" \
            --communication "jacobi=$jacobi_graph" \
            --communication "minimod=$portfolio/minimod/group-graph.json" \
            --communication "mm_minimal=$mm_graph" \
            --communication "mixed_lto=$portfolio/mixed_lto/group-graph.json" \
            --collective "collective_n8=$collective_graph" \
            --structural "coalescing_placement=$structural_graph" \
            --output-dir "$reused_refreeze_dir"
    fi
    current_suite="$reused_refreeze_dir/suite.json"
    current_prompts="$reused_refreeze_dir/prompts"
else
    set_state reused_skipped "confirmation_state=$reused_phase"
fi

python3 -c \
    'import json,sys; value=json.load(open(sys.argv[1])); assert value["boundary"]["source_visible"] is False; assert value["boundary"]["provider_call_supported"] is False' \
    "$current_suite"
set_state complete "suite=$current_suite prompts=$current_prompts"
trap - EXIT
