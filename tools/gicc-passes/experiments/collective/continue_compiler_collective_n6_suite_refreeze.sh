#!/usr/bin/env bash
# Conditionally refreeze the final compiler suite after confirmed N6 runtime.
set -euo pipefail

if [[ $# -ne 0 ]]; then
    echo "usage: $0" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
portfolio="$repo_root/build_ofi/compiler_fact_coverage_20260904/portfolio"
metadata="$repo_root/build_ofi/compiler_fact_coverage_20260904"
n8_graph="$repo_root/build_ofi/compiler_collective_capacity_n8_hierpipe_v1_20260903/discovery/graph.json"
n6_graph="$repo_root/build_ofi/compiler_collective_capacity_n6_hierpipe_v1_20260904/discovery/graph.json"
n6_confirmation="$repo_root/build_ofi/compiler_collective_n6_confirmation_20260904"
n6_chain_state="$n6_confirmation.chain.state"
headroom_state="$repo_root/build_ofi/compiler_headroom_graph_refreezes_20260904.state"
output_dir="$repo_root/build_ofi/compiler_collective_n6_suite_refreeze_20260904"
state_path="$output_dir.chain.state"
events_path="$output_dir.chain.events"
lock_path="$output_dir.chain.lock"
refreezer="$script_dir/prepare_collective_n6_suite_refreeze.py"
successor="$script_dir/continue_compiler_collective_n6_suite_refreeze.sh"
structural_graph="$repo_root/build_ofi/compiler_comm_plan_placement_current_20260904/generated/opportunity-graph.json"

exec 9>"$lock_path"
if ! flock -n 9; then
    echo "another N6 suite-refreeze successor is active" >&2
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

artifacts=(
    "$successor" "$refreezer" "$n8_graph" "$n6_graph"
    "$structural_graph"
    "$script_dir/analyze_compiler_collective_n6_confirmation.py"
    "$script_dir/analyze_compiler_collective_n8_confirmation.py"
    "$repo_root/tools/gicc-passes/python/gicc_compiler_decision_suite.py"
    "$repo_root/tools/gicc-passes/python/gicc_collective_plan_bridge.py"
    "$repo_root/tools/gicc-passes/python/gicc_comm_group_plan_bridge.py"
    "$repo_root/tools/gicc-passes/python/gicc_comm_plan_bridge.py"
    "$repo_root/tools/gicc-passes/python/gicc_llm_bridge.py"
    "$portfolio/jacobi/group-graph.json"
    "$portfolio/mm_minimal/group-graph.json"
    "$portfolio/loop_lto/group-graph.json"
    "$portfolio/minimod/group-graph.json"
    "$portfolio/mixed_lto/group-graph.json"
)
for optional in \
    "$repo_root/build_ofi/producer_fission_graph_expansion_20260904/expanded-group-graph.json" \
    "$repo_root/build_ofi/guarded_early_trigger_graph_expansion_20260904/expanded-group-graph.json" \
    "$repo_root/build_ofi/reused_loop_descriptor_graph_expansion_20260904/expanded-group-graph.json"; do
    if [[ -f $optional ]]; then
        artifacts+=("$optional")
    fi
done
hashes=()
for artifact in "${artifacts[@]}"; do
    if [[ ! -f $artifact ]]; then
        echo "missing frozen N6 refreeze artifact: $artifact" >&2
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
            echo "N6 suite-refreeze artifact changed: ${artifacts[$index]}" >&2
            exit 2
        fi
    done
}

if [[ -e $output_dir ]]; then
    echo "refusing pre-existing N6 suite refreeze: $output_dir" >&2
    exit 2
fi

set_state waiting_n6_confirmation "$n6_chain_state"
n6_phase=
while [[ -z $n6_phase ]]; do
    n6_phase=$(phase_of "$n6_chain_state")
    case $n6_phase in
        confirmed|negative|skipped|skipped_no_incremental_policy|failed) ;;
        *) n6_phase=; sleep 5 ;;
    esac
done
verify_artifacts
if [[ $n6_phase != confirmed ]]; then
    set_state skipped "n6_confirmation=$n6_phase"
    trap - EXIT
    exit 0
fi

set_state waiting_headroom_refreeze "$headroom_state"
headroom_phase=
while [[ -z $headroom_phase ]]; do
    headroom_phase=$(phase_of "$headroom_state")
    case $headroom_phase in
        complete|failed) ;;
        *) headroom_phase=; sleep 5 ;;
    esac
done
if [[ $headroom_phase != complete ]]; then
    echo "headroom suite-refreeze chain failed" >&2
    exit 2
fi
verify_artifacts

state_line=$(sed -n '1p' "$headroom_state")
state_detail=${state_line#*$'\t'complete$'\t'}
current_suite=${state_detail#suite=}
current_suite=${current_suite%% prompts=*}
current_prompts=${state_detail##* prompts=}
if [[ ! -f $current_suite || ! -d $current_prompts ]]; then
    echo "headroom finalizer did not name a valid suite and prompt directory" >&2
    exit 2
fi

resolve_graph() {
    local label=$1 expected candidate digest matches=0 selected=
    shift
    expected=$(python3 -c \
        'import json,sys; x=json.load(open(sys.argv[1])); e={i["label"]:i for i in x["entries"]}; print(e[sys.argv[2]]["graph_file_sha256"])' \
        "$current_suite" "$label")
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

jacobi_graph=$(resolve_graph jacobi \
    "$portfolio/jacobi/group-graph.json" \
    "$repo_root/build_ofi/producer_fission_graph_expansion_20260904/expanded-group-graph.json")
mm_graph=$(resolve_graph mm_minimal \
    "$portfolio/mm_minimal/group-graph.json" \
    "$repo_root/build_ofi/guarded_early_trigger_graph_expansion_20260904/expanded-group-graph.json")
loop_graph=$(resolve_graph loop_lto \
    "$portfolio/loop_lto/group-graph.json" \
    "$repo_root/build_ofi/reused_loop_descriptor_graph_expansion_20260904/expanded-group-graph.json")
minimod_graph=$(resolve_graph minimod "$portfolio/minimod/group-graph.json")
mixed_graph=$(resolve_graph mixed_lto "$portfolio/mixed_lto/group-graph.json")
placement_graph=$(resolve_graph coalescing_placement "$structural_graph")
verify_artifacts

set_state refreezing "suite=$current_suite"
python3 "$refreezer" prepare \
    --current-suite "$current_suite" \
    --current-prompt-dir "$current_prompts" \
    --confirmation-analysis "$n6_confirmation/analysis.json" \
    --n8-graph "$n8_graph" \
    --n6-graph "$n6_graph" \
    --communication "jacobi=$jacobi_graph" \
    --communication "loop_lto=$loop_graph" \
    --communication "minimod=$minimod_graph" \
    --communication "mixed_lto=$mixed_graph" \
    --communication "mm_minimal=$mm_graph" \
    --structural "coalescing_placement=$placement_graph" \
    --output-dir "$output_dir"
python3 "$refreezer" verify-contained --manifest "$output_dir/manifest.json"
set_state refrozen "suite=$output_dir/suite.json prompts=$output_dir/prompts"
trap - EXIT
