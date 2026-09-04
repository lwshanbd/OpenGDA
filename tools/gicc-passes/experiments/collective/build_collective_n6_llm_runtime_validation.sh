#!/usr/bin/env bash
# Materialize every deduplicated N6 representative through the LTO pass.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 PLAN_DIR" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
plan_dir=$(cd -- "$1" && pwd)
preparer="$script_dir/prepare_collective_n6_llm_runtime_validation.py"
builder="$script_dir/build_compiler_collective_eval.sh"
evaluator="$script_dir/compiler_collective_eval.py"

python3 "$preparer" verify-contained --plan-dir "$plan_dir"
if [[ -e $plan_dir/manifest.json ]]; then
    echo "refusing already finalized runtime bundle: $plan_dir" >&2
    exit 2
fi
mapfile -t names < <(python3 -c \
    'import json,sys; print("\n".join(x["name"] for x in json.load(open(sys.argv[1]))["policies"]))' \
    "$plan_dir/plan.json")
if [[ ${#names[@]} -lt 1 || ${#names[@]} -gt 9 ]]; then
    echo "runtime bundle requires 1..9 deduplicated policies" >&2
    exit 2
fi
graph=$(python3 -c \
    'import json,sys,pathlib; x=json.load(open(sys.argv[1])); p=pathlib.Path(x["evidence"]["compiler_graph"]["path"]); root=pathlib.Path(sys.argv[2]); print(p if p.is_absolute() else root/p)' \
    "$plan_dir/plan.json" "$repo_root")

for name in "${names[@]}"; do
    if [[ ! $name =~ ^policy[0-9][0-9]$ ]]; then
        echo "invalid runtime policy name: $name" >&2
        exit 2
    fi
    policy_dir="$plan_dir/policies/$name"
    if [[ -e $policy_dir/build ]]; then
        echo "refusing partial runtime policy build: $policy_dir/build" >&2
        exit 2
    fi
    bash "$builder" lower "$policy_dir/build" "$policy_dir/hint.json"
    python3 "$evaluator" verify-plan-ir \
        --graph "$graph" --hint "$policy_dir/hint.json" \
        --ir "$policy_dir/build/materialized.ll"
done
python3 "$preparer" finalize \
    --plan-dir "$plan_dir" --repo-root "$repo_root"
python3 "$preparer" verify-built \
    --plan-dir "$plan_dir" --repo-root "$repo_root"
