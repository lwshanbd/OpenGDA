#!/usr/bin/env bash
# Prepare and content-freeze every compiler-owned v3 collective artifact.
#
# This script is deliberately offline: it neither invokes a model nor calls
# Flux.  Run it only after the supplied Gate-A monitor has passed.  All eight
# uniform controls and the mixed-policy canary are built sequentially from the
# same application source, and every build must pass provenance and IR audits.
set -euo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "usage: $0 OUTPUT_DIR GATE_A_MONITOR [CALIBRATION_JSON [PLATFORM_JSON]]" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
output_dir=$1
gate_a_monitor=$2
calibration_input=${3:-$repo_root/build_ofi/compiler_lto_calibration/generated/dossier.json}
platform_input=${4:-$repo_root/tools/gicc-passes/python/profiles/tioga-mi250x-cxi-collective-capacity-n2.json}

if [[ -e "$output_dir" ]]; then
    echo "refusing to overwrite existing v3 output: $output_dir" >&2
    exit 2
fi
for required in "$gate_a_monitor" "$calibration_input" "$platform_input"; do
    if [[ ! -f "$required" ]]; then
        echo "missing required v3 input: $required" >&2
        exit 2
    fi
done

evaluator="$script_dir/compiler_collective_eval.py"
python3 "$evaluator" verify-gate-a-monitor --monitor "$gate_a_monitor"
gate_a_stdout=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["stdout"])' \
    "$gate_a_monitor")
gate_a_stderr=$(python3 -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["paths"]["stderr"])' \
    "$gate_a_monitor")
for required in "$gate_a_stdout" "$gate_a_stderr"; do
    if [[ ! -f "$required" ]]; then
        echo "missing Gate-A raw log: $required" >&2
        exit 2
    fi
done

mkdir -p "$output_dir/inputs" "$output_dir/discovery" \
    "$output_dir/prompts" "$output_dir/controls" \
    "$output_dir/binaries" "$output_dir/canary"
output_dir=$(cd -- "$output_dir" && pwd)
gate_a_monitor=$(cd -- "$(dirname -- "$gate_a_monitor")" && pwd)/$(basename -- "$gate_a_monitor")
cp -- "$calibration_input" "$output_dir/inputs/primitive-calibration.json"
cp -- "$platform_input" "$output_dir/inputs/platform.json"
cp -- "$gate_a_monitor" "$output_dir/inputs/gate-a.monitor.json"
cp -- "$gate_a_stdout" "$output_dir/inputs/gate-a.out"
cp -- "$gate_a_stderr" "$output_dir/inputs/gate-a.err"

builder="$script_dir/build_compiler_collective_eval.sh"
bridge="$repo_root/tools/gicc-passes/python/gicc_collective_plan_bridge.py"
source_file="$script_dir/compiler_collective_eval.cpp"
catalog_file="$script_dir/compiler_collective_catalog.hpp"
platform="$output_dir/inputs/platform.json"
calibration="$output_dir/inputs/primitive-calibration.json"
graph="$output_dir/discovery/graph.json"

echo "=== v3 discovery build and dependency closure ==="
bash "$builder" discover "$output_dir/discovery/build"

echo "=== v3 graph and three compiler-input views ==="
graph_sha=
for view in relational descriptors opaque; do
    python3 "$bridge" emit \
        --inventory "$output_dir/discovery/build/inventory.json" \
        --platform "$platform" \
        --calibration-artifact "$calibration" \
        --graph "$graph" \
        --prompt "$output_dir/prompts/$view.txt" \
        --prompt-view "$view"
    current_sha=$(sha256sum "$graph")
    current_sha=${current_sha%% *}
    if [[ -z "$graph_sha" ]]; then
        graph_sha=$current_sha
    elif [[ "$current_sha" != "$graph_sha" ]]; then
        echo "graph changed across prompt views" >&2
        exit 2
    fi
done

source_sha=$(sha256sum "$source_file")
source_sha=${source_sha%% *}
catalog_sha=$(sha256sum "$catalog_file")
catalog_sha=${catalog_sha%% *}

echo "=== v3 uniform compiler controls ==="
python3 "$evaluator" controls \
    --graph "$graph" \
    --out "$output_dir/controls" \
    --source-sha256 "$source_sha" \
    --catalog-sha256 "$catalog_sha"
python3 "$evaluator" verify \
    --graph "$graph" \
    --manifest "$output_dir/controls/manifest.json"

echo "=== v3 same-build uniform binaries (sequential) ==="
while IFS= read -r name; do
    hint="$output_dir/controls/$name-hint.json"
    arm_dir="$output_dir/binaries/$name"
    echo "--- $name"
    bash "$builder" lower "$arm_dir" "$hint"
    python3 "$evaluator" verify-plan-ir \
        --graph "$graph" --hint "$hint" \
        --ir "$arm_dir/materialized.ll"
done < <(python3 -c \
    'import json,sys; print("\n".join(a["name"] for a in json.load(open(sys.argv[1]))["arms"]))' \
    "$output_dir/controls/manifest.json")
python3 "$evaluator" verify-ir \
    --manifest "$output_dir/controls/manifest.json" \
    --ir "$output_dir/binaries"

echo "=== v3 compiler-owned mixed-policy canary ==="
python3 "$evaluator" canary \
    --graph "$graph" \
    --decision "$output_dir/canary/decision.json" \
    --hint "$output_dir/canary/hint.json"
bash "$builder" lower "$output_dir/canary/build" \
    "$output_dir/canary/hint.json"
python3 "$evaluator" verify-plan-ir \
    --graph "$graph" \
    --hint "$output_dir/canary/hint.json" \
    --ir "$output_dir/canary/build/materialized.ll"

echo "=== v3 complete compiler action-space audit ==="
python3 "$evaluator" capacity-audit \
    --graph "$graph" --out "$output_dir/capacity-audit.json"

echo "=== v3 authoritative offline freeze ==="
python3 "$evaluator" freeze-offline \
    --bundle-root "$output_dir" \
    --repo-root "$repo_root" \
    --platform "$platform" \
    --calibration-artifact "$calibration" \
    --gate-a-monitor "$output_dir/inputs/gate-a.monitor.json" \
    --gate-a-stdout "$output_dir/inputs/gate-a.out" \
    --gate-a-stderr "$output_dir/inputs/gate-a.err" \
    --preparation-script "$script_dir/prepare_compiler_collective_v3.sh" \
    --out "$output_dir/FROZEN_V3_MANIFEST.json"
python3 "$evaluator" verify-offline-freeze \
    --manifest "$output_dir/FROZEN_V3_MANIFEST.json" \
    --repo-root "$repo_root"

echo "V3_OFFLINE_READY=$output_dir/FROZEN_V3_MANIFEST.json"
