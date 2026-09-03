#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 <discover|lower> <output-dir> [collective-hint.json]" >&2
    exit 2
fi

mode=$1
output_dir=$2
hint=${3:-}
if [[ "$mode" != discover && "$mode" != lower ]]; then
    echo "mode must be discover or lower" >&2
    exit 2
fi
if [[ "$mode" == lower && -z "$hint" ]]; then
    echo "lower mode requires a compiler collective hint" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
clang=/opt/rocm-6.4.0/lib/llvm/bin/clang++
plugin="$repo_root/tools/gicc-passes/build/libgicc-passes.so"
source_file="$script_dir/compiler_collective_eval.cpp"
catalog_file="$script_dir/compiler_collective_catalog.hpp"
common_header="$repo_root/examples/proxy/coll_common.hpp"
evaluator="$script_dir/compiler_collective_eval.py"
mkdir -p "$output_dir/meta" "$output_dir/obj"
output_dir=$(cd -- "$output_dir" && pwd)
cd "$repo_root"

command_log="$output_dir/commands.jsonl"
python3 "$evaluator" record-command --out "$command_log" --reset

run_recorded() {
    python3 "$evaluator" record-command --out "$command_log" -- "$@"
    "$@"
}

common=(
    "-DGICC_BOOTSTRAP_MPI=1"
    "-DGICC_CPU_PROXY=1"
    "-DGICC_GPU_HIP=1"
    -DGICC_PLATFORM_OFI
    "-DUSE_PROF_API=1"
    "-D__HIP_PLATFORM_AMD__=1"
    "-D__HIP_ROCclr__=1"
    "-I$repo_root/src"
    "-I$repo_root/src/gicc/platform/ofi/internal"
    -isystem /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/include
    "--offload-arch=gfx90a"
    -O3
    -gline-tables-only
    -fno-exceptions
    -flto
    "-std=gnu++17"
    "-fpass-plugin=$plugin"
)

# Runtime support is a same-build input, not a set of opaque objects borrowed
# from an unrelated CMake target.  It deliberately does not load compiler
# planning passes: those passes own the benchmark collective call only.
runtime_common=(
    "-DGICC_BOOTSTRAP_MPI=1"
    "-DGICC_CPU_PROXY=1"
    "-DGICC_GPU_HIP=1"
    -DGICC_PLATFORM_OFI
    "-DUSE_PROF_API=1"
    "-D__HIP_PLATFORM_AMD__=1"
    "-D__HIP_ROCclr__=1"
    "-I$repo_root/src"
    "-I$repo_root/src/gicc/platform/ofi/internal"
    -isystem /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/include
    "--offload-arch=gfx90a"
    -O3
    -gline-tables-only
    -fno-exceptions
    -flto
    "-std=gnu++17"
)

runtime_names=(runtime_helpers proxy_thread proxy_libfabric)
runtime_sources=(
    "$repo_root/src/gicc/platform/ofi/runtime_helpers.cpp"
    "$repo_root/src/gicc/platform/ofi/proxy/proxy_thread.cpp"
    "$repo_root/src/gicc/platform/ofi/proxy/proxy_libfabric.cpp"
)
link_inputs=(
    /usr/lib64/libhugetlbfs.so
    /usr/lib64/libfabric.so
    /usr/lib64/libhwloc.so
    /opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/lib/libmpi_gnu_112.so
    /opt/cray/pe/lib64/libmpi_gtl_hsa.so
    /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400
    /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib/libmpi_cray.so
)

export GICC_META_DIR="$output_dir/meta"
export GICC_COLLECTIVE_OUT="$output_dir/inventory.json"
# This binary evaluates only the compiler's collective-plan decision. Do not
# run the independent per-transfer lowering pipeline without its own hint: it
# would erase proxy put/quiet calls while this experiment has no synthesized
# host trace to replace them.
export GICC_COLLECTIVE_ONLY=1
unset GICC_HINT_IN GICC_FEATURES_OUT || true
if [[ "$mode" == discover ]]; then
    export GICC_MODE=feature-extract
    unset GICC_COLLECTIVE_HINT_IN || true
else
    export GICC_MODE=lower
    GICC_COLLECTIVE_HINT_IN=$(cd -- "$(dirname -- "$hint")" && pwd)/$(basename -- "$hint")
    export GICC_COLLECTIVE_HINT_IN
fi

source_before=$(sha256sum "$source_file" "$catalog_file" "$common_header")

run_recorded "$clang" "${common[@]}" -MMD -MF "$output_dir/eval.d" \
    -o "$output_dir/eval.o" -x hip -c "$source_file" \
    >"$output_dir/compile.log" 2>&1

# A host-only textual artifact preserves the compiler decision metadata and is
# audited before any scheduler job. It is not an alternative source build.
run_recorded "$clang" "${common[@]}" --offload-host-only -emit-llvm -S \
    -o "$output_dir/materialized.ll" -x hip "$source_file" \
    >"$output_dir/ir.log" 2>&1

# Preserve the matching device IR as evidence that collective-only planning
# did not erase the proxy-ring communication bodies. This is emitted from the
# same source, flags, plugin, and environment as eval.o.
run_recorded "$clang" "${common[@]}" --offload-device-only -emit-llvm -S \
    -o "$output_dir/materialized-device.ll" -x hip "$source_file" \
    >"$output_dir/device-ir.log" 2>&1
run_recorded python3 "$evaluator" verify-device-ir \
    --ir "$output_dir/materialized-device.ll" \
    >"$output_dir/device-ir-audit.log" 2>&1

emit_provenance() {
    local args=(
        build-provenance
        --mode "$mode"
        --repo-root "$repo_root"
        --build-root "$output_dir"
        --commands "$command_log"
        --out "$output_dir/build-provenance.json"
        --input "benchmark_source=$source_file"
        --input "catalog_source=$catalog_file"
        --input "build_script=$script_dir/build_compiler_collective_eval.sh"
        --input "evaluator=$evaluator"
        --input "compiler=$clang"
        --input "compiler_config=/opt/rocm-6.4.0/lib/llvm/bin/clang++.cfg"
        --input "pass_plugin=$plugin"
        --artifact "eval_object=$output_dir/eval.o"
        --artifact "inventory=$output_dir/inventory.json"
        --artifact "host_ir=$output_dir/materialized.ll"
        --artifact "device_ir=$output_dir/materialized-device.ll"
        --artifact "device_ir_audit=$output_dir/device-ir-audit.log"
        --artifact "compile_log=$output_dir/compile.log"
        --artifact "host_ir_log=$output_dir/ir.log"
        --artifact "device_ir_log=$output_dir/device-ir.log"
        --artifact "command_log=$command_log"
        --dependency-file "$output_dir/eval.d"
        --environment "GICC_MODE=$GICC_MODE"
        --environment "GICC_COLLECTIVE_ONLY=$GICC_COLLECTIVE_ONLY"
        --environment "GICC_META_DIR=$GICC_META_DIR"
        --environment "GICC_COLLECTIVE_OUT=$GICC_COLLECTIVE_OUT"
        --environment GICC_HINT_IN
        --environment GICC_FEATURES_OUT
    )
    if [[ "$mode" == discover ]]; then
        args+=(--environment GICC_COLLECTIVE_HINT_IN)
    else
        args+=(
            --environment "GICC_COLLECTIVE_HINT_IN=$GICC_COLLECTIVE_HINT_IN"
            --input "collective_hint=$GICC_COLLECTIVE_HINT_IN"
            --artifact "binary=$output_dir/compiler_collective_eval"
            --artifact "link_log=$output_dir/link.log"
        )
        local index name
        for index in "${!runtime_names[@]}"; do
            name=${runtime_names[$index]}
            args+=(
                --input "${name}_source=${runtime_sources[$index]}"
                --artifact "${name}_object=$output_dir/obj/$name.o"
                --artifact "${name}_compile_log=$output_dir/$name-compile.log"
                --dependency-file "$output_dir/obj/$name.d"
            )
        done
        for index in "${!link_inputs[@]}"; do
            args+=(--input "link_input_${index}=${link_inputs[$index]}")
        done
    fi
    python3 "$evaluator" "${args[@]}"
    python3 "$evaluator" verify-build-provenance \
        --manifest "$output_dir/build-provenance.json" \
        --repo-root "$repo_root"
}

if [[ "$mode" == discover ]]; then
    source_after=$(sha256sum "$source_file" "$catalog_file" "$common_header")
    [[ "$source_before" == "$source_after" ]]
    emit_provenance
    exit 0
fi

for index in "${!runtime_names[@]}"; do
    name=${runtime_names[$index]}
    runtime_source=${runtime_sources[$index]}
    run_recorded env -u GICC_MODE -u GICC_META_DIR -u GICC_COLLECTIVE_OUT \
        -u GICC_COLLECTIVE_ONLY -u GICC_COLLECTIVE_HINT_IN -u GICC_HINT_IN \
        -u GICC_FEATURES_OUT \
        "$clang" "${runtime_common[@]}" -MMD -MF "$output_dir/obj/$name.d" \
        -o "$output_dir/obj/$name.o" -x hip -c "$runtime_source" \
        >"$output_dir/$name-compile.log" 2>&1
done

run_recorded "$clang" --offload-arch=gfx90a -flto --hip-link \
    --rtlib=compiler-rt \
    -unwindlib=libgcc -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive \
    "$output_dir/eval.o" \
    "$output_dir/obj/runtime_helpers.o" \
    "$output_dir/obj/proxy_thread.o" \
    "$output_dir/obj/proxy_libfabric.o" \
    -o "$output_dir/compiler_collective_eval" \
    -Wl,-rpath,/opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/lib:/opt/cray/pe/lib64:/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib \
    /usr/lib64/libfabric.so /usr/lib64/libhwloc.so \
    /opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/lib/libmpi_gnu_112.so \
    -lpthread /opt/cray/pe/lib64/libmpi_gtl_hsa.so \
    /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 \
    /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib/libmpi_cray.so \
    >"$output_dir/link.log" 2>&1

source_after=$(sha256sum "$source_file" "$catalog_file" "$common_header")
[[ "$source_before" == "$source_after" ]]
emit_provenance
