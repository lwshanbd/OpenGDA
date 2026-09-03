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
mkdir -p "$output_dir/meta"
output_dir=$(cd -- "$output_dir" && pwd)

common=(
    -DGICC_BOOTSTRAP_MPI=1
    -DGICC_CPU_PROXY=1
    -DGICC_GPU_HIP=1
    -DGICC_PLATFORM_OFI
    -DUSE_PROF_API=1
    -D__HIP_PLATFORM_AMD__=1
    -D__HIP_ROCclr__=1
    "-I$repo_root/src"
    "-I$repo_root/src/gicc/platform/ofi/internal"
    -isystem /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/include
    --offload-arch=gfx90a
    -O3
    -gline-tables-only
    -fno-exceptions
    -flto
    -std=gnu++17
    "-fpass-plugin=$plugin"
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
    export GICC_COLLECTIVE_HINT_IN=$(cd -- "$(dirname -- "$hint")" && pwd)/$(basename -- "$hint")
fi

"$clang" "${common[@]}" -o "$output_dir/eval.o" -x hip -c "$source_file" \
    >"$output_dir/compile.log" 2>&1

# A host-only textual artifact preserves the compiler decision metadata and is
# audited before any scheduler job. It is not an alternative source build.
"$clang" "${common[@]}" --offload-host-only -emit-llvm -S \
    -o "$output_dir/materialized.ll" -x hip "$source_file" \
    >"$output_dir/ir.log" 2>&1

# Preserve the matching device IR as evidence that collective-only planning
# did not erase the proxy-ring communication bodies. This is emitted from the
# same source, flags, plugin, and environment as eval.o.
"$clang" "${common[@]}" --offload-device-only -emit-llvm -S \
    -o "$output_dir/materialized-device.ll" -x hip "$source_file" \
    >"$output_dir/device-ir.log" 2>&1
python3 "$script_dir/compiler_collective_eval.py" verify-device-ir \
    --ir "$output_dir/materialized-device.ll" \
    >"$output_dir/device-ir-audit.log" 2>&1

if [[ "$mode" == discover ]]; then
    exit 0
fi

proxy_dir="$repo_root/build_ofi/examples/proxy/CMakeFiles/jacobi3d.dir"
objects=(
    "$proxy_dir/__/__/src/gicc/platform/ofi/proxy/proxy_libfabric.cpp.o"
    "$proxy_dir/__/__/src/gicc/platform/ofi/proxy/proxy_thread.cpp.o"
    "$proxy_dir/__/__/src/gicc/platform/ofi/runtime_helpers.cpp.o"
)
for object in "${objects[@]}"; do
    if [[ ! -f "$object" ]]; then
        echo "missing reusable runtime object: $object" >&2
        exit 2
    fi
done

"$clang" --offload-arch=gfx90a -flto --hip-link --rtlib=compiler-rt \
    -unwindlib=libgcc -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive \
    "$output_dir/eval.o" "${objects[@]}" -o "$output_dir/compiler_collective_eval" \
    -Wl,-rpath,/opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/lib:/opt/cray/pe/lib64:/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib \
    /usr/lib64/libfabric.so /usr/lib64/libhwloc.so \
    /opt/cray/pe/mpich/8.1.32/ofi/gnu/11.2/lib/libmpi_gnu_112.so \
    -lpthread /opt/cray/pe/lib64/libmpi_gtl_hsa.so \
    /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 \
    /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib/libmpi_cray.so \
    >"$output_dir/link.log" 2>&1
