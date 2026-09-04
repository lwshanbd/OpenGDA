#!/usr/bin/env bash
# Build unchanged mm_minimal with a DWQ baseline and compiler early-trigger arm.
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 OUTPUT_DIR" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
if [[ $1 = /* ]]; then
    output_dir=$1
else
    output_dir="$repo_root/$1"
fi
if [[ -e $output_dir ]]; then
    echo "refusing existing build output: $output_dir" >&2
    exit 2
fi

llvm_root=/opt/rocm-6.4.0/lib/llvm/bin
compiler="$llvm_root/clang++"
opt="$llvm_root/opt"
llvm_dis="$llvm_root/llvm-dis"
plugin="$repo_root/tools/gicc-passes/build/libgicc-passes.so"
source_file="$repo_root/examples/ofi/mm_minimal.cpp"
helper_source="$repo_root/src/gicc/platform/ofi/runtime_helpers.cpp"
baseline_hint="$script_dir/hint_baseline_dwq.json"
guarded_hint="$script_dir/hint_guarded_early_dwq.json"
checksum_source="$script_dir/checksum_hip_memcpy.cpp"
auditor="$script_dir/audit_guarded_early_trigger_ir.py"
for path in "$compiler" "$opt" "$llvm_dis" "$plugin" "$source_file" \
            "$helper_source" "$baseline_hint" "$guarded_hint" \
            "$checksum_source" "$auditor"; do
    if [[ ! -f $path ]]; then
        echo "missing build input: $path" >&2
        exit 2
    fi
done

compile_flags=(
    "-DGICC_BOOTSTRAP_MPI=1"
    "-DGICC_GPU_HIP=1"
    -DGICC_PLATFORM_OFI
    "-DUSE_PROF_API=1"
    "-D__HIP_PLATFORM_AMD__=1"
    "-D__HIP_ROCclr__=1"
    -I"$repo_root/src"
    -I"$repo_root/src/gicc/platform/ofi/internal"
    -I/opt/cray/libfabric/2.1/include
    -isystem /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/include
    -O3 "--offload-arch=gfx90a" "-std=gnu++17" -flto
    "-fpass-plugin=$plugin"
    -x hip -c
)
link_flags=(
    -O3 "--offload-arch=gfx90a" -flto --hip-link
    "--rtlib=compiler-rt" "-unwindlib=libgcc"
    "-fpass-plugin=$plugin" "-Wl,--wrap=hipMemcpy"
    "-Wl,--whole-archive,-lhugetlbfs,--no-whole-archive"
    "-Wl,-rpath,/opt/cray/libfabric/2.1/lib64:/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib"
)
link_libraries=(
    /opt/cray/libfabric/2.1/lib64/libfabric.so
    /usr/lib64/libhwloc.so
    -lpthread
    /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400
    /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib/libmpi_cray.so
    /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400
)

mkdir -p "$output_dir/common"
"$compiler" -O3 -std=gnu++17 \
    -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1 \
    -I/opt/rocm-6.4.0/include \
    -isystem /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/include \
    -c "$checksum_source" -o "$output_dir/common/checksum_hip_memcpy.o"

build_arm() {
    local arm=$1
    local hint=$2
    local arm_dir="$output_dir/$arm"
    local meta="$arm_dir/meta"
    mkdir -p "$arm_dir/obj" "$meta"
    local env_args=(
        "GICC_MODE=lower"
        "GICC_TARGET=ofi-triggered"
        "GICC_META_DIR=$meta"
        "GICC_HINT_IN=$hint"
    )
    env "${env_args[@]}" "$compiler" -save-temps=obj "${compile_flags[@]}" \
        "$source_file" -o "$arm_dir/obj/mm_minimal.o" \
        >"$arm_dir/compile-mm.log" 2>&1
    env "${env_args[@]}" "$compiler" "${compile_flags[@]}" \
        "$helper_source" -o "$arm_dir/obj/runtime_helpers.o" \
        >"$arm_dir/compile-runtime.log" 2>&1
    env "${env_args[@]}" "$compiler" "${link_flags[@]}" \
        "$arm_dir/obj/mm_minimal.o" "$arm_dir/obj/runtime_helpers.o" \
        "$output_dir/common/checksum_hip_memcpy.o" \
        -o "$arm_dir/mm_minimal" "${link_libraries[@]}" \
        >"$arm_dir/link.log" 2>&1
}

build_arm baseline "$baseline_hint"
build_arm guarded "$guarded_hint"

if rg -q 'guarded-early-trigger-(host|device).*materialized' \
       "$output_dir/baseline/compile-mm.log" \
       "$output_dir/baseline/link.log"; then
    echo "baseline unexpectedly materialized guarded early trigger" >&2
    exit 1
fi
rg -q 'guarded-early-trigger-device.*materialized guarded early/original trigger partition' \
    "$output_dir/guarded/compile-mm.log"
rg -q 'guarded-early-trigger-host.*materialized allocation-guarded single launch' \
    "$output_dir/guarded/compile-mm.log"
cmp "$output_dir/baseline/meta/features.json" \
    "$output_dir/guarded/meta/features.json"

baseline_metadata=("$output_dir/baseline/meta/"*.json)
guarded_metadata=("$output_dir/guarded/meta/"*.json)
if [[ ${#baseline_metadata[@]} -ne 2 || ${#guarded_metadata[@]} -ne 2 ]]; then
    echo "each arm must emit features plus one kernel metadata file" >&2
    exit 1
fi
baseline_kernel=$(find "$output_dir/baseline/meta" -maxdepth 1 -type f \
    -name '*.json' ! -name features.json -print)
guarded_kernel=$(find "$output_dir/guarded/meta" -maxdepth 1 -type f \
    -name '*.json' ! -name features.json -print)
if rg -q '"guarded_early_trigger_device_materialized"' "$baseline_kernel"; then
    echo "baseline unexpectedly carries a guarded device attestation" >&2
    exit 1
fi
rg -q '"guarded_early_trigger_device_materialized": true' "$guarded_kernel"
cmp <(sed '/"guarded_early_trigger_device_materialized": true/d' \
          "$baseline_kernel") \
    <(sed '/"guarded_early_trigger_device_materialized": true/d' \
          "$guarded_kernel")
if cmp -s "$output_dir/baseline/mm_minimal" \
          "$output_dir/guarded/mm_minimal"; then
    echo "baseline and guarded executables are unexpectedly identical" >&2
    exit 1
fi

audit_dir="$output_dir/guarded/ir-audit"
mkdir -p "$audit_dir"
cp -a "$output_dir/guarded/meta" "$audit_dir/meta"
device_bc=("$output_dir/guarded/obj/"*-hip-amdgcn-amd-amdhsa-gfx90a.bc)
host_bc=("$output_dir/guarded/obj/"*-host-x86_64-unknown-linux-gnu.bc)
if [[ ${#device_bc[@]} -ne 1 || ${#host_bc[@]} -ne 1 ]]; then
    echo "could not identify one saved host/device mm_minimal bitcode input" >&2
    exit 1
fi
audit_env=(
    "GICC_MODE=lower"
    "GICC_TARGET=ofi-triggered"
    "GICC_META_DIR=$audit_dir/meta"
    "GICC_HINT_IN=$guarded_hint"
)
env "${audit_env[@]}" "$opt" -load-pass-plugin="$plugin" \
    '-passes=default<O3>' "${device_bc[0]}" \
    -o "$audit_dir/device-final.bc" >"$audit_dir/device-opt.log" 2>&1
env "${audit_env[@]}" "$opt" -load-pass-plugin="$plugin" \
    '-passes=default<O3>' "${host_bc[0]}" \
    -o "$audit_dir/host-final.bc" >"$audit_dir/host-opt.log" 2>&1
"$llvm_dis" "$audit_dir/device-final.bc" -o "$audit_dir/device-final.ll"
"$llvm_dis" "$audit_dir/host-final.bc" -o "$audit_dir/host-final.ll"
python3 "$auditor" --device "$audit_dir/device-final.ll" \
    --host "$audit_dir/host-final.ll" --out "$audit_dir/audit.json"

provenance="$output_dir/BUILD_PROVENANCE.txt"
{
    printf 'schema=gicc-guarded-early-trigger-build-v1\n'
    printf 'commit=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
    printf 'compiler=%s\n' "$compiler"
    "$compiler" --version | head -n 1
    for artifact in \
        "$source_file" "$helper_source" "$baseline_hint" "$guarded_hint" \
        "$checksum_source" "$auditor" "$plugin" \
        "$output_dir/common/checksum_hip_memcpy.o" \
        "$output_dir/baseline/meta/features.json" "$baseline_kernel" \
        "$output_dir/guarded/meta/features.json" "$guarded_kernel" \
        "$audit_dir/device-final.ll" "$audit_dir/host-final.ll" \
        "$audit_dir/audit.json" "$output_dir/baseline/mm_minimal" \
        "$output_dir/guarded/mm_minimal"; do
        sha256sum "$artifact"
    done
} >"$provenance"
