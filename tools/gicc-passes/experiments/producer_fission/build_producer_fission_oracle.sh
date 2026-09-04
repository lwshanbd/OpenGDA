#!/usr/bin/env bash
# Build unchanged Jacobi with and without the compiler fission oracle.
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

compiler=/opt/rocm-6.4.0/lib/llvm/bin/clang++
plugin="$repo_root/tools/gicc-passes/build/libgicc-passes.so"
source_file="$repo_root/examples/ofi/jacobi.cpp"
helper_source="$repo_root/src/gicc/platform/ofi/runtime_helpers.cpp"
hint="$repo_root/tools/gicc-passes/tests/lit/Inputs/hint_fission_dwq.json"
for path in "$compiler" "$plugin" "$source_file" "$helper_source" "$hint"; do
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
    "-fpass-plugin=$plugin"
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

build_arm() {
    local arm=$1
    local oracle=$2
    local arm_dir="$output_dir/$arm"
    local meta="$arm_dir/meta"
    mkdir -p "$arm_dir/obj" "$meta"
    local env_args=(
        "GICC_MODE=lower"
        "GICC_META_DIR=$meta"
        "GICC_HINT_IN=$hint"
    )
    if [[ $oracle == yes ]]; then
        env_args+=("GICC_PRODUCER_FISSION_ORACLE=1")
    fi
    env "${env_args[@]}" "$compiler" "${compile_flags[@]}" \
        "$source_file" -o "$arm_dir/obj/jacobi.o" \
        >"$arm_dir/compile-jacobi.log" 2>&1
    env "${env_args[@]}" "$compiler" "${compile_flags[@]}" \
        "$helper_source" -o "$arm_dir/obj/runtime_helpers.o" \
        >"$arm_dir/compile-runtime.log" 2>&1
    env "${env_args[@]}" "$compiler" "${link_flags[@]}" \
        "$arm_dir/obj/jacobi.o" "$arm_dir/obj/runtime_helpers.o" \
        -o "$arm_dir/jacobi" "${link_libraries[@]}" \
        >"$arm_dir/link.log" 2>&1
}

mkdir -p "$output_dir"
build_arm baseline no
build_arm fission yes

if rg -q 'producer-fission-(host|device).*materialized' \
       "$output_dir/baseline/compile-jacobi.log" \
       "$output_dir/baseline/link.log"; then
    echo "baseline unexpectedly materialized producer fission" >&2
    exit 1
fi
rg -q 'producer-fission-device.*materialized exact producer/remainder' \
    "$output_dir/fission/compile-jacobi.log"
rg -q 'producer-fission-host.*materialized guarded two-phase' \
    "$output_dir/fission/compile-jacobi.log"
cmp "$output_dir/baseline/meta/features.json" \
    "$output_dir/fission/meta/features.json"
kernel_metadata=("$output_dir/baseline/meta/"*.json)
if [[ ${#kernel_metadata[@]} -ne 2 ]]; then
    echo "baseline did not emit exactly features plus one kernel metadata file" >&2
    exit 1
fi
kernel_metadata=("$output_dir/fission/meta/"*.json)
if [[ ${#kernel_metadata[@]} -ne 2 ]]; then
    echo "fission did not emit exactly features plus one kernel metadata file" >&2
    exit 1
fi
baseline_kernel=$(find "$output_dir/baseline/meta" -maxdepth 1 -type f \
    -name '*.json' ! -name features.json -print)
fission_kernel=$(find "$output_dir/fission/meta" -maxdepth 1 -type f \
    -name '*.json' ! -name features.json -print)
if rg -q '"producer_fission_device_materialized"' "$baseline_kernel"; then
    echo "baseline unexpectedly carries a device-fission attestation" >&2
    exit 1
fi
rg -q '"producer_fission_device_materialized": true' "$fission_kernel"
cmp <(sed '/"producer_fission_device_materialized": true/d' \
          "$baseline_kernel") \
    <(sed '/"producer_fission_device_materialized": true/d' \
          "$fission_kernel")
if cmp -s "$output_dir/baseline/jacobi" "$output_dir/fission/jacobi"; then
    echo "baseline and fission executables are unexpectedly identical" >&2
    exit 1
fi

provenance="$output_dir/BUILD_PROVENANCE.txt"
{
    printf 'schema=gicc-producer-fission-build-v1\n'
    printf 'commit=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
    printf 'compiler=%s\n' "$compiler"
    "$compiler" --version | head -n 1
    for artifact in \
        "$source_file" "$helper_source" "$hint" "$plugin" \
        "$output_dir/baseline/meta/features.json" "$baseline_kernel" \
        "$output_dir/fission/meta/features.json" "$fission_kernel" \
        "$output_dir/baseline/jacobi" "$output_dir/fission/jacobi"; do
        sha256sum "$artifact"
    done
} >"$provenance"
