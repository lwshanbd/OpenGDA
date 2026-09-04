#!/usr/bin/env bash
# Build unchanged loop_lto source with generic-array and scalar-reuse traces.
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
source_file="$repo_root/examples/proxy/bench_pingpong_lto.cpp"
helper_source="$repo_root/src/gicc/platform/ofi/runtime_helpers.cpp"
proxy_thread="$repo_root/src/gicc/platform/ofi/proxy/proxy_thread.cpp"
proxy_fabric="$repo_root/src/gicc/platform/ofi/proxy/proxy_libfabric.cpp"
baseline_hint="$script_dir/hint_baseline_dwq.json"
reused_hint="$script_dir/hint_reused_dwq.json"
auditor="$script_dir/audit_reused_loop_descriptor_ir.py"
for path in "$compiler" "$opt" "$llvm_dis" "$plugin" "$source_file" \
            "$helper_source" "$proxy_thread" "$proxy_fabric" \
            "$baseline_hint" "$reused_hint" "$auditor"; do
    if [[ ! -f $path ]]; then
        echo "missing build input: $path" >&2
        exit 2
    fi
done

compile_flags=(
    "-DGICC_BOOTSTRAP_MPI=1" "-DGICC_GPU_HIP=1" -DGICC_PLATFORM_OFI
    "-DGICC_CPU_PROXY=1" "-DUSE_PROF_API=1"
    "-D__HIP_PLATFORM_AMD__=1" "-D__HIP_ROCclr__=1"
    -I"$repo_root/src" -I"$repo_root/src/gicc/platform/ofi/internal"
    -I/opt/cray/libfabric/2.1/include
    -isystem /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/include
    -O3 "--offload-arch=gfx90a" "-std=gnu++17" -flto
    "-fpass-plugin=$plugin" -x hip -c
)
link_flags=(
    -O3 "--offload-arch=gfx90a" -flto --hip-link
    "--rtlib=compiler-rt" "-unwindlib=libgcc" "-fpass-plugin=$plugin"
    "-Wl,--whole-archive,-lhugetlbfs,--no-whole-archive"
    "-Wl,-rpath,/opt/cray/libfabric/2.1/lib64:/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib"
)
link_libraries=(
    /opt/cray/libfabric/2.1/lib64/libfabric.so /usr/lib64/libhwloc.so
    -lpthread /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400
    /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib/libmpi_cray.so
)

source_before=$(sha256sum "$source_file")
source_before=${source_before%% *}

build_arm() {
    local arm=$1
    local hint=$2
    local arm_dir="$output_dir/$arm"
    local meta="$arm_dir/meta"
    mkdir -p "$arm_dir/obj" "$meta"
    local env_args=(
        "GICC_MODE=lower" "GICC_TARGET=ofi-triggered"
        "GICC_META_DIR=$meta" "GICC_HINT_IN=$hint"
    )
    env "${env_args[@]}" "$compiler" -save-temps=obj \
        "${compile_flags[@]}" "$source_file" \
        -o "$arm_dir/obj/bench_pingpong_lto.o" \
        >"$arm_dir/compile-app.log" 2>&1
    env "${env_args[@]}" "$compiler" "${compile_flags[@]}" \
        "$helper_source" -o "$arm_dir/obj/runtime_helpers.o" \
        >"$arm_dir/compile-runtime.log" 2>&1
    env "${env_args[@]}" "$compiler" "${compile_flags[@]}" \
        "$proxy_thread" -o "$arm_dir/obj/proxy_thread.o" \
        >"$arm_dir/compile-proxy-thread.log" 2>&1
    env "${env_args[@]}" "$compiler" "${compile_flags[@]}" \
        "$proxy_fabric" -o "$arm_dir/obj/proxy_libfabric.o" \
        >"$arm_dir/compile-proxy-fabric.log" 2>&1
    env "${env_args[@]}" "$compiler" "${link_flags[@]}" \
        "$arm_dir/obj/bench_pingpong_lto.o" \
        "$arm_dir/obj/runtime_helpers.o" "$arm_dir/obj/proxy_thread.o" \
        "$arm_dir/obj/proxy_libfabric.o" \
        -o "$arm_dir/bench_pingpong_lto" "${link_libraries[@]}" \
        >"$arm_dir/link.log" 2>&1
}

mkdir -p "$output_dir"
build_arm baseline "$baseline_hint"
build_arm reused "$reused_hint"

if rg -q 'materialized one invariant descriptor' \
       "$output_dir/baseline/compile-app.log"; then
    echo "baseline unexpectedly materialized descriptor reuse" >&2
    exit 1
fi
rg -q 'materialized one invariant descriptor' \
    "$output_dir/reused/compile-app.log"
cmp "$output_dir/baseline/meta/features.json" \
    "$output_dir/reused/meta/features.json"
rg -q '"descriptor_reusable": true' \
    "$output_dir/reused/meta/features.json"
rg -q '"bound_param_type": "i32"' \
    "$output_dir/reused/meta/features.json"

baseline_kernel=$(find "$output_dir/baseline/meta" -maxdepth 1 -type f \
    -name '*.json' ! -name features.json -print)
reused_kernel=$(find "$output_dir/reused/meta" -maxdepth 1 -type f \
    -name '*.json' ! -name features.json -print)
if [[ -z $baseline_kernel || -z $reused_kernel ]]; then
    echo "each arm must emit one kernel metadata file" >&2
    exit 1
fi
cmp "$baseline_kernel" "$reused_kernel"
if cmp -s "$output_dir/baseline/bench_pingpong_lto" \
          "$output_dir/reused/bench_pingpong_lto"; then
    echo "baseline and reused executables are unexpectedly identical" >&2
    exit 1
fi

audit_dir="$output_dir/ir-audit"
mkdir -p "$audit_dir"
for arm in baseline reused; do
    cp -a "$output_dir/$arm/meta" "$audit_dir/$arm-meta"
    device_bc=("$output_dir/$arm/obj/"*-hip-amdgcn-amd-amdhsa-gfx90a.bc)
    host_bc=("$output_dir/$arm/obj/"*-host-x86_64-unknown-linux-gnu.bc)
    if [[ ${#device_bc[@]} -ne 1 || ${#host_bc[@]} -ne 1 ]]; then
        echo "could not identify saved host/device bitcode for $arm" >&2
        exit 1
    fi
    if [[ $arm == baseline ]]; then hint=$baseline_hint; else hint=$reused_hint; fi
    audit_env=(
        "GICC_MODE=lower" "GICC_TARGET=ofi-triggered"
        "GICC_META_DIR=$audit_dir/$arm-meta" "GICC_HINT_IN=$hint"
    )
    env "${audit_env[@]}" "$opt" -load-pass-plugin="$plugin" \
        '-passes=default<O3>' "${device_bc[0]}" \
        -o "$audit_dir/$arm-device.bc" \
        >"$audit_dir/$arm-device-opt.log" 2>&1
    env "${audit_env[@]}" "$opt" -load-pass-plugin="$plugin" \
        '-passes=default<O3>' "${host_bc[0]}" \
        -o "$audit_dir/$arm-host.bc" \
        >"$audit_dir/$arm-host-opt.log" 2>&1
    "$llvm_dis" "$audit_dir/$arm-device.bc" \
        -o "$audit_dir/$arm-device.ll"
    "$llvm_dis" "$audit_dir/$arm-host.bc" \
        -o "$audit_dir/$arm-host.ll"
done
python3 "$auditor" \
    --baseline-host "$audit_dir/baseline-host.ll" \
    --reused-host "$audit_dir/reused-host.ll" \
    --baseline-device "$audit_dir/baseline-device.ll" \
    --reused-device "$audit_dir/reused-device.ll" \
    --out "$audit_dir/audit.json"

source_after=$(sha256sum "$source_file")
source_after=${source_after%% *}
if [[ $source_before != "$source_after" ]]; then
    echo "application source changed during compiler experiment build" >&2
    exit 1
fi

provenance="$output_dir/BUILD_PROVENANCE.txt"
{
    printf 'schema=gicc-reused-loop-descriptor-build-v1\n'
    printf 'commit=%s\n' "$(git -C "$repo_root" rev-parse HEAD)"
    printf 'application_source_sha256=%s\n' "$source_after"
    printf 'compiler=%s\n' "$compiler"
    "$compiler" --version | head -n 1
    for artifact in "$source_file" "$helper_source" "$baseline_hint" \
        "$reused_hint" "$auditor" "$plugin" \
        "$output_dir/baseline/meta/features.json" "$baseline_kernel" \
        "$output_dir/reused/meta/features.json" "$reused_kernel" \
        "$audit_dir/baseline-host.ll" "$audit_dir/reused-host.ll" \
        "$audit_dir/baseline-device.ll" "$audit_dir/reused-device.ll" \
        "$audit_dir/audit.json" \
        "$output_dir/baseline/bench_pingpong_lto" \
        "$output_dir/reused/bench_pingpong_lto"; do
        sha256sum "$artifact"
    done
} >"$provenance"

printf 'BUILT %s\n' "$output_dir"
printf 'SOURCE_SHA256 %s\n' "$source_after"
