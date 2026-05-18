#!/bin/bash
# Manual LTO build for bench_pingpong_lto.cpp.
# Mirrors tools/gicc-passes/tests/integration/01_minimal_put/run.sh up
# through the link step, since the existing CMake build doesn't wire
# -fpass-plugin into the proxy examples.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd ../.. && pwd)}"
PASSES_SO="${GICC_PASSES_SO:-${GICC_ROOT}/tools/gicc-passes/build/libgicc-passes.so}"

if [[ ! -f "${PASSES_SO}" ]]; then
    echo "ERROR: ${PASSES_SO} does not exist. Build the LTO pass plugin first."
    exit 1
fi

SCRATCH="${GICC_ROOT}/build_ofi/lto_bench_scratch"
mkdir -p "${SCRATCH}"
META_DIR="$(mktemp -d -p "${SCRATCH}" meta.XXXXXX)"
OBJ_DIR="$(mktemp -d -p "${SCRATCH}" obj.XXXXXX)"
trap 'rm -rf "${OBJ_DIR}"' EXIT

HIPCC=/opt/rocm-6.4.0/lib/llvm/bin/clang++
LIBFAB=/opt/cray/libfabric/2.1

CFLAGS=(
    -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1
    -DGICC_CPU_PROXY=1 -DUSE_PROF_API=1
    -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1
    -I"${GICC_ROOT}/src"
    -I"${GICC_ROOT}/src/gicc/platform/ofi/internal"
    -I"${LIBFAB}/include"
    -isystem /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/include
    -O3 --offload-arch=gfx90a -std=gnu++17
    -fpass-plugin="${PASSES_SO}" -flto
)

set -x

# 1) Discover-pass run on the kernel TU first (populates per-kernel JSON
#    in META_DIR so the host-side trace synthesis can match by kernel
#    mangled name). The same hipcc invocation in mode=lower would also
#    do this for us, but emitting an explicit two-pass shows the flow.
GICC_MODE=lower GICC_META_DIR="${META_DIR}" \
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c \
    "${GICC_ROOT}/examples/proxy/bench_pingpong_lto.cpp" \
    -o "${OBJ_DIR}/bench.o"

# 2) Compile the runtime helpers — these provide the gicc_runtime_*
#    C ABI symbols the LTO-synthesized trace function will call.
GICC_MODE=lower GICC_META_DIR="${META_DIR}" \
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c \
    "${GICC_ROOT}/src/gicc/platform/ofi/runtime_helpers.cpp" \
    -o "${OBJ_DIR}/runtime_helpers.o"

# 3) Proxy infrastructure (only needed because GICC_CPU_PROXY=1 is on;
#    bench_pingpong_lto itself never invokes the proxy path).
for src in proxy_thread.cpp proxy_libfabric.cpp; do
    GICC_MODE=lower GICC_META_DIR="${META_DIR}" \
        "${HIPCC}" "${CFLAGS[@]}" -x hip -c \
        "${GICC_ROOT}/src/gicc/platform/ofi/proxy/${src}" \
        -o "${OBJ_DIR}/${src%.cpp}.o"
done

# 4) Link.
"${HIPCC}" -O3 --offload-arch=gfx90a --hip-link \
    --rtlib=compiler-rt -unwindlib=libgcc \
    -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive \
    "${OBJ_DIR}/bench.o" \
    "${OBJ_DIR}/runtime_helpers.o" \
    "${OBJ_DIR}/proxy_thread.o" \
    "${OBJ_DIR}/proxy_libfabric.o" \
    -o "${GICC_ROOT}/build_ofi/bench_pingpong_lto" \
    -Wl,-rpath,"${LIBFAB}/lib64":/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so \
    -lpthread /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 \
    /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib/libmpi_cray.so

set +x
echo "BUILT: ${GICC_ROOT}/build_ofi/bench_pingpong_lto"
echo "META : ${META_DIR}  (per-kernel JSON written by gicc-passes)"
