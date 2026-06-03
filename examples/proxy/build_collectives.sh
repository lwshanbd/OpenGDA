#!/bin/bash
# Build the GICC collective examples (allreduce_ring, alltoall) in BOTH
# transports from a single source each:
#   <name>_proxy : -DGICC_CPU_PROXY, links the proxy worker objects.
#   <name>_dwq   : no proxy define, rt.enable_host_wait_mode() picks the
#                  CXI deferred-work-queue / GPU-trigger path.
# No LTO pass needed (puts are issued via the device API / host rt.put(),
# not put_no_db loops). Just GICC runtime + libfabric + MPI.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
HIPCC=/opt/rocm-6.4.0/lib/llvm/bin/clang++
LIBFAB=/opt/cray/libfabric/2.1
MPI=/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0
# Cray MPICH GPU Transport Layer (GTL) — required so the MPI baseline in
# coll_bench can MPI_Allreduce/Alltoall directly on device pointers
# (MPICH_GPU_SUPPORT_ENABLED=1).
GTL=/opt/cray/pe/mpich/9.0.1/gtl/lib

SCRATCH="${GICC_ROOT}/build_ofi/coll_scratch"
mkdir -p "${SCRATCH}"

COMMON_CFLAGS=(
    -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1
    -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1
    -I"${GICC_ROOT}/src"
    -I"${GICC_ROOT}/src/gicc/platform/ofi/internal"
    -I"${LIBFAB}/include"
    -isystem "${MPI}/include"
    -O3 --offload-arch=gfx90a -std=gnu++17
)

LINK_LIBS=(
    -Wl,-rpath,"${LIBFAB}/lib64":"${MPI}/lib":"${GTL}"
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so
    -lpthread /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400
    "${MPI}/lib/libmpi_cray.so"
    "${GTL}/libmpi_gtl_hsa.so"
)

# build_one <example.cpp> <out-name> <proxy|dwq>
build_one() {
    local src="$1" out="$2" mode="$3"
    local objdir; objdir="$(mktemp -d -p "${SCRATCH}" obj.XXXXXX)"
    local cflags=("${COMMON_CFLAGS[@]}")
    local objs=()

    if [[ "${mode}" == "proxy" ]]; then
        cflags+=(-DGICC_CPU_PROXY=1)
    fi

    echo ">>> building ${out} (${mode})"
    "${HIPCC}" "${cflags[@]}" -x hip -c \
        "${GICC_ROOT}/examples/proxy/${src}" -o "${objdir}/app.o"
    objs+=("${objdir}/app.o")

    "${HIPCC}" "${cflags[@]}" -x hip -c \
        "${GICC_ROOT}/src/gicc/platform/ofi/runtime_helpers.cpp" \
        -o "${objdir}/runtime_helpers.o"
    objs+=("${objdir}/runtime_helpers.o")

    if [[ "${mode}" == "proxy" ]]; then
        local p
        for p in proxy_thread.cpp proxy_libfabric.cpp; do
            "${HIPCC}" "${cflags[@]}" -x hip -c \
                "${GICC_ROOT}/src/gicc/platform/ofi/proxy/${p}" \
                -o "${objdir}/${p%.cpp}.o"
            objs+=("${objdir}/${p%.cpp}.o")
        done
    fi

    "${HIPCC}" -O3 --offload-arch=gfx90a --hip-link \
        --rtlib=compiler-rt -unwindlib=libgcc \
        -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive \
        "${objs[@]}" \
        -o "${GICC_ROOT}/build_ofi/${out}" \
        "${LINK_LIBS[@]}"
    rm -rf "${objdir}"
    echo "BUILT: ${GICC_ROOT}/build_ofi/${out}"
}

build_one allreduce_ring.cpp allreduce_ring_proxy proxy
build_one allreduce_ring.cpp allreduce_ring_dwq   dwq
build_one alltoall.cpp        alltoall_proxy        proxy
build_one alltoall.cpp        alltoall_dwq          dwq
build_one coll_bench.cpp      coll_bench_proxy      proxy
build_one coll_bench.cpp      coll_bench_dwq        dwq

echo "All collective binaries built into ${GICC_ROOT}/build_ofi/"
