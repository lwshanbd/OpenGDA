#!/bin/bash
# 01_minimal_put: compile + run the existing examples/ofi/test_gicc.cpp
# end-to-end via the LTO pipeline. Verifies (a) the full plugin pipeline
# transforms real HIP source without crashing, (b) the resulting binary
# links + runs on Tioga compute nodes, (c) the 4-iteration loop pattern
# correctly moves halo data via the LTO-emitted host trace function.
#
# Pass criterion: srun output contains "PASSED (16384 bytes)".
set -euo pipefail

if [[ -z "${GICC_ROOT:-}" ]]; then
    echo "ERROR: GICC_ROOT not set (driver should set it)" >&2
    exit 2
fi

# Build artifacts must live on a shared filesystem so srun on a
# different node can find the binary. /var/tmp is node-local on Tioga.
SCRATCH_ROOT="${GICC_ROOT}/build_ofi/integration_scratch"
mkdir -p "${SCRATCH_ROOT}"
META_DIR="$(mktemp -d -p "${SCRATCH_ROOT}" meta.XXXXXX)"
OBJ_DIR="$(mktemp -d -p "${SCRATCH_ROOT}" obj.XXXXXX)"
trap 'rm -rf "${META_DIR}" "${OBJ_DIR}"' EXIT

HIPCC=/opt/rocm-6.4.0/lib/llvm/bin/clang++

CFLAGS=(
    -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1 -DUSE_PROF_API=1
    -D__HIP_PLATFORM_AMD__=1 -D__HIP_ROCclr__=1
    -I"${GICC_ROOT}/src"
    -I"${GICC_ROOT}/src/gicc/platform/ofi/internal"
    -I/opt/cray/libfabric/2.1/include
    -isystem /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/include
    -O3 --offload-arch=gfx90a -std=gnu++17
    -fpass-plugin="${GICC_PASSES_SO}" -flto
)

GICC_MODE=lower GICC_META_DIR="${META_DIR}" \
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c \
    "${GICC_ROOT}/examples/ofi/test_gicc.cpp" \
    -o "${OBJ_DIR}/test_gicc.o"

GICC_MODE=lower GICC_META_DIR="${META_DIR}" \
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c \
    "${GICC_ROOT}/src/gicc/platform/ofi/runtime_helpers.cpp" \
    -o "${OBJ_DIR}/runtime_helpers.o"

"${HIPCC}" -O3 --offload-arch=gfx90a --hip-link \
    --rtlib=compiler-rt -unwindlib=libgcc \
    -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive \
    "${OBJ_DIR}/test_gicc.o" "${OBJ_DIR}/runtime_helpers.o" \
    -o "${OBJ_DIR}/test_gicc" \
    -Wl,-rpath,/opt/cray/libfabric/2.1/lib64:/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib \
    /opt/cray/libfabric/2.1/lib64/libfabric.so /usr/lib64/libhwloc.so \
    -lpthread /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 \
    /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib/libmpi_cray.so \
    /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400

OUT=$(FI_MR_CACHE_MAX_COUNT=0 srun -p pci -N 2 -n 2 --ntasks-per-node=1 -t 1 \
       "${OBJ_DIR}/test_gicc" 2>&1)
echo "${OUT}"

if echo "${OUT}" | grep -q "PASSED (16384 bytes)"; then
    exit 0
fi
echo "FAIL: expected 'PASSED (16384 bytes)' in output"
exit 1
