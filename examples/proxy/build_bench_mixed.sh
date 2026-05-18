#!/bin/bash
# Build bench_mixed_lto via LTO pipeline with a hint.json that splits
# the two put_no_db sites: site 0 → DWQ_TRIGGER, site 1 → CPU_PROXY_ENQUEUE.
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd ../.. && pwd)}"
PASSES_SO="${GICC_ROOT}/tools/gicc-passes/build/libgicc-passes.so"
SCRATCH="${GICC_ROOT}/build_ofi/lto_mixed_scratch"
mkdir -p "${SCRATCH}"
META_DIR="$(mktemp -d -p "${SCRATCH}" meta.XXXXXX)"
OBJ_DIR="$(mktemp -d -p "${SCRATCH}" obj.XXXXXX)"

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

echo "=== Phase 1: feature-extract to discover site IDs ==="
GICC_MODE=feature-extract GICC_META_DIR="${META_DIR}" \
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c \
    "${GICC_ROOT}/examples/proxy/bench_mixed_lto.cpp" \
    -o /dev/null 2>&1 | grep -E "(discovery|site|feature-extract)" | head -8

# Find the kernel JSON to read the put_no_db site IDs from.
KERNEL_JSON=$(ls "${META_DIR}"/_Z12mixed_kernel*.json 2>/dev/null | head -1)
if [[ -z "${KERNEL_JSON}" ]]; then
    echo "ERROR: kernel JSON not found in ${META_DIR}"
    ls -la "${META_DIR}"
    exit 1
fi
echo "kernel JSON: ${KERNEL_JSON}"

# Extract the put_no_db site_ids in order (site 0, site 1).
SITE0=$(python3 -c "
import json
d = json.load(open('${KERNEL_JSON}'))
puts = [o['site_id'] for o in d['ops'] if o['kind']=='put_no_db']
print(puts[0])
")
SITE1=$(python3 -c "
import json
d = json.load(open('${KERNEL_JSON}'))
puts = [o['site_id'] for o in d['ops'] if o['kind']=='put_no_db']
print(puts[1])
")
echo "  site 0 (DWQ): ${SITE0}"
echo "  site 1 (PROXY): ${SITE1}"

HINT_JSON="${SCRATCH}/hint_mixed.json"
cat > "${HINT_JSON}" <<EOF
{
  "version": 1,
  "schema_version": "gicc-hint-v1",
  "default_dispatch": "DWQ_TRIGGER",
  "sites": {
    "${SITE0}": { "dispatch": "DWQ_TRIGGER" },
    "${SITE1}": { "dispatch": "CPU_PROXY_ENQUEUE" }
  }
}
EOF
echo "wrote ${HINT_JSON}"

echo
echo "=== Phase 2: lower with hint.json ==="
# Re-run with mode=lower + hint to actually generate dispatch-specific code
GICC_MODE=lower GICC_PROXY_ENABLED=1 GICC_META_DIR="${META_DIR}" GICC_HINT_IN="${HINT_JSON}" \
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c \
    "${GICC_ROOT}/examples/proxy/bench_mixed_lto.cpp" \
    -o "${OBJ_DIR}/bench.o" 2>&1 | grep -E "(dispatch|hint|preserved|proxy_aware)" | head -10

# Compile dependencies
GICC_MODE=lower GICC_PROXY_ENABLED=1 GICC_META_DIR="${META_DIR}" GICC_HINT_IN="${HINT_JSON}" \
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c \
    "${GICC_ROOT}/src/gicc/platform/ofi/runtime_helpers.cpp" \
    -o "${OBJ_DIR}/runtime_helpers.o"

for src in proxy_thread.cpp proxy_libfabric.cpp; do
    GICC_MODE=lower GICC_PROXY_ENABLED=1 GICC_META_DIR="${META_DIR}" GICC_HINT_IN="${HINT_JSON}" \
        "${HIPCC}" "${CFLAGS[@]}" -x hip -c \
        "${GICC_ROOT}/src/gicc/platform/ofi/proxy/${src}" \
        -o "${OBJ_DIR}/${src%.cpp}.o"
done

echo
echo "=== Phase 3: link ==="
"${HIPCC}" -O3 --offload-arch=gfx90a --hip-link \
    --rtlib=compiler-rt -unwindlib=libgcc \
    -Wl,--whole-archive,-lhugetlbfs,--no-whole-archive \
    "${OBJ_DIR}/bench.o" \
    "${OBJ_DIR}/runtime_helpers.o" \
    "${OBJ_DIR}/proxy_thread.o" \
    "${OBJ_DIR}/proxy_libfabric.o" \
    -o "${GICC_ROOT}/build_ofi/bench_mixed_lto" \
    -Wl,-rpath,"${LIBFAB}/lib64":/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so \
    -lpthread /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 \
    /opt/cray/pe/mpich/9.0.1/ofi/cray/20.0/lib/libmpi_cray.so

echo "BUILT: ${GICC_ROOT}/build_ofi/bench_mixed_lto"
echo "HINT : ${HINT_JSON}"
