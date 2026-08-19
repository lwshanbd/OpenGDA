#!/bin/bash
# Build two binaries from identical source:
#   ml_path_default -- no hint, production IPC_OR_DWQ default
#   ml_path_gbt     -- GBT hint trained with the query size held out
set -euo pipefail

GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
OUT="${GICC_ROOT}/build_ofi/ml_path_e2e"
META="${OUT}/meta"
OBJ="${OUT}/obj"
PASSES="${GICC_ROOT}/tools/gicc-passes/build/libgicc-passes.so"
FEATURES="${OUT}/features.json"
HINT="${OUT}/gbt-hint.json"
GRID="${GICC_ROOT}/docs/experiments/grid/grid_big.csv"
SRC="${GICC_ROOT}/examples/proxy/ml_path_e2e.cpp"

mkdir -p "${META}" "${OBJ}"
find "${META}" -mindepth 1 -maxdepth 1 -type f -delete

HIPCC=/opt/rocm-6.4.0/lib/llvm/bin/clang++
LIBFAB=/opt/cray/libfabric/2.1
MPI=/opt/cray/pe/mpich/9.0.1/ofi/cray/20.0
CFLAGS=(
  -DGICC_BOOTSTRAP_MPI=1 -DGICC_PLATFORM_OFI -DGICC_GPU_HIP=1
  -DGICC_CPU_PROXY=1 -DUSE_PROF_API=1 -D__HIP_PLATFORM_AMD__=1
  -D__HIP_ROCclr__=1 -I"${GICC_ROOT}/src"
  -I"${GICC_ROOT}/src/gicc/platform/ofi/internal" -I"${LIBFAB}/include"
  -isystem "${MPI}/include" -O3 --offload-arch=gfx90a -std=gnu++17
  -fpass-plugin="${PASSES}" -flto
)

env GICC_MODE=discover GICC_META_DIR="${META}" GICC_PROXY_ENABLED=1 \
  "${HIPCC}" "${CFLAGS[@]}" -x hip -c "${SRC}" -o "${OBJ}/discover.o"
env GICC_MODE=feature-extract GICC_META_DIR="${META}" \
  GICC_FEATURES_OUT="${FEATURES}" GICC_PROXY_ENABLED=1 \
  "${HIPCC}" "${CFLAGS[@]}" -x hip -c "${SRC}" -o "${OBJ}/features.o"

env -u PYTHONPATH python3 "${GICC_ROOT}/examples/proxy/ml_path_decider.py" \
  --grid "${GRID}" --features "${FEATURES}" --output "${HINT}" \
  --heldout-bytes 4096

compile_one() {
  local name="$1" hint="$2"
  local hint_env=()
  if [ -n "$hint" ]; then hint_env=(GICC_HINT_IN="$hint"); fi
  env GICC_MODE=lower GICC_META_DIR="${META}" GICC_PROXY_ENABLED=1 \
    "${hint_env[@]}" "${HIPCC}" "${CFLAGS[@]}" -x hip -c "${SRC}" \
    -o "${OBJ}/${name}.o"
}

compile_one default ""
compile_one gbt "${HINT}"

for extra in runtime_helpers proxy_thread proxy_libfabric; do
  src="${GICC_ROOT}/src/gicc/platform/ofi/"
  if [ "$extra" = runtime_helpers ]; then src+="runtime_helpers.cpp";
  else src+="proxy/${extra}.cpp"; fi
  env GICC_MODE=lower GICC_META_DIR="${META}" GICC_PROXY_ENABLED=1 \
    "${HIPCC}" "${CFLAGS[@]}" -x hip -c "$src" -o "${OBJ}/${extra}.o"
done

for name in default gbt; do
  "${HIPCC}" -O3 --offload-arch=gfx90a --hip-link -flto \
    --rtlib=compiler-rt -unwindlib=libgcc \
    "${OBJ}/${name}.o" "${OBJ}/runtime_helpers.o" \
    "${OBJ}/proxy_thread.o" "${OBJ}/proxy_libfabric.o" \
    -o "${OUT}/ml_path_${name}" \
    -Wl,-rpath,"${LIBFAB}/lib64:${MPI}/lib" \
    "${LIBFAB}/lib64/libfabric.so" /usr/lib64/libhwloc.so -lpthread \
    /opt/rocm-6.4.0/lib/libamdhip64.so.6.4.60400 "${MPI}/lib/libmpi_cray.so"
done

echo "built ${OUT}/ml_path_default and ${OUT}/ml_path_gbt"
