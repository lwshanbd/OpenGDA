#!/bin/bash
# Build the proxy halo benchmark against the prebuilt GiOMP library.
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
exec bash "${GICC_ROOT}/examples/omp/build_giomp_example.sh" \
    "${GICC_ROOT}/examples/omp/halo_omp_bench.cpp" \
    "${GICC_ROOT}/build_ofi/halo_omp_bench"
