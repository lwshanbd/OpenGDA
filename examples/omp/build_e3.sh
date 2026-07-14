#!/bin/bash
# Build E3 against the prebuilt GiOMP library.
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
exec bash "${GICC_ROOT}/examples/omp/build_giomp_example.sh" \
    "${GICC_ROOT}/examples/omp/e3_halo.cpp" \
    "${GICC_ROOT}/build_ofi/e3_halo"
