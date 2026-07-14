#!/bin/bash
# Build E4 against the prebuilt GiOMP library.
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
exec bash "${GICC_ROOT}/examples/omp/build_giomp_example.sh" \
    "${GICC_ROOT}/examples/omp/e4_get.cpp" \
    "${GICC_ROOT}/build_ofi/e4_get"
