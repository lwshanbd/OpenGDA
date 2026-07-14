#!/bin/bash
# Build E1 against the prebuilt GiOMP library.
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
exec bash "${GICC_ROOT}/examples/omp/build_giomp_example.sh" \
    "${GICC_ROOT}/examples/omp/e1_single_put.cpp" \
    "${GICC_ROOT}/build_ofi/e1_single_put"
