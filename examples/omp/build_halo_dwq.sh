#!/bin/bash
# Build the DWQ halo against libgicc_omp; only the app TU uses the LTO pass.
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export GIOMP_DWQ_SOURCE="${GICC_ROOT}/examples/omp/halo_dwq.cpp"
export GIOMP_DWQ_OUT="${GICC_ROOT}/build_ofi/halo_dwq"
exec bash "${GICC_ROOT}/examples/omp/build_omp_dwq.sh"
