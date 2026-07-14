#!/bin/bash
# Compatibility entry point: build E_DWQ against libgicc_omp.
set -euo pipefail
GICC_ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export GIOMP_DWQ_SOURCE="${GICC_ROOT}/examples/omp/e_dwq_single.cpp"
export GIOMP_DWQ_OUT="${GICC_ROOT}/build_ofi/e_dwq_single"
exec bash "${GICC_ROOT}/examples/omp/build_omp_dwq.sh"
