#!/bin/bash
# Build allreduce_bench through build_giomp_example.sh (every mode is in the
# one binary; the MPI modes do not initialize GiOMP).
# Usage: build_allreduce.sh [OUTDIR]   (default: build_ofi/allreduce)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GICC_ROOT="${GICC_ROOT:-$(cd "${HERE}/../../.." && pwd)}"
OUT="${1:-${GICC_ROOT}/build_ofi/allreduce}"
GICC_ROOT="${GICC_ROOT}" bash "${HERE}/../build_giomp_example.sh" \
    "${HERE}/allreduce_bench.cpp" "${OUT}/allreduce_bench"
