#!/bin/bash
# Build an example whose kernels are followed by ordinary ompx_put calls, so
# that the gicc-passes plugin moves those puts into the kernels
# (tools/gicc-passes/include/GICCAfterPut.h). Two compiles of the source:
#
#   1. host side only, GICC_MODE=put-discover: records the puts after each
#      launch in the metadata directory;
#   2. the real build, GICC_MODE=chunk-lower: each kernel sends what it can
#      of the puts after it, and the host skips those.
#
# Usage: build_giomp_after.sh SOURCE OUTPUT
# GICC_PASSES_SO names the plugin (default: tools/gicc-passes/build), and
# GICC_META_DIR the metadata directory (default: OUTPUT.meta, emptied first).
# Everything else as for build_giomp_example.sh.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 SOURCE OUTPUT" >&2
    exit 2
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
GICC_ROOT="${GICC_ROOT:-$(cd "${HERE}/../.." && pwd)}"
PASSES="${GICC_PASSES_SO:-${GICC_ROOT}/tools/gicc-passes/build/libgicc-passes.so}"
SOURCE="$1"
OUTPUT="$2"
EXTRA="${GIOMP_EXTRA_FLAGS:-}"

META="${GICC_META_DIR:-${OUTPUT}.meta}"
rm -rf "${META}"
mkdir -p "${META}"
export GICC_META_DIR="${META}"

GICC_MODE=put-discover \
GIOMP_EXTRA_FLAGS="--offload-host-only -fpass-plugin=${PASSES} ${EXTRA}" \
    bash "${HERE}/build_giomp_example.sh" "${SOURCE}" "${OUTPUT}.discover"
rm -f "${OUTPUT}.discover"

GICC_MODE=chunk-lower \
GIOMP_EXTRA_FLAGS="-foffload-lto -fpass-plugin=${PASSES} ${EXTRA}" \
    bash "${HERE}/build_giomp_example.sh" "${SOURCE}" "${OUTPUT}"
