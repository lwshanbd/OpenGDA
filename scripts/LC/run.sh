#!/bin/bash
# Run script for GICC on LC — OFI/CXI backend
# Usage: ./run.sh <binary> [args...]
#   e.g. ./run.sh test_gicc
#        ./run.sh mm_gda_minimal_gicc 1024
#        ./run.sh gda_benchmark_gicc
#
# Environment variables:
#   NODES       — number of nodes (default: 2)
#   TIME        — time limit in minutes (default: 2)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$SCRIPT_DIR/../../build_ofi"

NODES="${NODES:-2}"
TIME="${TIME:-2}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <binary> [args...]"
    echo ""
    echo "Available binaries:"
    ls "$BUILD_DIR"/test_gicc "$BUILD_DIR"/gda_benchmark_gicc "$BUILD_DIR"/mm_gda_minimal_gicc 2>/dev/null | xargs -n1 basename
    exit 1
fi

BINARY="$1"
shift

# Resolve binary path
if [ -f "$BUILD_DIR/$BINARY" ]; then
    BINARY="$BUILD_DIR/$BINARY"
elif [ ! -f "$BINARY" ]; then
    echo "Error: binary '$BINARY' not found in $BUILD_DIR or as absolute path"
    exit 1
fi

echo "=== GICC Run (LC) ==="
echo "Binary:  $BINARY"
echo "Nodes:   $NODES"
echo "Args:    $@"
echo ""

FI_MR_CACHE_MAX_COUNT=0 srun \
    -N "$NODES" \
    -n "$NODES" \
    --ntasks-per-node=1 \
    -t "$TIME" \
    "$BINARY" "$@"
