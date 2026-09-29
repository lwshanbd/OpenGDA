#!/bin/bash
# Build the graph examples, each once as plain GPU-aware MPI and once as
# GiOMP. Both go through build_giomp_example.sh -- same compiler, same flags,
# and neither through the gicc-passes plugin -- so the kernels they share are
# compiled alike and only the communication differs.
#
# Usage: build_graph.sh [OUTDIR] [APP...]   (default: build_ofi/graph, all apps)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
GICC_ROOT="${GICC_ROOT:-$(cd "${HERE}/../../.." && pwd)}"
OUT="${1:-${GICC_ROOT}/build_ofi/graph}"
shift || true
APPS=("$@")
[[ ${#APPS[@]} -eq 0 ]] && APPS=(pagerank bfs)
for app in "${APPS[@]}"; do
    for backend in mpi giomp; do
        GICC_ROOT="${GICC_ROOT}" GIOMP_EXTRA_FLAGS="-DGRAPH_BACKEND_${backend^^}" \
            bash "${HERE}/../build_giomp_example.sh" "${HERE}/${app}.cpp" \
            "${OUT}/${app}_${backend}"
    done
done
