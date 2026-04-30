#!/bin/bash
# tools/gicc-passes/tests/integration/run_all.sh
#
# L2 integration test driver: compiles real C++/HIP sources end-to-end
# through the LTO pipeline, runs them via srun, and verifies output.
# These tests need a GPU + libfabric + the ROCm Clang at the canonical
# path. Lit (L1) tests are sufficient for CI; these are nightly /
# pre-release.
#
# Each test is a self-contained subdirectory with:
#   - sources or a reference to an in-tree binary
#   - a `run.sh` that compiles + runs + checks output
#
# Usage:  ./run_all.sh [test_name_filter]
set -uo pipefail

cd "$(dirname "$0")"

GICC_ROOT="${GICC_ROOT:-$(cd ../../../.. && pwd)}"
PASSES_SO="${GICC_PASSES_SO:-${GICC_ROOT}/tools/gicc-passes/build/libgicc-passes.so}"

if [[ ! -f "${PASSES_SO}" ]]; then
    echo "ERROR: ${PASSES_SO} does not exist. Build the LTO pass plugin first."
    exit 1
fi

export GICC_ROOT
export GICC_PASSES_SO="${PASSES_SO}"

filter="${1:-}"
fail=0
pass=0
for d in */; do
    name="${d%/}"
    if [[ -n "${filter}" && "${name}" != *"${filter}"* ]]; then
        continue
    fi
    if [[ ! -x "${d}run.sh" ]]; then
        continue
    fi
    echo "=== ${name} ==="
    if (cd "${d}" && ./run.sh); then
        pass=$((pass + 1))
        echo "PASS: ${name}"
    else
        fail=$((fail + 1))
        echo "FAIL: ${name}"
    fi
    echo
done

echo "=========================================="
echo "L2 integration: ${pass} passed, ${fail} failed"
exit "${fail}"
