#!/bin/bash
# Build the early/late trigger-placement variant without changing benchmark
# source.  All arguments are the same as build_compiler_comm_plan_calibration.sh.
set -euo pipefail

ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
export GICC_COMM_PLAN_VARIANT=placement
exec bash "${ROOT}/examples/proxy/build_compiler_comm_plan_calibration.sh" "$@"
