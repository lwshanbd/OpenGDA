#!/bin/bash
# Tioga's login environment can put Python-3.9 user packages ahead of the
# system Python 3.13 packages.  Isolate the interpreter used for NumPy/sklearn.
set -euo pipefail
ROOT="${GICC_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
exec env -u PYTHONPATH PYTHONNOUSERSITE=1 /usr/tce/bin/python3 \
    "$ROOT/examples/minimod_paper/analyze.py" "$@"
