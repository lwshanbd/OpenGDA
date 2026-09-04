#!/usr/bin/env python3
"""Audit three independent N6 compiler-policy confirmation allocations."""

import importlib.util
from pathlib import Path
import sys

import prepare_compiler_collective_n6_confirmation as n6_transition


BASE_SCRIPT = (
    Path(__file__).resolve().parent
    / "analyze_compiler_collective_n8_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gicc_collective_confirmation_analysis_base_n6", BASE_SCRIPT
)
base = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

base.transition_base = n6_transition.base
base.SIZES = tuple(n6_transition.base.SIZES)
base.ConfirmError = n6_transition.base.TransitionError
base.sha256 = n6_transition.base.sha256_file
base.fingerprint = n6_transition.base.bridge._fingerprint
base.TOPOLOGY_LABEL = "N6"
base.NODES = 6
base.RANKS = 48
base.PPN = 8
base.RUNNER_FILENAME = "run_compiler_collective_n6_confirmation.sh"
base.CONTROLLER_FILENAME = "continue_compiler_collective_n6_confirmation.sh"
base.ANALYZER_PATH = Path(__file__).resolve()
base.SUPPORT_ANALYZER_FILES = (BASE_SCRIPT,)
base.LOG_PREFIX = "COLLECTIVE_N6_CONFIRM"
base.RESULT_SCHEMA = "gicc-collective-n6-confirmation-v1"
base.RESULT_SCOPE = (
    "Three sequential independent N6 pdebug allocations with Latin-"
    "rotated compiler/LTO policies; no model or source modification."
)
base.PROGRAM_NAME = "compiler-collective-n6-confirm"


if __name__ == "__main__":
    raise SystemExit(base.main())
