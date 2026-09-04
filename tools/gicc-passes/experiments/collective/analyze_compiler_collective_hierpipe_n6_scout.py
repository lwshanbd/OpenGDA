#!/usr/bin/env python3
"""Audit the topology-matched n6 rotated hierarchy-pipeline scout."""

import importlib.util
from pathlib import Path
import sys


BASE_SCRIPT = (
    Path(__file__).resolve().parent
    / "analyze_compiler_collective_hierpipe_n8_scout.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gicc_hierpipe_scale_scout_base_n6", BASE_SCRIPT
)
base = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)


base.GRAPH_ID = (
    "sha256:a7aa1f681a64574251aa8e2511550fb8d46b2ea08b5a4aae35fd3865b860a0c0"
)
base.BUNDLE_ID = (
    "sha256:dc243d4f358d3c6b64eb9ff26ff5851ef458f9eed2f7be079424657125edc113"
)
base.TOPOLOGY_LABEL = "n6"
base.NODES = 6
base.RANKS = 48
base.PPN = 8
base.LOG_PREFIX = "HIERPIPE_N6"
base.PROTOCOL_FILENAME = "HIERPIPE_N6_PROTOCOL.md"
base.RUNNER_FILENAME = "run_compiler_collective_hierpipe_n6_scout.sh"
base.CONTROLLER_FILENAME = "continue_compiler_collective_hierpipe_n6_scout.sh"
base.ANALYZER_PATH = Path(__file__).resolve()
base.SUPPORT_FILENAMES = (
    "analyze_compiler_collective_hierpipe_n8_scout.py",
)
base.RESULT_SCHEMA = "gicc-collective-hierpipe-n6-scout-v1"
base.CAPACITY_GATE_KEY = "n6_capacity_gate"
base.RESULT_SCOPE = (
    "One six-node pdebug allocation with three rotated compiler-control "
    "blocks; no model or application-source change."
)
base.PROGRAM_NAME = "compiler-collective-hierpipe-n6-scout"


if __name__ == "__main__":
    raise SystemExit(base.main())
