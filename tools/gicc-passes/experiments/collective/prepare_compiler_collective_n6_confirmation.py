#!/usr/bin/env python3
"""Derive the preregistered N6 confirmation from a passed N6 scout."""

import importlib.util
from pathlib import Path
import sys

import analyze_compiler_collective_hierpipe_n6_scout as n6_scout


BASE_SCRIPT = (
    Path(__file__).resolve().parent
    / "prepare_compiler_collective_n8_confirmation.py"
)
SPEC = importlib.util.spec_from_file_location(
    "gicc_collective_confirmation_transition_base_n6", BASE_SCRIPT
)
base = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)

base.scout = n6_scout.base
base.TRANSITION_SCHEMA = "gicc-collective-n6-confirmation-transition-v1"
base.GRAPH_ID = n6_scout.base.GRAPH_ID
base.BUNDLE_ID = n6_scout.base.BUNDLE_ID
base.SYSTEM_PROTOCOL = (
    Path(__file__).resolve().parent / "HIERPIPE_N6_CONFIRMATION_TRANSITION.md"
)
base.SCOUT_SCHEMA = "gicc-collective-hierpipe-n6-scout-v1"
base.CAPACITY_GATE_KEY = "n6_capacity_gate"
base.TOPOLOGY_LABEL = "N6"
base.NODES = 6
base.RANKS = 48
base.RANKS_PER_NODE = 8
base.SCOUT_FILE_ROLE = "passed_n6_scout"
base.DECISION_ORIGIN = (
    "preregistered_n6_scout_to_confirmation_selector"
)
base.DERIVED_RATIONALE = (
    "preregistered per-bin geometric-mean selector from the N6 scout"
)
base.UNIFORM_RATIONALE = (
    "regenerated best uniform hierarchy-pipeline arm from the N6 scout"
)
base.PREPARER_PATH = Path(__file__).resolve()
base.SUPPORT_PREPARER_FILES = ((BASE_SCRIPT, "transition_preparer_base"),)
base.TRANSITION_SUPPORT_ROLES = {"transition_preparer_base"}
base.PROGRAM_NAME = "compiler-collective-n6-confirmation"


if __name__ == "__main__":
    raise SystemExit(base.main())
