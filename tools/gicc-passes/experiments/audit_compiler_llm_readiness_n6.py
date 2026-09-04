#!/usr/bin/env python3
"""Run the compiler-only readiness audit after an N6 suite transition."""

from pathlib import Path

import audit_compiler_llm_readiness as base
import analyze_compiler_collective_n6_confirmation as n6_confirmation
import prepare_collective_n6_suite_refreeze as n6_refreeze


base.EXPECTED_LABELS = (
    base.BASE_EXPECTED_LABELS - {"collective_n8"} | {"collective_n6"}
)
base.COLLECTIVE_LABEL = "collective_n6"
base.COLLECTIVE_TOPOLOGY_LABEL = "n6"
base.COLLECTIVE_SCOUT_SCHEMA = "gicc-collective-hierpipe-n6-scout-v1"
base.COLLECTIVE_CAPACITY_GATE_KEY = "n6_capacity_gate"
base.COLLECTIVE_CONFIRMATION_SCHEMA = n6_confirmation.base.RESULT_SCHEMA
base.COLLECTIVE_CONFIRMATION = n6_confirmation.base
base.COLLECTIVE_CONFIRMATION_ANALYZER = Path(n6_confirmation.__file__).resolve()
base.COLLECTIVE_REFREEZE = n6_refreeze
base.COLLECTIVE_REFREEZE_ERROR = n6_refreeze.RefreezeError
base.COLLECTIVE_REFREEZER = Path(n6_refreeze.__file__).resolve()
base.COLLECTIVE_CONFIRMATION_STATE_DEFAULT = (
    base.ROOT / "build_ofi/compiler_collective_n6_confirmation_20260904.state"
)
base.COLLECTIVE_CONFIRMATION_ANALYSIS_DEFAULT = (
    base.ROOT
    / "build_ofi/compiler_collective_n6_confirmation_20260904/analysis.json"
)
base.COLLECTIVE_REFREEZE_MANIFEST_DEFAULT = (
    base.ROOT
    / "build_ofi/compiler_collective_n6_suite_refreeze_20260904/manifest.json"
)
base.PROGRAM_NAME = "compiler-llm-readiness-n6"


if __name__ == "__main__":
    raise SystemExit(base.main())
