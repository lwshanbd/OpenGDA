#!/usr/bin/env python3
"""Build the held-out collective policy screen for the N6 topology."""

import analyze_compiler_collective_n6_confirmation as n6_confirmation
import prepare_collective_llm_policy_screen as base


base.CONFIRMATION_ANALYZER = n6_confirmation.base
base.CONFIRMATION_SCHEMA = n6_confirmation.base.RESULT_SCHEMA
base.TOPOLOGY_LABEL = "N6"
base.CONTROL_NODES = 6
base.CONTROL_RANKS = 48
base.RUNNER_NAME = "run_compiler_collective_n6_llm_controls.sh"
base.CONTROLLER_NAME = "continue_compiler_collective_n6_llm_controls.sh"


if __name__ == "__main__":
    raise SystemExit(base.main())
