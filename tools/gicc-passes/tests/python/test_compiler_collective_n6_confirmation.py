import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


PASS_ROOT = Path(__file__).resolve().parents[2]
COLLECTIVE = PASS_ROOT / "experiments" / "collective"
REPO_ROOT = PASS_ROOT.parents[1]
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(COLLECTIVE))

import analyze_compiler_collective_hierpipe_n6_scout as scout
import analyze_compiler_collective_n6_confirmation as analysis
import gicc_llm_bridge as bridge
import prepare_compiler_collective_n6_confirmation as transition


def crossover_rows():
    rows = {}
    for replicate in (1, 2, 3):
        rows[replicate] = {arm: {} for arm in scout.base.ARMS}
        for size in scout.base.SIZES:
            key = str(size)
            large = size >= 4194304
            rows[replicate][scout.base.ARMS[0]][key] = (
                300.0 if large else 100.0
            )
            rows[replicate][scout.base.ARMS[1]][key] = (
                100.0 if large else 200.0
            )
            rows[replicate][scout.base.ARMS[2]][key] = 500.0
    return rows


def passed_scout():
    analyzed = scout.base.analyze_rows(crossover_rows())
    payload = {
        "schema_version": scout.base.RESULT_SCHEMA,
        "scope": "unit compiler-only N6 scout",
        "model_invoked": False,
        "application_source_modified": False,
        "graph_id": transition.base.GRAPH_ID,
        "bundle_id": transition.base.BUNDLE_ID,
        "job_id": "unit-job",
        "nodelist": [f"unit{index}" for index in range(6)],
        "block_monitors": [],
        **analyzed,
    }
    return {**payload, "result_id": bridge._fingerprint(payload)}


class CompilerCollectiveN6ConfirmationTests(unittest.TestCase):
    def test_passed_n6_scout_maps_to_preregistered_bins(self):
        selected, bins, uniform = transition.base.derive_bin_algorithms(
            passed_scout()
        )
        self.assertEqual(
            {
                "message-bin-0": scout.base.ARMS[0],
                "message-bin-1": scout.base.ARMS[0],
                "message-bin-2": scout.base.ARMS[1],
                "message-bin-3": scout.base.ARMS[1],
            },
            selected,
        )
        self.assertEqual(scout.base.ARMS[0], uniform)
        self.assertEqual(
            [1048576, 4194304, 8388608],
            bins["message-bin-2"]["scout_sizes"],
        )

    def test_wrong_n6_gate_or_n8_schema_fails_closed(self):
        negative = passed_scout()
        negative["n6_capacity_gate"]["passed"] = False
        payload = dict(negative)
        payload.pop("result_id")
        negative["result_id"] = bridge._fingerprint(payload)
        with self.assertRaises(transition.base.TransitionError):
            transition.base.validate_scout(negative)

        wrong_schema = passed_scout()
        wrong_schema["schema_version"] = (
            "gicc-collective-hierpipe-n8-scout-v1"
        )
        payload = dict(wrong_schema)
        payload.pop("result_id")
        wrong_schema["result_id"] = bridge._fingerprint(payload)
        with self.assertRaises(transition.base.TransitionError):
            transition.base.validate_scout(wrong_schema)

    def test_n6_contract_and_compiler_origin_are_distinct_from_n8(self):
        self.assertEqual(6, transition.base.NODES)
        self.assertEqual(48, transition.base.RANKS)
        self.assertEqual("passed_n6_scout", transition.base.SCOUT_FILE_ROLE)
        self.assertEqual(
            "gicc-collective-n6-confirmation-v1",
            analysis.base.RESULT_SCHEMA,
        )
        graph_path = (
            REPO_ROOT
            / "build_ofi/compiler_collective_capacity_n6_hierpipe_v1_20260904"
            / "discovery/graph.json"
        )
        if not graph_path.is_file():
            self.skipTest("N6 offline bundle is unavailable")
        graph = json.loads(graph_path.read_text())
        algorithms = {
            slot: scout.base.ARMS[0]
            for slot in transition.base.SLOT_SIZES
        }
        _, hint = transition.base.make_decision(
            graph, algorithms, "unit N6 decision"
        )
        self.assertEqual(
            "preregistered_n6_scout_to_confirmation_selector",
            hint["llm_metadata"]["decision_origin"],
        )
        self.assertFalse(hint["llm_metadata"]["model_invoked"])

    def test_n6_transition_round_trips_exact_content_addressed_roles(self):
        graph_path = (
            REPO_ROOT
            / "build_ofi/compiler_collective_capacity_n6_hierpipe_v1_20260904"
            / "discovery/graph.json"
        )
        heuristic_path = (
            REPO_ROOT
            / "build_ofi/compiler_collective_n6_structural_control_20260904"
            / "decision.json"
        )
        if not graph_path.is_file() or not heuristic_path.is_file():
            self.skipTest("N6 offline controls are unavailable")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            scout_path = root / "analysis.json"
            scout_path.write_text(
                json.dumps(passed_scout(), indent=2, sort_keys=True) + "\n"
            )
            output_dir = root / "transition"
            # Raw-monitor regeneration is covered by the scout analyzer tests.
            # Here the synthetic, content-addressed scout isolates the successor
            # transition's graph/policy/file-closure contract.
            with mock.patch.object(
                transition.base, "validate_scout_provenance"
            ):
                value = transition.base.prepare(
                    graph_path, scout_path, heuristic_path, output_dir
                )
                regenerated, paths = analysis.base.validate_transition(
                    output_dir / "transition.json"
                )
            self.assertEqual(value, regenerated)
            self.assertEqual("confirmation_plan_ready", value["status"])
            self.assertEqual(
                {
                    "compiler_graph",
                    "passed_n6_scout",
                    "frozen_structural_heuristic",
                    "preregistered_transition_protocol",
                    "transition_preparer",
                    "transition_preparer_base",
                    "derived_decision",
                    "derived_hint",
                    "uniform_decision",
                    "uniform_hint",
                    "heuristic_hint",
                },
                set(paths),
            )
            self.assertFalse(value["boundary"]["model_invoked"])
            self.assertFalse(
                value["boundary"]["application_source_modified"]
            )

    def test_n6_shell_entrypoints_are_pdebug_only_and_serial(self):
        runner = COLLECTIVE / "run_compiler_collective_n6_confirmation.sh"
        controller = (
            COLLECTIVE / "continue_compiler_collective_n6_confirmation.sh"
        )
        successor = (
            COLLECTIVE / "continue_compiler_collective_n6_after_scout.sh"
        )
        for script in (runner, controller, successor):
            subprocess.run(["bash", "-n", script], check=True)
            text = script.read_text()
            self.assertNotIn("-q pci", text)
            self.assertNotIn("flux cancel", text)
        for script in (runner, controller):
            text = script.read_text()
            self.assertIn("-N6", text)
            self.assertIn("-n48", text)
        controller_text = controller.read_text()
        self.assertIn("-q pdebug", controller_text)
        self.assertIn("for replicate in 1 2 3", controller_text)
        self.assertIn('python3 "$monitor"', controller_text)
        successor_text = successor.read_text()
        self.assertIn("--filter=active", successor_text)
        self.assertIn("skipped_no_incremental_policy", successor_text)


if __name__ == "__main__":
    unittest.main()
