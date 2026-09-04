import importlib.util
import math
import re
import subprocess
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
COLLECTIVE = PASS_ROOT / "experiments" / "collective"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(PASS_ROOT / "experiments"))
sys.path.insert(0, str(COLLECTIVE))
SCRIPT = COLLECTIVE / "analyze_collective_n6_llm_runtime_validation.py"
CONTROLLER = COLLECTIVE / "continue_collective_n6_llm_runtime_validation.sh"
RUNNER = COLLECTIVE / "run_collective_n6_llm_runtime_validation.sh"
SPEC = importlib.util.spec_from_file_location("collective_runtime_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)


ROLES = (
    "control:anchor",
    "control:deterministic",
    "control:oracle",
    "descriptors:primary_modal_representative",
    "descriptors:posthoc_capability_upper_bound",
    "relational:primary_modal_representative",
    "relational:posthoc_capability_upper_bound",
    "opaque:primary_modal_representative",
    "opaque:posthoc_capability_upper_bound",
)


def minimal_plan():
    return {
        "policies": [
            {"name": f"policy{index:02d}", "roles": [role]}
            for index, role in enumerate(ROLES, start=1)
        ],
    }


def unit_weights():
    return {
        f"message_bytes:{size}": 1.0 / len(analysis.SIZES)
        for size in analysis.SIZES
    }


def synthetic_rows(relational_primary=75.0):
    costs = {
        "control:anchor": 100.0,
        "control:deterministic": 90.0,
        "control:oracle": 65.0,
        "descriptors:primary_modal_representative": 92.0,
        "descriptors:posthoc_capability_upper_bound": 80.0,
        "relational:primary_modal_representative": relational_primary,
        "relational:posthoc_capability_upper_bound": 60.0,
        "opaque:primary_modal_representative": 96.0,
        "opaque:posthoc_capability_upper_bound": 85.0,
    }
    policies = minimal_plan()["policies"]
    return {
        replicate: {
            policy["name"]: {
                str(size): costs[policy["roles"][0]] * factor
                for size in analysis.SIZES
            }
            for policy in policies
        }
        for replicate, factor in ((1, 1.00), (2, 1.01), (3, 0.99))
    }


class AnalyzeCollectiveN6LlmRuntimeValidationTests(unittest.TestCase):
    def test_weighted_geomean_uses_frozen_message_distribution(self):
        values = {str(size): 1.0 for size in analysis.SIZES}
        values[str(analysis.SIZES[0])] = 4.0
        weights = {
            f"message_bytes:{size}": 1.0 for size in analysis.SIZES
        }
        weights[f"message_bytes:{analysis.SIZES[0]}"] = 9.0
        expected = math.exp(9.0 * math.log(4.0) / 17.0)
        self.assertAlmostEqual(expected, analysis.weighted_cost(values, weights))

    def test_relational_modal_and_context_pass_preregistered_gates(self):
        result = analysis.analyze_rows(
            minimal_plan(), synthetic_rows(), unit_weights()
        )
        gates = result["claim_gates"]
        self.assertTrue(
            gates["stable_relational_modal_runtime_improvement_supported"]
        )
        self.assertTrue(gates["relational_context_runtime_effect_supported"])
        self.assertTrue(
            gates["posthoc_capability_ceiling_runtime_potential_observed"]
        )
        self.assertFalse(gates["portfolio_generalization_supported"])
        self.assertFalse(gates["paper_mainline_complete"])
        speedup = result["representative_comparisons"][
            "relational:primary_modal_representative"
        ]["over_deterministic"]["subject_speedup"]
        self.assertEqual(27, speedup["bootstrap_samples"])
        self.assertEqual(3, speedup["allocation_wins"])

    def test_posthoc_ceiling_cannot_rescue_weak_primary_policy(self):
        result = analysis.analyze_rows(
            minimal_plan(), synthetic_rows(relational_primary=98.5),
            unit_weights(),
        )
        gates = result["claim_gates"]
        self.assertFalse(
            gates["stable_relational_modal_runtime_improvement_supported"]
        )
        self.assertTrue(
            gates["posthoc_capability_ceiling_runtime_potential_observed"]
        )
        self.assertFalse(gates["paper_mainline_complete"])

    def test_entrypoints_are_serial_pdebug_and_model_free(self):
        for script in (CONTROLLER, RUNNER):
            subprocess.run(["bash", "-n", script], check=True)
            text = script.read_text(encoding="utf-8")
            self.assertNotIn("-q pci", text)
            self.assertNotIn("flux cancel", text)
            self.assertIn("-N6", text)
            self.assertIn("-n48", text)
        controller = CONTROLLER.read_text(encoding="utf-8")
        runner = RUNNER.read_text(encoding="utf-8")
        self.assertEqual(1, controller.count("flux batch"))
        self.assertIn("flux batch -q pdebug -N6 -n48", controller)
        self.assertIn("for replicate in 1 2 3; do", controller)
        self.assertRegex(
            controller,
            re.compile(
                r"for replicate in 1 2 3; do.*flux batch.*"
                r"python3 \"\$monitor\".*done\n\nset_state analyzing",
                re.DOTALL,
            ),
        )
        self.assertEqual(0, runner.count("flux batch"))
        self.assertNotIn("provider", runner.lower())
        self.assertNotIn("model", runner.lower())


if __name__ == "__main__":
    unittest.main()
