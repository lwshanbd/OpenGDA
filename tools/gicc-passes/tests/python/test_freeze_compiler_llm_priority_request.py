import importlib.util
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PASS_ROOT / "experiments" / "freeze_compiler_llm_priority_request.py"
)
SPEC = importlib.util.spec_from_file_location("priority_request", SCRIPT)
selector = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = selector
SPEC.loader.exec_module(selector)


def entry(label, count, family="communication"):
    return {
        "label": label,
        "entry_id": f"entry:{label}",
        "graph_id": f"graph:{label}",
        "decision_family": family,
        "decision_space": {"independent_policy_count": count},
    }


class FreezeCompilerLlmPriorityRequestTests(unittest.TestCase):
    def test_widest_eligible_graph_is_selected(self):
        suite = {"entries": [
            entry("jacobi", 10),
            entry("loop_lto", 3),
            entry("collective_n6", 4096, "collective"),
        ]}
        protocol = {
            "entries": {
                label: {"eligible_to_freeze_provider_request": True}
                for label in ("jacobi", "loop_lto", "collective_n6")
            },
            "summary": {"runtime_eligible_entries": [
                "collective_n6", "jacobi", "loop_lto",
            ]},
        }
        ranked = selector.prioritized_entries(suite, protocol)
        self.assertEqual("collective_n6", ranked[0]["label"])
        self.assertEqual(4096, ranked[0]["independent_policy_count"])

    def test_ineligible_entries_are_never_ranked(self):
        suite = {"entries": [entry("collective_n6", 4096)]}
        protocol = {
            "entries": {"collective_n6": {
                "eligible_to_freeze_provider_request": False,
            }},
            "summary": {"runtime_eligible_entries": []},
        }
        self.assertEqual([], selector.prioritized_entries(suite, protocol))

    def test_tie_break_is_frozen_and_call_unauthorized(self):
        suite = {"entries": [
            entry("coalescing_placement", 4096),
            entry("collective_n6", 4096),
        ]}
        protocol = {
            "entries": {
                label: {"eligible_to_freeze_provider_request": True}
                for label in ("coalescing_placement", "collective_n6")
            },
            "summary": {"runtime_eligible_entries": [
                "coalescing_placement", "collective_n6",
            ]},
        }
        ranked = selector.prioritized_entries(suite, protocol)
        self.assertEqual("collective_n6", ranked[0]["label"])
        payload = selector.selection_payload(
            suite={"suite_id": "suite"},
            protocol={"protocol_id": "protocol"},
            ranked=ranked,
            request_id="request",
            evidence_paths={"selector": SCRIPT},
        )
        report = selector.verify_selection(selector.with_id(payload))
        self.assertFalse(report["authorization_granted"])
        self.assertEqual(0, report["currently_permitted_provider_calls"])


if __name__ == "__main__":
    unittest.main()
