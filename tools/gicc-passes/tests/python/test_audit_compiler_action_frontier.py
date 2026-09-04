import copy
import importlib.util
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PASS_ROOT / "experiments" / "audit_compiler_action_frontier.py"
SPEC = importlib.util.spec_from_file_location("compiler_action_frontier", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def candidate(candidate_id):
    return {
        "candidate_id": candidate_id,
        "kind": "group_uniform_default",
        "routes": {},
        "effects": {},
        "materializer": {"sites": {}},
    }


class CompilerActionFrontierTests(unittest.TestCase):
    def test_boundary_forbids_every_external_or_performance_input(self):
        for field in (
            "application_source_read",
            "application_source_modified",
            "compiler_invoked",
            "scheduler_invoked",
            "model_invoked",
            "provider_invoked",
            "provider_call_authorized",
            "runtime_evidence_used",
            "expanded_graphs_written",
            "conditional_candidates_model_visible",
        ):
            self.assertIs(False, audit.BOUNDARY[field])

    def test_report_verifier_rejects_a_performance_claim(self):
        records = []
        for label, counts in sorted(audit.EXPECTED_COUNTS.items()):
            records.append({
                "label": label,
                "current_independent_policy_count": counts[0],
                "conditional_independent_policy_count": counts[1],
                "per_entry_policy_delta": 1,
                "runtime_status": "unconfirmed",
                "performance_claim_supported": False,
            })
        payload = {
            "schema_version": audit.REPORT_SCHEMA,
            "boundary": dict(audit.BOUNDARY),
            "conditional_frontier": records,
        }
        report = {"frontier_id": audit.bridge._fingerprint(payload), **payload}
        audit.verify_report(report)
        overstated = copy.deepcopy(report)
        overstated["conditional_frontier"][0]["performance_claim_supported"] = True
        changed_payload = dict(overstated)
        changed_payload.pop("frontier_id")
        overstated["frontier_id"] = audit.bridge._fingerprint(changed_payload)
        with self.assertRaisesRegex(audit.FrontierError, "overstates evidence"):
            audit.verify_report(overstated)

    def test_graph_summary_rejects_duplicate_candidate_ids(self):
        graph = {
            "opportunities": [{
                "opportunity_id": "opportunity:a",
                "candidates": [candidate("candidate:a"), candidate("candidate:a")],
            }]
        }
        original = audit.producer_expansion.groups.verified_graph
        audit.producer_expansion.groups.verified_graph = lambda value: value
        try:
            with self.assertRaisesRegex(audit.FrontierError, "duplicate"):
                audit.graph_summary(graph)
        finally:
            audit.producer_expansion.groups.verified_graph = original


if __name__ == "__main__":
    unittest.main()
