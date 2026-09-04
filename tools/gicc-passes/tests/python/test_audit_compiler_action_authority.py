import copy
import importlib.util
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PASS_ROOT / "experiments" / "audit_compiler_action_authority.py"
SPEC = importlib.util.spec_from_file_location("compiler_action_authority", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class CompilerActionAuthorityTests(unittest.TestCase):
    def test_route_materializer_has_exact_dispatch_and_no_transform(self):
        self.assertEqual(
            "proxy",
            audit.route_materializer({
                "dispatch": "CPU_PROXY_ENQUEUE", "transform": "NONE",
            }),
        )
        self.assertIsNone(audit.route_materializer({
            "dispatch": "DWQ_TRIGGER", "transform": "COALESCE_LOOP",
        }))
        self.assertIsNone(audit.route_materializer({
            "dispatch": "unknown", "transform": "NONE",
        }))

    def test_boundary_forbids_external_execution(self):
        for field in (
            "application_source_read", "application_source_modified",
            "compiler_invoked", "scheduler_invoked",
            "runtime_benchmark_invoked", "model_invoked", "provider_invoked",
            "provider_call_authorized",
        ):
            self.assertIs(False, audit.BOUNDARY[field])
        self.assertIs(True, audit.BOUNDARY["compiler_lto_decisions_only"])

    def test_verifier_rejects_intrinsic_llm_superiority_claim(self):
        payload = {
            "schema_version": audit.REPORT_SCHEMA,
            "boundary": dict(audit.BOUNDARY),
            "claim_separation": {
                "wider_authority_comes_from_compiler_interface_not_llm_identity": True,
                "structured_ml_could_use_the_same_interface": True,
                "llm_is_required_for_these_actions": False,
                "valid_model_intelligence_comparison_requires_equal_action_authority": True,
                "llm_performance_superiority_claimed": False,
            },
        }
        report = {"authority_id": audit.bridge._fingerprint(payload), **payload}
        audit.verify_report(report)
        invalid = copy.deepcopy(report)
        invalid["claim_separation"]["llm_is_required_for_these_actions"] = True
        invalid_payload = dict(invalid)
        invalid_payload.pop("authority_id")
        invalid["authority_id"] = audit.bridge._fingerprint(invalid_payload)
        with self.assertRaisesRegex(audit.AuthorityError, "interface width"):
            audit.verify_report(invalid)


if __name__ == "__main__":
    unittest.main()
