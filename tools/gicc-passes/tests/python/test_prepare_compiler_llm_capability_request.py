import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PASS_ROOT / "experiments" / "prepare_compiler_llm_capability_request.py"
)
SPEC = importlib.util.spec_from_file_location("capability_request", SCRIPT)
request = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = request
SPEC.loader.exec_module(request)


def suite_and_protocol(eligible=True):
    suite_entry = {
        "label": "unit",
        "entry_id": "sha256:" + "1" * 64,
        "graph_id": "sha256:" + "2" * 64,
        "decision_family": "communication_route_or_schedule",
    }
    suite = {
        "suite_id": "sha256:" + "3" * 64,
        "entries": [suite_entry],
    }
    protocol_entry = {
        "suite_entry_id": suite_entry["entry_id"],
        "compiler_graph_id": suite_entry["graph_id"],
        "decision_family": suite_entry["decision_family"],
        "runtime_readiness_status": (
            "provider_protocol_permitted" if eligible else "awaiting_scout"
        ),
        "eligible_to_freeze_provider_request": eligible,
        "provider_call_authorized": False,
        "current_permitted_provider_calls": 0,
        "conditional_protocol": {
            "views": list(request.VIEWS),
            "independent_responses_per_view": request.TRIALS_PER_VIEW,
            "same_selectable_ids_across_views": True,
        },
    }
    payload = {
        "schema_version": request.capability.REPORT_SCHEMA,
        "suite_id": suite["suite_id"],
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_visible_to_model": False,
            "application_source_modified": False,
            "model_output_is_existing_graph_bound_ids_only": True,
            "compiler_revalidates_before_materialization": True,
            "evaluation_runtime_labels_visible_to_model": False,
            "provider_request_frozen": False,
            "provider_call_authorized": False,
            "provider_invoked": False,
            "scheduler_invoked": False,
        },
        "entries": {"unit": protocol_entry},
        "summary": {
            "provider_calls_made_count": 0,
            "provider_calls_authorized_count": 0,
        },
    }
    protocol = dict(payload)
    protocol["protocol_id"] = request.capability.fingerprint(payload)
    return suite, protocol, suite_entry, protocol_entry


class CompilerLlmCapabilityRequestTests(unittest.TestCase):
    def test_ineligible_entry_cannot_freeze_request(self):
        suite, protocol, _, _ = suite_and_protocol(eligible=False)
        with self.assertRaisesRegex(
            request.CapabilityRequestError, "does not permit request freezing"
        ):
            request.eligible_entry(protocol, suite, "unit")

    def test_protocol_mutation_or_unequal_authority_fails_closed(self):
        suite, protocol, _, _ = suite_and_protocol()
        changed = copy.deepcopy(protocol)
        changed["boundary"]["application_source_visible_to_model"] = True
        with self.assertRaisesRegex(
            request.CapabilityRequestError, "protocol ID"
        ):
            request.eligible_entry(changed, suite, "unit")

        changed = copy.deepcopy(protocol)
        changed["entries"]["unit"]["conditional_protocol"][
            "same_selectable_ids_across_views"
        ] = False
        payload = dict(changed)
        payload.pop("protocol_id")
        changed["protocol_id"] = request.capability.fingerprint(payload)
        with self.assertRaisesRegex(
            request.CapabilityRequestError, "unequal action authority"
        ):
            request.eligible_entry(changed, suite, "unit")

    def test_bundle_is_source_free_unauthorized_and_reproducible(self):
        suite, protocol, suite_entry, protocol_entry = suite_and_protocol()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            sources.mkdir()
            schema = sources / "response-schema.json"
            graph = sources / "graph.json"
            schema.write_text(json.dumps({"type": "object"}) + "\n")
            graph.write_text(json.dumps({"schema_version": "unit"}) + "\n")
            prompts = {}
            for view in request.VIEWS:
                path = sources / f"{view}.txt"
                path.write_text(f"{view} compiler facts\n")
                prompts[view] = path
            evidence_files = {}
            for name in (
                "suite", "readiness", "separation", "sampling_null", "protocol",
            ):
                path = sources / f"{name}.json"
                path.write_text("{}\n")
                evidence_files[name] = path
            inputs = {
                "suite_path": evidence_files["suite"],
                "suite": suite,
                "readiness_path": evidence_files["readiness"],
                "separation_path": evidence_files["separation"],
                "sampling_null_path": evidence_files["sampling_null"],
                "protocol_path": evidence_files["protocol"],
                "protocol": {
                    **protocol,
                    "preregistered_scoring": {"primary": "modal"},
                    "runtime_validation": {"queue": "pdebug"},
                },
                "suite_entry": suite_entry,
                "protocol_entry": protocol_entry,
                "graph_path": graph,
                "graph": {"schema_version": "unit"},
                "schema_path": schema,
                "schema": {"type": "object"},
                "prompts": prompts,
            }
            bundle = root / "request"
            frozen = request.prepare(inputs, bundle)
            self.assertFalse(frozen["authorization"]["granted"])
            self.assertEqual(
                0, frozen["authorization"]["currently_permitted_provider_calls"]
            )
            self.assertEqual(
                60, frozen["provider_delivery"]["total_conditional_calls"]
            )
            self.assertNotIn("source", frozen["provider_delivery"])
            visible_roles = {
                row["role"] for row in frozen["evidence"]["bundle_files"]
                if row["provider_visible"]
            }
            self.assertNotIn("compiler_graph", visible_roles)
            self.assertEqual(frozen, request.verify_bundle(inputs, bundle))

            (bundle / "prompts/opaque.txt").write_text("changed\n")
            with self.assertRaisesRegex(
                request.CapabilityRequestError, "opaque prompt changed"
            ):
                request.verify_bundle(inputs, bundle)


if __name__ == "__main__":
    unittest.main()
