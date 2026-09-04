import copy
import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


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

    def test_collective_label_is_configurable_without_changing_capacity(self):
        parser = audit.parser()
        required = [
            "--suite", "suite.json",
            "--prompt-dir", "prompts",
            "--gbt-report", "gbt.json",
            "--input-separation", "separation.json",
            "--communication", "jacobi=jacobi.json",
            "--structural-graph", "structural.json",
            "--collective-graph", "collective.json",
            "--conditional-frontier", "frontier.json",
            "--out", "out.json",
        ]
        self.assertEqual(
            "collective_n8", parser.parse_args(required).collective_label,
        )
        self.assertEqual(
            "collective_n6",
            parser.parse_args([
                *required, "--collective-label", "collective_n6",
            ]).collective_label,
        )

    def test_confirmed_communication_transform_expands_current_authority(self):
        dispatch = {
            "default": "IPC_OR_DWQ",
            "proxy": "CPU_PROXY_ENQUEUE",
            "trigger": "DWQ_TRIGGER",
        }
        graphs = {}
        multi_index = 0
        for label, count in audit.BASELINE_COMMUNICATION_POLICY_COUNTS.items():
            opportunities = []
            for index in range(count):
                multi = index > 0
                sites = [f"{label}:{index}:0"]
                if multi:
                    sites.append(f"{label}:{index}:1")
                mixed = multi and multi_index < 18
                actions = {
                    site: ("default" if not mixed or offset == 0 else "proxy")
                    for offset, site in enumerate(sites)
                }
                materializers = {
                    site: {
                        "dispatch": dispatch[action], "transform": "NONE",
                    }
                    for site, action in actions.items()
                }
                opportunities.append({
                    "site_ids": sites,
                    "candidates": [{
                        "effects": {"site_actions": actions},
                        "materializer": {"sites": materializers},
                    }],
                })
                if multi:
                    multi_index += 1
            transform = audit.CONFIRMED_TRANSFORM_BY_LABEL.get(label)
            if label == "jacobi":
                sites = ["jacobi:confirmed:0"]
                opportunities.append({
                    "site_ids": sites,
                    "candidates": [{
                        "effects": {"site_actions": {sites[0]: "trigger"}},
                        "materializer": {"sites": {
                            sites[0]: {
                                "dispatch": "DWQ_TRIGGER",
                                "transform": transform,
                            },
                        }},
                    }],
                })
                extra = 1
            else:
                extra = 0
            graphs[label] = (
                {"opportunities": opportunities},
                {"decision_space": {
                    "selectable_candidate_id_count": count + extra,
                    "independent_policy_count": count + extra,
                }},
            )
        result = audit.analyze_communication(graphs)
        self.assertEqual(32, result["route_composable_candidate_count"])
        self.assertEqual(1, result["compiler_transform_candidate_count"])
        self.assertEqual(
            ["PRODUCER_FRONTIER_TWO_PHASE"],
            result["compiler_transforms"],
        )
        self.assertEqual(27, result["atomic_multi_site_candidate_count"])
        self.assertEqual(18, result["mixed_route_candidate_count"])

    def test_frozen_frontier_is_partitioned_by_current_graph_identity(self):
        transforms = {
            "jacobi": "PRODUCER_FRONTIER_TWO_PHASE",
            "loop_lto": "REUSE_LOOP_DESCRIPTOR",
            "mm_minimal": "GUARDED_EARLY_TRIGGER",
        }
        records = [{
            "label": label,
            "current_graph_id": f"current-{label}",
            "conditional_graph_id": f"expanded-{label}",
            "compiler_materializer_transform": transform,
            "runtime_status": "unconfirmed",
            "model_visibility": (
                "forbidden_until_positive_confirmation_and_refreeze"
            ),
            "performance_claim_supported": False,
        } for label, transform in transforms.items()]
        report = {
            "suite_id": "old-suite",
            "frontier_id": "frontier-id",
            "conditional_frontier": records,
            "evidence": {"unit": {}},
        }
        graphs = {
            label: ({
                "graph_id": (
                    f"expanded-{label}" if label == "jacobi"
                    else f"current-{label}"
                ),
            }, {})
            for label in transforms
        }
        with (
            mock.patch.object(
                audit.action_frontier, "verify_report", return_value=report,
            ),
            mock.patch.object(audit, "verify_recorded_evidence"),
        ):
            result = audit.analyze_conditional_frontier(report, graphs)
        self.assertEqual(["jacobi"], result["realized_current_entries"])
        self.assertEqual(
            ["loop_lto", "mm_minimal"],
            result["unresolved_conditional_entries"],
        )
        self.assertEqual(1, result["model_visible_transform_count"])


if __name__ == "__main__":
    unittest.main()
