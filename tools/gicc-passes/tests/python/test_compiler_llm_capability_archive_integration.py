import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


PASS_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PASS_ROOT / "experiments"))
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(TEST_ROOT))

import analyze_compiler_llm_capability_trials as analysis
import gicc_comm_plan_bridge as structural
import gicc_llm_bridge as llm
import prepare_compiler_llm_capability_request as freezer
import run_compiler_llm_capability_trials as trials
from test_gicc_comm_plan_bridge import PLATFORM, coalescable_feature


class CompilerLlmCapabilityArchiveIntegrationTests(unittest.TestCase):
    def test_frozen_request_sixty_archived_trials_and_chance_analysis(self):
        graph = structural.make_opportunity_graph(llm.make_dossier(
            [coalescable_feature()], PLATFORM,
        ))
        opportunity = graph["opportunities"][0]
        fast = next(
            candidate for candidate in opportunity["candidates"]
            if candidate["kind"] == "trigger_coalesced_loop"
        )
        response = {
            "schema_version": structural.DECISION_SCHEMA,
            "graph_id": graph["graph_id"],
            "selections": {
                opportunity["opportunity_id"]: {
                    "candidate_id": fast["candidate_id"],
                    "confidence": 0.9,
                    "rationale": "synthetic compiler-fact integration test",
                },
            },
        }
        schema = structural.decision_response_schema(graph)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sources = root / "sources"
            sources.mkdir()
            schema_path = sources / "response-schema.json"
            graph_path = sources / "graph.json"
            schema_path.write_text(
                json.dumps(schema, indent=2, sort_keys=True) + "\n"
            )
            graph_path.write_text(
                json.dumps(graph, indent=2, sort_keys=True) + "\n"
            )
            prompts = {}
            for view in freezer.VIEWS:
                path = sources / f"{view}.txt"
                path.write_text(f"{view} source-free compiler facts\n")
                prompts[view] = path
            evidence_paths = {}
            for name in (
                "suite", "readiness", "separation", "sampling_null",
                "protocol",
            ):
                path = sources / f"{name}.json"
                path.write_text("{}\n")
                evidence_paths[name] = path

            suite_entry = {
                "label": "unit",
                "entry_id": "sha256:" + "1" * 64,
                "graph_id": graph["graph_id"],
                "decision_family": (
                    "communication_coalescing_and_trigger_placement"
                ),
            }
            protocol = {
                "protocol_id": "sha256:" + "2" * 64,
                "preregistered_scoring": {"primary": "modal"},
                "runtime_validation": {"queue": "pdebug"},
            }
            inputs = {
                "suite_path": evidence_paths["suite"],
                "suite": {
                    "suite_id": "sha256:" + "3" * 64,
                    "entries": [suite_entry],
                },
                "readiness_path": evidence_paths["readiness"],
                "separation_path": evidence_paths["separation"],
                "sampling_null_path": evidence_paths["sampling_null"],
                "protocol_path": evidence_paths["protocol"],
                "protocol": protocol,
                "suite_entry": suite_entry,
                "protocol_entry": {
                    "runtime_readiness_status": (
                        "provider_protocol_permitted"
                    ),
                },
                "graph_path": graph_path,
                "graph": graph,
                "schema_path": schema_path,
                "schema": schema,
                "prompts": prompts,
            }
            request_dir = root / "request"
            request = freezer.prepare(inputs, request_dir)
            self.assertEqual(60, request["provider_delivery"][
                "total_conditional_calls"
            ])

            authorization_payload = {
                "schema_version": trials.AUTHORIZATION_SCHEMA,
                "granted": True,
                "request_id": request["request_id"],
                "provider_delivery": trials.delivery_contract(request),
                "data_boundary": {
                    "model_tools": [],
                    "source_visible": False,
                    "source_locations_visible": False,
                    "llvm_ir_visible": False,
                    "evaluation_runtime_labels_visible": False,
                    "evaluation_oracle_visible": False,
                    "model_may_generate_code_or_ir": False,
                },
                "provider": {
                    "kind": "anthropic-claude-cli",
                    "executable": "claude",
                    "cli_version": "synthetic-version",
                    "requested_model": "synthetic-model",
                    "effort": "high",
                    "fresh_session_per_trial": True,
                    "structured_output": True,
                },
                "transport_retry": {
                    "max_attempts_per_trial": 1,
                    "timeout_seconds": 30,
                },
            }
            authorization = dict(authorization_payload)
            authorization["authorization_id"] = llm._fingerprint(
                authorization_payload
            )
            verified_authorization = trials.verify_authorization(
                authorization, request
            )
            envelope = json.dumps({
                "model": "synthetic-model",
                "structured_output": response,
            })
            archive_dir = root / "archive"
            archive_dir.mkdir()
            trials.archive_authorization(authorization, archive_dir)
            records = []
            with mock.patch.object(trials.subprocess, "run") as provider:
                provider.return_value = SimpleNamespace(
                    returncode=0, stdout=envelope, stderr="",
                )
                for view, trial in trials.expected_trials(request):
                    records.append(trials.run_one(
                        view=view,
                        trial=trial,
                        prompt=(request_dir / f"prompts/{view}.txt").read_text(),
                        graph=graph,
                        request=request,
                        authorization=verified_authorization,
                        authorization_id=authorization["authorization_id"],
                        system_prompt=(
                            request_dir / "system-prompt.txt"
                        ).read_text(),
                        response_schema=schema,
                        output_dir=archive_dir,
                        observed_cli_version="synthetic-version",
                    ))
                self.assertEqual(60, provider.call_count)
            index = {
                "schema_version": trials.INDEX_SCHEMA,
                "status": "complete",
                "request_id": request["request_id"],
                "authorization_id": authorization["authorization_id"],
                "runs": records,
            }
            trials.write_json_atomic(archive_dir / "run-index.json", index)
            verified_index, _ = trials.verify_complete_archive(
                request=request,
                authorization_value=authorization,
                graph=graph,
                output_dir=archive_dir,
            )

            anchor = analysis.policy_bridge.decision_to_policy(graph, None)
            selected = analysis.policy_bridge.decision_to_policy(
                graph, response
            )
            screen = {
                "controls": {
                    "oracle": {
                        "selected_ids_by_slot": selected[
                            "selected_ids_by_slot"
                        ],
                        "cost_by_unit": {"u": 1.0},
                    },
                    "anchor": {
                        "selected_ids_by_slot": anchor[
                            "selected_ids_by_slot"
                        ],
                        "cost_by_unit": {"u": 2.0},
                    },
                    "deterministic": {
                        "selected_ids_by_slot": anchor[
                            "selected_ids_by_slot"
                        ],
                        "cost_by_unit": {"u": 2.0},
                    },
                },
                "unit_weights": {"u": 1.0},
                "policy_cost_by_id": {
                    selected["policy_id"]: {"u": 1.0},
                },
            }
            scored = analysis.score_archive(verified_index, graph, screen)
            calibrated = analysis.chance_calibration(
                scored["metrics"],
                analysis.sampling_null.uniform_oracle_null(3, 20),
            )
            self.assertEqual(60, scored["screened_trial_count"])
            for view in freezer.VIEWS:
                self.assertEqual(20, calibrated["views"][view][
                    "observed_exact_oracle_hits"
                ])
                self.assertTrue(calibrated["views"][view][
                    "meets_one_sided_alpha_0_05_hit_threshold"
                ])


if __name__ == "__main__":
    unittest.main()
