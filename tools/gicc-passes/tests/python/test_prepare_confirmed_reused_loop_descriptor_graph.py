import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = PASS_ROOT / "experiments" / "reused_loop_descriptor"
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(TEST_ROOT))

import gicc_llm_bridge as bridge
import prepare_confirmed_reused_loop_descriptor_graph as expansion
from test_gicc_comm_group_plan_bridge import (  # noqa: E402
    PLATFORM, reused_loop_feature, template,
)


SITE_ID = "unit.cpp:10:group_kernel::0"


class ConfirmedReusedLoopDescriptorGraphTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.full_feature = reused_loop_feature(SITE_ID)
        current_feature = copy.deepcopy(self.full_feature)
        current_feature["loop"].pop("bound_param_type")
        platform = copy.deepcopy(PLATFORM)
        platform["compiler_transforms"] = {
            "reused_loop_descriptor": False,
        }
        self.dossier = bridge.make_dossier([current_feature], platform)
        self.template = template()
        self.template["ops"] = [
            op for op in self.template["ops"]
            if op["site_id"] != "unit.cpp:11:group_kernel::1"
        ]
        self.graph = expansion.json_value(
            expansion.groups.make_group_graph(self.dossier, [self.template])
        )
        transfer = next(
            op for op in self.template["ops"] if op["site_id"] == SITE_ID
        )
        proof = {
            "descriptor_reusable": True,
            "buffer_reusable": True,
            "host_knowable_interval": True,
            "loop": copy.deepcopy(self.full_feature["loop"]),
            "descriptor_arguments": copy.deepcopy(transfer["args"]),
            "network_operation_count": {
                "kind": "runtime_loop_bound", "kernel_param_index": 4,
            },
            "network_operation_order_preserved": True,
        }
        candidate_payload = {
            "kind": expansion.CANDIDATE_KIND,
            "compiler_materializer": expansion.TRANSFORM,
            "kernel_mangled": self.template["kernel_mangled"],
            "transfer_site_id": SITE_ID,
            "compiler_proof": proof,
        }
        self.candidate = {
            **candidate_payload,
            "candidate_id": bridge._fingerprint({
                "schema_version": "gicc-compiler-schedule-candidate-v1",
                **candidate_payload,
            }),
            "model_visible": False,
        }

        self.dossier_path = self.root / "dossier.json"
        self.template_path = self.root / "template.json"
        self.graph_path = self.root / "graph.json"
        self.baseline_features_path = self.root / "baseline-features.json"
        self.reused_features_path = self.root / "reused-features.json"
        self.baseline_metadata_path = self.root / "baseline-metadata.json"
        self.reused_metadata_path = self.root / "reused-metadata.json"
        self.audit_path = self.root / "audit.json"
        for path, value in (
            (self.dossier_path, self.dossier),
            (self.template_path, self.template),
            (self.graph_path, self.graph),
            (self.baseline_features_path, [self.full_feature]),
            (self.reused_features_path, [self.full_feature]),
            (self.baseline_metadata_path, self.template),
            (self.reused_metadata_path, self.template),
            (self.audit_path, {
                "schema_version": "gicc-reused-loop-descriptor-ir-audit-v1",
                "passed": True,
                "host_only_device_identity": True,
            }),
        ):
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        self.transition_path = self.root / "transition.json"
        self.transition_path.write_text("{}\n", encoding="utf-8")
        self.transition = {
            "transition_id": "sha256:" + "1" * 64,
            "dormant_compiler_candidate": self.candidate,
        }
        payload = {
            "schema_version": expansion.CONFIRMATION_SCHEMA,
            "scope": "unit reused confirmation",
            "model_invoked": False,
            "application_source_visible_to_model": False,
            "application_source_modified": False,
            "provider_call_authorized": False,
            "transition_id": self.transition["transition_id"],
            "dormant_compiler_candidate_id": self.candidate["candidate_id"],
            "transition": str(self.transition_path.resolve()),
            "transition_sha256": expansion.sha256_file(self.transition_path),
            "allocation_monitors": [
                {
                    "allocation": allocation,
                    "monitor": str(
                        (self.root / f"monitor{allocation}.json").resolve()
                    ),
                }
                for allocation in (1, 2, 3)
            ],
            "correctness_gate": {"passed": True},
            "confirmation_gate": {"passed": True},
        }
        self.confirmation = {
            **payload, "result_id": bridge._fingerprint(payload),
        }
        self.confirmation_path = self.root / "confirmation.json"
        self._write_confirmation()
        self.transition_paths = {
            "baseline_features": self.baseline_features_path,
            "reused_features": self.reused_features_path,
            "baseline_kernel_metadata": self.baseline_metadata_path,
            "reused_kernel_metadata": self.reused_metadata_path,
            "frozen_ir_audit": self.audit_path,
        }

    def tearDown(self):
        self.temporary.cleanup()

    def _write_confirmation(self):
        self.confirmation_path.write_text(
            json.dumps(self.confirmation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def replay_patches(self):
        return (
            mock.patch.object(
                expansion.confirmation, "validate_transition",
                return_value=(self.transition, self.transition_paths),
            ),
            mock.patch.object(
                expansion.confirmation, "analyze_monitors",
                return_value=self.confirmation,
            ),
        )

    def test_expansion_adds_only_confirmed_descriptor_reuse(self):
        dossier, graph, change = expansion.expand_graph(
            self.dossier, self.template, self.graph, self.candidate,
            [self.full_feature], self.template,
        )
        old = self.graph["opportunities"][0]
        new = graph["opportunities"][0]
        self.assertEqual(len(old["candidates"]) + 1, len(new["candidates"]))
        self.assertEqual(
            [item["candidate_id"] for item in old["candidates"]],
            [item["candidate_id"] for item in new["candidates"]
             if item["kind"] != expansion.CANDIDATE_KIND],
        )
        candidate = next(
            item for item in new["candidates"]
            if item["kind"] == expansion.CANDIDATE_KIND
        )
        self.assertEqual(
            expansion.TRANSFORM,
            candidate["materializer"]["sites"][SITE_ID]["transform"],
        )
        self.assertEqual(
            {"kind": "runtime_loop_bound", "kernel_param_index": 4},
            candidate["effects"]["network_operations"],
        )
        self.assertEqual(
            len(old["candidates"]) + 1,
            change["expanded_selectable_candidate_count"],
        )
        self.assertEqual(
            "i32", dossier["sites"][0]["loop"]["bound_param_type"]
        )
        self.assertTrue(
            dossier["platform_profile"]["compiler_transforms"]
            ["reused_loop_descriptor"]
        )

    def test_bundle_replays_confirmation_and_verifies_outputs(self):
        first, second = self.replay_patches()
        with first, second:
            rendered, manifest = expansion.build_bundle(
                self.confirmation_path, self.dossier_path,
                self.template_path, self.graph_path,
            )
            output = self.root / "expanded"
            expansion.write_bundle(output, rendered)
            verified = expansion.verify_contained(output / "manifest.json")
        self.assertEqual(manifest, verified)
        self.assertFalse(manifest["boundary"]["provider_call_authorized"])
        self.assertFalse(
            manifest["boundary"]["application_source_visible_to_model"]
        )
        prompt = (output / "prompt-relational.txt").read_text(
            encoding="utf-8"
        )
        self.assertIn(expansion.CANDIDATE_KIND, prompt)
        self.assertNotIn("SECRET_SOURCE_SENTINEL", prompt)
        self.assertNotIn("unit.cpp", prompt)
        with self.assertRaisesRegex(expansion.ExpansionError, "overwrite"):
            expansion.write_bundle(output, rendered)

    def test_failed_gate_or_unexpected_fact_delta_is_rejected(self):
        failed = copy.deepcopy(self.confirmation)
        failed["confirmation_gate"]["passed"] = False
        payload = dict(failed)
        payload.pop("result_id")
        failed["result_id"] = bridge._fingerprint(payload)
        self.confirmation = failed
        self._write_confirmation()
        with self.assertRaisesRegex(expansion.ExpansionError, "did not pass"):
            expansion.replay_confirmation(self.confirmation_path)

        changed = copy.deepcopy(self.full_feature)
        changed["loop"]["iv_step"] = 2
        with self.assertRaisesRegex(
            expansion.ExpansionError, "change more than loop type"
        ):
            expansion.expand_graph(
                self.dossier, self.template, self.graph, self.candidate,
                [changed], self.template,
            )

    def test_current_graph_must_exactly_regenerate(self):
        changed = copy.deepcopy(self.graph)
        changed["objective"]["instruction"] = "changed"
        payload = dict(changed)
        payload.pop("graph_id")
        changed["graph_id"] = bridge._fingerprint(payload)
        with self.assertRaisesRegex(
            expansion.ExpansionError, "does not regenerate"
        ):
            expansion.expand_graph(
                self.dossier, self.template, changed, self.candidate,
                [self.full_feature], self.template,
            )


if __name__ == "__main__":
    unittest.main()
