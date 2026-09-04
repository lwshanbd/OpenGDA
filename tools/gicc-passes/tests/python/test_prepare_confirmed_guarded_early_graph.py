import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = PASS_ROOT / "experiments" / "guarded_early_trigger"
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(TEST_ROOT))

import gicc_llm_bridge as bridge
import prepare_confirmed_guarded_early_graph as expansion
from test_gicc_comm_group_plan_bridge import (  # noqa: E402
    PLATFORM, guarded_early_feature, template,
)


SITE_ID = "unit.cpp:10:group_kernel::0"


class ConfirmedGuardedEarlyGraphTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        full_feature = guarded_early_feature(SITE_ID)
        current_feature = copy.deepcopy(full_feature)
        for key in expansion.GUARD_FACT_KEYS:
            current_feature["producer_frontier"].pop(key)
        platform = copy.deepcopy(PLATFORM)
        platform["compiler_transforms"] = {"guarded_early_trigger": False}
        self.dossier = bridge.make_dossier([current_feature], platform)
        self.template = template()
        self.template["ops"] = [
            op for op in self.template["ops"]
            if op["site_id"] != "unit.cpp:11:group_kernel::1"
        ]
        self.graph = expansion.json_value(
            expansion.groups.make_group_graph(self.dossier, [self.template])
        )
        self.baseline_metadata = copy.deepcopy(self.template)
        transfer = next(
            op for op in self.baseline_metadata["ops"]
            if op["site_id"] == SITE_ID
        )
        transfer["producer_frontier"] = full_feature["producer_frontier"]
        self.guarded_metadata = copy.deepcopy(self.baseline_metadata)
        self.guarded_metadata[
            "guarded_early_trigger_device_materialized"
        ] = True
        guard_facts = {
            "source_pointer_candidates": [1, 2],
            "source_buffer_index_param": 9,
            "write_pointer_params": [3],
            "unsafe_side_effect_sites": 0,
            "guardable": True,
        }
        candidate_payload = {
            "kind": "guarded_early_trigger",
            "compiler_materializer": "allocation_guarded_phase3_trigger",
            "kernel_mangled": self.template["kernel_mangled"],
            "transfer_site_id": SITE_ID,
            "schedule_phase": 3,
            "guard_facts": guard_facts,
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
        self.baseline_path = self.root / "baseline-meta.json"
        self.guarded_path = self.root / "guarded-meta.json"
        self.audit_path = self.root / "audit.json"
        for path, value in (
            (self.dossier_path, self.dossier),
            (self.template_path, self.template),
            (self.graph_path, self.graph),
            (self.baseline_path, self.baseline_metadata),
            (self.guarded_path, self.guarded_metadata),
            (self.audit_path, {
                "schema_version": "gicc-guarded-early-trigger-ir-audit-v1",
                "passed": True,
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
            "scope": "unit guarded confirmation",
            "model_invoked": False,
            "application_source_modified": False,
            "provider_call_authorized": False,
            "transition_id": self.transition["transition_id"],
            "dormant_compiler_candidate_id": self.candidate["candidate_id"],
            "transition": str(self.transition_path.resolve()),
            "transition_sha256": expansion.sha256_file(self.transition_path),
            "allocation_monitors": [
                {
                    "allocation": allocation,
                    "monitor": str((self.root / f"monitor{allocation}.json").resolve()),
                }
                for allocation in (1, 2, 3)
            ],
            "correctness_gate": {"passed": True},
            "runtime_guard_gate": {"passed": True},
            "confirmation_gate": {"passed": True},
        }
        self.confirmation = {
            **payload, "result_id": bridge._fingerprint(payload),
        }
        self.confirmation_path = self.root / "confirmation.json"
        self.confirmation_path.write_text(
            json.dumps(self.confirmation, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.transition_paths = {
            "baseline_kernel_metadata": self.baseline_path,
            "guarded_kernel_metadata": self.guarded_path,
            "frozen_ir_audit": self.audit_path,
        }

    def tearDown(self):
        self.temporary.cleanup()

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

    def test_expansion_adds_only_the_guarded_lto_candidate(self):
        dossier, graph, change = expansion.expand_graph(
            self.dossier, self.template, self.graph,
            self.candidate, self.baseline_metadata,
        )
        old = self.graph["opportunities"][0]
        new = graph["opportunities"][0]
        self.assertEqual(3, len(old["candidates"]))
        self.assertEqual(4, len(new["candidates"]))
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
        self.assertEqual(4, change["expanded_selectable_candidate_count"])
        frontier = dossier["sites"][0]["producer_frontier"]
        self.assertTrue(expansion.GUARD_FACT_KEYS.issubset(frontier))

    def test_bundle_replays_confirmation_and_verifies_all_outputs(self):
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
        prompt = (output / "prompt-relational.txt").read_text(encoding="utf-8")
        self.assertIn(expansion.CANDIDATE_KIND, prompt)
        self.assertNotIn("unit.cpp", prompt)

    def test_failed_gate_or_unexpected_fact_delta_is_rejected(self):
        failed = copy.deepcopy(self.confirmation)
        failed["runtime_guard_gate"]["passed"] = False
        payload = dict(failed)
        payload.pop("result_id")
        failed["result_id"] = bridge._fingerprint(payload)
        self.confirmation_path.write_text(
            json.dumps(failed), encoding="utf-8",
        )
        with self.assertRaisesRegex(expansion.ExpansionError, "runtime-guard"):
            expansion.replay_confirmation(self.confirmation_path)

        metadata = copy.deepcopy(self.baseline_metadata)
        transfer = next(
            op for op in metadata["ops"] if op["site_id"] == SITE_ID
        )
        transfer["producer_frontier"]["invented_fact"] = True
        with self.assertRaisesRegex(expansion.ExpansionError, "unexpected fact"):
            expansion.expand_graph(
                self.dossier, self.template, self.graph,
                self.candidate, metadata,
            )


if __name__ == "__main__":
    unittest.main()
