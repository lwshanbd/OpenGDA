import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PASS_ROOT = Path(__file__).resolve().parents[2]
COLLECTIVE = PASS_ROOT / "experiments" / "collective"
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(COLLECTIVE))
sys.path.insert(0, str(TEST_ROOT))

import gicc_compiler_decision_suite as suites
import gicc_llm_bridge as bridge
import prepare_collective_n6_suite_refreeze as refreeze
from test_gicc_compiler_decision_suite import (  # noqa: E402
    communication_graph, structural_graph,
)


def collective_graph(nodes):
    collective = suites.collective
    anchor = {
        "family": "allreduce", "contract": "sum_f32",
        "algorithm": "baseline_auto",
    }
    candidates = []
    for index in range(1, 8):
        descriptor = {
            "family": "allreduce", "contract": "sum_f32",
            "algorithm": f"algorithm_{index}",
        }
        candidates.append({
            "catalog_id": collective._target_id("catalog", descriptor),
            "descriptor": descriptor,
            "compiler_legality": {
                "same_family": True,
                "same_semantic_contract": True,
                "exact_function_type": True,
                "void_call_materializer": True,
            },
        })
    inventory = {
        "schema_version": collective.INVENTORY_SCHEMA,
        "compiler_only": True,
        "source_visible": False,
        "target_triple": "amdgcn-amd-amdhsa",
        "opportunities": [{
            "opportunity_id": "opportunity:" + "4" * 24,
            "family": "allreduce",
            "contract": "sum_f32",
            "anchor_descriptor": anchor,
            "anchor_id": collective._target_id("anchor", anchor),
            "call_facts": {"element_bytes": 4, "message_shape": "dynamic"},
            "compiler_policy_thresholds_bytes": [4096, 1048576, 8388608],
            "candidates": candidates,
        }],
    }
    profile = {
        "schema_version": collective.PROFILE_SCHEMA,
        "platform_id": f"unit-collective-n{nodes}",
        "topology": {
            "nodes": nodes, "ranks_per_node": 8, "gpus_per_node": 8,
        },
    }
    return collective.make_graph(inventory, profile)


class CollectiveN6SuiteRefreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.communication_path = self.root / "communication.json"
        self.structural_path = self.root / "structural.json"
        self.n8_path = self.root / "collective-n8.json"
        self.n6_path = self.root / "collective-n6.json"
        self.confirmation_path = self.root / "confirmation.json"
        values = {
            self.communication_path: communication_graph(),
            self.structural_path: structural_graph(),
            self.n8_path: collective_graph(8),
            self.n6_path: collective_graph(6),
            self.confirmation_path: {"unit": "confirmation"},
        }
        for path, value in values.items():
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self.prompts = self.root / "prompts"
        self.current = suites.make_suite(
            [("minimod", self.communication_path)],
            [(refreeze.OLD_LABEL, self.n8_path)],
            [("coalescing_placement", self.structural_path)],
            prompt_dir=self.prompts,
        )
        self.suite_path = self.root / "suite.json"
        self.suite_path.write_text(
            json.dumps(self.current, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.confirmation = {
            "result_id": "sha256:" + "5" * 64,
        }
        self.communication = [("minimod", self.communication_path)]
        self.structural = [("coalescing_placement", self.structural_path)]

    def tearDown(self):
        self.temporary.cleanup()

    def verified_confirmation(self):
        return mock.patch.object(
            refreeze, "verify_confirmation", return_value=self.confirmation,
        )

    def test_replaces_only_collective_label_and_preserves_action_capacity(self):
        with self.verified_confirmation():
            value, _, transition = refreeze.make_refrozen_suite(
                self.suite_path, self.prompts, self.confirmation_path,
                self.n8_path, self.n6_path, self.communication,
                self.structural,
            )
        old = {entry["label"]: entry for entry in self.current["entries"]}
        new = {entry["label"]: entry for entry in value["entries"]}
        self.assertEqual(
            (set(old) - {refreeze.OLD_LABEL}) | {refreeze.NEW_LABEL},
            set(new),
        )
        for label in set(old) - {refreeze.OLD_LABEL}:
            self.assertEqual(old[label], new[label])
        self.assertEqual(4096, transition["old_policy_count"])
        self.assertEqual(4096, transition["new_policy_count"])
        self.assertTrue(transition["all_noncollective_entries_preserved"])
        self.assertFalse(value["boundary"]["provider_call_supported"])

    def test_bundle_is_atomic_self_verifying_and_provider_unauthorized(self):
        output = self.root / "refrozen"
        with self.verified_confirmation():
            manifest = refreeze.prepare_bundle(
                output, self.suite_path, self.prompts,
                self.confirmation_path, self.n8_path, self.n6_path,
                self.communication, self.structural,
            )
            verified = refreeze.verify_contained(output / "manifest.json")
        self.assertEqual(manifest, verified)
        self.assertEqual(
            "refrozen_suite_ready_for_readiness_audit", manifest["status"]
        )
        self.assertEqual(refreeze.BOUNDARY, manifest["boundary"])
        self.assertFalse(manifest["boundary"]["provider_call_authorized"])
        with self.verified_confirmation(), self.assertRaisesRegex(
            refreeze.RefreezeError, "overwrite"
        ):
            refreeze.prepare_bundle(
                output, self.suite_path, self.prompts,
                self.confirmation_path, self.n8_path, self.n6_path,
                self.communication, self.structural,
            )

    def test_missing_or_changed_noncollective_entry_fails_closed(self):
        with self.verified_confirmation(), self.assertRaisesRegex(
            refreeze.RefreezeError, "exactly every"
        ):
            refreeze.make_refrozen_suite(
                self.suite_path, self.prompts, self.confirmation_path,
                self.n8_path, self.n6_path, [], self.structural,
            )
        changed = communication_graph()
        changed["platform_profile"]["platform_id"] = "changed-platform"
        payload = dict(changed)
        payload.pop("graph_id")
        changed["graph_id"] = bridge._fingerprint(payload)
        changed_path = self.root / "changed.json"
        changed_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.verified_confirmation(), self.assertRaisesRegex(
            refreeze.RefreezeError, "unrelated entry"
        ):
            refreeze.make_refrozen_suite(
                self.suite_path, self.prompts, self.confirmation_path,
                self.n8_path, self.n6_path,
                [("minimod", changed_path)], self.structural,
            )

    def test_negative_confirmation_is_rejected_before_transition_replay(self):
        payload = {
            "schema_version": refreeze.n6_confirmation.base.RESULT_SCHEMA,
            "model_invoked": False,
            "application_source_modified": False,
            "provider_call_authorized": False,
            "confirmation_gate": {"passed": False},
        }
        value = {**payload, "result_id": bridge._fingerprint(payload)}
        self.confirmation_path.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(refreeze.RefreezeError, "did not pass"):
            refreeze.verify_confirmation(
                self.confirmation_path, self.n6_path
            )


if __name__ == "__main__":
    unittest.main()
