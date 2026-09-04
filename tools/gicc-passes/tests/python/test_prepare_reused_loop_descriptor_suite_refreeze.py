import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PASS_ROOT = Path(__file__).resolve().parents[2]
REUSED = PASS_ROOT / "experiments" / "reused_loop_descriptor"
PRODUCER = PASS_ROOT / "experiments" / "producer_fission"
TEST_ROOT = Path(__file__).resolve().parent
for path in (PASS_ROOT / "python", REUSED, PRODUCER, TEST_ROOT):
    sys.path.insert(0, str(path))

import gicc_compiler_decision_suite as suites
import gicc_llm_bridge as bridge
import prepare_confirmed_reused_loop_descriptor_graph as expansion
import prepare_reused_loop_descriptor_suite_refreeze as refreeze
from test_gicc_comm_group_plan_bridge import (  # noqa: E402
    PLATFORM, reused_loop_feature, template,
)
from test_gicc_compiler_decision_suite import (  # noqa: E402
    collective_graph, communication_graph, structural_graph,
)


SITE_ID = "unit.cpp:10:group_kernel::0"


class ReusedLoopDescriptorSuiteRefreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        full = reused_loop_feature(SITE_ID)
        old_feature = copy.deepcopy(full)
        old_feature["loop"].pop("bound_param_type")
        platform = copy.deepcopy(PLATFORM)
        platform["compiler_transforms"] = {
            "reused_loop_descriptor": False,
        }
        dossier = bridge.make_dossier([old_feature], platform)
        kernel_template = template()
        kernel_template["ops"] = [
            op for op in kernel_template["ops"]
            if op["site_id"] != "unit.cpp:11:group_kernel::1"
        ]
        transfer = next(
            op for op in kernel_template["ops"] if op["site_id"] == SITE_ID
        )
        proof = {
            "descriptor_reusable": True,
            "buffer_reusable": True,
            "host_knowable_interval": True,
            "loop": copy.deepcopy(full["loop"]),
            "descriptor_arguments": copy.deepcopy(transfer["args"]),
            "network_operation_count": {
                "kind": "runtime_loop_bound", "kernel_param_index": 4,
            },
            "network_operation_order_preserved": True,
        }
        candidate_payload = {
            "kind": expansion.CANDIDATE_KIND,
            "compiler_materializer": expansion.TRANSFORM,
            "kernel_mangled": kernel_template["kernel_mangled"],
            "transfer_site_id": SITE_ID,
            "compiler_proof": proof,
        }
        dormant = {
            **candidate_payload,
            "candidate_id": bridge._fingerprint({
                "schema_version": "gicc-compiler-schedule-candidate-v1",
                **candidate_payload,
            }),
            "model_visible": False,
        }
        current = expansion.json_value(
            expansion.groups.make_group_graph(dossier, [kernel_template])
        )
        _, expanded, change = expansion.expand_graph(
            dossier, kernel_template, current, dormant,
            [full], kernel_template,
        )
        self.current_path = self.root / "loop-current.json"
        self.expanded_path = self.root / "loop-expanded.json"
        self.other_comm_path = self.root / "minimod.json"
        self.collective_path = self.root / "collective.json"
        self.structural_path = self.root / "structural.json"
        for path, value in (
            (self.current_path, current),
            (self.expanded_path, expanded),
            (self.other_comm_path, communication_graph()),
            (self.collective_path, collective_graph()),
            (self.structural_path, structural_graph()),
        ):
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self.prompts = self.root / "prompts"
        self.current_suite = suites.make_suite(
            [("loop_lto", self.current_path),
             ("minimod", self.other_comm_path)],
            [("collective_n8", self.collective_path)],
            [("coalescing_placement", self.structural_path)],
            prompt_dir=self.prompts,
        )
        self.suite_path = self.root / "suite.json"
        self.suite_path.write_text(
            json.dumps(self.current_suite, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.manifest = {
            "schema_version": expansion.EXPANSION_SCHEMA,
            "expansion_id": "sha256:" + "5" * 64,
            "graph_transition": {
                "current_graph_id": current["graph_id"],
                "expanded_graph_id": expanded["graph_id"],
                **change,
            },
            "inputs": [{
                "role": "frozen_current_graph",
                "path": str(self.current_path.resolve()),
                "sha256": refreeze.base.sha256_file(self.current_path),
                "bytes": self.current_path.stat().st_size,
            }],
            "outputs": [{
                "role": "expanded_graph",
                "path": self.expanded_path.name,
                "sha256": refreeze.base.sha256_file(self.expanded_path),
                "bytes": self.expanded_path.stat().st_size,
            }],
        }
        self.manifest_path = self.root / "expansion.json"
        self.manifest_path.write_text(
            json.dumps(self.manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.communication = [("minimod", self.other_comm_path)]
        self.collective = [("collective_n8", self.collective_path)]
        self.structural = [("coalescing_placement", self.structural_path)]

    def tearDown(self):
        self.temporary.cleanup()

    def verified_expansion(self):
        return mock.patch.object(
            refreeze.expansion, "verify_contained", return_value=self.manifest,
        )

    def test_only_loop_entry_gains_one_policy(self):
        with self.verified_expansion():
            value, _, delta = refreeze.make_refrozen_suite(
                self.suite_path, self.prompts, self.manifest_path,
                self.communication, self.collective, self.structural,
            )
        old = {entry["label"]: entry for entry in self.current_suite["entries"]}
        new = {entry["label"]: entry for entry in value["entries"]}
        for label in set(old) - {"loop_lto"}:
            self.assertEqual(old[label], new[label])
        self.assertEqual(
            delta["old_policy_count"] + 1, delta["new_policy_count"]
        )
        self.assertEqual(0, delta["old_masked_candidate_count"])
        self.assertEqual(0, delta["new_masked_candidate_count"])
        self.assertFalse(value["boundary"]["provider_call_supported"])

    def test_bundle_is_self_verifying_and_never_authorizes_provider(self):
        output = self.root / "refrozen"
        with self.verified_expansion():
            manifest = refreeze.prepare_bundle(
                output, self.suite_path, self.prompts, self.manifest_path,
                self.communication, self.collective, self.structural,
            )
            verified = refreeze.verify_contained(output / "manifest.json")
        self.assertEqual(manifest, verified)
        self.assertFalse(manifest["boundary"]["provider_call_authorized"])
        self.assertFalse(
            manifest["boundary"]["application_source_visible_to_model"]
        )

    def test_target_graph_cannot_be_supplied_by_caller(self):
        with self.verified_expansion(), self.assertRaisesRegex(
                refreeze.RefreezeError, "loop_lto graph"):
            refreeze.make_refrozen_suite(
                self.suite_path, self.prompts, self.manifest_path,
                [("loop_lto", self.current_path), *self.communication],
                self.collective, self.structural,
            )


if __name__ == "__main__":
    unittest.main()
