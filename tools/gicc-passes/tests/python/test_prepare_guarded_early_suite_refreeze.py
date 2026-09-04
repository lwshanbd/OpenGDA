import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PASS_ROOT = Path(__file__).resolve().parents[2]
GUARDED = PASS_ROOT / "experiments" / "guarded_early_trigger"
PRODUCER = PASS_ROOT / "experiments" / "producer_fission"
TEST_ROOT = Path(__file__).resolve().parent
for path in (PASS_ROOT / "python", GUARDED, PRODUCER, TEST_ROOT):
    sys.path.insert(0, str(path))

import gicc_compiler_decision_suite as suites
import gicc_llm_bridge as bridge
import prepare_confirmed_guarded_early_graph as expansion
import prepare_guarded_early_suite_refreeze as refreeze
from test_gicc_comm_group_plan_bridge import (  # noqa: E402
    PLATFORM, guarded_early_feature, template,
)
from test_gicc_compiler_decision_suite import (  # noqa: E402
    collective_graph, communication_graph, structural_graph,
)


SITE_ID = "unit.cpp:10:group_kernel::0"


class GuardedEarlySuiteRefreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        full = guarded_early_feature(SITE_ID)
        old_feature = copy.deepcopy(full)
        for key in expansion.GUARD_FACT_KEYS:
            old_feature["producer_frontier"].pop(key)
        platform = copy.deepcopy(PLATFORM)
        platform["compiler_transforms"] = {"guarded_early_trigger": False}
        dossier = bridge.make_dossier([old_feature], platform)
        kernel_template = template()
        kernel_template["ops"] = [
            op for op in kernel_template["ops"]
            if op["site_id"] != "unit.cpp:11:group_kernel::1"
        ]
        baseline_metadata = copy.deepcopy(kernel_template)
        next(op for op in baseline_metadata["ops"]
             if op["site_id"] == SITE_ID)["producer_frontier"] = (
                 full["producer_frontier"]
        )
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
            "kernel_mangled": kernel_template["kernel_mangled"],
            "transfer_site_id": SITE_ID,
            "schedule_phase": 3,
            "guard_facts": guard_facts,
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
            dossier, kernel_template, current, dormant, baseline_metadata,
        )
        self.current_path = self.root / "mm-current.json"
        self.expanded_path = self.root / "mm-expanded.json"
        self.jacobi_path = self.root / "jacobi.json"
        self.collective_path = self.root / "collective.json"
        self.structural_path = self.root / "structural.json"
        for path, value in (
            (self.current_path, current),
            (self.expanded_path, expanded),
            (self.jacobi_path, communication_graph()),
            (self.collective_path, collective_graph()),
            (self.structural_path, structural_graph()),
        ):
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self.prompts = self.root / "prompts"
        self.current_suite = suites.make_suite(
            [("jacobi", self.jacobi_path),
             ("mm_minimal", self.current_path)],
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
        self.communication = [("jacobi", self.jacobi_path)]
        self.collective = [("collective_n8", self.collective_path)]
        self.structural = [("coalescing_placement", self.structural_path)]

    def tearDown(self):
        self.temporary.cleanup()

    def verified_expansion(self):
        return mock.patch.object(
            refreeze.expansion, "verify_contained", return_value=self.manifest,
        )

    def test_only_mm_entry_changes_from_three_to_four_policies(self):
        with self.verified_expansion():
            value, _, delta = refreeze.make_refrozen_suite(
                self.suite_path, self.prompts, self.manifest_path,
                self.communication, self.collective, self.structural,
            )
        old = {entry["label"]: entry for entry in self.current_suite["entries"]}
        new = {entry["label"]: entry for entry in value["entries"]}
        for label in set(old) - {"mm_minimal"}:
            self.assertEqual(old[label], new[label])
        self.assertEqual(3, delta["old_policy_count"])
        self.assertEqual(4, delta["new_policy_count"])
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
                refreeze.RefreezeError, "mm_minimal graph"):
            refreeze.make_refrozen_suite(
                self.suite_path, self.prompts, self.manifest_path,
                [("mm_minimal", self.current_path), *self.communication],
                self.collective, self.structural,
            )


if __name__ == "__main__":
    unittest.main()
