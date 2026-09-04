import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = PASS_ROOT / "experiments" / "producer_fission"
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(TEST_ROOT))

import gicc_compiler_decision_suite as suites
import gicc_llm_bridge as bridge
import prepare_confirmed_producer_fission_graph as expansion
import prepare_producer_fission_suite_refreeze as refreeze
from test_gicc_comm_group_plan_bridge import (  # noqa: E402
    PLATFORM, fission_feature, template,
)
from test_gicc_compiler_decision_suite import (  # noqa: E402
    collective_graph, communication_graph, structural_graph,
)


SITE_IDS = [
    "unit.cpp:10:group_kernel::0",
    "unit.cpp:11:group_kernel::1",
]


class ProducerFissionSuiteRefreezeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        platform = copy.deepcopy(PLATFORM)
        platform["compiler_transforms"] = {"group_early_trigger": False}
        dossier = bridge.make_dossier(
            [fission_feature(site_id) for site_id in SITE_IDS], platform,
        )
        kernel_template = template()
        self.current_graph = expansion.json_value(
            expansion.groups.make_group_graph(dossier, [kernel_template])
        )
        _, self.expanded_graph, self.change = expansion.expand_graph(
            dossier, [kernel_template], self.current_graph,
        )
        self.current_graph_path = self.root / "jacobi-current.json"
        self.expanded_graph_path = self.root / "jacobi-expanded.json"
        self.other_comm_path = self.root / "minimod.json"
        self.collective_path = self.root / "collective.json"
        self.structural_path = self.root / "structural.json"
        for path, value in (
            (self.current_graph_path, self.current_graph),
            (self.expanded_graph_path, self.expanded_graph),
            (self.other_comm_path, communication_graph()),
            (self.collective_path, collective_graph()),
            (self.structural_path, structural_graph()),
        ):
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        self.current_prompts = self.root / "current-prompts"
        self.current_suite = suites.make_suite(
            [("jacobi", self.current_graph_path),
             ("minimod", self.other_comm_path)],
            [("collective_n8", self.collective_path)],
            [("coalescing_placement", self.structural_path)],
            prompt_dir=self.current_prompts,
        )
        self.current_suite_path = self.root / "current-suite.json"
        self.current_suite_path.write_text(
            json.dumps(self.current_suite, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.expansion_manifest = {
            "schema_version": expansion.EXPANSION_SCHEMA,
            "expansion_id": "sha256:" + "4" * 64,
            "graph_transition": {
                "current_graph_id": self.current_graph["graph_id"],
                "expanded_graph_id": self.expanded_graph["graph_id"],
                **self.change,
            },
            "inputs": [{
                "role": "frozen_current_graph",
                "path": str(self.current_graph_path.resolve()),
                "sha256": refreeze.sha256_file(self.current_graph_path),
                "bytes": self.current_graph_path.stat().st_size,
            }],
            "outputs": [{
                "role": "expanded_graph",
                "path": self.expanded_graph_path.name,
                "sha256": refreeze.sha256_file(self.expanded_graph_path),
                "bytes": self.expanded_graph_path.stat().st_size,
            }],
        }
        self.expansion_manifest_path = self.root / "expansion-manifest.json"
        self.expansion_manifest_path.write_text(
            json.dumps(self.expansion_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.communication = [("minimod", self.other_comm_path)]
        self.collective = [("collective_n8", self.collective_path)]
        self.structural = [("coalescing_placement", self.structural_path)]

    def tearDown(self):
        self.temporary.cleanup()

    def verified_expansion(self):
        return mock.patch.object(
            refreeze.expansion, "verify_contained",
            return_value=self.expansion_manifest,
        )

    def test_only_jacobi_entry_changes_and_provider_stays_unauthorized(self):
        with self.verified_expansion():
            value, _, delta = refreeze.make_refrozen_suite(
                self.current_suite_path, self.current_prompts,
                self.expansion_manifest_path, self.communication,
                self.collective, self.structural,
            )
        old = {entry["label"]: entry for entry in self.current_suite["entries"]}
        new = {entry["label"]: entry for entry in value["entries"]}
        for label in set(old) - {"jacobi"}:
            self.assertEqual(old[label], new[label])
        self.assertNotEqual(old["jacobi"], new["jacobi"])
        self.assertEqual(9, delta["old_policy_count"])
        self.assertEqual(10, delta["new_policy_count"])
        self.assertEqual(self.change["candidate_id"], delta["new_candidate_id"])
        self.assertFalse(value["boundary"]["provider_call_supported"])

    def test_bundle_is_atomic_and_self_verifying(self):
        output = self.root / "refrozen"
        with self.verified_expansion():
            manifest = refreeze.prepare_bundle(
                output, self.current_suite_path, self.current_prompts,
                self.expansion_manifest_path, self.communication,
                self.collective, self.structural,
            )
            verified = refreeze.verify_contained(output / "manifest.json")
        self.assertEqual(manifest, verified)
        self.assertEqual(
            "refrozen_suite_ready_for_readiness_audit", manifest["status"]
        )
        self.assertFalse(manifest["boundary"]["provider_call_authorized"])
        with self.verified_expansion(), self.assertRaisesRegex(
                refreeze.RefreezeError, "overwrite"):
            refreeze.prepare_bundle(
                output, self.current_suite_path, self.current_prompts,
                self.expansion_manifest_path, self.communication,
                self.collective, self.structural,
            )

    def test_missing_or_changed_non_jacobi_entry_is_rejected(self):
        with self.verified_expansion(), self.assertRaisesRegex(
                refreeze.RefreezeError, "exactly every"):
            refreeze.make_refrozen_suite(
                self.current_suite_path, self.current_prompts,
                self.expansion_manifest_path, [], self.collective,
                self.structural,
            )
        changed = communication_graph()
        changed["platform_profile"]["platform_id"] = "changed-platform"
        payload = dict(changed)
        payload.pop("graph_id")
        changed["graph_id"] = bridge._fingerprint(payload)
        changed_path = self.root / "changed-minimod.json"
        changed_path.write_text(json.dumps(changed), encoding="utf-8")
        with self.verified_expansion(), self.assertRaisesRegex(
                refreeze.RefreezeError, "unrelated entry"):
            refreeze.make_refrozen_suite(
                self.current_suite_path, self.current_prompts,
                self.expansion_manifest_path,
                [("minimod", changed_path)], self.collective, self.structural,
            )


if __name__ == "__main__":
    unittest.main()
