import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = PASS_ROOT / "experiments" / "producer_fission"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import gicc_comm_group_plan_bridge as groups
import gicc_llm_bridge as bridge
import prepare_confirmed_producer_fission_graph as expansion
from test_gicc_comm_group_plan_bridge import (  # noqa: E402
    PLATFORM, fission_feature, template,
)


SITE_IDS = [
    "unit.cpp:10:group_kernel::0",
    "unit.cpp:11:group_kernel::1",
]


class ConfirmedProducerFissionGraphTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        platform = copy.deepcopy(PLATFORM)
        platform["compiler_transforms"] = {"group_early_trigger": False}
        self.dossier = bridge.make_dossier(
            [fission_feature(site_id) for site_id in SITE_IDS], platform,
        )
        self.template = template()
        self.graph = groups.make_group_graph(self.dossier, [self.template])
        self.meta = self.root / "meta"
        self.meta.mkdir()
        self.features_path = self.meta / "features.json"
        self.features_path.write_text(
            json.dumps(
                [fission_feature(site_id) for site_id in SITE_IDS],
                indent=2, sort_keys=True,
            ) + "\n",
            encoding="utf-8",
        )
        self.dossier_path = self.root / "dossier.json"
        self.template_path = self.meta / "template.json"
        self.graph_path = self.root / "graph.json"
        for path, value in (
            (self.dossier_path, self.dossier),
            (self.template_path, self.template),
            (self.graph_path, self.graph),
        ):
            path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

        private_case, model_case = (
            expansion.confirmation.transition_base.coverage.load_case(
                "unit", self.features_path,
            )
        )
        model_graph = {
            "schema_version": "gicc-source-free-schedule-coverage-v1",
            "cases": [model_case],
            "policy": {
                "application_source_present": False,
                "model_output_scope": "existing compiler candidate IDs only",
                "provider_call_authorized": False,
            },
        }
        model_graph["graph_id"] = (
            expansion.confirmation.transition_base.coverage.canonical_id(
                "gicc-source-free-schedule-coverage-v1", model_graph,
            )
        )
        self.coverage_path = self.root / "coverage.json"
        self.coverage_path.write_text(json.dumps({
            "schema_version": "gicc-compiler-schedule-coverage-report-v1",
            "created_at": "unit",
            "cases": {"unit": private_case},
            "model_graph": model_graph,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        dormant = model_case["dormant_compiler_oracle"]
        self.transition_path = self.root / "transition.json"
        self.transition_path.write_text("{}\n", encoding="utf-8")
        self.transition = {
            "transition_id": "sha256:" + "1" * 64,
            "schedule_coverage_graph_id": model_graph["graph_id"],
            "dormant_compiler_candidate": dormant,
        }
        payload = {
            "schema_version": expansion.CONFIRMATION_SCHEMA,
            "scope": "unit confirmation",
            "model_invoked": False,
            "application_source_modified": False,
            "provider_call_authorized": False,
            "transition_id": self.transition["transition_id"],
            "dormant_compiler_candidate_id": self.transition[
                "dormant_compiler_candidate"
            ]["candidate_id"],
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
            "confirmation_gate": {"passed": True},
        }
        self.confirmation = {
            **payload, "result_id": bridge._fingerprint(payload),
        }
        self.confirmation_path = self.root / "confirmation.json"
        self._write_confirmation()

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
                return_value=(self.transition, {
                    "source_free_schedule_coverage": self.coverage_path,
                }),
            ),
            mock.patch.object(
                expansion.confirmation, "analyze_monitors",
                return_value=self.confirmation,
            ),
        )

    def test_expansion_preserves_old_candidates_and_exposes_only_fission(self):
        old_text = self.graph_path.read_text(encoding="utf-8")
        expanded_dossier, expanded_graph, change = expansion.expand_graph(
            self.dossier, [self.template], self.graph,
        )
        opportunity = expanded_graph["opportunities"][0]
        old_opportunity = self.graph["opportunities"][0]
        self.assertEqual(9, len(old_opportunity["candidates"]))
        self.assertEqual(10, len(opportunity["candidates"]))
        self.assertEqual(
            [candidate["candidate_id"] for candidate in old_opportunity["candidates"]],
            [candidate["candidate_id"] for candidate in opportunity["candidates"]
             if candidate["kind"] != expansion.FISSION_KIND],
        )
        candidate = next(
            candidate for candidate in opportunity["candidates"]
            if candidate["kind"] == expansion.FISSION_KIND
        )
        self.assertTrue(all(
            request["transform"] == expansion.FISSION_TRANSFORM
            for request in candidate["materializer"]["sites"].values()
        ))
        self.assertEqual([], opportunity["masked_candidates"])
        self.assertTrue(change["prior_candidate_ids_preserved"])
        self.assertTrue(
            expanded_dossier["platform_profile"]["compiler_transforms"]
            ["producer_frontier_fission"]
        )
        self.assertNotEqual(self.graph["graph_id"], expanded_graph["graph_id"])
        self.assertEqual(old_text, self.graph_path.read_text(encoding="utf-8"))

    def test_bundle_replays_confirmation_and_is_self_verifying(self):
        first, second = self.replay_patches()
        with first, second:
            rendered, manifest = expansion.build_bundle(
                self.confirmation_path, self.dossier_path,
                [self.template_path], self.graph_path,
            )
            output = self.root / "expanded"
            expansion.write_bundle(output, rendered)
            verified = expansion.verify_contained(output / "manifest.json")
        self.assertEqual(manifest, verified)
        self.assertEqual(
            "expanded_graph_ready_for_suite_refreeze", manifest["status"]
        )
        self.assertFalse(manifest["boundary"]["provider_call_authorized"])
        self.assertFalse(manifest["boundary"]["current_decision_suite_modified"])
        prompt = (output / "prompt-relational.txt").read_text(encoding="utf-8")
        self.assertNotIn("SECRET_SOURCE_SENTINEL", prompt)
        self.assertNotIn("unit.cpp", prompt)
        self.assertIn(expansion.FISSION_KIND, prompt)
        with self.assertRaisesRegex(expansion.ExpansionError, "overwrite"):
            expansion.write_bundle(output, rendered)

    def test_failed_or_tampered_confirmation_cannot_expand(self):
        tampered = copy.deepcopy(self.confirmation)
        tampered["confirmation_gate"]["passed"] = False
        payload = dict(tampered)
        payload.pop("result_id")
        tampered["result_id"] = bridge._fingerprint(payload)
        self.confirmation = tampered
        self._write_confirmation()
        with self.assertRaisesRegex(expansion.ExpansionError, "did not pass"):
            expansion.replay_confirmation(self.confirmation_path)

        self.confirmation["confirmation_gate"]["passed"] = True
        self._write_confirmation()
        with self.assertRaisesRegex(expansion.ExpansionError, "result ID"):
            expansion.replay_confirmation(self.confirmation_path)

    def test_current_graph_must_exactly_regenerate(self):
        changed = copy.deepcopy(self.graph)
        changed["objective"]["instruction"] = "changed"
        payload = dict(changed)
        payload.pop("graph_id")
        changed["graph_id"] = bridge._fingerprint(payload)
        with self.assertRaisesRegex(
            expansion.ExpansionError, "does not regenerate"
        ):
            expansion.expand_graph(self.dossier, [self.template], changed)


if __name__ == "__main__":
    unittest.main()
