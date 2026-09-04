import copy
import importlib.util
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(TEST_ROOT))
SCRIPT = (
    PASS_ROOT / "experiments" / "analyze_compiler_llm_capability_trials.py"
)
SPEC = importlib.util.spec_from_file_location("capability_analysis", SCRIPT)
analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = analysis
SPEC.loader.exec_module(analysis)

import gicc_comm_plan_bridge as structural
import gicc_llm_bridge as llm
from test_gicc_comm_plan_bridge import PLATFORM, coalescable_feature


class CompilerLlmCapabilityAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.graph = structural.make_opportunity_graph(llm.make_dossier(
            [coalescable_feature()], PLATFORM,
        ))
        fallback = analysis.policy_bridge.decision_to_policy(self.graph, None)
        self.anchor = fallback["selected_ids_by_slot"]
        opportunity = self.graph["opportunities"][0]
        fast_candidate = next(
            candidate for candidate in opportunity["candidates"]
            if candidate["kind"] == "trigger_coalesced_loop"
        )
        self.fast = {
            opportunity["opportunity_id"]: fast_candidate["candidate_id"]
        }
        self.anchor_id = analysis.policy_bridge.verified_policy(
            self.graph, self.anchor
        )["policy_id"]
        self.fast_id = analysis.policy_bridge.verified_policy(
            self.graph, self.fast
        )["policy_id"]
        runs = []
        for view in analysis.request_freezer.VIEWS:
            for trial in range(1, 21):
                if view == "relational":
                    accepted = trial <= 15
                    selected = self.fast if accepted else self.anchor
                elif view == "descriptors":
                    accepted = True
                    selected = self.fast if trial <= 10 else self.anchor
                else:
                    accepted = True
                    selected = self.anchor
                policy_id = (
                    self.fast_id if selected == self.fast else self.anchor_id
                )
                runs.append({
                    "view": view,
                    "trial": trial,
                    "bridge_accepted": accepted,
                    "fallback_applied": not accepted,
                    "selected_ids_by_slot": selected,
                    "policy_id": policy_id,
                })
        self.index = {"status": "complete", "runs": runs}
        self.controls = {
            "oracle": {
                "selected_ids_by_slot": self.fast,
                "cost_by_unit": {"u": 1.0},
            },
            "anchor": {
                "selected_ids_by_slot": self.anchor,
                "cost_by_unit": {"u": 2.0},
            },
            "deterministic": {
                "selected_ids_by_slot": self.anchor,
                "cost_by_unit": {"u": 2.0},
            },
        }
        self.screen = {
            "controls": self.controls,
            "unit_weights": {"u": 1.0},
            "policy_cost_by_id": {
                self.fast_id: {"u": 1.0},
                self.anchor_id: {"u": 2.0},
            },
        }

    def test_itt_modal_and_posthoc_outputs_share_verified_archive(self):
        result = analysis.score_archive(
            self.index, self.graph, self.screen
        )
        relational = result["metrics"]["views"]["relational"]
        self.assertEqual(60, result["screened_trial_count"])
        self.assertEqual(2, result["unique_policy_count"])
        self.assertEqual(0.25, relational["invalid_output_rate"])
        self.assertAlmostEqual(
            math.pow(2.0, 15.0 / 20.0),
            relational["intention_to_treat"]["geomean_speedup_over_anchor"],
        )
        self.assertEqual(
            self.fast_id,
            relational["primary_modal_representative"]["policy_id"],
        )
        self.assertEqual(
            self.fast_id,
            relational["posthoc_capability_upper_bound"]["policy_id"],
        )
        provenance = result["representative_provenance"]["relational"]
        self.assertFalse(provenance["primary_modal"]["posthoc"])
        self.assertTrue(provenance["posthoc_best_of_20"]["posthoc"])
        self.assertEqual(1, provenance["primary_modal"]["representative_trial"])

    def test_missing_screen_cost_and_non_anchor_fallback_fail_closed(self):
        screen = copy.deepcopy(self.screen)
        screen["policy_cost_by_id"].pop(self.fast_id)
        with self.assertRaisesRegex(
            analysis.CapabilityAnalysisError, "lacks an observed policy"
        ):
            analysis.score_archive(self.index, self.graph, screen)

        index = copy.deepcopy(self.index)
        rejected = next(
            row for row in index["runs"] if not row["bridge_accepted"]
        )
        rejected["selected_ids_by_slot"] = self.fast
        rejected["policy_id"] = self.fast_id
        with self.assertRaisesRegex(
            analysis.CapabilityAnalysisError, "compiler anchor"
        ):
            analysis.score_archive(index, self.graph, self.screen)

    def test_exact_oracle_rate_is_chance_calibrated(self):
        scored = analysis.score_archive(self.index, self.graph, self.screen)
        null_entry = analysis.sampling_null.uniform_oracle_null(4, 20)
        result = analysis.chance_calibration(scored["metrics"], null_entry)
        self.assertEqual(15, result["views"]["relational"][
            "observed_exact_oracle_hits"
        ])
        self.assertTrue(result["views"]["relational"][
            "meets_one_sided_alpha_0_05_hit_threshold"
        ])
        self.assertEqual(0, result["views"]["opaque"][
            "observed_exact_oracle_hits"
        ])
        self.assertFalse(result[
            "null_is_model_distribution_or_performance_evidence"
        ])

    def test_screen_binds_archive_boundary_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter.py"
            controls_file = root / "controls.json"
            index_path = root / "archive/run-index.json"
            adapter.write_text("# family adapter\n")
            controls_file.write_text("{}\n")
            index_path.parent.mkdir()
            index_path.write_text(json.dumps(self.index, sort_keys=True) + "\n")

            def record(path):
                return {
                    "path": path.relative_to(root).as_posix(),
                    "sha256": analysis.sha256_file(path),
                    "bytes": path.stat().st_size,
                }

            payload = {
                "schema_version": analysis.SCREEN_SCHEMA,
                "graph_id": self.graph["graph_id"],
                "request_id": "sha256:" + "1" * 64,
                "run_index_sha256": analysis.sha256_file(index_path),
                "boundary": dict(analysis.SCREEN_BOUNDARY),
                "controls": self.controls,
                "unit_weights": self.screen["unit_weights"],
                "policy_cost_by_id": self.screen["policy_cost_by_id"],
                "evidence": {
                    "family_adapter": record(adapter),
                    "runtime_controls": [record(controls_file)],
                },
            }
            screen = dict(payload)
            screen["screen_id"] = llm._fingerprint(payload)
            verified = analysis.verified_screen(
                screen, graph=self.graph,
                request={"request_id": payload["request_id"]},
                index=self.index, index_path=index_path, repo_root=root,
            )
            self.assertEqual(screen, verified)

            changed = copy.deepcopy(screen)
            changed["boundary"]["provider_visible"] = True
            changed_payload = dict(changed)
            changed_payload.pop("screen_id")
            changed["screen_id"] = llm._fingerprint(changed_payload)
            with self.assertRaisesRegex(
                analysis.CapabilityAnalysisError, "held-out boundary"
            ):
                analysis.verified_screen(
                    changed, graph=self.graph,
                    request={"request_id": payload["request_id"]},
                    index=self.index, index_path=index_path, repo_root=root,
                )


if __name__ == "__main__":
    unittest.main()
