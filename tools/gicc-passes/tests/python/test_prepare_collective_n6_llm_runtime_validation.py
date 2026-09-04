import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = Path(__file__).resolve().parent
COLLECTIVE = PASS_ROOT / "experiments" / "collective"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(PASS_ROOT / "experiments"))
sys.path.insert(0, str(COLLECTIVE))
sys.path.insert(0, str(TEST_ROOT))
SCRIPT = COLLECTIVE / "prepare_collective_n6_llm_runtime_validation.py"
BUILDER = COLLECTIVE / "build_collective_n6_llm_runtime_validation.sh"
SPEC = importlib.util.spec_from_file_location("collective_runtime_plan", SCRIPT)
runtime = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = runtime
SPEC.loader.exec_module(runtime)

import gicc_collective_plan_bridge as plans
import gicc_compiler_policy_bridge as policy_bridge
import gicc_llm_bridge as bridge
from test_gicc_collective_plan_bridge import PROFILE, inventory


class PrepareCollectiveLlmRuntimeValidationTests(unittest.TestCase):
    def setUp(self):
        profile = copy.deepcopy(PROFILE)
        profile["topology"]["nodes"] = 6
        profile["topology"]["ranks_per_node"] = 8
        self.graph = plans.make_graph(inventory(), profile)
        opportunity = self.graph["opportunities"][0]
        self.anchor = policy_bridge.decision_to_policy(
            self.graph, None
        )["selected_ids_by_slot"]
        mixed = {}
        for index, slot in enumerate(opportunity["decision_slots"]):
            option = slot["options"][index % len(slot["options"])]
            mixed[f"{opportunity['opportunity_id']}/{slot['slot_id']}"] = (
                option["option_id"]
            )
        self.mixed = policy_bridge.verified_policy(
            self.graph, mixed
        )["selected_ids_by_slot"]
        screen_payload = {
            "schema_version": runtime.capability_analysis.SCREEN_SCHEMA,
            "graph_id": self.graph["graph_id"],
            "controls": {
                "anchor": {"selected_ids_by_slot": self.anchor},
                "deterministic": {"selected_ids_by_slot": self.mixed},
                "oracle": {"selected_ids_by_slot": self.mixed},
            },
        }
        self.screen = {
            **screen_payload,
            "screen_id": bridge._fingerprint(screen_payload),
        }
        views = {
            view: {
                "primary_modal_representative": {
                    "selected_ids_by_slot": self.anchor,
                },
                "posthoc_capability_upper_bound": {
                    "selected_ids_by_slot": self.mixed,
                },
            }
            for view in runtime.request_freezer.VIEWS
        }
        analysis_payload = {
            "schema_version": runtime.capability_analysis.ANALYSIS_SCHEMA,
            "status": "offline_screen_complete_runtime_validation_required",
            "decision_family": "collective_algorithm_and_size_policy",
            "compiler_graph_id": self.graph["graph_id"],
            "screen_id": self.screen["screen_id"],
            "boundary": {
                "compiler_lto_decisions_only": True,
                "application_source_visible_or_modified": False,
                "representative_runtime_validation_still_required": True,
            },
            "results": {"metrics": {"views": views}},
        }
        self.analysis = {
            **analysis_payload,
            "analysis_id": bridge._fingerprint(analysis_payload),
        }

    def test_representatives_and_controls_are_deduplicated(self):
        policies = runtime.representative_policies(
            self.graph, self.analysis, self.screen,
        )
        self.assertEqual(2, len(policies))
        roles = {role for policy in policies for role in policy["roles"]}
        self.assertEqual(9, len(roles))
        self.assertIn("control:anchor", roles)
        self.assertIn("relational:primary_modal_representative", roles)
        self.assertIn("opaque:posthoc_capability_upper_bound", roles)

    def test_every_runtime_policy_maps_back_to_private_compiler_hint(self):
        policies = runtime.representative_policies(
            self.graph, self.analysis, self.screen,
        )
        for policy in policies:
            decision, hint = runtime.decision_and_hint(self.graph, policy)
            rebuilt = policy_bridge.decision_to_policy(self.graph, decision)
            self.assertTrue(rebuilt["bridge_accepted"])
            self.assertEqual(policy["policy_id"], rebuilt["policy_id"])
            self.assertFalse(hint["llm_metadata"]["model_invoked"])
            self.assertEqual(
                policy["policy_id"], hint["llm_metadata"]["policy_id"],
            )

    def test_three_allocation_order_is_frozen_and_rotated(self):
        policies = [{"name": f"policy{index:02d}"} for index in range(1, 6)]
        self.assertEqual({
            "1": ["policy01", "policy02", "policy03", "policy04", "policy05"],
            "2": ["policy03", "policy04", "policy05", "policy01", "policy02"],
            "3": ["policy05", "policy01", "policy02", "policy03", "policy04"],
        }, runtime.runtime_orders(policies))

    def test_contained_plan_rejects_changed_private_hint(self):
        policies = runtime.representative_policies(
            self.graph, self.analysis, self.screen,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evidence_files = {}
            for role in (
                "preparer", "compiler_graph", "capability_analysis",
                "policy_screen", "compiler_build_script",
                "compiler_evaluator", "collective_bridge",
            ):
                path = root / f"{role}.json"
                path.write_text("{}\n")
                evidence_files[role] = path
            evidence_files["compiler_graph"].write_text(
                json.dumps(self.graph) + "\n"
            )
            for policy in policies:
                policy_root = root / "policies" / policy["name"]
                decision, hint = runtime.decision_and_hint(self.graph, policy)
                runtime.write_json_atomic(policy_root / "decision.json", decision)
                runtime.write_json_atomic(policy_root / "hint.json", hint)
            payload = runtime.plan_payload(
                graph=self.graph, analysis=self.analysis, screen=self.screen,
                policies=policies, output_dir=root,
                input_paths=evidence_files,
            )
            runtime.write_json_atomic(root / "plan.json", {
                "plan_id": bridge._fingerprint(payload), **payload,
            })
            plan, graph = runtime.verify_contained(root)
            self.assertEqual(self.graph["graph_id"], graph["graph_id"])
            self.assertEqual(len(policies), plan["unique_policy_count"])

            changed = copy.deepcopy(plan)
            changed["runtime_contract"]["queue"] = "pci"
            changed_payload = dict(changed)
            changed_payload.pop("plan_id")
            changed["plan_id"] = bridge._fingerprint(changed_payload)
            runtime.write_json_atomic(root / "plan.json", changed)
            with self.assertRaisesRegex(
                runtime.RuntimeValidationError, "execution contract changed"
            ):
                runtime.verify_contained(root)
            runtime.write_json_atomic(root / "plan.json", plan)

            hint_path = root / "policies" / policies[0]["name"] / "hint.json"
            hint_path.write_text("{}\n")
            with self.assertRaisesRegex(
                runtime.RuntimeValidationError, "evidence changed"
            ):
                runtime.verify_contained(root)

    def test_builder_is_lto_only_and_scheduler_free(self):
        subprocess.run(["bash", "-n", BUILDER], check=True)
        text = BUILDER.read_text(encoding="utf-8")
        self.assertNotIn("flux ", text)
        self.assertNotIn("provider", text.lower())
        self.assertIn('"$builder" lower', text)
        self.assertIn("verify-plan-ir", text)
        self.assertIn('"$preparer" verify-built', text)


if __name__ == "__main__":
    unittest.main()
