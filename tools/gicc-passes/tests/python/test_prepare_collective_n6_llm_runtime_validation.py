import copy
import importlib.util
import sys
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


if __name__ == "__main__":
    unittest.main()
