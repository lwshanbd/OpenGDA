import copy
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
TEST_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(TEST_ROOT))

import gicc_collective_plan_bridge as collective
import gicc_comm_group_plan_bridge as communication
import gicc_comm_plan_bridge as structural
import gicc_compiler_policy_bridge as unified
import gicc_llm_bridge as llm
import test_gicc_collective_plan_bridge as collective_fixture
import test_gicc_comm_group_plan_bridge as communication_fixture
import test_gicc_comm_plan_bridge as structural_fixture


class UnifiedCompilerPolicyBridgeTests(unittest.TestCase):
    def test_structural_response_normalizes_and_falls_back(self):
        graph = structural.make_opportunity_graph(llm.make_dossier(
            [structural_fixture.coalescable_feature()],
            structural_fixture.PLATFORM,
        ))
        opportunity = graph["opportunities"][0]
        chosen = next(
            candidate for candidate in opportunity["candidates"]
            if candidate["kind"] == "trigger_coalesced_loop"
        )
        decision = {
            "schema_version": structural.DECISION_SCHEMA,
            "graph_id": graph["graph_id"],
            "selections": {opportunity["opportunity_id"]: {
                "candidate_id": chosen["candidate_id"],
                "confidence": 0.8,
                "rationale": "compiler facts",
            }},
        }
        accepted = unified.decision_to_policy(graph, decision)
        self.assertTrue(accepted["bridge_accepted"])
        self.assertEqual(
            chosen["candidate_id"],
            accepted["selected_ids_by_slot"][opportunity["opportunity_id"]],
        )
        changed_rationale = copy.deepcopy(decision)
        changed_rationale["selections"][opportunity["opportunity_id"]][
            "rationale"
        ] = "different prose"
        self.assertEqual(
            accepted["policy_id"],
            unified.decision_to_policy(graph, changed_rationale)["policy_id"],
        )

        rejected = copy.deepcopy(decision)
        rejected["source"] = "forbidden"
        fallback = unified.decision_to_policy(graph, rejected)
        trigger = next(
            candidate for candidate in opportunity["candidates"]
            if candidate["kind"] == "trigger_descriptor_batch"
        )
        self.assertFalse(fallback["bridge_accepted"])
        self.assertTrue(fallback["fallback_applied"])
        self.assertEqual(
            trigger["candidate_id"],
            fallback["selected_ids_by_slot"][opportunity["opportunity_id"]],
        )

    def test_group_response_uses_same_policy_shape(self):
        site_ids = [
            "unit.cpp:10:group_kernel::0", "unit.cpp:11:group_kernel::1",
        ]
        graph = communication.make_group_graph(
            llm.make_dossier(
                [communication_fixture.feature(site_id) for site_id in site_ids],
                communication_fixture.PLATFORM,
            ),
            [communication_fixture.template()],
        )
        opportunity = graph["opportunities"][0]
        chosen = next(
            candidate for candidate in opportunity["candidates"]
            if candidate["kind"] == "group_trigger_early"
        )
        decision = {
            "schema_version": communication.DECISION_SCHEMA,
            "graph_id": graph["graph_id"],
            "selections": {opportunity["opportunity_id"]: {
                "candidate_id": chosen["candidate_id"],
                "confidence": 0.9,
                "rationale": "proved group frontier",
            }},
        }
        accepted = unified.decision_to_policy(graph, decision)
        self.assertTrue(accepted["bridge_accepted"])
        self.assertEqual(
            "communication_route_or_schedule", accepted["decision_family"]
        )
        rejected = copy.deepcopy(decision)
        rejected["selections"][opportunity["opportunity_id"]][
            "candidate_id"
        ] = "candidate:" + "0" * 24
        fallback = unified.decision_to_policy(graph, rejected)
        baseline = next(
            candidate for candidate in opportunity["candidates"]
            if candidate["kind"] == "group_uniform_default"
        )
        self.assertFalse(fallback["bridge_accepted"])
        self.assertEqual(
            baseline["candidate_id"],
            fallback["selected_ids_by_slot"][opportunity["opportunity_id"]],
        )

    def test_collective_slots_flatten_without_losing_identity(self):
        graph = collective.make_graph(
            collective_fixture.inventory(), collective_fixture.PROFILE,
        )
        opportunity = graph["opportunities"][0]
        decision = {}
        fixture = collective_fixture.CollectivePlanBridgeTests()
        fixture.graph = graph
        fixture.opportunity = opportunity
        decision = fixture.decision()
        accepted = unified.decision_to_policy(graph, decision)
        self.assertTrue(accepted["bridge_accepted"])
        self.assertEqual(4, len(accepted["selected_ids_by_slot"]))
        self.assertTrue(all(
            key.startswith(opportunity["opportunity_id"] + "/")
            for key in accepted["selected_ids_by_slot"]
        ))

        rejected = copy.deepcopy(decision)
        rejected["llvm_ir"] = "call void @invented()"
        fallback = unified.decision_to_policy(graph, rejected)
        self.assertFalse(fallback["bridge_accepted"])
        for slot in opportunity["decision_slots"]:
            anchor = next(
                option for option in slot["options"] if option["role"] == "anchor"
            )
            self.assertEqual(
                anchor["option_id"], fallback["selected_ids_by_slot"][
                    f"{opportunity['opportunity_id']}/{slot['slot_id']}"
                ],
            )

    def test_unknown_graph_schema_is_rejected(self):
        with self.assertRaisesRegex(
            unified.CompilerPolicyBridgeError, "unsupported"
        ):
            unified.decision_to_policy({"schema_version": "invented"}, {})

    def test_public_graph_and_schema_dispatch_match_family_bridge(self):
        graph = collective.make_graph(
            collective_fixture.inventory(), collective_fixture.PROFILE,
        )
        self.assertEqual(graph, unified.verified_graph(graph))
        self.assertEqual(
            collective.decision_response_schema(graph),
            unified.decision_response_schema(graph),
        )


if __name__ == "__main__":
    unittest.main()
