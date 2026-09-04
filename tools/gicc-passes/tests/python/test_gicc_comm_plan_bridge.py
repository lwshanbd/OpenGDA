import copy
import json
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "python"))

import gicc_comm_plan_bridge as plans
import gicc_llm_bridge as bridge


PLATFORM = {
    "schema_version": "gicc-platform-profile-v1",
    "platform_id": "unit-platform",
    "measurements": {"proxy_fixed_us": 5.0, "trigger_fixed_us": 8.0},
}


def coalescable_feature(site_id="unit.cpp:10:k::0"):
    return {
        "schema_version": 6,
        "site_id": site_id,
        "kernel": "k",
        "op_kind": "put_no_db",
        "hk_capable": True,
        "size_kind": "const",
        "size_bytes": 1024,
        "size_log2": 10,
        "peer_kind": "param",
        "peer_locality": None,
        "in_loop": True,
        "loop": {
            "bound_known": True,
            "bound_const": 8,
            "iv_start": 0,
            "iv_step": 1,
        },
        "guard_density": 1.0,
        "guard_kind": "always",
        "fan_out": 1,
        "static_launch_sites": 1,
        "launch_contexts": [],
        "launch_grid": {"x": 1, "y": 1, "z": 1},
        "launch_block": {"x": 1, "y": 1, "z": 1},
        "grid_blocks": 1,
        "threads_per_block": 1,
        "compute_before_flops": 0,
        "flops_to_first_use": 32,
        "trip_count": 8,
        "distance_exact": True,
        "iter_estimate": None,
        "descriptor_reusable": False,
        "buffer_reusable": True,
        "coalescable": True,
        "max_vector_bytes": 16,
        "batch_size": 8,
        "legal_paths": ["proxy", "trigger"],
    }


def fixed_feature(site_id, legal_paths):
    feature = coalescable_feature(site_id)
    feature.update({
        "kernel": "fixed",
        "in_loop": False,
        "loop": None,
        "trip_count": None,
        "coalescable": False,
        "legal_paths": legal_paths,
    })
    return feature


def reusable_feature(site_id="unit.cpp:50:reuse::0"):
    feature = coalescable_feature(site_id)
    feature.update({
        "kernel": "reuse",
        "size_kind": "param",
        "size_bytes": None,
        "size_log2": None,
        "loop": {
            "bound_known": True,
            "bound_param_idx": 4,
            "bound_param_type": "i32",
            "iv_start": 0,
            "iv_step": 1,
        },
        "trip_count": None,
        "descriptor_reusable": True,
        "coalescable": False,
        "batch_size": 1,
        "distance_exact": False,
    })
    return feature


class CommunicationPlanBridgeTests(unittest.TestCase):
    def setUp(self):
        self.dossier = bridge.make_dossier(
            [coalescable_feature()], PLATFORM
        )
        self.graph = plans.make_opportunity_graph(self.dossier)
        self.opportunity = self.graph["opportunities"][0]

    def candidate(self, kind):
        return next(
            candidate for candidate in self.opportunity["candidates"]
            if candidate["kind"] == kind
        )

    def decision(self, kind="trigger_coalesced_loop"):
        return {
            "schema_version": plans.DECISION_SCHEMA,
            "graph_id": self.graph["graph_id"],
            "selections": {
                self.opportunity["opportunity_id"]: {
                    "candidate_id": self.candidate(kind)["candidate_id"],
                    "confidence": 0.9,
                    "rationale": "one descriptor replaces eight proven-adjacent PUTs",
                }
            },
        }

    def test_graph_is_content_addressed_and_compiler_only(self):
        plans.verified_graph(self.graph)
        self.assertFalse(self.graph["boundary"]["source_visible"])
        self.assertFalse(self.graph["boundary"]["model_may_generate_code"])
        self.assertFalse(self.graph["boundary"]["model_may_generate_ir"])
        self.assertEqual(
            {
                "proxy_device",
                "trigger_descriptor_batch",
                "trigger_coalesced_loop",
            },
            {candidate["kind"] for candidate in self.opportunity["candidates"]},
        )
        prompt = plans.render_prompt(self.graph)
        self.assertNotIn("SECRET_SOURCE_SENTINEL", prompt)
        self.assertNotIn("unit.cpp", prompt)
        self.assertNotIn('"site_ids"', prompt)
        self.assertNotIn('"materializer"', prompt)
        self.assertIn("candidate IDs only", prompt)

    def test_model_views_ablate_relations_without_changing_actions(self):
        second = coalescable_feature("unit.cpp:20:k2::0")
        second["kernel"] = "k2"
        second["size_bytes"] = 2048
        second["size_log2"] = 11
        dossier = bridge.make_dossier(
            [coalescable_feature(), second], PLATFORM
        )
        graph = plans.make_opportunity_graph(dossier)
        views = {
            kind: plans.model_view(graph, kind)
            for kind in plans.MODEL_VIEW_KINDS
        }
        expected_ids = {
            opportunity["opportunity_id"]: {
                candidate["candidate_id"]
                for candidate in opportunity["candidates"]
            }
            for opportunity in graph["opportunities"]
        }
        for kind, view in views.items():
            self.assertEqual(plans.MODEL_VIEW_SCHEMA, view["schema_version"])
            self.assertEqual(kind, view["view_kind"])
            self.assertEqual(graph["graph_id"], view["compiler_graph_id"])
            self.assertEqual(
                expected_ids,
                {
                    opportunity["opportunity_id"]: {
                        candidate["candidate_id"]
                        for candidate in opportunity["candidates"]
                    }
                    for opportunity in view["opportunities"]
                },
            )
            rendered = json.dumps(view, sort_keys=True)
            self.assertNotIn("unit.cpp", rendered)
            self.assertNotIn('"site_ids"', rendered)
            self.assertNotIn('"materializer"', rendered)
        self.assertTrue(views["relational"]["relations"])
        size_order = next(
            relation
            for relation in views["relational"]["relations"]
            if relation["kind"] == "ascending_numeric_compiler_fact"
            and relation["fact"] == "size_bytes"
        )
        self.assertEqual(
            [1024, 2048],
            [member["value"] for member in size_order["members"]],
        )
        self.assertEqual([], views["descriptors"]["relations"])
        self.assertEqual([], views["opaque"]["relations"])
        self.assertIn(
            "summary",
            views["descriptors"]["opportunities"][0]["candidates"][0],
        )
        self.assertEqual(
            [
                {"candidate_id": candidate["candidate_id"]}
                for candidate in graph["opportunities"][0]["candidates"]
            ],
            views["opaque"]["opportunities"][0]["candidates"],
        )

    def test_response_schema_is_exactly_graph_bound(self):
        schema = plans.decision_response_schema(self.graph)
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            {"const": plans.DECISION_SCHEMA},
            schema["properties"]["schema_version"],
        )
        self.assertEqual(
            {"const": self.graph["graph_id"]},
            schema["properties"]["graph_id"],
        )
        selections = schema["properties"]["selections"]
        self.assertEqual(
            [self.opportunity["opportunity_id"]], selections["required"]
        )
        self.assertEqual(
            [candidate["candidate_id"]
             for candidate in self.opportunity["candidates"]],
            selections["properties"][self.opportunity["opportunity_id"]][
                "properties"
            ]["candidate_id"]["enum"],
        )
        rendered = json.dumps(schema, sort_keys=True)
        self.assertNotIn("unit.cpp", rendered)
        self.assertNotIn("materializer", rendered)

    def test_unknown_model_view_and_identity_leak_are_rejected(self):
        with self.assertRaisesRegex(plans.PlanBridgeError, "unknown model view"):
            plans.model_view(self.graph, "invented")
        platform = copy.deepcopy(PLATFORM)
        platform["leaked_identity"] = "unit.cpp:10:k::0"
        graph = plans.make_opportunity_graph(
            bridge.make_dossier([coalescable_feature()], platform)
        )
        with self.assertRaisesRegex(
            plans.PlanBridgeError, "identity escaped"
        ):
            plans.model_view(graph)

    def test_valid_selection_becomes_narrow_compiler_hint(self):
        hint, accepted, errors = plans.plan_to_hint(
            self.graph, self.decision()
        )
        self.assertTrue(accepted)
        self.assertEqual([], errors)
        site = hint["sites"]["unit.cpp:10:k::0"]
        self.assertEqual("DWQ_TRIGGER", site["dispatch"])
        self.assertEqual("COALESCE_LOOP", site["transform"])
        self.assertTrue(hint["llm_metadata"]["compiler_only_output"])

    def test_profile_enabled_early_trigger_is_candidate_id_only(self):
        platform = copy.deepcopy(PLATFORM)
        platform["compiler_transforms"] = {"early_trigger": True}
        dossier = bridge.make_dossier([coalescable_feature()], platform)
        graph = plans.make_opportunity_graph(dossier)
        opportunity = graph["opportunities"][0]
        self.assertEqual(
            {
                "proxy_device",
                "trigger_descriptor_batch",
                "trigger_coalesced_loop",
                "trigger_coalesced_early",
            },
            {candidate["kind"] for candidate in opportunity["candidates"]},
        )
        early = next(
            candidate for candidate in opportunity["candidates"]
            if candidate["kind"] == "trigger_coalesced_early"
        )
        decision = {
            "schema_version": plans.DECISION_SCHEMA,
            "graph_id": graph["graph_id"],
            "selections": {
                opportunity["opportunity_id"]: {
                    "candidate_id": early["candidate_id"],
                    "confidence": 0.8,
                    "rationale": "overlap the compiler-measured distance",
                }
            },
        }
        hint, accepted, errors = plans.plan_to_hint(graph, decision)
        self.assertTrue(accepted, errors)
        self.assertEqual(
            "COALESCE_LOOP_EARLY",
            hint["sites"]["unit.cpp:10:k::0"]["transform"],
        )

    def test_early_trigger_requires_exact_compiler_distance(self):
        platform = copy.deepcopy(PLATFORM)
        platform["compiler_transforms"] = {"early_trigger": True}
        feature = coalescable_feature()
        feature["distance_exact"] = False
        dossier = bridge.make_dossier([feature], platform)
        graph = plans.make_opportunity_graph(dossier)
        self.assertNotIn(
            "trigger_coalesced_early",
            {candidate["kind"] for candidate in graph["opportunities"][0]["candidates"]},
        )

    def test_reused_descriptor_is_invisible_until_profile_enabled(self):
        dossier = bridge.make_dossier([reusable_feature()], PLATFORM)
        with self.assertRaisesRegex(
            plans.PlanBridgeError, "no compiler-proved structural opportunity"
        ):
            plans.make_opportunity_graph(dossier)

        platform = copy.deepcopy(PLATFORM)
        platform["compiler_transforms"] = {
            "reused_loop_descriptor": True,
        }
        graph = plans.make_opportunity_graph(
            bridge.make_dossier([reusable_feature()], platform)
        )
        opportunity = graph["opportunities"][0]
        self.assertEqual(
            {
                "proxy_device",
                "trigger_descriptor_batch",
                "trigger_reused_descriptor_loop",
            },
            {candidate["kind"] for candidate in opportunity["candidates"]},
        )
        reused = next(
            candidate for candidate in opportunity["candidates"]
            if candidate["kind"] == "trigger_reused_descriptor_loop"
        )
        self.assertEqual(
            {"kind": "runtime_loop_bound", "kernel_param_index": 4},
            reused["effects"]["network_operations"],
        )
        decision = {
            "schema_version": plans.DECISION_SCHEMA,
            "graph_id": graph["graph_id"],
            "selections": {
                opportunity["opportunity_id"]: {
                    "candidate_id": reused["candidate_id"],
                    "confidence": 0.8,
                    "rationale": "reuse the compiler-proved invariant descriptor",
                }
            },
        }
        hint, accepted, errors = plans.plan_to_hint(graph, decision)
        self.assertTrue(accepted, errors)
        self.assertEqual(
            "REUSE_LOOP_DESCRIPTOR",
            hint["sites"]["unit.cpp:50:reuse::0"]["transform"],
        )

    def test_reused_descriptor_rejects_unproved_bound_type(self):
        platform = copy.deepcopy(PLATFORM)
        platform["compiler_transforms"] = {
            "reused_loop_descriptor": True,
        }
        feature = reusable_feature()
        feature["loop"]["bound_param_type"] = "i64"
        with self.assertRaisesRegex(
            plans.PlanBridgeError, "no compiler-proved structural opportunity"
        ):
            plans.make_opportunity_graph(
                bridge.make_dossier([feature], platform)
            )

    def test_invented_candidate_falls_back_without_transform(self):
        decision = self.decision()
        selection = decision["selections"][self.opportunity["opportunity_id"]]
        selection["candidate_id"] = "candidate:" + "0" * 24
        hint, accepted, errors = plans.plan_to_hint(self.graph, decision)
        self.assertFalse(accepted)
        self.assertTrue(any("not compiler-generated" in e for e in errors))
        site = hint["sites"]["unit.cpp:10:k::0"]
        self.assertEqual("DWQ_TRIGGER", site["dispatch"])
        self.assertNotIn("transform", site)

    def test_code_or_ir_payload_is_rejected(self):
        decision = self.decision()
        selection = decision["selections"][self.opportunity["opportunity_id"]]
        selection["llvm_ir"] = "call void @invented()"
        hint, accepted, errors = plans.plan_to_hint(self.graph, decision)
        self.assertFalse(accepted)
        self.assertTrue(any("unknown field" in e for e in errors))
        self.assertNotIn("transform", hint["sites"]["unit.cpp:10:k::0"])

    def test_tampered_graph_is_rejected(self):
        graph = copy.deepcopy(self.graph)
        graph["opportunities"][0]["candidates"][-1]["effects"][
            "coalesced_bytes"
        ] = 1
        with self.assertRaisesRegex(plans.PlanBridgeError, "graph_id"):
            plans.verified_graph(graph)

    def test_legality_disqualifiers_produce_no_opportunity(self):
        mutations = [
            ("not adjacent", lambda f: f.update(coalescable=False)),
            ("GET unsupported", lambda f: f.update(op_kind="get_no_db")),
            ("dynamic size", lambda f: f.update(size_kind="param")),
            ("per-iteration guard", lambda f: f.update(
                guard_density=0.5, guard_kind="field_not_null"
            )),
            ("non-unit step", lambda f: f["loop"].update(iv_step=2)),
            ("inconsistent trips", lambda f: f.update(trip_count=7)),
        ]
        for label, mutate in mutations:
            with self.subTest(label=label):
                feature = coalescable_feature()
                mutate(feature)
                dossier = bridge.make_dossier([feature], PLATFORM)
                with self.assertRaisesRegex(
                    plans.PlanBridgeError, "no compiler-proved"
                ):
                    plans.make_opportunity_graph(dossier)

    def test_graph_and_candidate_ids_are_deterministic(self):
        again = plans.make_opportunity_graph(self.dossier)
        self.assertEqual(self.graph, again)

    def test_non_opportunity_sites_are_compiler_fixed(self):
        dossier = bridge.make_dossier(
            [
                coalescable_feature(),
                fixed_feature("unit.cpp:20:default::0", [
                    "proxy", "trigger", "ipc"
                ]),
                fixed_feature("unit.cpp:30:trigger::0", ["proxy", "trigger"]),
                fixed_feature("unit.cpp:40:proxy::0", ["proxy"]),
            ],
            PLATFORM,
        )
        graph = plans.make_opportunity_graph(dossier)
        fixed = {
            item["site_id"]: item["materializer"]
            for item in graph["fixed_sites"]
        }
        self.assertEqual(
            {"dispatch": "IPC_OR_DWQ", "transform": "NONE"},
            fixed["unit.cpp:20:default::0"],
        )
        self.assertEqual(
            {"dispatch": "DWQ_TRIGGER", "transform": "NONE"},
            fixed["unit.cpp:30:trigger::0"],
        )
        self.assertEqual(
            {"dispatch": "CPU_PROXY_ENQUEUE", "transform": "NONE"},
            fixed["unit.cpp:40:proxy::0"],
        )
        hint, accepted, errors = plans.plan_to_hint(
            graph,
            {
                "schema_version": plans.DECISION_SCHEMA,
                "graph_id": graph["graph_id"],
                "selections": {
                    graph["opportunities"][0]["opportunity_id"]: {
                        "candidate_id": graph["opportunities"][0][
                            "candidates"
                        ][-1]["candidate_id"],
                        "confidence": 1.0,
                        "rationale": "exact compiler control",
                    }
                },
            },
        )
        self.assertTrue(accepted, errors)
        self.assertNotIn("unit.cpp:20:default::0", hint["sites"])
        self.assertEqual(
            "DWQ_TRIGGER", hint["sites"]["unit.cpp:30:trigger::0"]["dispatch"]
        )
        self.assertEqual(
            "CPU_PROXY_ENQUEUE",
            hint["sites"]["unit.cpp:40:proxy::0"]["dispatch"],
        )


if __name__ == "__main__":
    unittest.main()
