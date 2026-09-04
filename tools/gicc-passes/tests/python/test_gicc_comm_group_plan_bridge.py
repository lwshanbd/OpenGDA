import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "python"))

import gicc_comm_group_plan_bridge as groups
import gicc_llm_bridge as bridge


PLATFORM = {
    "schema_version": "gicc-platform-profile-v1",
    "platform_id": "unit-group-platform",
    "compiler_transforms": {"group_early_trigger": True},
    "measurements": {"proxy_fixed_us": 5.0, "trigger_fixed_us": 8.0},
}


def feature(site_id):
    return {
        "schema_version": 6,
        "site_id": site_id,
        "kernel": "group_kernel",
        "op_kind": "put_no_db",
        "hk_capable": True,
        "size_kind": "param",
        "size_bytes": None,
        "size_log2": None,
        "peer_kind": "param",
        "peer_locality": None,
        "in_loop": False,
        "loop": None,
        "guard_density": 1.0,
        "guard_kind": "always",
        "fan_out": 2,
        "static_launch_sites": 1,
        "launch_contexts": [{
            "static_callsite_count": 1,
            "launch_grid": {"x": None, "y": None, "z": None},
            "launch_block": {"x": None, "y": None, "z": None},
            "grid_blocks": None,
            "threads_per_block": None,
            "size_bytes": None,
            "trip_count": None,
        }],
        "launch_grid": {"x": None, "y": None, "z": None},
        "launch_block": {"x": None, "y": None, "z": None},
        "grid_blocks": None,
        "threads_per_block": None,
        "compute_before_flops": 0,
        "flops_to_first_use": 37,
        "trip_count": None,
        "distance_exact": True,
        "iter_estimate": None,
        "descriptor_reusable": False,
        "buffer_reusable": True,
        "coalescable": False,
        "max_vector_bytes": 1,
        "batch_size": 2,
        "legal_paths": ["proxy", "trigger", "ipc"],
    }


def template(*, early_legal=True):
    reason = (
        "proved mandatory flush and no intervening memory writes"
        if early_legal
        else "intervening instruction may write a registered source buffer"
    )
    sites = ["unit.cpp:10:group_kernel::0", "unit.cpp:11:group_kernel::1"]
    ops = []
    for index, site_id in enumerate(sites):
        ops.append({
            "site_id": site_id,
            "kind": "put_no_db",
            "args": {
                "target_rank": {"kind": "param", "param": 8 + index * 2},
                "dst_buf": {"kind": "param", "param": 9 + index * 2},
                "dst_off": {"kind": "param", "param": 15 + index},
                "src_buf": {"kind": "param", "param": 12},
                "src_off": {"kind": "param", "param": 13 + index},
                "size": {"kind": "param", "param": 17},
            },
            "guard": {"kind": "always"},
            "hk_capable": True,
            "compute_after": 37,
            "distance_exact": True,
            "batch_size": 2,
            "completion_site_id": "unit.cpp:20:group_kernel::2",
            "group_early_trigger_legal": early_legal,
            "group_early_trigger_reason": reason,
        })
    ops.extend([
        {
            "site_id": "unit.cpp:20:group_kernel::2",
            "kind": "flush",
            "args": {},
            "guard": {"kind": "always"},
            "hk_capable": True,
        },
        {
            "site_id": "unit.cpp:21:group_kernel::3",
            "kind": "quiet",
            "args": {},
            "guard": {"kind": "always"},
            "hk_capable": True,
        },
    ])
    return {
        "version": 1,
        "kernel_mangled": "_Z12group_kernelv",
        "kernel_simple": "group_kernel",
        "params": [{"idx": 0, "name": "SECRET_SOURCE_SENTINEL", "type": "ptr"}],
        "ops": ops,
        "proxy_aware": False,
    }


def fission_feature(site_id):
    row = feature(site_id)
    row["phase_launch_supported"] = True
    row["producer_frontier"] = {
        "analyzed": True,
        "write_footprint_known": True,
        "unknown_write_sites": 0,
        "buffer_identity_guardable": True,
        "producer_domains_known": True,
        "ordinary_store_sites": 1,
        "producer_pointer_param": 1,
        "source_buffer_index_param": 12,
        "completion_site_id": "unit.cpp:20:group_kernel::2",
        "producer_store_domains": [{
            "domain_exact": True,
            "partition_region_exact": True,
        }],
        "overlap_partition": {
            "exact": True,
            "full_compute_region_partition_proved": True,
            "store_instance_partition_complete": True,
            "store_instance_partition_disjoint": True,
            "side_effect_safety_exact": True,
            "side_effects_excluded_on_optimized_path": True,
        },
    }
    return row


class CommunicationGroupPlanBridgeTests(unittest.TestCase):
    def setUp(self):
        self.site_ids = [
            "unit.cpp:10:group_kernel::0",
            "unit.cpp:11:group_kernel::1",
        ]
        self.dossier = bridge.make_dossier(
            [feature(site_id) for site_id in self.site_ids], PLATFORM
        )
        self.graph = groups.make_group_graph(self.dossier, [template()])
        self.opportunity = self.graph["opportunities"][0]

    def candidate(self, kind):
        return next(
            candidate for candidate in self.opportunity["candidates"]
            if candidate["kind"] == kind
        )

    def decision(self, kind="group_trigger_early"):
        candidate = self.candidate(kind)
        return {
            "schema_version": groups.DECISION_SCHEMA,
            "graph_id": self.graph["graph_id"],
            "selections": {
                self.opportunity["opportunity_id"]: {
                    "candidate_id": candidate["candidate_id"],
                    "confidence": 0.9,
                    "rationale": "use the compiler-proved group frontier",
                }
            },
        }

    def test_graph_is_relational_content_addressed_and_source_free(self):
        groups.verified_graph(self.graph)
        self.assertEqual(2, self.opportunity["compiler_facts"]["group_size"])
        relations = self.opportunity["compiler_facts"]["argument_relations"]
        self.assertEqual("distinct_formals", relations["target_rank"])
        self.assertEqual("same_expression", relations["src_buf"])
        self.assertEqual("same_expression", relations["size"])
        self.assertEqual(10, len(self.opportunity["candidates"]))
        prompt = groups.render_prompt(self.graph)
        self.assertNotIn("SECRET_SOURCE_SENTINEL", prompt)
        self.assertNotIn("unit.cpp", prompt)
        self.assertNotIn("group_kernel", prompt)
        self.assertNotIn('"site_ids"', prompt)
        self.assertNotIn('"materializer"', prompt)
        self.assertIn("program-dependence legality", prompt)
        self.assertIn("transfer_semantic_facts", prompt)
        self.assertIn("producer_frontier", prompt)
        self.assertFalse(self.graph["boundary"]["source_visible"])
        self.assertEqual("candidate IDs only", self.graph["boundary"]["model_output"])

    def test_early_candidate_becomes_only_a_narrow_lto_hint(self):
        hint, accepted, errors = groups.plan_to_hint(
            self.graph, self.decision()
        )
        self.assertTrue(accepted, errors)
        for site_id in self.site_ids:
            self.assertEqual("DWQ_TRIGGER", hint["sites"][site_id]["dispatch"])
            self.assertEqual(
                "TRIGGER_GROUP_EARLY", hint["sites"][site_id]["transform"]
            )
        self.assertTrue(hint["llm_metadata"]["compiler_only_output"])

    def test_intervening_write_masks_early_candidate(self):
        graph = groups.make_group_graph(
            self.dossier, [template(early_legal=False)]
        )
        opportunity = graph["opportunities"][0]
        self.assertNotIn(
            "group_trigger_early",
            {candidate["kind"] for candidate in opportunity["candidates"]},
        )
        self.assertEqual(9, len(opportunity["candidates"]))
        self.assertIn(
            "intervening instruction may write a registered source buffer",
            opportunity["masked_candidates"][0]["reason"],
        )

    def test_profile_must_enable_materializer_even_when_ir_is_safe(self):
        platform = copy.deepcopy(PLATFORM)
        platform["compiler_transforms"] = {"group_early_trigger": False}
        dossier = bridge.make_dossier(
            [feature(site_id) for site_id in self.site_ids], platform
        )
        graph = groups.make_group_graph(dossier, [template()])
        self.assertNotIn(
            "group_trigger_early",
            {candidate["kind"] for candidate in graph["opportunities"][0]["candidates"]},
        )

    def test_two_phase_candidate_requires_profile_gate_and_becomes_lto_hint(self):
        rows = [fission_feature(site_id) for site_id in self.site_ids]
        gated = groups.make_group_graph(
            bridge.make_dossier(rows, PLATFORM), [template()]
        )
        opportunity = gated["opportunities"][0]
        self.assertNotIn(
            "group_producer_frontier_two_phase",
            {candidate["kind"] for candidate in opportunity["candidates"]},
        )
        self.assertIn(
            "group_producer_frontier_two_phase",
            {candidate["kind"] for candidate in opportunity["masked_candidates"]},
        )

        platform = copy.deepcopy(PLATFORM)
        platform["compiler_transforms"]["producer_frontier_fission"] = True
        graph = groups.make_group_graph(
            bridge.make_dossier(rows, platform), [template()]
        )
        opportunity = graph["opportunities"][0]
        candidate = next(
            candidate for candidate in opportunity["candidates"]
            if candidate["kind"] == "group_producer_frontier_two_phase"
        )
        decision = {
            "schema_version": groups.DECISION_SCHEMA,
            "graph_id": graph["graph_id"],
            "selections": {
                opportunity["opportunity_id"]: {
                    "candidate_id": candidate["candidate_id"],
                    "confidence": 0.9,
                    "rationale": "select the compiler-proved two-phase schedule",
                }
            },
        }
        hint, accepted, errors = groups.plan_to_hint(graph, decision)
        self.assertTrue(accepted, errors)
        for site_id in self.site_ids:
            self.assertEqual(
                "PRODUCER_FRONTIER_TWO_PHASE",
                hint["sites"][site_id]["transform"],
            )

    def test_invented_candidate_or_code_payload_fails_closed(self):
        decision = self.decision()
        selection = decision["selections"][self.opportunity["opportunity_id"]]
        selection["candidate_id"] = "candidate:" + "0" * 24
        selection["llvm_ir"] = "call void @invented()"
        hint, accepted, errors = groups.plan_to_hint(self.graph, decision)
        self.assertFalse(accepted)
        self.assertTrue(any("unknown field" in error for error in errors))
        self.assertTrue(any("not compiler-generated" in error for error in errors))
        self.assertFalse(hint["llm_metadata"]["accepted"])
        for site in hint["sites"].values():
            self.assertNotIn("transform", site)

    def test_tampered_legality_or_candidate_is_rejected(self):
        graph = copy.deepcopy(self.graph)
        graph["opportunities"][0]["compiler_facts"]["dependence_legality"][
            "group_early_trigger_legal"
        ] = False
        with self.assertRaisesRegex(groups.GroupPlanError, "graph_id"):
            groups.verified_graph(graph)

    def test_graph_and_candidate_ids_are_deterministic(self):
        self.assertEqual(
            self.graph,
            groups.make_group_graph(self.dossier, [template()]),
        )

    def test_single_site_uses_same_opaque_candidate_protocol(self):
        dossier = bridge.make_dossier(
            [feature(self.site_ids[0])], PLATFORM
        )
        graph = groups.make_group_graph(dossier, [template()])
        opportunity = graph["opportunities"][0]
        self.assertEqual("single_site_communication_route",
                         opportunity["kind"])
        self.assertEqual(1, len(opportunity["site_ids"]))
        self.assertEqual(
            {"site_default", "site_proxy", "site_trigger"},
            {candidate["kind"] for candidate in opportunity["candidates"]},
        )
        trigger = next(
            candidate for candidate in opportunity["candidates"]
            if candidate["kind"] == "site_trigger"
        )
        decision = {
            "schema_version": groups.DECISION_SCHEMA,
            "graph_id": graph["graph_id"],
            "selections": {
                opportunity["opportunity_id"]: {
                    "candidate_id": trigger["candidate_id"],
                    "confidence": 0.8,
                    "rationale": "measured trigger cost is lower",
                }
            },
        }
        hint, accepted, errors = groups.plan_to_hint(graph, decision)
        self.assertTrue(accepted, errors)
        self.assertEqual(
            "DWQ_TRIGGER", hint["sites"][self.site_ids[0]]["dispatch"]
        )
        prompt = groups.render_prompt(graph)
        self.assertNotIn(self.site_ids[0], prompt)
        self.assertNotIn("group_kernel", prompt)

    def test_real_meta_directory_ignores_sibling_features_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            meta_dir = Path(temporary)
            (meta_dir / "features.json").write_text(
                json.dumps([feature(self.site_ids[0])]), encoding="utf-8"
            )
            (meta_dir / "kernel.json").write_text(
                json.dumps(template()), encoding="utf-8"
            )
            self.assertEqual([template()], groups._load_templates(meta_dir))

    def test_model_view_strips_nested_ids_and_rejects_identity_strings(self):
        rows = [feature(site_id) for site_id in self.site_ids]
        rows[0]["producer_frontier"] = {
            "completion_site_id": "unit.cpp:20:group_kernel::2",
            "source_identity_guardable": True,
        }
        dossier = bridge.make_dossier(rows, PLATFORM)
        graph = groups.make_group_graph(dossier, [template()])
        view = groups.model_view(graph)
        frontier = view["opportunities"][0]["compiler_facts"][
            "transfer_semantic_facts"
        ][0]["producer_frontier"]
        self.assertNotIn("completion_site_id", frontier)

        platform = copy.deepcopy(PLATFORM)
        platform["leaked_identity"] = self.site_ids[0]
        leaked_graph = groups.make_group_graph(
            bridge.make_dossier(rows, platform), [template()]
        )
        with self.assertRaisesRegex(
                groups.GroupPlanError, "identity escaped"):
            groups.model_view(leaked_graph)


if __name__ == "__main__":
    unittest.main()
