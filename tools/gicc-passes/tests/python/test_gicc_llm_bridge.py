import copy
import json
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PASS_ROOT / "python"))

import gicc_llm_bridge as bridge


def feature(site_id, *, hk=True, locality=None, batched_loop=False):
    legal = ["proxy"] + (["trigger"] if hk else [])
    if hk and not batched_loop:
        legal.append("ipc")
    return {
        "schema_version": 6,
        "site_id": site_id,
        "kernel": "kernel_from_ir",
        "op_kind": "put_no_db",
        "hk_capable": hk,
        "size_kind": "const",
        "size_bytes": 4096,
        "size_log2": 12,
        "peer_kind": "param",
        "peer_locality": locality,
        "in_loop": batched_loop,
        "loop": ({"bound_known": True, "bound_const": 64,
                  "iv_start": 0, "iv_step": 1}
                 if batched_loop else None),
        "guard_density": 1.0,
        "fan_out": 1,
        "static_launch_sites": 1,
        "launch_contexts": [
            {
                "static_callsite_count": 1,
                "launch_grid": {"x": 8, "y": 1, "z": 1},
                "launch_block": {"x": 1, "y": 1, "z": 1},
                "grid_blocks": 8,
                "threads_per_block": 1,
                "size_bytes": 4096,
                "trip_count": 64,
            }
        ],
        "launch_grid": {"x": 8, "y": 1, "z": 1},
        "launch_block": {"x": 1, "y": 1, "z": 1},
        "grid_blocks": 8,
        "threads_per_block": 1,
        "compute_before_flops": 0,
        "flops_to_first_use": 0,
        "trip_count": 64,
        "distance_exact": True,
        "iter_estimate": None,
        "descriptor_reusable": False,
        "buffer_reusable": True,
        "coalescable": False,
        "max_vector_bytes": 16,
        "batch_size": 64,
        "producer_frontier": {
            "analyzed": True,
            "write_footprint_known": True,
            "completion_site_id": "unit.cpp:12:kernel_from_ir::2",
            "ordinary_store_params": [1],
            "atomic_write_params": [3],
            "ordinary_store_sites": 1,
            "atomic_write_sites": 1,
            "unknown_write_sites": 0,
            "reason": "compiler fact only",
            "remaining_proofs": ["registered_buffer_identity"],
        },
        "legal_paths": legal,
    }


PLATFORM = {
    "schema_version": "gicc-platform-profile-v1",
    "platform_id": "test-platform",
    "measurements": {"trigger_us": 100.0, "proxy_us": 80.0},
}


class LlmBridgeTests(unittest.TestCase):
    def setUp(self):
        self.features = [
            feature("unit.cpp:10:kernel_from_ir::0"),
            feature("unit.cpp:11:kernel_from_ir::1", hk=False),
            {
                "schema_version": 6,
                "site_id": "unit.cpp:12:kernel_from_ir::2",
                "op_kind": "quiet",
            },
        ]
        self.dossier = bridge.make_dossier(self.features, PLATFORM)

    def decision(self):
        return {
            "schema_version": "gicc-llm-decision-v1",
            "dossier_id": self.dossier["dossier_id"],
            "decisions": {
                "unit.cpp:10:kernel_from_ir::0": {
                    "action": "default",
                    "confidence": 0.55,
                    "rationale": "measured margin is weak",
                },
                "unit.cpp:11:kernel_from_ir::1": {
                    "action": "proxy",
                    "confidence": 1.0,
                    "rationale": "the compiler says this is proxy-only",
                },
            },
        }

    def test_dossier_is_content_addressed_and_contains_only_decision_ops(self):
        self.assertEqual("gicc-llm-dossier-v1", self.dossier["schema_version"])
        self.assertEqual(2, len(self.dossier["sites"]))
        self.assertTrue(self.dossier["dossier_id"].startswith("sha256:"))
        bridge._verified_dossier(self.dossier)
        prompt = bridge.render_prompt(self.dossier)
        self.assertNotIn("SECRET_SOURCE_SENTINEL", prompt)
        self.assertIn(self.dossier["dossier_id"], prompt)
        self.assertEqual(8, self.dossier["sites"][0]["grid_blocks"])
        self.assertEqual(
            {"x": 8, "y": 1, "z": 1},
            self.dossier["sites"][0]["launch_grid"],
        )
        self.assertTrue(
            self.dossier["sites"][0]["producer_frontier"]
                        ["write_footprint_known"]
        )

    def test_unknown_locality_removes_forced_ipc_but_keeps_safe_default(self):
        site = self.dossier["sites"][0]
        self.assertEqual(["proxy", "trigger", "default"], site["legal_actions"])

    def test_proven_locality_exposes_forced_ipc(self):
        dossier = bridge.make_dossier(
            [feature("unit.cpp:10:kernel_from_ir::0", locality="same_node")],
            PLATFORM,
        )
        self.assertIn("ipc", dossier["sites"][0]["legal_actions"])

    def test_batched_loop_exposes_only_materializable_actions(self):
        site_id = "unit.cpp:20:kernel_from_ir::0"
        dossier = bridge.make_dossier(
            [feature(site_id, locality="same_node", batched_loop=True)],
            PLATFORM,
        )
        self.assertEqual(
            ["proxy", "trigger"], dossier["sites"][0]["legal_actions"]
        )
        bad = {
            "schema_version": bridge.DECISION_SCHEMA,
            "dossier_id": dossier["dossier_id"],
            "decisions": {
                site_id: {
                    "action": "default",
                    "confidence": 1.0,
                    "rationale": "the batched lowering cannot materialize this",
                }
            },
        }
        hint, accepted, errors = bridge.decision_to_hint(dossier, bad)
        self.assertFalse(accepted)
        self.assertTrue(any("illegal action" in error for error in errors))
        self.assertEqual(
            "DWQ_TRIGGER", hint["sites"][site_id]["dispatch"]
        )

    def test_valid_decision_becomes_hint_and_abstention_is_not_pinned(self):
        hint, accepted, errors = bridge.decision_to_hint(
            self.dossier, self.decision()
        )
        self.assertTrue(accepted)
        self.assertEqual([], errors)
        self.assertEqual("IPC_OR_DWQ", hint["default_dispatch"])
        self.assertNotIn("unit.cpp:10:kernel_from_ir::0", hint["sites"])
        self.assertEqual(
            "CPU_PROXY_ENQUEUE",
            hint["sites"]["unit.cpp:11:kernel_from_ir::1"]["dispatch"],
        )

    def test_illegal_action_falls_back_without_materializing_model_choice(self):
        decision = self.decision()
        decision["decisions"]["unit.cpp:11:kernel_from_ir::1"]["action"] = "trigger"
        hint, accepted, errors = bridge.decision_to_hint(self.dossier, decision)
        self.assertFalse(accepted)
        self.assertTrue(any("illegal action" in error for error in errors))
        self.assertFalse(hint["llm_metadata"]["accepted"])
        self.assertEqual(
            "CPU_PROXY_ENQUEUE",
            hint["sites"]["unit.cpp:11:kernel_from_ir::1"]["dispatch"],
        )
        self.assertNotIn("unit.cpp:10:kernel_from_ir::0", hint["sites"])

    def test_missing_or_unknown_sites_reject_the_entire_response(self):
        decision = self.decision()
        del decision["decisions"]["unit.cpp:10:kernel_from_ir::0"]
        decision["decisions"]["invented"] = {
            "action": "proxy",
            "confidence": 1.0,
            "rationale": "not a compiler site",
        }
        hint, accepted, errors = bridge.decision_to_hint(self.dossier, decision)
        self.assertFalse(accepted)
        self.assertTrue(any("missing decision" in error for error in errors))
        self.assertTrue(any("unknown decision" in error for error in errors))
        self.assertFalse(hint["llm_metadata"]["accepted"])

    def test_stale_dossier_binding_rejects_the_entire_response(self):
        decision = self.decision()
        decision["dossier_id"] = "sha256:" + "0" * 64
        hint, accepted, errors = bridge.decision_to_hint(self.dossier, decision)
        self.assertFalse(accepted)
        self.assertTrue(any("stale" in error for error in errors))
        self.assertFalse(hint["llm_metadata"]["accepted"])

    def test_tampered_dossier_cannot_even_create_a_fallback(self):
        dossier = copy.deepcopy(self.dossier)
        dossier["sites"][0]["size_log2"] = 20
        with self.assertRaises(bridge.BridgeError):
            bridge.decision_to_hint(dossier, self.decision())

    def test_duplicate_feature_site_is_rejected(self):
        with self.assertRaisesRegex(bridge.BridgeError, "duplicate"):
            bridge.make_dossier([self.features[0], self.features[0]], PLATFORM)


if __name__ == "__main__":
    unittest.main()
