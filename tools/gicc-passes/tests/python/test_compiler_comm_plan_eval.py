import copy
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PASS_ROOT = ROOT / "tools" / "gicc-passes"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(ROOT / "examples" / "proxy"))

import compiler_comm_plan_eval as controls
import gicc_comm_plan_bridge as plans
import gicc_llm_bridge as bridge

from test_gicc_comm_plan_bridge import PLATFORM, coalescable_feature


class CompilerCommunicationPlanControlTests(unittest.TestCase):
    def setUp(self):
        far = coalescable_feature("unit.cpp:20:eval_far_batch::0")
        far["loop"]["bound_const"] = 16
        far["trip_count"] = 16
        far["batch_size"] = 16
        dossier = bridge.make_dossier(
            [
                coalescable_feature("unit.cpp:10:eval_adjacent_batch::0"),
                far,
            ],
            PLATFORM,
        )
        self.graph = plans.make_opportunity_graph(dossier)

    def test_exact_controls_cover_three_by_three_product(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            manifest = controls.generate_controls(self.graph, out, "a" * 64)
            controls.verify_manifest(self.graph, manifest, out)
            self.assertEqual(9, len(manifest["arms"]))
            self.assertEqual(
                {
                    (first, second)
                    for first in controls.EXPECTED_KINDS
                    for second in controls.EXPECTED_KINDS
                },
                {
                    tuple(item["kind"] for item in arm["selections"])
                    for arm in manifest["arms"]
                },
            )
            self.assertFalse(manifest["model_invoked"])

    def test_manifest_tamper_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            manifest = controls.generate_controls(self.graph, out, "b" * 64)
            tampered = copy.deepcopy(manifest)
            tampered["arms"][0]["name"] = "invented"
            with self.assertRaisesRegex(controls.ControlError, "manifest_id"):
                controls.verify_manifest(self.graph, tampered, out)

    def test_ir_verifier_matches_every_materializer(self):
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary)
            ir_dir = out / "ir"
            ir_dir.mkdir()
            manifest = controls.generate_controls(self.graph, out, "c" * 64)
            facts = {
                item["opportunity_id"]: item["compiler_facts"]
                for item in self.graph["opportunities"]
            }
            for arm in manifest["arms"]:
                lines = [
                    "call void @gicc_runtime_dwq_enqueue_batched("
                    "ptr null, i32 32, ptr null)"
                ]
                for selection in arm["selections"]:
                    item = facts[selection["opportunity_id"]]
                    trips = item["trip_count"]
                    total = trips * item["size_bytes"]
                    if selection["kind"] == "trigger_descriptor_batch":
                        lines.append(
                            "call void @gicc_runtime_dwq_enqueue_batched("
                            f"ptr null, i32 {trips}, ptr null)"
                        )
                    elif selection["kind"] == "trigger_coalesced_loop":
                        lines.append(
                            "call void @gicc_runtime_dwq_enqueue("
                            f"ptr null, i64 {total})"
                        )
                (ir_dir / f"{arm['name']}.ll").write_text("\n".join(lines))
            controls.verify_ir(self.graph, manifest, out, ir_dir)


if __name__ == "__main__":
    unittest.main()
