import tempfile
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PROXY = ROOT / "examples/proxy"
sys.path.insert(0, str(PROXY))

import prepare_compiler_lto_llm_calibrated as prepare


CAL_DOSSIER = (
    ROOT / "docs/experiments/compiler-lto-calibration/frozen-v1/dossier.json"
)
CAL_RESULTS = ROOT / (
    "docs/experiments/compiler-lto-calibration/runs/"
    "compiler-lto-calibration-cuid-v2/summary.json"
)
EVAL_DOSSIER = ROOT / (
    "docs/experiments/compiler-lto-eval/frozen-v1/dossier.json"
)


class PrepareCompilerLtoLlmCalibratedTests(unittest.TestCase):
    def test_pack_matches_gbt_evidence_and_adds_relations(self):
        pack, _, evaluation = prepare.calibration_pack(
            CAL_DOSSIER, CAL_RESULTS, EVAL_DOSSIER,
        )
        self.assertEqual(17, len(pack["cases"]))
        self.assertEqual([], pack["message_size_overlap_with_evaluation"])
        self.assertTrue(pack["evidence_parity"]["same_action_cost_rows"])
        self.assertTrue(pack["evidence_parity"]["same_scalar_compiler_facts"])
        for case in pack["cases"]:
            self.assertNotIn(
                "proxy_producer_concurrency", case["compiler_facts"],
            )
        relations = prepare.evaluation_relations(evaluation)
        self.assertEqual(7, len(relations))
        static = next(
            row for row in relations
            if row["kernel"] == "eval_static4_parallel"
        )
        self.assertEqual(4, len(static["site_ids"]))
        self.assertTrue(static["shared_decision_context"])
        self.assertEqual(8, static["resource_relations"]["proxy_worker_lanes"])
        self.assertEqual(
            4,
            static["resource_relations"]["max_concurrent_proxy_producers"],
        )
        self.assertNotIn(
            "proxy_producer_concurrency", static["compiler_facts"],
        )

    def test_package_regenerates_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "package"
            request = prepare.prepare(
                calibration_dossier_path=CAL_DOSSIER,
                calibration_results_path=CAL_RESULTS,
                evaluation_dossier_path=EVAL_DOSSIER,
                output_dir=output,
            )
            self.assertFalse(request["authorization"]["granted"])
            self.assertTrue(request["implementation"]["provider_runner_frozen"])
            control = request["matched_evidence_control"]
            self.assertTrue(control["calibration_scalar_facts_match_comparator"])
            self.assertIn(
                "full source-free compiler evaluation dossier",
                control["llm_additional_evidence"],
            )
            self.assertNotIn("scalar_evidence_is_identical", control)
            self.assertEqual(
                20, request["provider_delivery"]["independent_responses"],
            )
            prepare.verify(
                request_path=output / "request.json",
                calibration_dossier_path=CAL_DOSSIER,
                calibration_results_path=CAL_RESULTS,
                evaluation_dossier_path=EVAL_DOSSIER,
            )

            prompt = output / "prompt.txt"
            prompt.write_text(prompt.read_text() + "tampered\n")
            with self.assertRaisesRegex(prepare.PrepareError, "prompt changed"):
                prepare.verify(
                    request_path=output / "request.json",
                    calibration_dossier_path=CAL_DOSSIER,
                    calibration_results_path=CAL_RESULTS,
                    evaluation_dossier_path=EVAL_DOSSIER,
                )


if __name__ == "__main__":
    unittest.main()
