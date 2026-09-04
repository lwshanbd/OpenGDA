import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT / "tools" / "gicc-passes" / "experiments" / "producer_fission"
sys.path.insert(0, str(EXPERIMENT))

import analyze_compiler_schedule_coverage as coverage


class CompilerScheduleCoverageTests(unittest.TestCase):
    def write_case(self, root, frontier=None, guard="always", loop=False,
                   phase=True, group_reason=""):
        site = "source.cpp:1:k::0"
        feature = {
            "schema_version": 6,
            "site_id": site,
            "kernel": "secret_kernel_name",
            "op_kind": "put_no_db",
            "guard_kind": guard,
            "in_loop": loop,
            "batch_size": 1,
            "flops_to_first_use": 8,
            "phase_launch_supported": phase,
            "phase_launch_reason": "invoke launch sites are not supported",
            "transfer_interval": {
                "symbolically_exact": True,
                "host_knowable": True,
            },
        }
        if frontier is not None:
            feature["producer_frontier"] = frontier
        features = root / "features.json"
        features.write_text(json.dumps([feature]), encoding="utf-8")
        template = {
            "version": 1,
            "ops": [{
                "site_id": site,
                "group_early_trigger_reason": group_reason,
            }],
        }
        (root / "kernel.json").write_text(json.dumps(template), encoding="utf-8")
        return features

    def test_exact_case_is_dormant_and_source_free(self):
        frontier = {
            "ordinary_store_sites": 1,
            "atomic_write_sites": 0,
            "unknown_write_sites": 0,
            "overlap_partition": {"exact": True},
            "reason": "exact compiler proof",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_case(Path(temporary), frontier=frontier)
            internal, model = coverage.load_case("jacobi", path)
            self.assertEqual("intra_kernel_exact_partition",
                             internal["facts"]["schedule_class"])
            self.assertEqual("producer_frontier_two_phase",
                             model["dormant_compiler_oracle"]["kind"])
            original = model["legal_candidates"][0]
            self.assertEqual("original_fused", original["kind"])
            self.assertTrue(original["candidate_id"].startswith("sha256:"))
            self.assertEqual([original["candidate_id"]],
                             model["model_visible_candidate_ids"])
            self.assertNotIn("secret_kernel_name", json.dumps(model))
            self.assertNotIn("source.cpp", json.dumps(model))

            tampered = json.loads(json.dumps(model))
            tampered["legal_candidates"][0]["phase_count"] = 9
            with self.assertRaisesRegex(coverage.CoverageError,
                                        "candidate ID.*does not match"):
                coverage.validate_model_case(tampered)

    def test_source_free_validation_rejects_identity_leaks(self):
        for escaped in [
            {"site_id": "opaque"},
            {"note": "/private/build/features.json"},
            {"note": "secret_kernel_name"},
            {"note": "source.cpp:1:k::0"},
            {"note": "jacobi"},
        ]:
            with self.subTest(escaped=escaped):
                with self.assertRaises(coverage.CoverageError):
                    coverage.validate_source_free_model(
                        escaped,
                        label="jacobi",
                        forbidden_fragments={
                            "secret_kernel_name",
                            "source.cpp",
                            "source.cpp:1:k::0",
                        },
                    )

    def test_main_classifies_unresolved_atomic_alias(self):
        frontier = {
            "ordinary_store_sites": 0,
            "atomic_write_sites": 1,
            "atomic_write_params": [3],
            "unknown_write_sites": 0,
            "atomic_domains_known": False,
            "buffer_identity_guardable": False,
            "write_footprint_known": False,
            "overlap_partition": {"exact": False},
            "reason": "no ordinary producer store",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            features = self.write_case(root, frontier=frontier, phase=False)
            output = root / "report.json"
            markdown = root / "report.md"
            argv = [
                "analyze_compiler_schedule_coverage.py",
                "--case", f"atomic={features}",
                "--out", str(output),
                "--markdown", str(markdown),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(0, coverage.main())
            report = json.loads(output.read_text(encoding="utf-8"))
            facts = report["cases"]["atomic"]["facts"]
            self.assertEqual("unresolved_atomic_alias_frontier",
                             facts["schedule_class"])
            self.assertIn("host_phase_materialization",
                          facts["missing_proof_families"])
            self.assertIn("registered_source_alias_disambiguation",
                          markdown.read_text(encoding="utf-8"))

    def test_exact_atomic_producer_keeps_partition_gap(self):
        frontier = {
            "ordinary_store_sites": 0,
            "atomic_write_sites": 1,
            "unknown_write_sites": 0,
            "atomic_domains_known": True,
            "buffer_identity_guardable": True,
            "write_footprint_known": True,
            "overlap_partition": {"exact": False},
            "reason": "exact atomic producer",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_case(Path(temporary), frontier=frontier)
            internal, _ = coverage.load_case("atomic", path)
            facts = internal["facts"]
            self.assertEqual("atomic_producer_no_store_remainder",
                             facts["schedule_class"])
            self.assertIn("atomic_producer_partition",
                          facts["missing_proof_families"])

    def test_source_identity_relation_retains_disjointness_obligation(self):
        frontier = {
            "ordinary_store_sites": 0,
            "atomic_write_sites": 1,
            "atomic_write_params": [3],
            "unknown_write_sites": 0,
            "atomic_domains_known": False,
            "buffer_identity_guardable": False,
            "write_footprint_known": False,
            "source_identity_guardable": True,
            "source_pointer_candidates": [2, 1],
            "source_identity_buffer_index_param": 9,
            "overlap_partition": {"exact": False},
            "reason": "runtime source identity candidate",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = self.write_case(Path(temporary), frontier=frontier)
            internal, model = coverage.load_case("synthetic_mm", path)
            facts = internal["facts"]
            self.assertEqual("guarded_source_identity_candidate",
                             facts["schedule_class"])
            self.assertEqual([{
                "source_buffer_formal": 9,
                "pointer_formals": [1, 2],
                "write_pointer_formals": [3],
                "proof": "runtime_source_identity_candidate",
                "required_proofs": [
                    "whole_write_allocation_disjointness",
                    "communication_side_effect_ordering",
                ],
            }], facts["source_identity_relations"])
            self.assertIn("write_allocation_disjointness",
                          facts["missing_proof_families"])
            self.assertIn("guarded_early_trigger_materialization",
                          facts["missing_proof_families"])
            self.assertIn("communication_side_effect_ordering",
                          facts["missing_proof_families"])
            self.assertEqual(facts, model["compiler_facts"])


if __name__ == "__main__":
    unittest.main()
