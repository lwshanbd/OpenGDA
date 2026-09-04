import importlib.util
import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = PASS_ROOT / "experiments" / "audit_compiler_input_separation.py"
SPEC = importlib.util.spec_from_file_location("compiler_input_separation", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


class CompilerInputSeparationTests(unittest.TestCase):
    def test_extracts_embedded_graph_and_selectable_ids(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "relational.txt"
            path.write_text(
                "instructions\nCOMPILER-GENERATED COMMUNICATION GRAPH:\n"
                '{"view_kind":"relational","candidates":['
                '{"candidate_id":"candidate:a"},'
                '{"candidate_id":"candidate:b"}]}\ntrailer\n',
                encoding="utf-8",
            )
            graph = audit.prompt_graph(path)
            self.assertEqual("relational", graph["view_kind"])
            self.assertEqual(
                {"candidate:a", "candidate:b"}, audit.selectable_ids(graph)
            )

    def test_detects_relational_compiler_semantics(self):
        graph = {
            "relations": [{"kind": "operation_order"}],
            "transfer_argument_expressions": [{"kind": "param"}],
            "dependence_legality": {"legal": True},
            "transfer_interval": {"byte_offset": 0, "byte_size": 4},
            "launch_contexts": [{"grid_blocks": 2}],
            "completion_kind": "flush",
            "effects": {"network_operations": 1},
            "masked_candidates": [],
        }
        families = audit.semantic_families(graph)
        for family in (
            "operation_and_cross_opportunity_relations",
            "symbolic_argument_expressions",
            "dependence_and_legality_proofs",
            "symbolic_transfer_intervals",
            "launch_and_resource_structure",
            "completion_and_compute_distance",
            "compiler_candidate_semantics_and_effects",
            "masked_compiler_transform_reasons",
        ):
            self.assertTrue(families[family])

    def test_gbt_contract_is_exactly_seven_scalar_features(self):
        report = {
            "schema_version": audit.GBT_SCHEMA,
            "model_features": list(audit.GBT_FEATURES),
            "source_read": False,
            "oracle_or_runtime_results_read": False,
            "predictions": [{
                "compiler_facts_not_mapped_to_historical_grid": {
                    name: [] for name in audit.GBT_UNMAPPED_FACTS
                }
            }],
        }
        checked = audit.verify_gbt_report(report)
        self.assertEqual(7, checked["feature_count"])
        self.assertEqual(0, checked["explicit_relational_edges"])

        report["source_read"] = True
        with self.assertRaisesRegex(audit.SeparationError, "source_read"):
            audit.verify_gbt_report(report)

    def test_missing_llm_semantics_are_not_silently_claimed(self):
        families = audit.semantic_families({"size_bytes": 4096})
        self.assertFalse(families["operation_and_cross_opportunity_relations"])
        self.assertFalse(families["dependence_and_legality_proofs"])


if __name__ == "__main__":
    unittest.main()
