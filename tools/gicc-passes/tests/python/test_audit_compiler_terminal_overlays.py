import copy
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = PASS_ROOT / "experiments"
sys.path.insert(0, str(EXPERIMENTS))


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, EXPERIMENTS / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


readiness = load(
    "readiness_terminal_test_module",
    "audit_compiler_llm_readiness_terminal.py",
)
authority = load(
    "authority_terminal_test_module",
    "audit_compiler_action_authority_terminal.py",
)


def terminal_report():
    candidates = {}
    for label, reason in {
        "jacobi": "compiler_correctness_gate_failed",
        "loop_lto": "preregistered_oracle_headroom_gate_failed",
        "mm_minimal": "positive_gate_mathematically_unreachable",
    }.items():
        candidates[label] = {
            "status": "closed_negative",
            "reason_code": reason,
            "model_visible": False,
            "confirmation_eligible": False,
            "paper_performance_claim": False,
            "runtime_values_are_positive_performance_evidence": False,
            "terminal_report_sha256": label * 4,
        }
    return {
        "terminal_negatives_id": "sha256:" + "a" * 64,
        "candidates": candidates,
    }


class CompilerTerminalOverlayTests(unittest.TestCase):
    def test_readiness_overlay_can_only_reduce_authority(self):
        base_report = {
            "readiness_id": "sha256:" + "b" * 64,
            "entries": {
                "jacobi": {
                    "status": "scout_failed",
                    "provider_protocol_permitted": False,
                    "provider_call_authorized": False,
                    "llm_performance_measured": False,
                },
                "loop_lto": {
                    "status": "closed_negative",
                    "provider_protocol_permitted": False,
                    "provider_call_authorized": False,
                    "llm_performance_measured": False,
                },
                "mm_minimal": {
                    "status": "scout_failed",
                    "provider_protocol_permitted": False,
                    "provider_call_authorized": False,
                    "llm_performance_measured": False,
                },
                "collective_n8": {
                    "status": "scout_failed",
                    "provider_protocol_permitted": False,
                },
            },
            "summary": {
                "provider_protocol_permitted_entries": [],
                "provider_protocol_permitted_count": 0,
            },
            "evidence": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terminal.json"
            path.write_text("{}\n", encoding="utf-8")
            result = readiness.apply_terminal_negatives(
                base_report, terminal_report(), path,
            )
        for label in readiness.TERMINAL_LABELS:
            self.assertEqual("closed_negative", result["entries"][label]["status"])
            self.assertFalse(
                result["entries"][label]["provider_protocol_permitted"]
            )
            self.assertFalse(result["entries"][label]["candidate_model_visible"])
        self.assertEqual(3, result["summary"]["terminal_negative_count"])
        self.assertEqual([], result["summary"]["provider_protocol_permitted_entries"])
        payload = dict(result)
        observed = payload.pop("readiness_id")
        self.assertEqual(readiness.base.bridge._fingerprint(payload), observed)

        invalid = copy.deepcopy(base_report)
        invalid["entries"]["jacobi"]["provider_protocol_permitted"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "terminal.json"
            path.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(
                readiness.base.ReadinessError, "reduce existing authority"
            ):
                readiness.apply_terminal_negatives(
                    invalid, terminal_report(), path,
                )

    def test_authority_overlay_closes_instead_of_realizing_frontier(self):
        base_report = {
            "authority_id": "sha256:" + "c" * 64,
            "conditional_compiler_policy_authority": {
                "unresolved_conditional_entries": [
                    "jacobi", "loop_lto", "mm_minimal",
                ],
                "realized_current_entries": [],
                "model_visible_transform_count": 0,
                "runtime_unconfirmed_transform_count": 3,
                "performance_claim_supported": False,
            },
            "claim_separation": {
                "conditional_runtime_unconfirmed_nonroute_transforms": 3,
                "llm_performance_superiority_claimed": False,
            },
            "evidence": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            terminal_path = root / "terminal.json"
            terminal_path.write_text("{}\n", encoding="utf-8")
            prompt_dir = root / "prompts"
            prompt_dir.mkdir()
            result = authority.apply_terminal_negatives(
                base_report, terminal_report(), terminal_path, prompt_dir,
            )
        conditional = result["conditional_compiler_policy_authority"]
        self.assertEqual([], conditional["unresolved_conditional_entries"])
        self.assertEqual([], conditional["realized_current_entries"])
        self.assertEqual(
            ["jacobi", "loop_lto", "mm_minimal"],
            conditional["closed_terminal_entries"],
        )
        self.assertEqual(0, conditional["runtime_unconfirmed_transform_count"])
        self.assertEqual(3, conditional["terminal_negative_transform_count"])


if __name__ == "__main__":
    unittest.main()
