import copy
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
PROXY = ROOT / "examples/proxy"
sys.path.insert(0, str(PROXY))

import analyze_compiler_lto_llm_calibrated as analyze
import gicc_llm_bridge as bridge
from analyze_compiler_lto_candidates import read_json, validated_response


DOSSIER_PATH = ROOT / (
    "docs/experiments/compiler-lto-eval/frozen-v1/dossier.json"
)
ORACLE_PATH = ROOT / (
    "docs/experiments/compiler-lto-eval/models/measured-oracle-v1/"
    "measured-oracle-response.json"
)
GBT_PATH = ROOT / (
    "docs/experiments/compiler-lto-eval/models/gbt-calibrated-v2/response.json"
)


def observed(dossier, response, trial):
    sites = [site["site_id"] for site in dossier["sites"]]
    return {
        "trial": trial,
        "bridge_accepted": True,
        "fallback_applied": False,
        "response": f"trial{trial:02d}/response.json",
        "response_sha256": str(trial) * 64,
        "response_sha256_canonical": str(trial) * 64,
        "actions": {
            site: response["decisions"][site]["action"] for site in sites
        },
    }


class AnalyzeCompilerLtoLlmCalibratedTests(unittest.TestCase):
    def test_policy_analysis_counts_oracle_and_gbt_matches(self):
        dossier = bridge._verified_dossier(read_json(DOSSIER_PATH))
        oracle = validated_response(dossier, ORACLE_PATH)
        gbt = validated_response(dossier, GBT_PATH)
        rows = [
            observed(dossier, gbt, 1),
            observed(dossier, gbt, 2),
            observed(dossier, oracle, 3),
            {
                "trial": 4,
                "bridge_accepted": False,
                "fallback_applied": True,
            },
        ]
        policies, distributions = analyze.policy_analysis(
            dossier, rows, oracle, gbt,
        )
        self.assertEqual(1, len(policies))
        policy = next(iter(policies.values()))
        self.assertEqual(3, policy["frequency"])
        self.assertTrue(policy["matches_calibrated_gbt"])
        self.assertEqual(6, policy["oracle_action_agreement"]["scenario_matches"])
        self.assertTrue(rows[0]["exact_scenario_oracle_match"])
        self.assertNotIn("actions", rows[3])
        for site in distributions.values():
            self.assertEqual(
                3, sum(site["accepted_response_counts"].values()),
            )

    def test_policy_analysis_does_not_mutate_response_inputs(self):
        dossier = bridge._verified_dossier(read_json(DOSSIER_PATH))
        oracle = validated_response(dossier, ORACLE_PATH)
        gbt = validated_response(dossier, GBT_PATH)
        frozen_oracle = copy.deepcopy(oracle)
        frozen_gbt = copy.deepcopy(gbt)
        analyze.policy_analysis(
            dossier, [observed(dossier, gbt, 1)], oracle, gbt,
        )
        self.assertEqual(frozen_oracle, oracle)
        self.assertEqual(frozen_gbt, gbt)


if __name__ == "__main__":
    unittest.main()
