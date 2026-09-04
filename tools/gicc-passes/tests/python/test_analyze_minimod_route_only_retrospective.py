import sys
import unittest
from pathlib import Path


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = PASS_ROOT / "experiments"
sys.path.insert(0, str(PASS_ROOT / "python"))
sys.path.insert(0, str(EXPERIMENTS))

import analyze_minimod_route_only_retrospective as retrospective


def samples(schedule="overlap"):
    cells = ((1, 8), (2, 1), (2, 8), (4, 1), (4, 8))
    result = []
    for nodes, rpn in cells:
        for replicate in range(1, 6):
            for route in retrospective.ROUTES:
                cost = 1.0
                if (nodes, rpn) == (2, 1):
                    cost = {"default": 1.0, "trigger": 1.2, "proxy": 0.8}[route]
                elif route != "default":
                    cost = 1.2
                result.append(retrospective.minimod.Sample(
                    kind="standard",
                    arm=f"{route}_{schedule}",
                    nodes=nodes,
                    rpn=rpn,
                    ranks=nodes * rpn,
                    rep=replicate,
                    grid=400,
                    steps=100,
                    kernel_s=cost,
                    comm_s=cost / 10,
                    comp_s=cost * 0.9,
                    checksum_set="unit",
                    staged=0,
                    pushed=0,
                    path="unit",
                ))
    return result


class MinimodRouteOnlyRetrospectiveTests(unittest.TestCase):
    def test_route_oracle_is_conditioned_on_one_fixed_schedule(self):
        candidate_ids = {
            route: f"candidate:{route}" for route in retrospective.ROUTES
        }
        result = retrospective.analyze_schedule(
            samples(), "overlap", candidate_ids,
        )
        self.assertEqual("default", result["best_fixed_route"])
        self.assertEqual(
            "proxy", result["cells"]["n2-rpn1"]["winner"]
        )
        self.assertGreater(result["best_fixed_over_pointwise_oracle"], 1.0)
        self.assertIn(
            "retrospectively selected",
            result["conditional_paired_bootstrap"]["interpretation"],
        )

    def test_bootstrap_is_deterministic_and_descriptive(self):
        left = retrospective.paired_bootstrap([1.0, 1.1, 1.2], draws=100)
        right = retrospective.paired_bootstrap([1.0, 1.1, 1.2], draws=100)
        self.assertEqual(left, right)
        self.assertEqual(100, left["bootstrap_draws"])

    def test_feature_signature_exposes_schema_drift(self):
        version, signature = retrospective.feature_signature([{
            "schema_version": 4,
            "site_id": "site",
            "op_kind": "put_no_db",
            "legal_paths": ["proxy", "trigger", "ipc"],
        }])
        self.assertEqual(4, version)
        self.assertEqual("site", signature[0]["site_id"])

    def test_incomplete_route_matrix_is_rejected(self):
        values = samples()
        values.pop()
        with self.assertRaises(retrospective.RetrospectiveError):
            retrospective.analyze_schedule(
                values, "overlap",
                {
                    route: f"candidate:{route}"
                    for route in retrospective.ROUTES
                },
            )


if __name__ == "__main__":
    unittest.main()
