import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "examples" / "proxy"))

import analyze_compiler_comm_plan_runtime as runtime


class CompilerCommunicationPlanRuntimeTests(unittest.TestCase):
    def test_parse_validates_data_and_route_contract(self):
        arm = {
            "name": "plan_ct",
            "selections": [
                {
                    "kernel": "eval_adjacent_batch",
                    "kind": "trigger_coalesced_loop",
                },
                {
                    "kernel": "eval_far_batch",
                    "kind": "trigger_descriptor_batch",
                },
            ],
        }
        routes = runtime.routes_for(arm)
        runs = 2
        lines = []
        for scenario in runtime.SCENARIOS:
            base, size = runtime.OFFSETS[scenario]
            staged, pushed = routes[scenario]
            lines.append(
                f"COMPILER_LTO_EVAL scenario={scenario} runs={runs} "
                f"median_us=10.0 p25_us=9.0 p75_us=11.0 "
                f"staged={staged * runs} pushed={pushed * runs} "
                f"hash={runtime.expected_hash(base, size)} data=OK"
            )
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "rep1-plan_ct.log"
            log.write_text("\n".join(lines) + "\n")
            parsed_arm, rep, rows = runtime.parse_log(
                log, ["plan_ct"], {"plan_ct": arm}
            )
            self.assertEqual("plan_ct", parsed_arm)
            self.assertEqual(1, rep)
            self.assertEqual(1 * runs, rows["adjacent-k16-grid1"]["staged"])
            self.assertEqual(64 * runs, rows["far-k64-grid8"]["staged"])

            text = log.read_text().replace("staged=2 pushed=0", "staged=4 pushed=0")
            log.write_text(text)
            with self.assertRaisesRegex(ValueError, "routes="):
                runtime.parse_log(log, ["plan_ct"], {"plan_ct": arm})


if __name__ == "__main__":
    unittest.main()
