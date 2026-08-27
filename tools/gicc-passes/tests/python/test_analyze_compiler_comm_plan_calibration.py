import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "examples" / "proxy"))

import analyze_compiler_comm_plan_calibration as runtime


class CompilerCommunicationPlanCalibrationRuntimeTests(unittest.TestCase):
    def test_parse_validates_all_fixed_and_structural_routes(self):
        structural_kernels = [
            kernel for kernel, scenario in runtime.SCENARIO_FOR_KERNEL.items()
            if scenario not in runtime.FIXED_ROUTES
        ]
        arm = {
            "name": "uniform_c",
            "selections": [
                {
                    "kernel": kernel,
                    "kind": "trigger_coalesced_loop",
                    "effects": {"host_descriptors": 1, "network_operations": 1},
                }
                for kernel in structural_kernels
            ],
        }
        routes = runtime.routes_for(arm)
        runs = 2
        kernel_for_scenario = {
            scenario: kernel
            for kernel, scenario in runtime.SCENARIO_FOR_KERNEL.items()
        }
        lines = [
            "RUN rep=1 arm=uniform_c "
            "binary=/tmp/compiler_comm_plan_calibration_uniform_c "
            f"sha256={'a' * 64}"
        ]
        for scenario in runtime.SCENARIO_FOR_KERNEL.values():
            base, size = runtime.REGION[scenario]
            staged, pushed = routes[scenario]
            lines.append(
                "COMPILER_LTO_CALIBRATION "
                f"scenario={scenario} kernel={kernel_for_scenario[scenario]} "
                f"runs={runs} median_us=10.0 p25_us=9.0 p75_us=11.0 "
                f"staged={staged * runs} pushed={pushed * runs} "
                f"hash={runtime.expected_hash(base, size)} data=OK"
            )
        with tempfile.TemporaryDirectory() as temporary:
            log = Path(temporary) / "rep1-uniform_c.log"
            log.write_text("\n".join(lines) + "\n")
            parsed_arm, rep, binary_sha, rows = runtime.parse_log(
                log, ["uniform_c"], {"uniform_c": routes}
            )
            self.assertEqual("uniform_c", parsed_arm)
            self.assertEqual(1, rep)
            self.assertEqual("a" * 64, binary_sha)
            self.assertEqual(
                runs, rows["adjacent-1k-k6-g1"]["staged"]
            )
            self.assertEqual(
                48 * runs, rows["reuse-8k-k48-g4"]["staged"]
            )

            log.write_text(log.read_text().replace(
                "scenario=adjacent-1k-k6-g1 ",
                "scenario=adjacent-1k-k6-g1 ",
            ).replace("staged=2 pushed=0", "staged=4 pushed=0", 1))
            with self.assertRaisesRegex(runtime.AnalysisError, "routes="):
                runtime.parse_log(log, ["uniform_c"], {"uniform_c": routes})

    def test_proxy_structural_route_includes_completion_push(self):
        kernel = "cal_adjacent_1k_k6_g1"
        arm = {
            "name": "uniform_p",
            "selections": [
                {
                    "kernel": candidate_kernel,
                    "kind": "proxy_device",
                    "effects": {
                        "network_operations": 6 if candidate_kernel == kernel else 1,
                        "host_descriptors": 0,
                    },
                }
                for candidate_kernel, scenario in runtime.SCENARIO_FOR_KERNEL.items()
                if scenario not in runtime.FIXED_ROUTES
            ],
        }
        routes = runtime.routes_for(arm)
        self.assertEqual((0, 7), routes["adjacent-1k-k6-g1"])


if __name__ == "__main__":
    unittest.main()
