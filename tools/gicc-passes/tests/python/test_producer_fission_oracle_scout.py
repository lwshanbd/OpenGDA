import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT / "tools" / "gicc-passes" / "experiments" / "producer_fission"
sys.path.insert(0, str(EXPERIMENT))

import analyze_producer_fission_oracle_scout as analyzer
import monitor_producer_fission_oracle_scout as monitor


class ProducerFissionOracleScoutTests(unittest.TestCase):
    def test_parse_run_checks_decomposed_mesh_and_result(self):
        text = (
            "GICC/OFI Jacobi (unified launch): 16 ranks, mesh 1010 x 1024, "
            "chunk 63, buf 266240 bytes, 200 iters\n"
            "Done: 200 iters in 1.2345 s, final l2=1.000000e-03\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run.out"
            path.write_text(text, encoding="utf-8")
            result = monitor.parse_run(path, 1024)
            self.assertEqual(200, result["iterations"])
            self.assertEqual(1010, result["actual_ny"])
            self.assertEqual(1.2345, result["seconds"])

            path.write_text(text.replace("mesh 1010", "mesh 1024"),
                            encoding="utf-8")
            with self.assertRaisesRegex(monitor.common.MonitorError,
                                        "unexpected Jacobi config"):
                monitor.parse_run(path, 1024)

    def test_analyzer_applies_paired_headroom_gate(self):
        monitor_value = {"state": "passed", "runs": {}}
        for replicate, speedup in enumerate((1.04, 1.06, 1.05, 0.99), 1):
            monitor_value["runs"][str(replicate)] = {
                "1024": {"speedup": speedup},
                "4096": {"speedup": 0.98},
            }
        with tempfile.TemporaryDirectory() as temporary:
            monitor_path = Path(temporary) / "monitor.json"
            output_path = Path(temporary) / "analysis.json"
            monitor_path.write_text(json.dumps(monitor_value), encoding="utf-8")
            argv = [
                "analyze_producer_fission_oracle_scout.py",
                "--monitor", str(monitor_path),
                "--out", str(output_path),
            ]
            with mock.patch.object(sys, "argv", argv):
                self.assertEqual(0, analyzer.main())
            result = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertTrue(result["correctness_gate"]["passed"])
            self.assertEqual([1024],
                             result["oracle_headroom_gate"]["promising_sizes"])
            self.assertFalse(result["oracle_headroom_gate"]["paper_claim"])


if __name__ == "__main__":
    unittest.main()
