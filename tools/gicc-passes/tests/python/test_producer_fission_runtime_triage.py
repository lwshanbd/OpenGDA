import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[4]
EXPERIMENT = ROOT / "tools" / "gicc-passes" / "experiments" / "producer_fission"
sys.path.insert(0, str(EXPERIMENT))

import audit_producer_fission_runtime_triage as triage


def jacobi_output(norm: float, seconds: float = 0.01) -> str:
    return (
        "GICC/OFI Jacobi (unified launch): 16 ranks, mesh 1010 x 1024, "
        "chunk 63, buf 266240 bytes, 200 iters\n"
        f"Done: 200 iters in {seconds:.4f} s, final l2={norm:.9e}\n"
    )


class ProducerFissionRuntimeTriageTests(unittest.TestCase):
    def test_pair_preserves_frozen_gate_and_rejects_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            baseline = directory / "baseline.out"
            fission = directory / "fission.out"
            baseline.write_text(jacobi_output(2.109035e-2), encoding="utf-8")
            fission.write_text(jacobi_output(2.098591e-2), encoding="utf-8")

            result = triage.compare_pair(baseline, fission)

        gate = result["correctness_gate"]
        self.assertFalse(gate["passed"])
        self.assertEqual(1e-6, gate["relative_tolerance"])
        self.assertEqual(1e-7, gate["absolute_tolerance"])
        self.assertGreater(gate["excess_over_allowed"], 1000)

    def test_completed_job_waits_before_reading_scheduler_records(self):
        jobspec = {
            "resources": [{
                "type": "node", "count": 2,
                "with": [{
                    "type": "slot", "count": 8,
                    "with": [
                        {"type": "core", "count": 8},
                        {"type": "gpu", "count": 1},
                    ],
                }],
            }],
            "tasks": [{"command": ["runner"]}],
            "attributes": {"system": {
                "queue": "pdebug",
                "job": {"name": "producer-fission-triage"},
            }},
        }
        resources = {
            "version": 1,
            "execution": {
                "nodelist": ["node[1-2]"],
                "properties": {"pdebug": "0-1"},
                "R_lite": [{"rank": "0-1"}],
            },
        }
        eventlog = "\n".join((
            '{"name":"finish","context":{"status":0}}',
            '{"name":"clean"}',
        ))
        responses = ["", eventlog, __import__("json").dumps(jobspec),
                     __import__("json").dumps(resources)]
        with mock.patch.object(triage.common, "run_flux",
                               side_effect=responses) as run_flux:
            result = triage.completed_job(
                "jobid", "producer-fission-triage")
        self.assertEqual(0, result["scheduler"]["exit_code"])
        self.assertEqual(("job", "wait-event", "jobid", "clean"),
                         run_flux.call_args_list[0].args)


if __name__ == "__main__":
    unittest.main()
