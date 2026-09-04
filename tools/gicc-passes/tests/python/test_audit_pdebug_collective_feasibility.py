import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock


PASS_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = (
    PASS_ROOT / "experiments" / "audit_pdebug_collective_feasibility.py"
)
SPEC = importlib.util.spec_from_file_location("pdebug_feasibility", SCRIPT)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


RESOURCE_LIST = """\
     STATE QUEUE       NNODES NCORES NGPUS NODELIST
      free pllm,pall        9    576    72 tioga[13-19,21-22]
 allocated pdebug,pall      7    448    56 tioga[34-40]
      down pdebug,pall      1     64     8 tioga41
"""
DRAIN_STATUS = """\
        TIME         STATE NNODES REASON NODELIST
 Jul21 13:20       drained      1 node falls out consistently. tioga41
"""


class PdebugCollectiveFeasibilityTests(unittest.TestCase):
    def test_snapshot_derives_seven_usable_and_exact_drain(self):
        with mock.patch.object(
            audit, "run_read_only", side_effect=[RESOURCE_LIST, DRAIN_STATUS]
        ):
            snapshot = audit.scheduler_snapshot()
        self.assertEqual(
            {"free": 0, "allocated": 7, "down": 1},
            snapshot["counts"],
        )
        self.assertEqual("tioga41", snapshot["drained"][0]["nodelist"])
        self.assertEqual(
            "node falls out consistently.",
            snapshot["drained"][0]["reason"],
        )

    def test_resource_parser_sums_free_and_allocated_pdebug_rows(self):
        text = """\
STATE QUEUE NNODES NCORES NGPUS NODELIST
free pdebug,pall 2 128 16 tioga[34-35]
allocated pdebug,pall 5 320 40 tioga[36-40]
down pdebug,pall 1 64 8 tioga41
"""
        self.assertEqual(
            {"free": 2, "allocated": 5, "down": 1},
            audit.parse_resource_list(text),
        )

    def test_missing_drain_fails_closed(self):
        with self.assertRaisesRegex(
            audit.FeasibilityError, "no drained pdebug record"
        ):
            audit.parse_drain_status("TIME STATE NNODES REASON NODELIST\n")


if __name__ == "__main__":
    unittest.main()
