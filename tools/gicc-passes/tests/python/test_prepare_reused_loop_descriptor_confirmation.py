import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest


PASS_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = PASS_ROOT / "experiments" / "reused_loop_descriptor"
for path in (PASS_ROOT / "python", EXPERIMENT):
    sys.path.insert(0, str(path))

import prepare_reused_loop_descriptor_confirmation as transition


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


class ReusedLoopDescriptorConfirmationTransitionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.binary_dir = self.root / "binary"
        baseline = self.binary_dir / "baseline"
        reused = self.binary_dir / "reused"
        (baseline / "meta").mkdir(parents=True)
        (reused / "meta").mkdir(parents=True)
        (self.binary_dir / "ir-audit").mkdir(parents=True)
        (baseline / "bench_pingpong_lto").write_bytes(b"baseline")
        (reused / "bench_pingpong_lto").write_bytes(b"reused")
        (self.binary_dir / "BUILD_PROVENANCE.txt").write_text(
            "unit provenance\n", encoding="utf-8",
        )
        self.kernel = {
            "version": 1,
            "kernel_mangled": transition.KERNEL,
            "kernel_simple": "dwq_loop_kernel",
            "ops": [{
                "kind": "put_no_db",
                "site_id": transition.SITE,
                "loop": {
                    "in_loop": True, "iv_param": 4,
                    "iv_start": 0, "iv_step": 1,
                },
                "args": {
                    "target_rank": {"kind": "param", "param": 1},
                    "dst_buf": {"kind": "param", "param": 2},
                    "dst_off": {"kind": "const_i64", "value": 0},
                    "src_buf": {"kind": "param", "param": 2},
                    "src_off": {"kind": "const_i64", "value": 0},
                    "size": {"kind": "param", "param": 3},
                },
            }],
        }
        self.features = [{
            "site_id": transition.SITE,
            "op_kind": "put_no_db",
            "hk_capable": True,
            "descriptor_reusable": True,
            "buffer_reusable": True,
            "in_loop": True,
            "guard_kind": "unknown",
            "legal_paths": ["proxy", "trigger"],
            "loop": {
                "bound_known": True, "bound_param_idx": 4,
                "bound_param_type": "i32", "iv_start": 0, "iv_step": 1,
            },
            "transfer_interval": {
                "host_knowable": True, "symbolically_exact": True,
            },
        }]
        for directory in (baseline, reused):
            write_json(
                directory / "meta" / f"{transition.KERNEL}.json",
                self.kernel,
            )
            write_json(directory / "meta/features.json", self.features)
        self.audit = {
            "schema_version": "gicc-reused-loop-descriptor-ir-audit-v1",
            "passed": True,
            "host_only_device_identity": True,
            "baseline": {
                "host": {
                    "kernel_launch_calls": 4,
                    "batched_helper_calls": 1,
                    "repeated_helper_calls": 0,
                    "descriptor_array_name_occurrences": 6,
                },
                "device": {
                    "kernel": transition.KERNEL,
                    "kernel_sha256": "a" * 64,
                    "trigger_stores": 1,
                },
            },
            "reused": {
                "host": {
                    "kernel_launch_calls": 4,
                    "batched_helper_calls": 0,
                    "repeated_helper_calls": 2,
                    "descriptor_array_name_occurrences": 0,
                },
                "device": {
                    "kernel": transition.KERNEL,
                    "kernel_sha256": "a" * 64,
                    "trigger_stores": 1,
                },
            },
        }
        write_json(self.binary_dir / "ir-audit/audit.json", self.audit)

        runs = {}
        for replicate in transition.scout_monitor.REPLICATES:
            batches = {}
            for batch in transition.BATCHES:
                expected = 31 * batch * len(transition.SIZES)
                batches[str(batch)] = {
                    "baseline": {
                        "enqueue_actual": expected,
                        "enqueue_expected": expected,
                    },
                    "reused": {
                        "enqueue_actual": expected,
                        "enqueue_expected": expected,
                    },
                    "per_size_speedup": {
                        size: 1.03 for size in transition.SIZES
                    },
                }
            runs[str(replicate)] = batches
        self.monitor = {
            "schema_version": transition.MONITOR_SCHEMA,
            "state": "passed",
            "job_id": "unit-job",
            "expected": {
                "queue": "pdebug", "nodes": 2, "ranks": 2, "ppn": 1,
                "replicates": 6, "batches": list(transition.BATCHES),
                "message_sizes": list(transition.SIZES),
                "warmup": 10, "measured": 21,
            },
            "scheduler": {"exit_code": 0, "exception_types": []},
            "jobspec": {
                "queue": "pdebug", "duration_seconds": 1800.0,
                "resources": [{
                    "type": "node", "count": 2,
                    "with": [{
                        "type": "slot", "label": "task", "count": 1,
                        "with": [
                            {"type": "core", "count": 64},
                            {"type": "gpu", "count": 1},
                        ],
                    }],
                }],
            },
            "resource_set": {"pdebug_ranks": "34-35"},
            "runs": runs,
        }
        artifact_paths = [
            baseline / "bench_pingpong_lto",
            reused / "bench_pingpong_lto",
            self.binary_dir / "BUILD_PROVENANCE.txt",
            transition.ROOT / "examples/proxy/bench_pingpong_lto.cpp",
            transition.HERE / "hint_baseline_dwq.json",
            transition.HERE / "hint_reused_dwq.json",
            baseline / "meta/features.json",
            reused / "meta/features.json",
            baseline / "meta" / f"{transition.KERNEL}.json",
            reused / "meta" / f"{transition.KERNEL}.json",
            self.binary_dir / "ir-audit/audit.json",
        ]
        self.monitor["artifacts"] = [
            {"path": str(path), "sha256": transition.sha256_file(path)}
            for path in artifact_paths
        ]
        self.monitor_path = self.root / "monitor.json"
        write_json(self.monitor_path, self.monitor)
        self.analysis = transition.scout_analysis.analyze(self.monitor)
        self.analysis["monitor"] = str(self.monitor_path.resolve())
        self.analysis_path = self.root / "analysis.json"
        write_json(self.analysis_path, self.analysis)

    def tearDown(self):
        self.temporary.cleanup()

    def test_builds_model_invisible_content_addressed_transition(self):
        value = transition.build_transition(
            self.monitor_path, self.analysis_path, self.binary_dir,
        )
        self.assertEqual("confirmation_plan_ready", value["status"])
        self.assertFalse(value["dormant_compiler_candidate"]["model_visible"])
        self.assertEqual(
            "trigger_reused_descriptor_loop",
            value["dormant_compiler_candidate"]["kind"],
        )
        self.assertTrue(value["boundary"]["application_source_hash_verified"])
        self.assertFalse(value["boundary"]["model_invoked"])
        self.assertFalse(value["boundary"]["provider_call_authorized"])
        self.assertFalse(value["boundary"]["scheduler_job_submitted"])
        self.assertEqual([4, 64], value["selection"]["confirmation_batches"])

        report = self.root / "transition.json"
        write_json(report, value)
        self.assertEqual(value, transition.verify_contained_report(report))

    def test_negative_or_changed_scout_cannot_advance(self):
        negative = copy.deepcopy(self.analysis)
        negative["oracle_headroom_gate"]["passed"] = False
        write_json(self.analysis_path, negative)
        with self.assertRaises(transition.TransitionError):
            transition.build_transition(
                self.monitor_path, self.analysis_path, self.binary_dir,
            )

    def test_ir_mechanism_attestation_is_required(self):
        changed = copy.deepcopy(self.audit)
        changed["reused"]["host"]["repeated_helper_calls"] = 0
        write_json(self.binary_dir / "ir-audit/audit.json", changed)
        with self.assertRaisesRegex(transition.TransitionError, "IR audit"):
            transition.compiler_candidate(self.binary_dir)


if __name__ == "__main__":
    unittest.main()
