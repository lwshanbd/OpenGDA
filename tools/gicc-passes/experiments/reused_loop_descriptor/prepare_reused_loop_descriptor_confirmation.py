#!/usr/bin/env python3
"""Freeze a reused-loop-descriptor confirmation after a passed scout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PASS_PYTHON = ROOT / "tools/gicc-passes/python"
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(HERE))

import analyze_reused_loop_descriptor_scout as scout_analysis  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import monitor_reused_loop_descriptor_scout as scout_monitor  # noqa: E402


TRANSITION_SCHEMA = "gicc-reused-loop-descriptor-confirmation-transition-v1"
MONITOR_SCHEMA = "gicc-reused-loop-descriptor-monitor-v1"
ANALYSIS_SCHEMA = "gicc-reused-loop-descriptor-analysis-v1"
BATCHES = tuple(scout_monitor.BATCHES)
SIZES = tuple(scout_monitor.SIZES)
SMALL_SIZES = tuple(scout_analysis.SMALL_SIZES)
KERNEL = "_Z15dwq_loop_kernelPN4gicc9DeviceCtxEiimi"
SITE = f"?:?:{KERNEL}::2"
PROTOCOL = HERE / "REUSED_LOOP_DESCRIPTOR_CONFIRMATION_TRANSITION.md"


class TransitionError(RuntimeError):
    """The scout evidence cannot enter the frozen confirmation stage."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransitionError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise TransitionError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def file_record(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise TransitionError(f"missing {role}: {path}")
    return {
        "role": role,
        "path": display_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def validate_runtime_contract(monitor: Any) -> dict[str, Any]:
    if (not isinstance(monitor, dict)
            or monitor.get("schema_version") != MONITOR_SCHEMA
            or monitor.get("state") != "passed"
            or not isinstance(monitor.get("job_id"), str)
            or not monitor["job_id"]):
        raise TransitionError("reused-loop-descriptor monitor did not pass")
    expected = {
        "queue": "pdebug", "nodes": 2, "ranks": 2, "ppn": 1,
        "replicates": 6, "batches": list(BATCHES),
        "message_sizes": list(SIZES), "warmup": 10, "measured": 21,
    }
    if monitor.get("expected") != expected:
        raise TransitionError("reused-loop-descriptor scout contract changed")
    scheduler = monitor.get("scheduler")
    jobspec = monitor.get("jobspec")
    resource_set = monitor.get("resource_set")
    if (not isinstance(scheduler, dict)
            or scheduler.get("exit_code") != 0
            or scheduler.get("exception_types") != []
            or not isinstance(jobspec, dict)
            or jobspec.get("queue") != "pdebug"
            or jobspec.get("duration_seconds") != 1800.0
            or not isinstance(resource_set, dict)):
        raise TransitionError("reused-loop-descriptor Flux contract changed")
    try:
        scout_monitor.validate_allocation_shape(jobspec, resource_set)
    except scout_monitor.common.MonitorError as exc:
        raise TransitionError(str(exc)) from exc
    runs = monitor.get("runs")
    if (not isinstance(runs, dict)
            or set(runs) != {str(value) for value in scout_monitor.REPLICATES}):
        raise TransitionError("reused-loop-descriptor scout runs changed")
    return monitor


def validate_analysis(value: Any, monitor: dict[str, Any],
                      monitor_path: Path) -> dict[str, Any]:
    if (not isinstance(value, dict)
            or value.get("schema_version") != ANALYSIS_SCHEMA
            or not isinstance(value.get("created_at"), str)
            or Path(value.get("monitor", "")).resolve()
            != monitor_path.resolve()):
        raise TransitionError("unexpected reused-loop-descriptor analysis")
    regenerated = scout_analysis.analyze(monitor)
    for key in (
        "correctness_gate", "target_stratum", "batches",
        "all_cluster_small_message_geomean_speedup", "oracle_headroom_gate",
    ):
        if value.get(key) != regenerated.get(key):
            raise TransitionError(
                f"reused-loop-descriptor analysis does not regenerate: {key}"
            )
    gate = value.get("oracle_headroom_gate", {})
    if (gate.get("passed") is not True
            or gate.get("paper_claim") is not False
            or gate.get("provider_protocol_permitted") is not False):
        raise TransitionError("reused-loop-descriptor scout gate did not pass")
    return value


def validate_artifacts(monitor: dict[str, Any]) -> dict[Path, str]:
    records = monitor.get("artifacts")
    if not isinstance(records, list) or not records:
        raise TransitionError("reused-loop-descriptor monitor lacks artifacts")
    artifacts: dict[Path, str] = {}
    for record in records:
        if (not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)):
            raise TransitionError("reused-loop-descriptor artifact is invalid")
        path = Path(record["path"]).resolve()
        if (path in artifacts or not path.is_file()
                or sha256_file(path) != record["sha256"]):
            raise TransitionError(f"reused-loop-descriptor artifact changed: {path}")
        artifacts[path] = record["sha256"]
    return artifacts


def _verified_ir_audit(path: Path) -> dict[str, Any]:
    value = read_json(path)
    baseline = value.get("baseline", {})
    reused = value.get("reused", {})
    baseline_host = baseline.get("host", {})
    reused_host = reused.get("host", {})
    baseline_device = baseline.get("device", {})
    reused_device = reused.get("device", {})
    if (value.get("schema_version")
            != "gicc-reused-loop-descriptor-ir-audit-v1"
            or value.get("passed") is not True
            or value.get("host_only_device_identity") is not True
            or baseline_host.get("kernel_launch_calls") != 4
            or reused_host.get("kernel_launch_calls") != 4
            or baseline_host.get("batched_helper_calls") != 1
            or baseline_host.get("repeated_helper_calls") != 0
            or baseline_host.get("descriptor_array_name_occurrences", 0) < 6
            or reused_host.get("batched_helper_calls") != 0
            or reused_host.get("repeated_helper_calls", 0) < 1
            or reused_host.get("descriptor_array_name_occurrences") != 0
            or baseline_device.get("kernel") != KERNEL
            or reused_device.get("kernel") != KERNEL
            or baseline_device.get("kernel_sha256")
            != reused_device.get("kernel_sha256")
            or baseline_device.get("trigger_stores") != 1
            or reused_device.get("trigger_stores") != 1):
        raise TransitionError("reused-loop-descriptor final-IR audit changed")
    return value


def compiler_candidate(binary_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    baseline_meta = binary_dir / "baseline/meta"
    reused_meta = binary_dir / "reused/meta"
    baseline_kernel_paths = [
        path for path in baseline_meta.glob("*.json")
        if path.name != "features.json"
    ]
    reused_kernel_paths = [
        path for path in reused_meta.glob("*.json")
        if path.name != "features.json"
    ]
    if len(baseline_kernel_paths) != 1 or len(reused_kernel_paths) != 1:
        raise TransitionError("reused-loop-descriptor build lacks one kernel")
    baseline_kernel = baseline_kernel_paths[0]
    reused_kernel = reused_kernel_paths[0]
    if baseline_kernel.read_bytes() != reused_kernel.read_bytes():
        raise TransitionError("reused-loop-descriptor kernel facts differ")
    kernel = read_json(baseline_kernel)
    if kernel.get("kernel_mangled") != KERNEL:
        raise TransitionError("reused-loop-descriptor kernel identity changed")
    puts = [
        op for op in kernel.get("ops", [])
        if isinstance(op, dict) and op.get("kind") == "put_no_db"
    ]
    if (len(puts) != 1 or puts[0].get("site_id") != SITE
            or puts[0].get("loop") != {
                "in_loop": True, "iv_param": 4,
                "iv_start": 0, "iv_step": 1,
            }):
        raise TransitionError("reused-loop-descriptor loop metadata changed")

    baseline_features = baseline_meta / "features.json"
    reused_features = reused_meta / "features.json"
    if (not baseline_features.is_file() or not reused_features.is_file()
            or baseline_features.read_bytes() != reused_features.read_bytes()):
        raise TransitionError("reused-loop-descriptor feature facts differ")
    features = read_json(baseline_features)
    reusable = [
        row for row in features
        if isinstance(row, dict) and row.get("site_id") == SITE
    ] if isinstance(features, list) else []
    if len(reusable) != 1:
        raise TransitionError("reused-loop-descriptor feature site changed")
    row = reusable[0]
    loop = row.get("loop")
    interval = row.get("transfer_interval")
    if (row.get("op_kind") != "put_no_db"
            or row.get("hk_capable") is not True
            or row.get("descriptor_reusable") is not True
            or row.get("buffer_reusable") is not True
            or row.get("in_loop") is not True
            or row.get("guard_kind") == "field_not_null"
            or "trigger" not in row.get("legal_paths", [])
            or loop != {
                "bound_known": True, "bound_param_idx": 4,
                "bound_param_type": "i32", "iv_start": 0, "iv_step": 1,
            }
            or not isinstance(interval, dict)
            or interval.get("host_knowable") is not True
            or interval.get("symbolically_exact") is not True):
        raise TransitionError("reused-loop-descriptor compiler proof changed")
    ir_audit = binary_dir / "ir-audit/audit.json"
    _verified_ir_audit(ir_audit)
    proof = {
        "descriptor_reusable": True,
        "buffer_reusable": True,
        "host_knowable_interval": True,
        "loop": loop,
        "descriptor_arguments": puts[0].get("args"),
        "network_operation_count": {
            "kind": "runtime_loop_bound", "kernel_param_index": 4,
        },
        "network_operation_order_preserved": True,
    }
    candidate_payload = {
        "kind": "trigger_reused_descriptor_loop",
        "compiler_materializer": "REUSE_LOOP_DESCRIPTOR",
        "kernel_mangled": KERNEL,
        "transfer_site_id": SITE,
        "compiler_proof": proof,
    }
    candidate = {
        **candidate_payload,
        "candidate_id": bridge._fingerprint({
            "schema_version": "gicc-compiler-schedule-candidate-v1",
            **candidate_payload,
        }),
        "model_visible": False,
    }
    return candidate, {
        "baseline_features": baseline_features,
        "reused_features": reused_features,
        "baseline_kernel_metadata": baseline_kernel,
        "reused_kernel_metadata": reused_kernel,
        "frozen_ir_audit": ir_audit,
    }


def build_transition(monitor_path: Path, analysis_path: Path,
                     binary_dir: Path) -> dict[str, Any]:
    monitor = validate_runtime_contract(read_json(monitor_path))
    analysis = validate_analysis(
        read_json(analysis_path), monitor, monitor_path,
    )
    artifacts = validate_artifacts(monitor)
    candidate, compiler_paths = compiler_candidate(binary_dir)
    baseline = binary_dir / "baseline/bench_pingpong_lto"
    reused = binary_dir / "reused/bench_pingpong_lto"
    provenance = binary_dir / "BUILD_PROVENANCE.txt"
    source = ROOT / "examples/proxy/bench_pingpong_lto.cpp"
    baseline_hint = HERE / "hint_baseline_dwq.json"
    reused_hint = HERE / "hint_reused_dwq.json"
    required = [
        baseline, reused, provenance, source, baseline_hint, reused_hint,
        *compiler_paths.values(),
    ]
    for path in required:
        if artifacts.get(path.resolve()) != sha256_file(path):
            raise TransitionError(f"scout monitor did not bind artifact: {path}")
    if sha256_file(baseline) == sha256_file(reused):
        raise TransitionError("reused-loop-descriptor executables are identical")
    payload = {
        "schema_version": TRANSITION_SCHEMA,
        "status": "confirmation_plan_ready",
        "scout_job_id": monitor["job_id"],
        "scout_monitor_sha256": sha256_file(monitor_path),
        "scout_analysis_sha256": sha256_file(analysis_path),
        "dormant_compiler_candidate": candidate,
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_hash_verified": True,
            "application_source_visible_to_model": False,
            "application_source_modified": False,
            "model_invoked": False,
            "provider_call_authorized": False,
            "scheduler_job_submitted": False,
            "candidate_model_visible_before_confirmation": False,
        },
        "selection": {
            "scout_batches": list(BATCHES),
            "confirmation_batches": list(BATCHES),
            "scout_target_sizes": analysis["target_stratum"]["sizes"],
            "confirmation_target_sizes": list(SMALL_SIZES),
            "posthoc_batch_or_size_pruning_allowed": False,
        },
        "confirmation_contract": {
            "queue": "pdebug", "nodes": 2, "ranks": 2,
            "ranks_per_node": 1, "cpu_cores_per_rank": 64,
            "gpus_per_rank": 1, "independent_allocations": 3,
            "maximum_active_or_queued_jobs": 1,
            "balanced_blocks_per_allocation": ["AB", "BA"],
            "batches": list(BATCHES),
            "all_message_sizes": list(SIZES),
            "target_message_sizes": list(SMALL_SIZES),
            "warmup_iterations": 10, "measured_iterations": 21,
            "primary_metric": (
                "geometric mean of baseline/reuse median-time speedups over "
                "both batches, both order-balanced blocks, and ten fixed "
                "small-message sizes, clustered by independent allocation"
            ),
            "pass_rule": (
                "point estimate >=1.01, exact allocation-cluster paired-"
                "bootstrap 95% lower bound >1.0, at least 2/3 allocation "
                "geometric means >1.0, each batch median >=0.98, and all "
                "scheduler/artifact/IR/enqueue audits pass"
            ),
            "scout_or_confirmation_labels_visible_to_model": False,
        },
        "post_confirmation": {
            "on_failure": "keep reused loop descriptor model-invisible",
            "on_pass": (
                "permit a new content-addressed compiler graph to expose "
                "the compiler-defined candidate"
            ),
            "single_positive_case_supports_llm_selection_claim": False,
            "provider_call_authorized": False,
        },
        "files": [
            file_record(monitor_path, "passed_scout_monitor"),
            file_record(analysis_path, "passed_scout_analysis"),
            file_record(source, "unchanged_application_source"),
            file_record(baseline, "frozen_baseline_binary"),
            file_record(reused, "frozen_reused_binary"),
            file_record(provenance, "frozen_build_provenance"),
            file_record(baseline_hint, "baseline_lto_hint"),
            file_record(reused_hint, "reused_lto_hint"),
            *[
                file_record(path, role)
                for role, path in compiler_paths.items()
            ],
            file_record(PROTOCOL, "preregistered_transition_protocol"),
            file_record(Path(__file__), "transition_preparer"),
        ],
    }
    return {**payload, "transition_id": bridge._fingerprint(payload)}


def verify_contained_report(report_path: Path) -> dict[str, Any]:
    value = read_json(report_path)
    if (not isinstance(value, dict)
            or value.get("schema_version") != TRANSITION_SCHEMA):
        raise TransitionError("unexpected reused-loop transition schema")
    payload = dict(value)
    transition_id = payload.pop("transition_id", None)
    if transition_id != bridge._fingerprint(payload):
        raise TransitionError("reused-loop transition ID changed")
    records = value.get("files")
    if not isinstance(records, list):
        raise TransitionError("reused-loop transition lacks file records")
    paths: dict[str, Path] = {}
    for record in records:
        if (not isinstance(record, dict)
                or not isinstance(record.get("role"), str)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)):
            raise TransitionError("reused-loop transition file is invalid")
        raw = Path(record["path"])
        path = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
        role = record["role"]
        if (role in paths or not path.is_file()
                or sha256_file(path) != record["sha256"]
                or path.stat().st_size != record["bytes"]):
            raise TransitionError(f"reused-loop transition file changed: {path}")
        paths[role] = path
    expected_roles = {
        "passed_scout_monitor", "passed_scout_analysis",
        "unchanged_application_source", "frozen_baseline_binary",
        "frozen_reused_binary", "frozen_build_provenance",
        "baseline_lto_hint", "reused_lto_hint", "baseline_features",
        "reused_features", "baseline_kernel_metadata",
        "reused_kernel_metadata", "frozen_ir_audit",
        "preregistered_transition_protocol", "transition_preparer",
    }
    if set(paths) != expected_roles:
        raise TransitionError("reused-loop transition file roles changed")
    binary_dir = paths["frozen_baseline_binary"].parent.parent
    if paths["frozen_reused_binary"].parent.parent != binary_dir:
        raise TransitionError("reused-loop binaries use different roots")
    regenerated = build_transition(
        paths["passed_scout_monitor"], paths["passed_scout_analysis"],
        binary_dir,
    )
    if regenerated != value:
        raise TransitionError("reused-loop transition does not regenerate")
    return value


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--binary-dir", type=Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    add_inputs(prepare)
    prepare.add_argument("--out", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    add_inputs(verify)
    verify.add_argument("--report", type=Path, required=True)
    contained = subparsers.add_parser("verify-contained")
    contained.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "verify-contained":
            transition = verify_contained_report(args.report)
            action = "verified-contained"
        else:
            transition = build_transition(
                args.monitor, args.analysis, args.binary_dir,
            )
            if args.command == "prepare":
                if args.out.exists():
                    raise TransitionError(f"refusing to overwrite {args.out}")
                write_json_atomic(args.out, transition)
                action = "prepared"
            else:
                if read_json(args.report) != transition:
                    raise TransitionError("reused-loop transition changed")
                action = "verified"
        print(
            f"reused-loop-confirmation: {action}; model_invoked=false; "
            f"scheduler_job_submitted=false; transition_id="
            f"{transition['transition_id']}"
        )
        return 0
    except (
        TransitionError, scout_monitor.common.MonitorError,
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"reused-loop-confirmation: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
