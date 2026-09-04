#!/usr/bin/env python3
"""Freeze a guarded-early-trigger confirmation after a passed scout."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PASS_PYTHON = ROOT / "tools/gicc-passes/python"
sys.path.insert(0, str(PASS_PYTHON))

import gicc_llm_bridge as bridge  # noqa: E402


TRANSITION_SCHEMA = "gicc-guarded-early-trigger-confirmation-transition-v1"
MONITOR_SCHEMA = "gicc-guarded-early-trigger-monitor-v1"
ANALYSIS_SCHEMA = "gicc-guarded-early-trigger-analysis-v1"
SIZES = (4096, 8192)
REPLICATES = (1, 2, 3, 4)
KERNEL = "_Z18matmul_step_kernelPN4gicc9DeviceCtxEPKfS3_Pfiiiiiim"
SITE = f"?:?:{KERNEL}::0"
PROTOCOL = HERE / "GUARDED_EARLY_CONFIRMATION_TRANSITION.md"


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


def geomean(values: list[float]) -> float:
    if not values or any(
            value <= 0 or not math.isfinite(value) for value in values):
        raise TransitionError("speedups must be finite and positive")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _runtime_arm(pair: dict[str, Any], arm: str) -> tuple[float, dict[str, str]]:
    value = pair.get(arm)
    if not isinstance(value, dict):
        raise TransitionError(f"guarded-trigger pair lacks {arm}")
    runtime = value.get("measured_mean_us")
    checksums = value.get("checksums")
    launches = value.get("successful_kernel_launches")
    expected_launches = 161 if arm == "baseline" else 483
    if (isinstance(runtime, bool) or not isinstance(runtime, (int, float))
            or float(runtime) <= 0 or not math.isfinite(float(runtime))
            or not isinstance(checksums, dict)
            or set(checksums) != {str(rank) for rank in range(16)}
            or not isinstance(launches, dict)
            or set(launches) != {str(rank) for rank in range(16)}
            or set(launches.values()) != {expected_launches}):
        raise TransitionError(f"guarded-trigger {arm} runtime attestation changed")
    if any(not isinstance(value, str) or len(value) != 16
           for value in checksums.values()):
        raise TransitionError(f"guarded-trigger {arm} checksums are invalid")
    return float(runtime), checksums


def recompute_scout(monitor: Any) -> dict[str, Any]:
    if (not isinstance(monitor, dict)
            or monitor.get("schema_version") != MONITOR_SCHEMA
            or monitor.get("state") != "passed"):
        raise TransitionError("guarded-early-trigger monitor did not pass")
    required = {
        "queue": "pdebug", "nodes": 2, "ranks": 16, "ppn": 8,
        "replicates": 4, "sizes": list(SIZES),
        "application_runs": 10, "application_warmup": 2,
    }
    if monitor.get("expected") != required:
        raise TransitionError("guarded-trigger monitor contract changed")
    runs = monitor.get("runs")
    if not isinstance(runs, dict) or set(runs) != {
            str(replicate) for replicate in REPLICATES}:
        raise TransitionError("guarded-trigger monitor lacks four blocks")
    sizes = {}
    promising = []
    all_speedups = []
    for size in SIZES:
        speedups = []
        for replicate in REPLICATES:
            pair = runs[str(replicate)].get(str(size))
            if not isinstance(pair, dict):
                raise TransitionError("guarded-trigger size coverage changed")
            baseline, baseline_checksums = _runtime_arm(pair, "baseline")
            guarded, guarded_checksums = _runtime_arm(pair, "guarded")
            if baseline_checksums != guarded_checksums:
                raise TransitionError("guarded-trigger correctness pair changed")
            speedup = baseline / guarded
            recorded = pair.get("speedup")
            if (isinstance(recorded, bool)
                    or not isinstance(recorded, (int, float))
                    or not math.isclose(
                        float(recorded), speedup, rel_tol=1e-12, abs_tol=1e-12,
                    )):
                raise TransitionError("guarded-trigger speedup does not regenerate")
            speedups.append(speedup)
        all_speedups.extend(speedups)
        median = statistics.median(speedups)
        wins = sum(value > 1.0 for value in speedups)
        passed = wins >= 3 and median >= 1.02
        if passed:
            promising.append(size)
        sizes[str(size)] = {
            "paired_speedups": speedups,
            "median_speedup": median,
            "min_speedup": min(speedups),
            "max_speedup": max(speedups),
            "wins": wins,
            "gate_passed": passed,
        }
    return {
        "correctness_gate": {
            "passed": True,
            "paired_checks": 8,
            "checksums_per_arm_per_pair": 16,
        },
        "sizes": sizes,
        "all_pairs_geometric_mean_speedup": geomean(all_speedups),
        "oracle_headroom_gate": {
            "passed": bool(promising),
            "promising_sizes": promising,
            "criterion": (
                "at least one size has >=3/4 wins and median speedup >=1.02"
            ),
            "interpretation": (
                "eligible for an independently frozen compiler-oracle confirmation"
                if promising else
                "keep guarded early trigger hidden from model evaluation"
            ),
            "paper_claim": False,
        },
    }


def validate_runtime_contract(monitor: dict[str, Any]) -> None:
    resources = [{
        "type": "node", "count": 2,
        "with": [{
            "type": "slot", "count": 8, "label": "task",
            "with": [
                {"type": "core", "count": 8},
                {"type": "gpu", "count": 1},
            ],
        }],
    }]
    scheduler = monitor.get("scheduler")
    jobspec = monitor.get("jobspec")
    nodelist = monitor.get("resource_set", {}).get("nodelist")
    if (not isinstance(monitor.get("job_id"), str) or not monitor["job_id"]
            or not isinstance(scheduler, dict)
            or scheduler.get("exit_code") != 0
            or scheduler.get("exception_types") != []
            or not isinstance(jobspec, dict)
            or jobspec.get("queue") != "pdebug"
            or jobspec.get("duration_seconds") != 1800.0
            or jobspec.get("resources") != resources
            or not isinstance(nodelist, list) or len(nodelist) != 2
            or len(set(nodelist)) != 2):
        raise TransitionError("guarded-trigger Flux contract changed")


def validate_analysis(value: Any, monitor: Any, monitor_path: Path) -> Any:
    if (not isinstance(value, dict)
            or value.get("schema_version") != ANALYSIS_SCHEMA):
        raise TransitionError("unexpected guarded-trigger analysis schema")
    regenerated = recompute_scout(monitor)
    for field, expected in regenerated.items():
        if value.get(field) != expected:
            raise TransitionError(
                f"guarded-trigger analysis field does not regenerate: {field}"
            )
    if Path(value.get("monitor", "")).resolve() != monitor_path.resolve():
        raise TransitionError("guarded-trigger analysis binds another monitor")
    if value["oracle_headroom_gate"]["passed"] is not True:
        raise TransitionError("guarded-trigger scout gate did not pass")
    return value


def validate_artifacts(monitor: dict[str, Any]) -> dict[Path, str]:
    artifacts = monitor.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise TransitionError("guarded-trigger monitor lacks frozen artifacts")
    result: dict[Path, str] = {}
    for record in artifacts:
        if (not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)):
            raise TransitionError("guarded-trigger artifact record is invalid")
        path = Path(record["path"]).resolve()
        if path in result or not path.is_file():
            raise TransitionError(f"guarded-trigger artifact is missing: {path}")
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise TransitionError(f"guarded-trigger artifact changed: {path}")
        result[path] = actual
    return result


def compiler_candidate(binary_dir: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    baseline_meta = list((binary_dir / "baseline/meta").glob("*.json"))
    guarded_meta = list((binary_dir / "guarded/meta").glob("*.json"))
    baseline_kernel = [path for path in baseline_meta if path.name != "features.json"]
    guarded_kernel = [path for path in guarded_meta if path.name != "features.json"]
    if len(baseline_kernel) != 1 or len(guarded_kernel) != 1:
        raise TransitionError("guarded-trigger build lacks exact kernel metadata")
    baseline_value = read_json(baseline_kernel[0])
    guarded_value = read_json(guarded_kernel[0])
    if "guarded_early_trigger_device_materialized" in baseline_value:
        raise TransitionError("baseline metadata carries guarded attestation")
    if guarded_value.get("guarded_early_trigger_device_materialized") is not True:
        raise TransitionError("guarded metadata lacks device attestation")
    normalized = dict(guarded_value)
    normalized.pop("guarded_early_trigger_device_materialized")
    if normalized != baseline_value:
        raise TransitionError("arm metadata differs beyond device attestation")
    if guarded_value.get("kernel_mangled") != KERNEL:
        raise TransitionError("guarded-trigger kernel identity changed")
    puts = [op for op in guarded_value.get("ops", []) if op.get("kind") == "put_no_db"]
    if len(puts) != 1 or puts[0].get("site_id") != SITE:
        raise TransitionError("guarded-trigger transfer identity changed")
    frontier = puts[0].get("producer_frontier", {})
    facts = {
        "source_pointer_candidates": frontier.get("source_pointer_candidates"),
        "source_buffer_index_param": frontier.get(
            "source_identity_buffer_index_param"
        ),
        "write_pointer_params": frontier.get("guarded_early_trigger_write_params"),
        "unsafe_side_effect_sites": frontier.get(
            "guarded_early_trigger_unsafe_side_effect_sites"
        ),
        "guardable": frontier.get("guarded_early_trigger_guardable"),
    }
    if facts != {
        "source_pointer_candidates": [1, 2],
        "source_buffer_index_param": 9,
        "write_pointer_params": [3],
        "unsafe_side_effect_sites": 0,
        "guardable": True,
    }:
        raise TransitionError("guarded-trigger compiler proof changed")
    features_baseline = binary_dir / "baseline/meta/features.json"
    features_guarded = binary_dir / "guarded/meta/features.json"
    if (not features_baseline.is_file() or not features_guarded.is_file()
            or features_baseline.read_bytes() != features_guarded.read_bytes()):
        raise TransitionError("guarded-trigger feature facts differ between arms")
    ir_audit = binary_dir / "guarded/ir-audit/audit.json"
    audit_value = read_json(ir_audit)
    if (audit_value.get("schema_version")
            != "gicc-guarded-early-trigger-ir-audit-v1"
            or audit_value.get("passed") is not True):
        raise TransitionError("guarded-trigger final IR audit did not pass")
    candidate_payload = {
        "kind": "guarded_early_trigger",
        "compiler_materializer": "allocation_guarded_phase3_trigger",
        "kernel_mangled": KERNEL,
        "transfer_site_id": SITE,
        "schedule_phase": 3,
        "guard_facts": facts,
    }
    candidate = {
        **candidate_payload,
        "candidate_id": bridge._fingerprint({
            "schema_version": "gicc-compiler-schedule-candidate-v1",
            **candidate_payload,
        }),
        "model_visible": False,
    }
    paths = {
        "baseline_kernel_metadata": baseline_kernel[0],
        "guarded_kernel_metadata": guarded_kernel[0],
        "frozen_ir_audit": ir_audit,
    }
    return candidate, paths


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


def build_transition(monitor_path: Path, analysis_path: Path,
                     binary_dir: Path) -> dict[str, Any]:
    monitor = read_json(monitor_path)
    validate_runtime_contract(monitor)
    analysis = validate_analysis(read_json(analysis_path), monitor, monitor_path)
    artifacts = validate_artifacts(monitor)
    candidate, compiler_paths = compiler_candidate(binary_dir)
    baseline = binary_dir / "baseline/mm_minimal"
    guarded = binary_dir / "guarded/mm_minimal"
    provenance = binary_dir / "BUILD_PROVENANCE.txt"
    source = ROOT / "examples/ofi/mm_minimal.cpp"
    baseline_hint = HERE / "hint_baseline_dwq.json"
    guarded_hint = HERE / "hint_guarded_early_dwq.json"
    required = [
        baseline, guarded, provenance, source, baseline_hint, guarded_hint,
        *compiler_paths.values(),
    ]
    for path in required:
        if artifacts.get(path.resolve()) != sha256_file(path):
            raise TransitionError(f"scout monitor did not bind artifact: {path}")
    if sha256_file(baseline) == sha256_file(guarded):
        raise TransitionError("guarded-trigger executables are identical")
    payload = {
        "schema_version": TRANSITION_SCHEMA,
        "status": "confirmation_plan_ready",
        "scout_job_id": monitor["job_id"],
        "scout_monitor_sha256": sha256_file(monitor_path),
        "scout_analysis_sha256": sha256_file(analysis_path),
        "dormant_compiler_candidate": candidate,
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_modified": False,
            "model_invoked": False,
            "provider_call_authorized": False,
            "scheduler_job_submitted": False,
            "candidate_model_visible_before_confirmation": False,
        },
        "selection": {
            "scout_promising_sizes": analysis["oracle_headroom_gate"][
                "promising_sizes"
            ],
            "confirmation_sizes": list(SIZES),
            "posthoc_size_pruning_allowed": False,
        },
        "confirmation_contract": {
            "queue": "pdebug", "nodes": 2, "ranks": 16,
            "ranks_per_node": 8, "cpu_cores_per_rank": 8,
            "gpus_per_rank": 1, "independent_allocations": 3,
            "maximum_active_or_queued_jobs": 1,
            "balanced_blocks_per_allocation": ["AB", "BA"],
            "sizes": list(SIZES), "application_runs": 10,
            "application_warmup": 2,
            "baseline_launches_per_rank": 161,
            "guarded_launches_per_rank": 483,
            "primary_metric": (
                "geometric mean of paired baseline/guarded speedups across "
                "both sizes and both order-balanced blocks, clustered by "
                "independent allocation"
            ),
            "pass_rule": (
                "point estimate >=1.02, exact allocation-cluster paired-"
                "bootstrap 95% lower bound >1.0, at least 2/3 allocation "
                "geometric means >1.0, and all checksum/launch audits pass"
            ),
            "scout_or_confirmation_labels_visible_to_model": False,
        },
        "post_confirmation": {
            "on_failure": "keep guarded early trigger model-invisible",
            "on_pass": (
                "permit a new content-addressed compiler graph to expose "
                "the existing candidate ID"
            ),
            "single_positive_case_supports_llm_selection_claim": False,
            "provider_call_authorized": False,
        },
        "files": [
            file_record(monitor_path, "passed_scout_monitor"),
            file_record(analysis_path, "passed_scout_analysis"),
            file_record(source, "unchanged_application_source"),
            file_record(baseline, "frozen_baseline_binary"),
            file_record(guarded, "frozen_guarded_binary"),
            file_record(provenance, "frozen_build_provenance"),
            file_record(baseline_hint, "baseline_lto_hint"),
            file_record(guarded_hint, "guarded_lto_hint"),
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
        raise TransitionError("unexpected guarded-trigger transition schema")
    payload = dict(value)
    transition_id = payload.pop("transition_id", None)
    if transition_id != bridge._fingerprint(payload):
        raise TransitionError("guarded-trigger transition ID changed")
    records = value.get("files")
    if not isinstance(records, list):
        raise TransitionError("guarded-trigger transition lacks file records")
    paths: dict[str, Path] = {}
    for record in records:
        if (not isinstance(record, dict)
                or not isinstance(record.get("role"), str)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)):
            raise TransitionError("guarded-trigger transition file is invalid")
        role = record["role"]
        raw = Path(record["path"])
        path = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
        if (role in paths or not path.is_file()
                or sha256_file(path) != record["sha256"]
                or path.stat().st_size != record["bytes"]):
            raise TransitionError(f"guarded-trigger transition file changed: {path}")
        paths[role] = path
    expected_roles = {
        "passed_scout_monitor", "passed_scout_analysis",
        "unchanged_application_source", "frozen_baseline_binary",
        "frozen_guarded_binary", "frozen_build_provenance",
        "baseline_lto_hint", "guarded_lto_hint", "baseline_kernel_metadata",
        "guarded_kernel_metadata", "frozen_ir_audit",
        "preregistered_transition_protocol", "transition_preparer",
    }
    if set(paths) != expected_roles:
        raise TransitionError("guarded-trigger transition file roles changed")
    binary_dir = paths["frozen_baseline_binary"].parent.parent
    if paths["frozen_guarded_binary"].parent.parent != binary_dir:
        raise TransitionError("guarded-trigger binaries use different roots")
    regenerated = build_transition(
        paths["passed_scout_monitor"], paths["passed_scout_analysis"],
        binary_dir,
    )
    if regenerated != value:
        raise TransitionError("guarded-trigger transition does not regenerate")
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
    prepare_parser = subparsers.add_parser("prepare")
    add_inputs(prepare_parser)
    prepare_parser.add_argument("--out", type=Path, required=True)
    verify_parser = subparsers.add_parser("verify")
    add_inputs(verify_parser)
    verify_parser.add_argument("--report", type=Path, required=True)
    contained_parser = subparsers.add_parser("verify-contained")
    contained_parser.add_argument("--report", type=Path, required=True)
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
        elif args.command == "verify":
            if read_json(args.report) != transition:
                raise TransitionError("guarded-trigger transition changed")
            action = "verified"
        print(
            f"guarded-early-confirmation: {action}; model_invoked=false; "
            f"scheduler_job_submitted=false; transition_id="
            f"{transition['transition_id']}"
        )
        return 0
    except (TransitionError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"guarded-early-confirmation: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
