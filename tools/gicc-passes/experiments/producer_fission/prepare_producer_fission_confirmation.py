#!/usr/bin/env python3
"""Freeze a Jacobi producer-fission confirmation after a passed scout.

This tool has no provider, compiler, scheduler, or source-edit path.  It
regenerates the scout statistics from its monitor and retains both
preregistered sizes in the confirmation, even if only one passed the
exploratory per-size gate.
"""

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
sys.path.insert(0, str(HERE))

import analyze_compiler_schedule_coverage as coverage  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


TRANSITION_SCHEMA = "gicc-producer-fission-confirmation-transition-v1"
MONITOR_SCHEMA = "gicc-producer-fission-oracle-monitor-v1"
ANALYSIS_SCHEMA = "gicc-producer-fission-oracle-analysis-v1"
COVERAGE_SCHEMA = "gicc-compiler-schedule-coverage-report-v1"
SIZES = (1024, 4096)
REPLICATES = (1, 2, 3, 4)
PROTOCOL = HERE / "PRODUCER_FISSION_CONFIRMATION_TRANSITION.md"


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


def recompute_scout(monitor: Any) -> dict[str, Any]:
    if (not isinstance(monitor, dict)
            or monitor.get("schema_version") != MONITOR_SCHEMA
            or monitor.get("state") != "passed"):
        raise TransitionError("producer-fission monitor did not pass")
    expected = monitor.get("expected")
    required = {
        "queue": "pdebug", "nodes": 2, "ranks": 16, "ppn": 8,
        "replicates": 4, "sizes": list(SIZES), "iterations": 200,
        "nccheck": 10,
    }
    if expected != required:
        raise TransitionError("producer-fission monitor contract changed")
    runs = monitor.get("runs")
    if not isinstance(runs, dict) or set(runs) != {
            str(replicate) for replicate in REPLICATES}:
        raise TransitionError("producer-fission monitor lacks four blocks")
    sizes = {}
    promising = []
    all_speedups = []
    for size in SIZES:
        speedups = []
        for replicate in REPLICATES:
            pair = runs[str(replicate)].get(str(size))
            if not isinstance(pair, dict):
                raise TransitionError("producer-fission size coverage changed")
            baseline = pair.get("baseline")
            fission = pair.get("fission")
            if not isinstance(baseline, dict) or not isinstance(fission, dict):
                raise TransitionError("producer-fission pair is incomplete")
            left = baseline.get("seconds")
            right = fission.get("seconds")
            if (isinstance(left, bool) or isinstance(right, bool)
                    or not isinstance(left, (int, float))
                    or not isinstance(right, (int, float))
                    or float(left) <= 0 or float(right) <= 0
                    or not math.isfinite(float(left))
                    or not math.isfinite(float(right))):
                raise TransitionError("producer-fission runtime is invalid")
            speedup = float(left) / float(right)
            recorded = pair.get("speedup")
            if (isinstance(recorded, bool)
                    or not isinstance(recorded, (int, float))
                    or not math.isclose(
                        float(recorded), speedup, rel_tol=1e-12, abs_tol=1e-12,
                    )):
                raise TransitionError("producer-fission speedup does not regenerate")
            if (baseline.get("iterations") != fission.get("iterations")
                    or not math.isclose(
                        float(baseline.get("final_l2")),
                        float(fission.get("final_l2")),
                        rel_tol=1e-6, abs_tol=1e-7,
                    )):
                raise TransitionError("producer-fission correctness pair changed")
            speedups.append(speedup)
        all_speedups.extend(speedups)
        median = statistics.median(speedups)
        wins = sum(value > 1.0 for value in speedups)
        passed = wins >= 3 and median >= 1.03
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
        "correctness_gate": {"passed": True, "paired_checks": 8},
        "sizes": sizes,
        "all_pairs_geometric_mean_speedup": geomean(all_speedups),
        "oracle_headroom_gate": {
            "passed": bool(promising),
            "promising_sizes": promising,
            "criterion": (
                "at least one size has >=3/4 wins and median speedup >=1.03"
            ),
            "interpretation": (
                "eligible for a larger compiler-oracle confirmation"
                if promising else
                "mask producer-frontier fission from model evaluation"
            ),
            "paper_claim": False,
        },
    }


def validate_runtime_contract(monitor: dict[str, Any]) -> None:
    scheduler = monitor.get("scheduler")
    jobspec = monitor.get("jobspec")
    resource_set = monitor.get("resource_set")
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
    if (not isinstance(monitor.get("job_id"), str)
            or not monitor["job_id"]
            or not isinstance(scheduler, dict)
            or scheduler.get("exit_code") != 0
            or scheduler.get("exception_types") != []
            or not isinstance(jobspec, dict)
            or jobspec.get("queue") != "pdebug"
            or jobspec.get("duration_seconds") != 1200.0
            or jobspec.get("resources") != resources
            or not isinstance(resource_set, dict)
            or not isinstance(resource_set.get("nodelist"), list)
            or len(resource_set["nodelist"]) != 2
            or len(set(resource_set["nodelist"])) != 2):
        raise TransitionError("producer-fission Flux contract changed")


def validate_analysis(value: Any, monitor: Any, monitor_path: Path) -> dict[str, Any]:
    if (not isinstance(value, dict)
            or value.get("schema_version") != ANALYSIS_SCHEMA):
        raise TransitionError("unexpected producer-fission analysis schema")
    regenerated = recompute_scout(monitor)
    for field, expected in regenerated.items():
        if value.get(field) != expected:
            raise TransitionError(
                f"producer-fission analysis field does not regenerate: {field}"
            )
    if Path(value.get("monitor", "")).resolve() != monitor_path.resolve():
        raise TransitionError("producer-fission analysis binds another monitor")
    if value["oracle_headroom_gate"]["passed"] is not True:
        raise TransitionError("producer-fission scout gate did not pass")
    return value


def validate_artifacts(monitor: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = monitor.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise TransitionError("producer-fission monitor lacks frozen artifacts")
    seen = set()
    records = []
    for record in artifacts:
        if (not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)):
            raise TransitionError("producer-fission artifact record is invalid")
        path = Path(record["path"]).resolve()
        if path in seen or not path.is_file():
            raise TransitionError(f"producer-fission artifact is missing: {path}")
        seen.add(path)
        actual = sha256_file(path)
        if actual != record["sha256"]:
            raise TransitionError(f"producer-fission artifact changed: {path}")
        records.append({"path": str(path), "sha256": actual})
    return sorted(records, key=lambda item: item["path"])


def validate_dormant_candidate(value: Any) -> tuple[dict[str, Any], str]:
    if (not isinstance(value, dict)
            or value.get("schema_version") != COVERAGE_SCHEMA):
        raise TransitionError("unexpected schedule-coverage schema")
    model_graph = value.get("model_graph")
    if (not isinstance(model_graph, dict)
            or model_graph.get("policy", {}).get("application_source_present")
            is not False
            or model_graph.get("policy", {}).get("provider_call_authorized")
            is not False):
        raise TransitionError("schedule coverage violates the source-free boundary")
    payload = dict(model_graph)
    graph_id = payload.pop("graph_id", None)
    if graph_id != coverage.canonical_id(
            "gicc-source-free-schedule-coverage-v1", payload):
        raise TransitionError("schedule-coverage model graph ID changed")
    dormant = [
        case for case in model_graph.get("cases", [])
        if case.get("dormant_compiler_oracle") is not None
    ]
    if len(dormant) != 1:
        raise TransitionError("expected exactly one dormant compiler oracle")
    case = dormant[0]
    candidate = case["dormant_compiler_oracle"]
    candidate_payload = {
        "case_id": candidate.get("case_id"),
        "kind": candidate.get("kind"),
        "compiler_materializer": candidate.get("compiler_materializer"),
        "phase_count": candidate.get("phase_count"),
    }
    if (candidate_payload != {
            "case_id": case.get("case_id"),
            "kind": "producer_frontier_two_phase",
            "compiler_materializer": "guarded_host_device_fission",
            "phase_count": 2,
            } or candidate.get("candidate_id") != coverage.canonical_id(
                "gicc-schedule-candidate-v1", candidate_payload
            ) or candidate["candidate_id"] in case.get(
                "model_visible_candidate_ids", []
            )):
        raise TransitionError("dormant producer-fission candidate changed")
    return candidate, graph_id


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
                     coverage_path: Path, binary_dir: Path) -> dict[str, Any]:
    monitor = read_json(monitor_path)
    validate_runtime_contract(monitor)
    analysis = validate_analysis(
        read_json(analysis_path), monitor, monitor_path,
    )
    artifacts = validate_artifacts(monitor)
    candidate, coverage_graph_id = validate_dormant_candidate(
        read_json(coverage_path)
    )
    baseline = binary_dir / "baseline/jacobi"
    fission = binary_dir / "fission/jacobi"
    provenance = binary_dir / "BUILD_PROVENANCE.txt"
    for binary in (baseline, fission, provenance):
        if not binary.is_file():
            raise TransitionError(f"missing frozen build artifact: {binary}")
    artifact_map = {Path(row["path"]).resolve(): row["sha256"] for row in artifacts}
    for binary in (baseline, fission, provenance):
        if artifact_map.get(binary.resolve()) != sha256_file(binary):
            raise TransitionError(
                f"scout monitor did not bind build artifact: {binary}"
            )
    payload = {
        "schema_version": TRANSITION_SCHEMA,
        "status": "confirmation_plan_ready",
        "scout_job_id": monitor.get("job_id"),
        "scout_monitor_sha256": sha256_file(monitor_path),
        "scout_analysis_sha256": sha256_file(analysis_path),
        "schedule_coverage_graph_id": coverage_graph_id,
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
            "reason": (
                "the compiler schedule has no runtime-size decision slot, so "
                "both preregistered sizes remain in the primary estimand"
            ),
        },
        "confirmation_contract": {
            "queue": "pdebug",
            "nodes": 2,
            "ranks": 16,
            "ranks_per_node": 8,
            "cpu_cores_per_rank": 8,
            "gpus_per_rank": 1,
            "independent_allocations": 3,
            "maximum_active_or_queued_jobs": 1,
            "balanced_blocks_per_allocation": ["AB", "BA"],
            "sizes": list(SIZES),
            "iterations": 200,
            "nccheck": 10,
            "primary_metric": (
                "geometric mean of paired baseline/fission speedups across "
                "both sizes and both order-balanced blocks, clustered by "
                "independent allocation"
            ),
            "pass_rule": (
                "point estimate >=1.03, exact allocation-cluster paired-"
                "bootstrap 95% lower bound >1.0, at least 2/3 allocation "
                "geometric means >1.0, and all correctness checks pass"
            ),
            "scout_or_confirmation_labels_visible_to_model": False,
        },
        "post_confirmation": {
            "on_failure": "keep producer-fission candidate dormant",
            "on_pass": (
                "permit a new content-addressed compiler graph to expose the "
                "candidate; do not mutate the frozen graph"
            ),
            "single_positive_case_supports_llm_selection_claim": False,
            "provider_call_authorized": False,
        },
        "files": [
            file_record(monitor_path, "passed_scout_monitor"),
            file_record(analysis_path, "passed_scout_analysis"),
            file_record(coverage_path, "source_free_schedule_coverage"),
            file_record(baseline, "frozen_baseline_binary"),
            file_record(fission, "frozen_fission_binary"),
            file_record(provenance, "frozen_build_provenance"),
            file_record(PROTOCOL, "preregistered_transition_protocol"),
            file_record(Path(__file__), "transition_preparer"),
        ],
    }
    transition = {**payload, "transition_id": bridge._fingerprint(payload)}
    return transition


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
    parser.add_argument("--coverage-report", type=Path, required=True)
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
    args = parser.parse_args()
    try:
        transition = build_transition(
            args.monitor, args.analysis, args.coverage_report,
            args.binary_dir,
        )
        if args.command == "prepare":
            if args.out.exists():
                raise TransitionError(f"refusing to overwrite {args.out}")
            write_json_atomic(args.out, transition)
            action = "prepared"
        else:
            if read_json(args.report) != transition:
                raise TransitionError(
                    "producer-fission confirmation transition changed"
                )
            action = "verified"
        print(
            f"producer-fission-confirmation: {action}; model_invoked=false; "
            f"scheduler_job_submitted=false; transition_id="
            f"{transition['transition_id']}"
        )
        return 0
    except (TransitionError, coverage.CoverageError, OSError, KeyError,
            TypeError, ValueError) as exc:
        print(f"producer-fission-confirmation: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
