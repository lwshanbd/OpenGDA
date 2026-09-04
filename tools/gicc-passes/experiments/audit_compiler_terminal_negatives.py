#!/usr/bin/env python3
"""Replay and bind the three terminal-negative compiler candidates.

The producer and guarded scouts ended through different fail-closed paths, so
their evidence cannot honestly be rewritten as ordinary completed-scout
analyses.  This audit preserves those distinctions while proving that neither
candidate may enter a model-visible compiler graph.  It has no provider,
scheduler, compiler, or application-source mutation path.
"""

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
ROOT = HERE.parents[2]
PASS_PYTHON = HERE.parent / "python"
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(HERE / "producer_fission"))
sys.path.insert(0, str(HERE / "guarded_early_trigger"))
sys.path.insert(0, str(HERE / "reused_loop_descriptor"))

import audit_partial_negative_scout as guarded  # noqa: E402
import audit_producer_fission_runtime_triage as producer  # noqa: E402
import analyze_reused_loop_descriptor_scout as reused  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


REPORT_SCHEMA = "gicc-compiler-terminal-negatives-v1"
BOUNDARY = {
    "compiler_lto_decisions_only": True,
    "application_source_modified": False,
    "model_invoked": False,
    "provider_invoked": False,
    "provider_call_authorized": False,
    "scheduler_job_submitted": False,
    "runtime_values_used_as_positive_performance_evidence": False,
}


class TerminalNegativeError(RuntimeError):
    """The supplied artifacts do not prove the terminal-negative result."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TerminalNegativeError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TerminalNegativeError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise TerminalNegativeError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def recorded_path(value: str) -> Path:
    raw = Path(value)
    return raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()


def file_record(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"missing {role}: {resolved}")
    return {
        "role": role,
        "path": display_path(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def verify_file_record(record: Any, *, label: str) -> Path:
    require(isinstance(record, dict), f"invalid {label} file record")
    value = record.get("path")
    require(isinstance(value, str), f"{label} lacks a path")
    path = recorded_path(value)
    require(path.is_file(), f"missing {label}: {path}")
    require(record.get("sha256") == sha256_file(path),
            f"{label} hash changed: {path}")
    if "bytes" in record:
        require(record.get("bytes") == path.stat().st_size,
                f"{label} size changed: {path}")
    return path


def state_phase(path: Path) -> str:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise TerminalNegativeError(f"cannot read state {path}: {exc}") from exc
    require(len(lines) == 1, f"invalid state file: {path}")
    fields = lines[0].split("\t")
    require(len(fields) == 3 and bool(fields[1]),
            f"invalid state record: {path}")
    return fields[1]


def _records_by_role(records: Any, expected: set[str], *, label: str) -> dict[str, Path]:
    require(isinstance(records, list), f"{label} evidence is not a list")
    paths: dict[str, Path] = {}
    for record in records:
        require(isinstance(record, dict) and isinstance(record.get("role"), str),
                f"invalid {label} evidence record")
        role = record["role"]
        require(role not in paths, f"duplicate {label} evidence role: {role}")
        paths[role] = verify_file_record(record, label=f"{label}:{role}")
    require(set(paths) == expected, f"{label} evidence roles changed")
    return paths


def verify_producer(path: Path, state: Path) -> dict[str, Any]:
    require(state_phase(state) == "negative",
            "producer terminal state is not negative")
    value = read_json(path)
    require(
        isinstance(value, dict)
        and value.get("schema_version")
        == "gicc-producer-fission-runtime-triage-v1"
        and value.get("status") == "negative_correctness_gate",
        "unexpected producer terminal-negative report",
    )
    disposition = value.get("candidate_disposition", {})
    require(
        disposition.get("candidate") == "producer_frontier_fission"
        and disposition.get("model_visible") is False
        and disposition.get("confirmation_eligible") is False
        and disposition.get("paper_performance_claim") is False,
        "producer candidate was not closed fail-safely",
    )
    evidence = _records_by_role(
        value.get("evidence"),
        {
            "runtime_triage_auditor", "runtime_triage_protocol",
            "fixed_build_provenance", "same_allocation_runner",
            "unchanged_application_source", "compiler_pass_repair",
            "compiler_pass_regression", "fixed_baseline_binary",
            "fixed_fission_binary",
        },
        label="producer",
    )
    original = value.get("original_failure", {})
    verify_file_record(original.get("monitor"), label="producer failed monitor")
    verify_file_record(original.get("stderr"), label="producer failed stderr")
    for index, record in enumerate(original.get("verified_artifacts", [])):
        verify_file_record(record, label=f"producer failed artifact {index}")
    require(original.get("state") == "failed"
            and original.get("scheduler", {}).get("exit_code") == 139
            and original.get("jobspec", {}).get("queue") == "pdebug",
            "producer original failure changed")

    validation = value.get("same_allocation_validation", {})
    job = validation.get("job", {})
    require(
        job.get("scheduler", {}).get("clean") is True
        and job.get("scheduler", {}).get("exit_code") == 0
        and not job.get("scheduler", {}).get("exception_types")
        and job.get("jobspec", {}).get("queue") == "pdebug",
        "producer diagnostic allocation was not a clean pdebug job",
    )
    try:
        producer.validate_topology(job["jobspec"])
    except (KeyError, producer.TriageError) as exc:
        raise TerminalNegativeError(
            f"producer diagnostic topology changed: {exc}"
        ) from exc
    baseline_stdout = verify_file_record(
        validation.get("baseline", {}).get("stdout"),
        label="producer baseline stdout",
    )
    fission_stdout = verify_file_record(
        validation.get("fission", {}).get("stdout"),
        label="producer fission stdout",
    )
    try:
        replay = producer.compare_pair(baseline_stdout, fission_stdout)
    except (producer.TriageError, ValueError, OSError) as exc:
        raise TerminalNegativeError(
            f"cannot replay producer correctness pair: {exc}"
        ) from exc
    for key in ("baseline", "fission", "correctness_gate"):
        require(validation.get(key) == replay[key],
                f"producer {key} does not replay")
    require(
        replay["correctness_gate"]["passed"] is False
        and validation.get("runtime_values_are_performance_evidence") is False,
        "producer correctness failure was promoted to performance evidence",
    )
    repair = value.get("compiler_repair", {})
    source_hash = evidence["unchanged_application_source"]
    require(
        repair.get("application_source_unchanged") is True
        and repair.get("failed_build_application_source_sha256")
        == repair.get("fixed_build_application_source_sha256")
        == sha256_file(source_hash),
        "producer repair changed the application source",
    )
    require(value.get("semantic_diagnosis", {}).get("status")
            == "inference_not_claim",
            "producer semantic diagnosis was promoted to a claim")
    return value


def verify_guarded(path: Path, state: Path, output_dir: Path) -> dict[str, Any]:
    require(state_phase(state) == "failed",
            "guarded scout must retain its factual failed state")
    value = read_json(path)
    require(
        isinstance(value, dict)
        and value.get("schema_version")
        == "gicc-guarded-early-partial-negative-audit-v1"
        and value.get("status") == "negative_gate_mathematically_unreachable"
        and value.get("audit_scope")
        == "one_sided_post_failure_reachability_only",
        "unexpected guarded terminal-negative report",
    )
    disposition = value.get("candidate_disposition", {})
    require(
        disposition.get("candidate") == "guarded_early_trigger"
        and disposition.get("model_visible") is False
        and disposition.get("confirmation_eligible") is False
        and disposition.get("paper_performance_claim") is False,
        "guarded candidate was not closed fail-safely",
    )
    monitor_path = verify_file_record(
        value.get("original_monitor"), label="guarded failed monitor"
    )
    monitor = read_json(monitor_path)
    require(
        monitor.get("state") == "failed"
        and monitor.get("scheduler", {}).get("exit_code") == 1
        and not monitor.get("scheduler", {}).get("exception_types")
        and monitor.get("jobspec", {}).get("queue") == "pdebug"
        and monitor.get("jobspec", {}).get("duration_seconds") == 1800.0,
        "guarded scout failure is not the frozen clean duration rejection",
    )
    for index, record in enumerate(value.get("verified_scout_artifacts", [])):
        verify_file_record(record, label=f"guarded scout artifact {index}")
    _records_by_role(
        value.get("evidence"),
        {"partial_negative_auditor", "partial_negative_audit_rules",
         "scout_driver_output"},
        label="guarded",
    )
    failure_stderr = verify_file_record(
        value.get("failure", {}).get("stderr"),
        label="guarded duration rejection",
    )
    require(
        "job duration (8m) exceeds remaining instance lifetime"
        in failure_stderr.read_text(encoding="utf-8", errors="replace"),
        "guarded missing pair is not the frozen duration rejection",
    )
    require(value.get("failure", {}).get("failed_arm")
            == "replicate=4 size=8192 arm=baseline",
            "guarded missing arm changed")
    require((output_dir / "rep4/size8192/baseline.out").stat().st_size == 0,
            "guarded rejected baseline unexpectedly has output")
    missing_guarded = output_dir / "rep4/size8192/guarded.out"
    require(not missing_guarded.exists() or missing_guarded.stat().st_size == 0,
            "guarded missing arm unexpectedly has output")
    try:
        completed = {
            "4096": [
                guarded.parse_pair(output_dir, replicate, 4096)
                for replicate in range(1, 5)
            ],
            "8192": [
                guarded.parse_pair(output_dir, replicate, 8192)
                for replicate in range(1, 4)
            ],
        }
        reachability = {
            size: guarded.gate_reachability(
                [pair["speedup"] for pair in pairs]
            )
            for size, pairs in completed.items()
        }
    except (guarded.PartialNegativeError, OSError, ValueError) as exc:
        raise TerminalNegativeError(
            f"cannot replay guarded partial scout: {exc}"
        ) from exc
    require(value.get("completed_pairs") == completed,
            "guarded completed pairs do not replay")
    require(value.get("frozen_gate_reachability") == reachability,
            "guarded gate reachability does not replay")
    require(all(
        item["gate_reachable_under_arbitrarily_favorable_missing_pair"] is False
        for item in reachability.values()
    ), "guarded positive gate is still reachable")
    limitations = value.get("limitations", {})
    for key in (
        "post_failure_audit_was_preregistered",
        "may_support_positive_result",
        "completed_timings_are_paper_performance_evidence",
        "frozen_monitor_accepts_scientific_timing_notation",
    ):
        require(limitations.get(key) is False,
                f"guarded limitation changed: {key}")
    require(limitations.get("rerun_needed_to_estimate_full_eight_pair_performance")
            is True, "guarded partial timing limitation changed")
    return value


def verify_reused(path: Path, state: Path) -> tuple[dict[str, Any], Path]:
    require(state_phase(state) == "negative",
            "reused-loop terminal state is not negative")
    value = read_json(path)
    require(
        isinstance(value, dict)
        and value.get("schema_version")
        == "gicc-reused-loop-descriptor-analysis-v1"
        and value.get("correctness_gate", {}).get("passed") is True
        and value.get("oracle_headroom_gate", {}).get("passed") is False
        and value.get("oracle_headroom_gate", {}).get(
            "provider_protocol_permitted"
        ) is False
        and value.get("oracle_headroom_gate", {}).get("paper_claim") is False,
        "unexpected reused-loop terminal-negative analysis",
    )
    monitor_value = value.get("monitor")
    require(isinstance(monitor_value, str), "reused-loop analysis lacks monitor")
    monitor_path = recorded_path(monitor_value)
    monitor = read_json(monitor_path)
    for index, record in enumerate(monitor.get("artifacts", [])):
        verify_file_record(record, label=f"reused scout artifact {index}")
    try:
        replay = reused.analyze(monitor)
    except (KeyError, TypeError, ValueError) as exc:
        raise TerminalNegativeError(
            f"cannot replay reused-loop analysis: {exc}"
        ) from exc
    observed = dict(value)
    observed.pop("created_at", None)
    observed.pop("monitor", None)
    replay.pop("created_at", None)
    require(observed == replay, "reused-loop analysis does not replay")
    return value, monitor_path


def build_report(
    producer_report: Path,
    producer_state: Path,
    guarded_report: Path,
    guarded_state: Path,
    guarded_output_dir: Path,
    reused_analysis: Path,
    reused_state: Path,
) -> dict[str, Any]:
    producer_value = verify_producer(producer_report, producer_state)
    guarded_value = verify_guarded(
        guarded_report, guarded_state, guarded_output_dir.resolve()
    )
    reused_value, reused_monitor = verify_reused(reused_analysis, reused_state)
    candidates = {
        "jacobi": {
            "candidate": "producer_frontier_fission",
            "status": "closed_negative",
            "reason_code": "compiler_correctness_gate_failed",
            "model_visible": False,
            "confirmation_eligible": False,
            "paper_performance_claim": False,
            "runtime_values_are_positive_performance_evidence": False,
            "terminal_report_sha256": sha256_file(producer_report),
        },
        "loop_lto": {
            "candidate": "reused_loop_descriptor",
            "status": "closed_negative",
            "reason_code": "preregistered_oracle_headroom_gate_failed",
            "model_visible": False,
            "confirmation_eligible": False,
            "paper_performance_claim": False,
            "runtime_values_are_positive_performance_evidence": False,
            "terminal_report_sha256": sha256_file(reused_analysis),
        },
        "mm_minimal": {
            "candidate": "guarded_early_trigger",
            "status": "closed_negative",
            "reason_code": "positive_gate_mathematically_unreachable",
            "model_visible": False,
            "confirmation_eligible": False,
            "paper_performance_claim": False,
            "runtime_values_are_positive_performance_evidence": False,
            "terminal_report_sha256": sha256_file(guarded_report),
        },
    }
    evidence = [
        file_record(Path(__file__), "terminal_negative_auditor"),
        file_record(producer_report, "producer_report"),
        file_record(producer_state, "producer_state"),
        file_record(guarded_report, "guarded_report"),
        file_record(guarded_state, "guarded_state"),
        file_record(reused_analysis, "reused_analysis"),
        file_record(reused_state, "reused_state"),
        file_record(reused_monitor, "reused_monitor"),
        file_record(Path(reused.__file__), "reused_analyzer"),
    ]
    payload = {
        "schema_version": REPORT_SCHEMA,
        "status": "three_candidates_closed_without_model_visibility",
        "boundary": dict(BOUNDARY),
        "candidates": candidates,
        "summary": {
            "terminal_negative_count": 3,
            "model_visible_count": 0,
            "confirmation_eligible_count": 0,
            "positive_performance_claim_count": 0,
            "provider_call_authorized_count": 0,
        },
        "guarded_output_dir": display_path(guarded_output_dir),
        "evidence": sorted(evidence, key=lambda item: item["role"]),
        "source_reports": {
            "producer_status": producer_value["status"],
            "guarded_status": guarded_value["status"],
            "reused_gate_passed": reused_value["oracle_headroom_gate"]["passed"],
        },
    }
    return {**payload, "terminal_negatives_id": bridge._fingerprint(payload)}


def verify_contained(report_path: Path) -> dict[str, Any]:
    report = read_json(report_path)
    require(isinstance(report, dict)
            and report.get("schema_version") == REPORT_SCHEMA,
            "unexpected terminal-negative schema")
    payload = dict(report)
    require(payload.pop("terminal_negatives_id", None)
            == bridge._fingerprint(payload),
            "terminal-negative ID does not match content")
    require(report.get("boundary") == BOUNDARY,
            "terminal-negative compiler-only boundary changed")
    paths = _records_by_role(
        report.get("evidence"),
        {
            "terminal_negative_auditor", "producer_report", "producer_state",
            "guarded_report", "guarded_state", "reused_analysis",
            "reused_state", "reused_monitor", "reused_analyzer",
        },
        label="terminal-negative",
    )
    require(paths["terminal_negative_auditor"] == Path(__file__).resolve(),
            "terminal-negative report names another auditor")
    require(paths["reused_analyzer"] == Path(reused.__file__).resolve(),
            "terminal-negative report names another reused analyzer")
    guarded_dir_value = report.get("guarded_output_dir")
    require(isinstance(guarded_dir_value, str),
            "terminal-negative report lacks guarded output directory")
    regenerated = build_report(
        paths["producer_report"], paths["producer_state"],
        paths["guarded_report"], paths["guarded_state"],
        recorded_path(guarded_dir_value), paths["reused_analysis"],
        paths["reused_state"],
    )
    require(regenerated == report,
            "terminal-negative report does not replay from raw evidence")
    return report


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
    parser.add_argument("--producer-report", type=Path, required=True)
    parser.add_argument("--producer-state", type=Path, required=True)
    parser.add_argument("--guarded-report", type=Path, required=True)
    parser.add_argument("--guarded-state", type=Path, required=True)
    parser.add_argument("--guarded-output-dir", type=Path, required=True)
    parser.add_argument("--reused-analysis", type=Path, required=True)
    parser.add_argument("--reused-state", type=Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit")
    add_inputs(emit)
    emit.add_argument("--out", type=Path, required=True)
    verify = subparsers.add_parser("verify-contained")
    verify.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "emit":
            result = build_report(
                args.producer_report, args.producer_state,
                args.guarded_report, args.guarded_state,
                args.guarded_output_dir, args.reused_analysis,
                args.reused_state,
            )
            write_json_atomic(args.out, result)
            action = "wrote"
        else:
            result = verify_contained(args.report)
            action = "verified"
        print(
            f"compiler-terminal-negatives: {action}; closed=3; "
            f"model_visible=0; provider_calls_authorized=0; "
            f"terminal_negatives_id={result['terminal_negatives_id']}"
        )
        return 0
    except (
        TerminalNegativeError, guarded.PartialNegativeError,
        producer.TriageError, OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"compiler-terminal-negatives: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
