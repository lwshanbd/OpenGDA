#!/usr/bin/env python3
"""Freeze a fail-closed runtime triage for the Jacobi fission candidate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re
from typing import Any

import monitor_producer_fission_oracle_scout as scout_monitor


common = scout_monitor.common
SHA_LINE_RE = re.compile(r"^(?P<sha>[0-9a-f]{64})  (?P<path>.+)$")
REL_TOL = 1e-6
ABS_TOL = 1e-7


class TriageError(RuntimeError):
    """The supplied evidence does not prove the frozen negative result."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise TriageError(message)


def file_record(path: Path, role: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {role}: {path}")
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": common.sha256(path),
    }


def parse_provenance(path: Path) -> tuple[dict[str, str], dict[Path, str]]:
    headers: dict[str, str] = {}
    artifacts: dict[Path, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if match := SHA_LINE_RE.match(line):
            artifact = Path(match.group("path")).resolve()
            require(artifact not in artifacts,
                    f"duplicate provenance artifact: {artifact}")
            artifacts[artifact] = match.group("sha")
        elif "=" in line:
            key, value = line.split("=", 1)
            headers[key] = value
    require(headers.get("schema") == "gicc-producer-fission-build-v1",
            "unexpected fixed-build provenance schema")
    require("commit" in headers and "compiler" in headers,
            "fixed-build provenance is incomplete")
    require(bool(artifacts), "fixed-build provenance has no artifacts")
    for artifact, expected in artifacts.items():
        require(artifact.is_file(), f"missing provenance artifact: {artifact}")
        require(common.sha256(artifact) == expected,
                f"provenance artifact changed: {artifact}")
    return headers, artifacts


def compare_pair(baseline_path: Path, fission_path: Path) -> dict[str, Any]:
    baseline = scout_monitor.parse_run(baseline_path, 1024)
    fission = scout_monitor.parse_run(fission_path, 1024)
    require(baseline["iterations"] == fission["iterations"],
            "paired diagnostic iteration counts differ")
    left = float(baseline["final_l2"])
    right = float(fission["final_l2"])
    difference = abs(left - right)
    allowed = max(ABS_TOL, REL_TOL * max(abs(left), abs(right)))
    passed = math.isclose(left, right, rel_tol=REL_TOL, abs_tol=ABS_TOL)
    return {
        "baseline": {
            "configured_iterations": baseline["configured_iterations"],
            "iterations": baseline["iterations"],
            "seconds": baseline["seconds"],
            "final_l2": left,
            "stdout": baseline["stdout"],
        },
        "fission": {
            "configured_iterations": fission["configured_iterations"],
            "iterations": fission["iterations"],
            "seconds": fission["seconds"],
            "final_l2": right,
            "stdout": fission["stdout"],
        },
        "correctness_gate": {
            "passed": passed,
            "relative_tolerance": REL_TOL,
            "absolute_tolerance": ABS_TOL,
            "absolute_difference": difference,
            "relative_difference_vs_baseline": difference / abs(left),
            "allowed_absolute_difference": allowed,
            "excess_over_allowed": difference / allowed,
        },
    }


def validate_topology(jobspec: dict[str, Any]) -> None:
    resources = jobspec.get("resources")
    require(isinstance(resources, list) and len(resources) == 1,
            "paired job must request one node resource entry")
    nodes = resources[0]
    require(nodes.get("type") == "node" and nodes.get("count") == 2,
            "paired job must request two nodes")
    slots = nodes.get("with")
    require(isinstance(slots, list) and len(slots) == 1,
            "paired job node entry must contain one slot entry")
    slot = slots[0]
    require(slot.get("type") == "slot" and slot.get("count") == 8,
            "paired job must request eight ranks per node")
    children = slot.get("with")
    require(isinstance(children, list), "paired job slot resources are missing")
    shape = {(item.get("type"), item.get("count")) for item in children}
    require(shape == {("core", 8), ("gpu", 1)},
            "paired job must request eight cores and one GPU per rank")


def completed_job(job_id: str, expected_name: str) -> dict[str, Any]:
    events = common.parse_json_lines(
        common.run_flux("job", "info", job_id, "eventlog"), "eventlog")
    jobspec_value = json.loads(
        common.run_flux("job", "info", "-o", job_id, "jobspec"))
    resources_value = json.loads(common.run_flux("job", "info", job_id, "R"))
    scheduler = common.scheduler_result(events)
    jobspec = common.summarize_jobspec(jobspec_value)
    resource_set = common.summarize_resources(resources_value)
    system = jobspec_value.get("attributes", {}).get("system", {})
    name = system.get("job", {}).get("name")
    require(name == expected_name,
            f"paired job name is {name!r}, expected {expected_name!r}")
    require(scheduler["exit_code"] == 0 and not scheduler["exception_types"],
            "paired diagnostic scheduler result is not clean")
    validate_topology(jobspec_value)
    return {
        "job_id": job_id,
        "job_name": name,
        "scheduler": scheduler,
        "jobspec": jobspec,
        "resource_set": resource_set,
    }


def failed_scout(path: Path, stderr_path: Path,
                 application_source: Path) -> tuple[dict[str, Any], str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("state") == "failed", "original scout did not fail")
    require(value.get("scheduler", {}).get("exit_code") == 139,
            "original scout did not record exit code 139")
    require(value.get("jobspec", {}).get("queue") == "pdebug",
            "original scout was not a pdebug job")
    stderr = stderr_path.read_text(encoding="utf-8", errors="replace")
    require("Segmentation fault" in stderr,
            "original fission stderr lacks the recorded segmentation fault")
    source = application_source.resolve()
    source_hash = None
    records = []
    for artifact in value.get("artifacts", []):
        artifact_path = Path(artifact["path"]).resolve()
        expected = artifact["sha256"]
        require(artifact_path.is_file(),
                f"missing original-scout artifact: {artifact_path}")
        require(common.sha256(artifact_path) == expected,
                f"original-scout artifact changed: {artifact_path}")
        records.append({"path": str(artifact_path), "sha256": expected})
        if artifact_path == source:
            source_hash = expected
    require(source_hash is not None,
            "original scout did not freeze the application source")
    return ({
        "job_id": value.get("job_id"),
        "state": value["state"],
        "scheduler": value["scheduler"],
        "jobspec": value["jobspec"],
        "resource_set": value.get("resource_set"),
        "error": value.get("error"),
        "monitor": file_record(path, "failed_scout_monitor"),
        "stderr": file_record(stderr_path, "failed_fission_stderr"),
        "verified_artifacts": records,
    }, source_hash)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failed-monitor", type=Path, required=True)
    parser.add_argument("--failed-stderr", type=Path, required=True)
    parser.add_argument("--fixed-provenance", type=Path, required=True)
    parser.add_argument("--fixed-binary-dir", type=Path, required=True)
    parser.add_argument("--paired-baseline", type=Path, required=True)
    parser.add_argument("--paired-fission", type=Path, required=True)
    parser.add_argument("--paired-runner", type=Path, required=True)
    parser.add_argument("--paired-job-id", required=True)
    parser.add_argument("--paired-job-name", default="pfission-pushfix-check")
    parser.add_argument("--application-source", type=Path, required=True)
    parser.add_argument("--pass-source", type=Path, required=True)
    parser.add_argument("--lit-test", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    original, original_source_hash = failed_scout(
        args.failed_monitor, args.failed_stderr, args.application_source)
    headers, provenance_artifacts = parse_provenance(args.fixed_provenance)
    application_source = args.application_source.resolve()
    fixed_source_hash = provenance_artifacts.get(application_source)
    require(fixed_source_hash == original_source_hash,
            "application source differs between failed and fixed builds")

    baseline_binary = (args.fixed_binary_dir / "baseline" / "jacobi").resolve()
    fission_binary = (args.fixed_binary_dir / "fission" / "jacobi").resolve()
    for binary in (baseline_binary, fission_binary):
        require(binary in provenance_artifacts,
                f"fixed binary is absent from provenance: {binary}")

    pass_text = args.pass_source.read_text(encoding="utf-8")
    test_text = args.lit_test.read_text(encoding="utf-8")
    require("cloneLaunch(phasedBuilder, *audited.configurationPush);" in pass_text,
            "compiler repair does not replay the audited HIP configuration push")
    require("STUB-NEXT: call i32 @__hipPushCallConfiguration" in test_text,
            "lit regression does not require the second HIP configuration push")

    pair = compare_pair(args.paired_baseline, args.paired_fission)
    require(pair["correctness_gate"]["passed"] is False,
            "negative triage requires a failed frozen correctness gate")
    runner_text = args.paired_runner.read_text(encoding="utf-8")
    require("for arm in baseline fission" in runner_text,
            "paired runner does not execute baseline then fission")
    require("flux run -N2 -n16 -c8 -g1" in runner_text,
            "paired runner topology is not the frozen N2/16-rank shape")

    evidence = [
        file_record(Path(__file__).resolve(), "runtime_triage_auditor"),
        file_record(
            Path(__file__).with_name("RUNTIME_TRIAGE_PROTOCOL.md"),
            "runtime_triage_protocol",
        ),
        file_record(args.fixed_provenance, "fixed_build_provenance"),
        file_record(args.paired_runner, "same_allocation_runner"),
        file_record(args.application_source, "unchanged_application_source"),
        file_record(args.pass_source, "compiler_pass_repair"),
        file_record(args.lit_test, "compiler_pass_regression"),
        file_record(baseline_binary, "fixed_baseline_binary"),
        file_record(fission_binary, "fixed_fission_binary"),
    ]
    result = {
        "schema_version": "gicc-producer-fission-runtime-triage-v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "negative_correctness_gate",
        "original_failure": original,
        "compiler_repair": {
            "classification": "replay_consumed_hip_launch_configuration",
            "application_source_unchanged": True,
            "failed_build_application_source_sha256": original_source_hash,
            "fixed_build_application_source_sha256": fixed_source_hash,
            "fixed_build_commit_field": headers["commit"],
            "fixed_build_compiler": headers["compiler"],
        },
        "same_allocation_validation": {
            "job": completed_job(args.paired_job_id, args.paired_job_name),
            **pair,
            "runtime_values_are_performance_evidence": False,
            "interpretation": "diagnostic only because numerical equivalence failed",
        },
        "candidate_disposition": {
            "candidate": "producer_frontier_fission",
            "model_visible": False,
            "confirmation_eligible": False,
            "paper_performance_claim": False,
            "action": "keep candidate masked for Jacobi",
            "reason": "post-repair output exceeds the predeclared correctness tolerance",
        },
        "semantic_diagnosis": {
            "status": "inference_not_claim",
            "observation": (
                "a compiler-inserted kernel boundary changes the final Jacobi norm"
            ),
            "likely_mechanism": (
                "the fused kernel has no grid-wide ordering point between distributed "
                "halo production and its communication trigger"
            ),
        },
        "evidence": evidence,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
