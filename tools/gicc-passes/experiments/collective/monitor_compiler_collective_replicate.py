#!/usr/bin/env python3
"""Audit one pdebug allocation containing a paired collective replicate.

All compiler-control arms execute sequentially on the same allocated nodes.
The monitor waits for one clean Flux completion, freezes the exact resource
set, and validates every per-arm benchmark log against the same runtime
contract.  It never submits, cancels, or modifies a scheduler job.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import monitor_compiler_collective_job as base


LABEL_RE = re.compile(r"^[a-z0-9_]+$")


def named_path(value: str) -> tuple[str, Path]:
    try:
        label, path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected LABEL=PATH") from exc
    if not LABEL_RE.fullmatch(label) or not path:
        raise argparse.ArgumentTypeError("invalid benchmark label or path")
    return label, Path(path)


def unique_paths(
    values: list[tuple[str, Path]], role: str,
) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for label, path in values:
        if label in result:
            raise base.MonitorError(f"duplicate {role} label: {label}")
        result[label] = path
    if not result:
        raise base.MonitorError(f"paired replicate has no {role} paths")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--driver-stdout", type=Path, required=True)
    parser.add_argument("--driver-stderr", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--replicate", type=int, required=True)
    parser.add_argument("--benchmark", type=named_path, action="append",
                        required=True, metavar="LABEL=PATH")
    parser.add_argument("--benchmark-stderr", type=named_path, action="append",
                        required=True, metavar="LABEL=PATH")
    parser.add_argument("--expected-size", type=int, action="append",
                        required=True)
    parser.add_argument("--expected-nodes", type=int, default=2)
    parser.add_argument("--expected-ranks", type=int, default=16)
    parser.add_argument("--expected-ppn", type=int, default=8)
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--expected-warmup", type=int, required=True)
    parser.add_argument(
        "--artifact", type=base.parse_artifact, action="append", default=[],
        metavar="PATH=SHA256",
    )
    args = parser.parse_args()

    state: dict[str, Any] = {
        "schema_version": "gicc-collective-replicate-job-monitor-v1",
        "job_id": args.job_id,
        "replicate": args.replicate,
        "monitor_started_at": base.utc_now(),
        "state": "monitoring",
        "expected": {
            "sizes": sorted(args.expected_size),
            "nodes": args.expected_nodes,
            "ranks": args.expected_ranks,
            "ppn": args.expected_ppn,
            "runs": args.expected_runs,
            "warmup": args.expected_warmup,
        },
        "paths": {
            "driver_stdout": str(args.driver_stdout.resolve()),
            "driver_stderr": str(args.driver_stderr.resolve()),
            "status": str(args.status.resolve()),
        },
    }
    base.atomic_write_json(args.status, state)

    try:
        stdout_paths = unique_paths(args.benchmark, "benchmark stdout")
        stderr_paths = unique_paths(
            args.benchmark_stderr, "benchmark stderr"
        )
        if set(stdout_paths) != set(stderr_paths):
            raise base.MonitorError(
                "benchmark stdout/stderr labels do not match"
            )
        state["artifacts"] = base.verify_artifacts(args.artifact)
        base.run_flux("job", "wait-event", args.job_id, "clean")
        eventlog_text = base.run_flux(
            "job", "info", args.job_id, "eventlog"
        )
        jobspec_text = base.run_flux(
            "job", "info", "-o", args.job_id, "jobspec"
        )
        resources_text = base.run_flux("job", "info", args.job_id, "R")
        scheduler = base.scheduler_result(
            base.parse_json_lines(eventlog_text, "eventlog")
        )
        state["scheduler"] = scheduler
        state["jobspec"] = base.summarize_jobspec(json.loads(jobspec_text))
        state["resource_set"] = base.summarize_resources(
            json.loads(resources_text)
        )
        state["driver_stdout"] = {
            "path": str(args.driver_stdout.resolve()),
            "bytes": args.driver_stdout.stat().st_size,
            "sha256": base.sha256(args.driver_stdout),
        }
        state["driver_stderr"] = {
            **base.summarize_stderr(args.driver_stderr),
            "path": str(args.driver_stderr.resolve()),
            "sha256": base.sha256(args.driver_stderr),
        }
        if scheduler["exit_code"] != 0 or scheduler["exception_types"]:
            exception_text = ",".join(scheduler["exception_types"]) or "none"
            raise base.MonitorError(
                f"scheduler failure: exit_code={scheduler['exit_code']} "
                f"exceptions={exception_text}"
            )

        results = {}
        for label in sorted(stdout_paths):
            stdout = stdout_paths[label]
            stderr = stderr_paths[label]
            benchmark = base.validate_output(
                stdout, label, args.expected_size,
                args.expected_nodes, args.expected_ranks, args.expected_ppn,
                args.expected_runs, args.expected_warmup,
            )
            results[label] = {
                "benchmark": benchmark,
                "stdout": {
                    "path": str(stdout.resolve()),
                    "bytes": stdout.stat().st_size,
                    "sha256": base.sha256(stdout),
                },
                "stderr": {
                    **base.summarize_stderr(stderr),
                    "path": str(stderr.resolve()),
                    "sha256": base.sha256(stderr),
                },
            }
        state["benchmarks"] = results
        state["state"] = "passed"
        return_code = 0
    except (base.MonitorError, OSError, json.JSONDecodeError) as exc:
        state["state"] = "failed"
        state["error"] = str(exc)
        return_code = 1
    finally:
        state["monitor_finished_at"] = base.utc_now()
        base.atomic_write_json(args.status, state)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
