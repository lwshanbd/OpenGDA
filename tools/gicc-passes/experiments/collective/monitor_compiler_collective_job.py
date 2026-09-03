#!/usr/bin/env python3
"""Wait for and audit one compiler-collective Flux job.

This monitor is deliberately read-only with respect to Flux: it never submits,
cancels, reprioritizes, or modifies a job.  It records an initial monitoring
state, waits for the named job's ``clean`` event, captures the final jobspec and
eventlog, validates the benchmark output contract, and atomically writes a JSON
status file.  A non-zero scheduler status, timeout/cancel exception, incomplete
log, topology mismatch, unexpected size, or correctness error is a failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any


FIELD_RE = re.compile(r"([A-Za-z_]+)=([^ ]+)")
SAFE_ENV_RE = re.compile(
    r"^(FI_MR_CACHE_MONITOR|GICC_COLLECTIVE_PLAN_LABEL|GICC_COLL_SIZES|"
    r"GICC_NUM_PROXY_THREADS|GICC_PROXY_ENABLED|HSA_ENABLE_IPC_MODE_LEGACY|"
    r"MPICH_GPU_SUPPORT_ENABLED)$"
)
HDIR_CHECKPOINT_RE = re.compile(
    r"^\[hdir r(?P<rank>[0-9]+) call(?P<call>[0-9]+) "
    r"(?P<bytes>[0-9]+) B\] (?P<stage>.*)$"
)


class MonitorError(RuntimeError):
    """A job or artifact failed the monitor contract."""


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def run_flux(*arguments: str) -> str:
    completed = subprocess.run(
        ["flux", *arguments],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise MonitorError(
            f"flux {' '.join(arguments)} failed with "
            f"status {completed.returncode}: {detail}"
        )
    return completed.stdout


def parse_json_lines(text: str, role: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MonitorError(f"{role} line {number} is not JSON") from exc
        if not isinstance(value, dict):
            raise MonitorError(f"{role} line {number} is not an object")
        records.append(value)
    if not records:
        raise MonitorError(f"{role} is empty")
    return records


def scheduler_result(events: list[dict[str, Any]]) -> dict[str, Any]:
    clean = any(event.get("name") == "clean" for event in events)
    finish = next(
        (event for event in reversed(events) if event.get("name") == "finish"),
        None,
    )
    exceptions = [
        event.get("context", {})
        for event in events
        if event.get("name") == "exception"
    ]
    if not clean:
        raise MonitorError("job has no clean event")
    if finish is None:
        raise MonitorError("job has no finish event")
    raw_status = finish.get("context", {}).get("status")
    if not isinstance(raw_status, int):
        raise MonitorError("job finish event has no integer status")
    exception_types = [
        item.get("type") for item in exceptions if isinstance(item.get("type"), str)
    ]
    return {
        "clean": True,
        "raw_wait_status": raw_status,
        "exit_code": os.waitstatus_to_exitcode(raw_status),
        "exceptions": exceptions,
        "exception_types": exception_types,
    }


def parse_fields(line: str) -> dict[str, str]:
    return dict(FIELD_RE.findall(line))


def required_int(fields: dict[str, str], name: str) -> int:
    try:
        return int(fields[name])
    except (KeyError, ValueError) as exc:
        raise MonitorError(f"missing or invalid integer field {name}") from exc


def validate_output(
    path: Path,
    expected_label: str,
    expected_sizes: list[int],
    expected_nodes: int,
    expected_ranks: int,
    expected_ppn: int,
    expected_runs: int,
    expected_warmup: int,
) -> dict[str, Any]:
    config: dict[str, str] | None = None
    done: dict[str, str] | None = None
    results: dict[int, dict[str, str]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("COLLECTIVE_CONFIG "):
            if config is not None:
                raise MonitorError("duplicate COLLECTIVE_CONFIG")
            config = parse_fields(line)
        elif line.startswith("RESULT "):
            fields = parse_fields(line)
            size = required_int(fields, "bytes")
            if size in results:
                raise MonitorError(f"duplicate RESULT for {size} bytes")
            if required_int(fields, "errors") != 0:
                raise MonitorError(f"correctness errors at {size} bytes")
            try:
                median = float(fields["median_us"])
            except (KeyError, ValueError) as exc:
                raise MonitorError(f"invalid median_us at {size} bytes") from exc
            if median <= 0:
                raise MonitorError(f"non-positive median_us at {size} bytes")
            results[size] = fields
        elif line.startswith("COLLECTIVE_DONE "):
            if done is not None:
                raise MonitorError("duplicate COLLECTIVE_DONE")
            done = parse_fields(line)

    if config is None or done is None:
        raise MonitorError("incomplete collective log")
    actual_config = {
        "label": config.get("plan"),
        "nodes": expected_nodes,
        "ranks": required_int(config, "ranks"),
        "ppn": required_int(config, "ppn"),
        "runs": required_int(config, "runs"),
        "warmup": required_int(config, "warmup"),
    }
    expected_config = {
        "label": expected_label,
        "nodes": expected_nodes,
        "ranks": expected_ranks,
        "ppn": expected_ppn,
        "runs": expected_runs,
        "warmup": expected_warmup,
    }
    if actual_config != expected_config:
        raise MonitorError(
            f"runtime config {actual_config!r}, expected {expected_config!r}"
        )
    if sorted(results) != sorted(expected_sizes):
        raise MonitorError(
            f"result sizes {sorted(results)!r}, expected {sorted(expected_sizes)!r}"
        )
    for size, fields in results.items():
        if fields.get("plan") != expected_label:
            raise MonitorError(f"RESULT label mismatch at {size} bytes")
        topology = (
            required_int(fields, "nodes"),
            required_int(fields, "ranks"),
            required_int(fields, "ppn"),
        )
        if topology != (expected_nodes, expected_ranks, expected_ppn):
            raise MonitorError(f"RESULT topology mismatch at {size} bytes")
    if done.get("plan") != expected_label:
        raise MonitorError("COLLECTIVE_DONE label mismatch")
    if required_int(done, "total_errors") != 0:
        raise MonitorError("COLLECTIVE_DONE reports correctness errors")
    return {
        "config": actual_config,
        "results": {
            str(size): float(fields["median_us"])
            for size, fields in sorted(results.items())
        },
        "total_errors": 0,
    }


def verify_artifacts(items: list[tuple[Path, str]]) -> list[dict[str, str]]:
    verified = []
    for path, expected in items:
        if not path.is_file():
            raise MonitorError(f"artifact does not exist: {path}")
        actual = sha256(path)
        if actual != expected:
            raise MonitorError(
                f"artifact hash mismatch for {path}: {actual}, expected {expected}"
            )
        verified.append({"path": str(path.resolve()), "sha256": actual})
    return verified


def summarize_stderr(path: Path, tail_lines: int = 40) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "bytes": 0, "tail": [], "hdir_last_by_rank": {}}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    last_by_rank: dict[str, dict[str, Any]] = {}
    for line in lines:
        match = HDIR_CHECKPOINT_RE.match(line)
        if not match:
            continue
        last_by_rank[match.group("rank")] = {
            "call": int(match.group("call")),
            "bytes": int(match.group("bytes")),
            "stage": match.group("stage"),
        }
    return {
        "exists": True,
        "bytes": path.stat().st_size,
        "line_count": len(lines),
        "tail": lines[-tail_lines:],
        "hdir_last_by_rank": {
            rank: last_by_rank[rank]
            for rank in sorted(last_by_rank, key=int)
        },
    }


def parse_artifact(value: str) -> tuple[Path, str]:
    try:
        name, expected = value.rsplit("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected PATH=SHA256") from exc
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise argparse.ArgumentTypeError("artifact SHA256 must be 64 lowercase hex digits")
    return Path(name), expected


def summarize_jobspec(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise MonitorError("jobspec is not an object")
    attributes = value.get("attributes", {})
    system = attributes.get("system", {}) if isinstance(attributes, dict) else {}
    if not isinstance(system, dict):
        raise MonitorError("jobspec system attributes are not an object")
    queue = system.get("queue")
    if queue != "pdebug":
        raise MonitorError(f"job queue is {queue!r}, expected 'pdebug'")
    environment = system.get("environment", {})
    if not isinstance(environment, dict):
        raise MonitorError("jobspec environment is not an object")
    tasks = value.get("tasks", [])
    command = None
    if isinstance(tasks, list) and tasks and isinstance(tasks[0], dict):
        command = tasks[0].get("command")
    embedded_script_sha256 = None
    files = system.get("files")
    if isinstance(files, dict):
        script = files.get("script")
        script_data = script.get("data") if isinstance(script, dict) else None
        if isinstance(script_data, str):
            embedded_script_sha256 = hashlib.sha256(
                script_data.encode()
            ).hexdigest()
    return {
        "queue": queue,
        "cwd": system.get("cwd"),
        "duration_seconds": system.get("duration"),
        "command": command,
        "embedded_script_sha256": embedded_script_sha256,
        "resources": value.get("resources"),
        "environment": {
            key: environment[key]
            for key in sorted(environment)
            if SAFE_ENV_RE.fullmatch(key)
        },
    }


def summarize_resources(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("version") != 1:
        raise MonitorError("resource set is not Flux R version 1")
    execution = value.get("execution")
    if not isinstance(execution, dict):
        raise MonitorError("resource set lacks execution details")
    nodelist = execution.get("nodelist")
    if (not isinstance(nodelist, list) or not nodelist
            or any(not isinstance(item, str) or not item for item in nodelist)):
        raise MonitorError("resource set lacks a node list")
    properties = execution.get("properties")
    if not isinstance(properties, dict) or "pdebug" not in properties:
        raise MonitorError("resource set is not a pdebug allocation")
    lite = execution.get("R_lite")
    if not isinstance(lite, list) or not lite:
        raise MonitorError("resource set lacks R_lite")
    return {
        "nodelist": nodelist,
        "ranks": [item.get("rank") for item in lite],
        "starttime": execution.get("starttime"),
        "expiration": execution.get("expiration"),
        "pdebug_ranks": properties["pdebug"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--stdout", type=Path, required=True)
    parser.add_argument("--stderr", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--expected-label", required=True)
    parser.add_argument("--expected-size", type=int, action="append", required=True)
    parser.add_argument("--expected-nodes", type=int, default=2)
    parser.add_argument("--expected-ranks", type=int, default=16)
    parser.add_argument("--expected-ppn", type=int, default=8)
    parser.add_argument("--expected-runs", type=int, required=True)
    parser.add_argument("--expected-warmup", type=int, required=True)
    parser.add_argument(
        "--artifact", type=parse_artifact, action="append", default=[],
        metavar="PATH=SHA256",
    )
    args = parser.parse_args()

    state: dict[str, Any] = {
        "schema_version": "gicc-collective-job-monitor-v1",
        "job_id": args.job_id,
        "monitor_started_at": utc_now(),
        "state": "monitoring",
        "expected": {
            "label": args.expected_label,
            "sizes": sorted(args.expected_size),
            "nodes": args.expected_nodes,
            "ranks": args.expected_ranks,
            "ppn": args.expected_ppn,
            "runs": args.expected_runs,
            "warmup": args.expected_warmup,
        },
        "paths": {
            "stdout": str(args.stdout.resolve()),
            "stderr": str(args.stderr.resolve()),
            "status": str(args.status.resolve()),
        },
    }
    atomic_write_json(args.status, state)

    try:
        state["artifacts"] = verify_artifacts(args.artifact)
        run_flux("job", "wait-event", args.job_id, "clean")
        eventlog_text = run_flux("job", "info", args.job_id, "eventlog")
        jobspec_text = run_flux("job", "info", "-o", args.job_id, "jobspec")
        resources_text = run_flux("job", "info", args.job_id, "R")
        events = parse_json_lines(eventlog_text, "eventlog")
        scheduler = scheduler_result(events)
        state["scheduler"] = scheduler
        state["jobspec"] = summarize_jobspec(json.loads(jobspec_text))
        state["resource_set"] = summarize_resources(json.loads(resources_text))
        state["stderr_summary"] = summarize_stderr(args.stderr)
        state["stderr_bytes"] = state["stderr_summary"]["bytes"]
        state["stdout_bytes"] = args.stdout.stat().st_size if args.stdout.exists() else 0

        if scheduler["exit_code"] != 0 or scheduler["exception_types"]:
            exception_text = ",".join(scheduler["exception_types"]) or "none"
            raise MonitorError(
                f"scheduler failure: exit_code={scheduler['exit_code']} "
                f"exceptions={exception_text}"
            )
        if not args.stdout.is_file():
            raise MonitorError(f"stdout log does not exist: {args.stdout}")
        state["benchmark"] = validate_output(
            args.stdout,
            args.expected_label,
            args.expected_size,
            args.expected_nodes,
            args.expected_ranks,
            args.expected_ppn,
            args.expected_runs,
            args.expected_warmup,
        )
        state["state"] = "passed"
        return_code = 0
    except (MonitorError, OSError, json.JSONDecodeError) as exc:
        state["state"] = "failed"
        state["error"] = str(exc)
        return_code = 1
    finally:
        state["monitor_finished_at"] = utc_now()
        atomic_write_json(args.status, state)
    return return_code


if __name__ == "__main__":
    sys.exit(main())
