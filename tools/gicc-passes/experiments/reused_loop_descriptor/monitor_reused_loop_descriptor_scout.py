#!/usr/bin/env python3
"""Wait for and strictly audit one reused-loop-descriptor pdebug scout."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


COLLECTIVE_DIR = Path(__file__).resolve().parent.parent / "collective"
sys.path.insert(0, str(COLLECTIVE_DIR))
import monitor_compiler_collective_job as common  # noqa: E402


BATCHES = (4, 64)
REPLICATES = tuple(range(1, 7))
ARMS = ("baseline", "reused")
SIZES = (
    "1B", "2B", "4B", "8B", "64B", "256B", "1KB", "4KB",
    "16KB", "64KB", "256KB", "512KB", "1MB", "2MB", "4MB",
    "16MB",
)
ROW_RE = re.compile(
    r"^(1B|2B|4B|8B|64B|256B|1KB|4KB|16KB|64KB|256KB|512KB|"
    r"1MB|2MB|4MB|16MB)\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+"
    r"([0-9]+(?:\.[0-9]+)?)$",
    re.MULTILINE,
)
AUDIT_RE = re.compile(
    r"\[enqueue-audit\] mono_total_ops=(\d+)\s+expected=(\d+)\s+match=(\w+)"
)


def idset_count(value: Any) -> int:
    """Count ranks in the simple comma/range syntax emitted by Flux R."""
    if not isinstance(value, str) or not value:
        raise common.MonitorError("pdebug rank set is missing")
    ranks: set[int] = set()
    for component in value.split(","):
        if re.fullmatch(r"\d+", component):
            ranks.add(int(component))
            continue
        match = re.fullmatch(r"(\d+)-(\d+)", component)
        if not match or int(match.group(1)) > int(match.group(2)):
            raise common.MonitorError(f"invalid Flux rank set: {value!r}")
        ranks.update(range(int(match.group(1)), int(match.group(2)) + 1))
    return len(ranks)


def validate_allocation_shape(
    jobspec: dict[str, Any], resource_set: dict[str, Any]
) -> None:
    """Require the frozen N2/n2, one-rank-per-node allocation shape."""
    resources = jobspec.get("resources")
    if not isinstance(resources, list) or len(resources) != 1:
        raise common.MonitorError("jobspec does not contain one node resource")
    node = resources[0]
    if not isinstance(node, dict) or node.get("type") != "node":
        raise common.MonitorError("jobspec top-level resource is not a node")
    if node.get("count") != 2:
        raise common.MonitorError("scout jobspec is not N2")
    children = node.get("with")
    if not isinstance(children, list) or len(children) != 1:
        raise common.MonitorError("scout jobspec lacks one task slot per node")
    slot = children[0]
    if (
        not isinstance(slot, dict)
        or slot.get("type") != "slot"
        or slot.get("label") != "task"
        or slot.get("count") != 1
    ):
        raise common.MonitorError("scout jobspec is not n2/ppn1")
    slot_resources = slot.get("with")
    if not isinstance(slot_resources, list):
        raise common.MonitorError("scout task slot has no resources")
    requested = {
        item.get("type"): item.get("count")
        for item in slot_resources
        if isinstance(item, dict)
    }
    if requested.get("core") != 64 or requested.get("gpu") != 1:
        raise common.MonitorError("scout task slot is not c64/g1")
    if idset_count(resource_set.get("pdebug_ranks")) != 2:
        raise common.MonitorError("resource set does not contain two pdebug nodes")


def parse_run(stdout: Path, stderr: Path, batch: int) -> dict[str, Any]:
    if not stdout.is_file() or not stderr.is_file():
        raise common.MonitorError(f"missing loop scout logs: {stdout}, {stderr}")
    stdout_data = stdout.read_bytes()
    stderr_data = stderr.read_bytes()
    text = stdout_data.decode("utf-8", errors="replace")
    err = stderr_data.decode("utf-8", errors="replace")
    failure_markers = (
        "VERIFY-FAIL", "FATAL", "FAILED", "job.exception", "timed out",
    )
    if any(marker in text or marker in err for marker in failure_markers):
        raise common.MonitorError(f"failure marker in {stdout} or {stderr}")
    if f"batch={batch}" not in text or "LTO-generated host trace" not in text:
        raise common.MonitorError(f"batch/header mismatch in {stdout}")

    matches = ROW_RE.findall(text)
    if len(matches) != len(SIZES) or tuple(row[0] for row in matches) != SIZES:
        raise common.MonitorError(
            f"expected exactly the canonical 16 rows in {stdout}"
        )
    rows: dict[str, Any] = {}
    for size, printed_iters, mean, median in matches:
        if int(printed_iters) != 21 * batch:
            raise common.MonitorError(f"iteration count mismatch in {stdout}")
        mean_value = float(mean)
        median_value = float(median)
        if (
            not math.isfinite(mean_value) or mean_value <= 0
            or not math.isfinite(median_value) or median_value <= 0
        ):
            raise common.MonitorError(f"invalid timing in {stdout}")
        rows[size] = {
            "printed_iters_total": int(printed_iters),
            "mean_us_per_message": mean_value,
            "median_us_per_message": median_value,
        }
    audit = AUDIT_RE.search(text)
    expected = 31 * batch * len(SIZES)
    if (
        not audit
        or int(audit.group(1)) != expected
        or int(audit.group(2)) != expected
        or audit.group(3) != "YES"
    ):
        raise common.MonitorError(f"enqueue audit failed in {stdout}")
    return {
        "batch": batch,
        "rows": rows,
        "enqueue_actual": expected,
        "enqueue_expected": expected,
        "stdout": {
            "path": str(stdout.resolve()),
            "bytes": len(stdout_data),
            "sha256": common.sha256(stdout),
        },
        "stderr": {
            **common.summarize_stderr(stderr),
            "path": str(stderr.resolve()),
            "sha256": common.sha256(stderr),
        },
    }


def geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise common.MonitorError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--status", required=True, type=Path)
    parser.add_argument(
        "--artifact", type=common.parse_artifact, action="append", default=[],
        metavar="PATH=SHA256",
    )
    args = parser.parse_args()

    state: dict[str, Any] = {
        "schema_version": "gicc-reused-loop-descriptor-monitor-v1",
        "job_id": args.job_id,
        "monitor_started_at": common.utc_now(),
        "state": "monitoring",
        "expected": {
            "queue": "pdebug", "nodes": 2, "ranks": 2, "ppn": 1,
            "replicates": 6, "batches": list(BATCHES),
            "message_sizes": list(SIZES), "warmup": 10, "measured": 21,
        },
    }
    common.atomic_write_json(args.status, state)
    result_code = 1
    try:
        state["artifacts"] = common.verify_artifacts(args.artifact)
        common.run_flux("job", "wait-event", args.job_id, "clean")
        eventlog = common.run_flux("job", "info", args.job_id, "eventlog")
        jobspec = common.run_flux("job", "info", "-o", args.job_id, "jobspec")
        resources = common.run_flux("job", "info", args.job_id, "R")
        scheduler = common.scheduler_result(
            common.parse_json_lines(eventlog, "eventlog")
        )
        state["scheduler"] = scheduler
        state["jobspec"] = common.summarize_jobspec(json.loads(jobspec))
        state["resource_set"] = common.summarize_resources(json.loads(resources))
        if scheduler["exit_code"] != 0 or scheduler["exception_types"]:
            raise common.MonitorError(
                "scheduler failure: "
                f"exit_code={scheduler['exit_code']} "
                f"exceptions={scheduler['exception_types']}"
            )
        if state["jobspec"].get("queue") != "pdebug":
            raise common.MonitorError("scout did not use pdebug")
        validate_allocation_shape(state["jobspec"], state["resource_set"])

        runs: dict[str, Any] = {}
        for replicate in REPLICATES:
            rep: dict[str, Any] = {}
            for batch in BATCHES:
                run_dir = args.output_dir / f"rep{replicate}" / f"batch{batch}"
                pair = {
                    arm: parse_run(
                        run_dir / f"{arm}.out", run_dir / f"{arm}.err", batch
                    )
                    for arm in ARMS
                }
                speedups = {
                    size: (
                        pair["baseline"]["rows"][size]["median_us_per_message"]
                        / pair["reused"]["rows"][size]["median_us_per_message"]
                    )
                    for size in SIZES
                }
                pair["per_size_speedup"] = speedups
                pair["all_size_geomean_speedup"] = geometric_mean(
                    list(speedups.values())
                )
                rep[str(batch)] = pair
            runs[str(replicate)] = rep
        state["runs"] = runs
        state["state"] = "passed"
        result_code = 0
    except (common.MonitorError, OSError, json.JSONDecodeError) as exc:
        state["state"] = "failed"
        state["error"] = str(exc)
    finally:
        state["monitor_finished_at"] = common.utc_now()
        common.atomic_write_json(args.status, state)
    return result_code


if __name__ == "__main__":
    raise SystemExit(main())
