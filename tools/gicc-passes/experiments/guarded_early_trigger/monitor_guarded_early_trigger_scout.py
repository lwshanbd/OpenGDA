#!/usr/bin/env python3
"""Wait for and audit one paired guarded-early-trigger pdebug allocation."""

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


RUN_RE = re.compile(
    r"^Run (?P<run>[0-9]+): (?P<usec>[0-9]+(?:\.[0-9]+)?) us"
    r"(?P<warmup> \(warmup\))?$"
)
AVERAGE_RE = re.compile(
    r"^gicc::launch average \(runs 2-9\): (?P<usec>[0-9]+(?:\.[0-9]+)?) us$"
)
CHECKSUM_RE = re.compile(
    r"^GICC_MM_CHECKSUM rank=(?P<rank>-?[0-9]+) "
    r"bytes=(?P<bytes>[0-9]+) fnv64=(?P<hash>[0-9a-f]{16})$"
)


def parse_run(stdout: Path, stderr: Path, size: int) -> dict[str, Any]:
    if not stdout.is_file() or not stderr.is_file():
        raise common.MonitorError(f"missing mm_minimal logs: {stdout}, {stderr}")
    timings: dict[int, float] = {}
    averages = []
    for line in stdout.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := RUN_RE.match(line):
            run = int(match.group("run"))
            if run in timings:
                raise common.MonitorError(f"duplicate Run {run} in {stdout}")
            if bool(match.group("warmup")) != (run < 2):
                raise common.MonitorError(f"warmup label mismatch in {stdout}")
            timings[run] = float(match.group("usec"))
        if match := AVERAGE_RE.match(line):
            averages.append(float(match.group("usec")))
    if sorted(timings) != list(range(10)) or len(averages) != 1:
        raise common.MonitorError(
            f"{stdout} has {len(timings)} run timings and {len(averages)} averages"
        )
    if any(not math.isfinite(value) or value <= 0 for value in timings.values()):
        raise common.MonitorError(f"invalid timing in {stdout}")
    measured_mean = sum(timings[index] for index in range(2, 10)) / 8
    if not math.isclose(averages[0], measured_mean, rel_tol=5e-5, abs_tol=0.5):
        raise common.MonitorError(
            f"reported average does not match runs 2-9 in {stdout}"
        )

    expected_bytes = size * (size // 16) * 4
    checksums: dict[int, str] = {}
    for line in stderr.read_text(encoding="utf-8", errors="replace").splitlines():
        match = CHECKSUM_RE.match(line)
        if not match or int(match.group("bytes")) != expected_bytes:
            continue
        rank = int(match.group("rank"))
        if rank in checksums:
            raise common.MonitorError(f"duplicate checksum for rank {rank} in {stderr}")
        checksums[rank] = match.group("hash")
    if sorted(checksums) != list(range(16)):
        raise common.MonitorError(
            f"{stderr} checksum ranks {sorted(checksums)}, expected 0..15"
        )
    return {
        "requested_size": size,
        "result_bytes_per_rank": expected_bytes,
        "run_us": [timings[index] for index in range(10)],
        "measured_mean_us": averages[0],
        "checksums": {str(rank): checksums[rank] for rank in sorted(checksums)},
        "stdout": {
            "path": str(stdout.resolve()),
            "bytes": stdout.stat().st_size,
            "sha256": common.sha256(stdout),
        },
        "stderr": {
            **common.summarize_stderr(stderr),
            "path": str(stderr.resolve()),
            "sha256": common.sha256(stderr),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument(
        "--artifact", type=common.parse_artifact, action="append", default=[],
        metavar="PATH=SHA256",
    )
    args = parser.parse_args()

    state: dict[str, Any] = {
        "schema_version": "gicc-guarded-early-trigger-monitor-v1",
        "job_id": args.job_id,
        "monitor_started_at": common.utc_now(),
        "state": "monitoring",
        "expected": {
            "queue": "pdebug", "nodes": 2, "ranks": 16, "ppn": 8,
            "replicates": 4, "sizes": [4096, 8192],
            "application_runs": 10, "application_warmup": 2,
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

        runs: dict[str, Any] = {}
        for replicate in range(1, 5):
            rep: dict[str, Any] = {}
            for size in (4096, 8192):
                run_dir = args.output_dir / f"rep{replicate}" / f"size{size}"
                pair = {
                    arm: parse_run(
                        run_dir / f"{arm}.out", run_dir / f"{arm}.err", size
                    )
                    for arm in ("baseline", "guarded")
                }
                if pair["baseline"]["checksums"] != pair["guarded"]["checksums"]:
                    raise common.MonitorError(
                        f"checksum mismatch in replicate {replicate}, size {size}"
                    )
                pair["speedup"] = (
                    pair["baseline"]["measured_mean_us"] /
                    pair["guarded"]["measured_mean_us"]
                )
                pair["correctness"] = "all_16_rank_checksums_equal"
                rep[str(size)] = pair
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
