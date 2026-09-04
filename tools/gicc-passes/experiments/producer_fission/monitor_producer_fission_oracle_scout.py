#!/usr/bin/env python3
"""Wait for and audit one paired producer-fission pdebug allocation."""

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


CONFIG_RE = re.compile(
    r"^GICC/OFI Jacobi \(unified launch\): (?P<ranks>[0-9]+) ranks, "
    r"mesh (?P<ny>[0-9]+) x (?P<nx>[0-9]+), chunk (?P<chunk>[0-9]+), "
    r"buf (?P<bytes>[0-9]+) bytes, (?P<configured>[0-9]+) iters$"
)
RESULT_RE = re.compile(
    r"^Done: (?P<iterations>[0-9]+) iters in "
    r"(?P<seconds>[0-9]+(?:\.[0-9]+)?) s, final l2="
    r"(?P<norm>[-+0-9.eE]+)$"
)


def parse_run(path: Path, expected_size: int) -> dict[str, Any]:
    if not path.is_file():
        raise common.MonitorError(f"missing Jacobi output: {path}")
    config = []
    results = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := CONFIG_RE.match(line):
            config.append(match.groupdict())
        if match := RESULT_RE.match(line):
            results.append(match.groupdict())
    if len(config) != 1 or len(results) != 1:
        raise common.MonitorError(
            f"{path} has {len(config)} configs and {len(results)} results"
        )
    cfg = config[0]
    result = results[0]
    ranks = int(cfg["ranks"])
    nx = int(cfg["nx"])
    ny = int(cfg["ny"])
    chunk = int(cfg["chunk"])
    configured = int(cfg["configured"])
    expected_chunk = (expected_size - 2) // 16
    expected_ny = expected_chunk * 16 + 2
    if (ranks, nx, ny, chunk, configured) != (
        16, expected_size, expected_ny, expected_chunk, 200
    ):
        raise common.MonitorError(
            f"unexpected Jacobi config in {path}: "
            f"ranks={ranks} nx={nx} ny={ny} chunk={chunk} iters={configured}"
        )
    iterations = int(result["iterations"])
    seconds = float(result["seconds"])
    norm = float(result["norm"])
    if not (0 < iterations <= 200):
        raise common.MonitorError(f"invalid iteration count in {path}")
    if not math.isfinite(seconds) or seconds <= 0:
        raise common.MonitorError(f"invalid runtime in {path}")
    if not math.isfinite(norm) or norm < 0:
        raise common.MonitorError(f"invalid final norm in {path}")
    return {
        "configured_iterations": configured,
        "iterations": iterations,
        "seconds": seconds,
        "final_l2": norm,
        "requested_size": expected_size,
        "actual_nx": nx,
        "actual_ny": ny,
        "chunk": chunk,
        "stdout": {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": common.sha256(path),
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
        "schema_version": "gicc-producer-fission-oracle-monitor-v1",
        "job_id": args.job_id,
        "monitor_started_at": common.utc_now(),
        "state": "monitoring",
        "expected": {
            "queue": "pdebug",
            "nodes": 2,
            "ranks": 16,
            "ppn": 8,
            "replicates": 4,
            "sizes": [1024, 4096],
            "iterations": 200,
            "nccheck": 10,
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
        state["resource_set"] = common.summarize_resources(
            json.loads(resources)
        )
        if scheduler["exit_code"] != 0 or scheduler["exception_types"]:
            raise common.MonitorError(
                "scheduler failure: "
                f"exit_code={scheduler['exit_code']} "
                f"exceptions={scheduler['exception_types']}"
            )

        runs: dict[str, Any] = {}
        for replicate in range(1, 5):
            rep: dict[str, Any] = {}
            for size in (1024, 4096):
                pair: dict[str, Any] = {}
                for arm in ("baseline", "fission"):
                    run_dir = args.output_dir / f"rep{replicate}" / f"size{size}"
                    parsed = parse_run(run_dir / f"{arm}.out", size)
                    stderr = run_dir / f"{arm}.err"
                    parsed["stderr"] = {
                        **common.summarize_stderr(stderr),
                        "path": str(stderr.resolve()),
                        "sha256": common.sha256(stderr),
                    }
                    pair[arm] = parsed
                left = pair["baseline"]
                right = pair["fission"]
                if left["iterations"] != right["iterations"]:
                    raise common.MonitorError(
                        f"iteration mismatch in replicate {replicate}, size {size}"
                    )
                if not math.isclose(
                    left["final_l2"], right["final_l2"],
                    rel_tol=1e-6, abs_tol=1e-7,
                ):
                    raise common.MonitorError(
                        f"norm mismatch in replicate {replicate}, size {size}: "
                        f"{left['final_l2']} vs {right['final_l2']}"
                    )
                pair["speedup"] = left["seconds"] / right["seconds"]
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
    sys.exit(main())
