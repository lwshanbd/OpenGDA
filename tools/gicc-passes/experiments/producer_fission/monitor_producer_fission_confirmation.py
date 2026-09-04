#!/usr/bin/env python3
"""Wait for and audit one independent producer-fission confirmation job."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import monitor_producer_fission_oracle_scout as scout_monitor


common = scout_monitor.common


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--allocation", type=int, choices=(1, 2, 3), required=True)
    parser.add_argument(
        "--artifact", type=common.parse_artifact, action="append", default=[],
        metavar="PATH=SHA256",
    )
    args = parser.parse_args()

    allocation_dir = args.output_dir / f"allocation{args.allocation}"
    state: dict[str, Any] = {
        "schema_version": "gicc-producer-fission-confirmation-monitor-v1",
        "job_id": args.job_id,
        "allocation": args.allocation,
        "monitor_started_at": common.utc_now(),
        "state": "monitoring",
        "expected": {
            "queue": "pdebug",
            "nodes": 2,
            "ranks": 16,
            "ppn": 8,
            "blocks": 2,
            "orders": ["AB", "BA"],
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
        driver = allocation_dir / "driver.out"
        driver_err = allocation_dir / "driver.err"
        state["driver_stdout"] = {
            "path": str(driver.resolve()),
            "bytes": driver.stat().st_size,
            "sha256": common.sha256(driver),
        }
        state["driver_stderr"] = {
            **common.summarize_stderr(driver_err),
            "path": str(driver_err.resolve()),
            "sha256": common.sha256(driver_err),
        }
        if scheduler["exit_code"] != 0 or scheduler["exception_types"]:
            raise common.MonitorError(
                "scheduler failure: "
                f"exit_code={scheduler['exit_code']} "
                f"exceptions={scheduler['exception_types']}"
            )

        blocks: dict[str, Any] = {}
        for block in (1, 2):
            block_value: dict[str, Any] = {}
            for size in (1024, 4096):
                pair: dict[str, Any] = {}
                run_dir = allocation_dir / f"block{block}" / f"size{size}"
                for arm in ("baseline", "fission"):
                    parsed = scout_monitor.parse_run(
                        run_dir / f"{arm}.out", size,
                    )
                    stderr = run_dir / f"{arm}.err"
                    parsed["stderr"] = {
                        **common.summarize_stderr(stderr),
                        "path": str(stderr.resolve()),
                        "sha256": common.sha256(stderr),
                    }
                    pair[arm] = parsed
                baseline = pair["baseline"]
                fission = pair["fission"]
                if baseline["iterations"] != fission["iterations"]:
                    raise common.MonitorError(
                        f"iteration mismatch in block {block}, size {size}"
                    )
                if not math.isclose(
                    baseline["final_l2"], fission["final_l2"],
                    rel_tol=1e-6, abs_tol=1e-7,
                ):
                    raise common.MonitorError(
                        f"norm mismatch in block {block}, size {size}: "
                        f"{baseline['final_l2']} vs {fission['final_l2']}"
                    )
                pair["speedup"] = baseline["seconds"] / fission["seconds"]
                block_value[str(size)] = pair
            blocks[str(block)] = block_value
        state["blocks"] = blocks
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
