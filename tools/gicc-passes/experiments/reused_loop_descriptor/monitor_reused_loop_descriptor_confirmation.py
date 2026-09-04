#!/usr/bin/env python3
"""Audit one independent reused-loop-descriptor confirmation job."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import monitor_reused_loop_descriptor_scout as scout_monitor


common = scout_monitor.common
BATCHES = tuple(scout_monitor.BATCHES)
SIZES = tuple(scout_monitor.SIZES)
ARMS = ("baseline", "reused")


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
        "schema_version": (
            "gicc-reused-loop-descriptor-confirmation-monitor-v1"
        ),
        "job_id": args.job_id,
        "allocation": args.allocation,
        "monitor_started_at": common.utc_now(),
        "state": "monitoring",
        "expected": {
            "queue": "pdebug", "nodes": 2, "ranks": 2, "ppn": 1,
            "blocks": 2, "orders": ["AB", "BA"],
            "batches": list(BATCHES), "message_sizes": list(SIZES),
            "warmup": 10, "measured": 21,
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
            "path": str(driver.resolve()), "bytes": driver.stat().st_size,
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
        if state["jobspec"].get("queue") != "pdebug":
            raise common.MonitorError("confirmation did not use pdebug")
        scout_monitor.validate_allocation_shape(
            state["jobspec"], state["resource_set"],
        )

        blocks: dict[str, Any] = {}
        for block in (1, 2):
            block_value: dict[str, Any] = {}
            for batch in BATCHES:
                run_dir = allocation_dir / f"block{block}" / f"batch{batch}"
                pair = {
                    arm: scout_monitor.parse_run(
                        run_dir / f"{arm}.out", run_dir / f"{arm}.err", batch,
                    )
                    for arm in ARMS
                }
                speedups = {
                    size: (
                        pair["baseline"]["rows"][size][
                            "median_us_per_message"
                        ]
                        / pair["reused"]["rows"][size][
                            "median_us_per_message"
                        ]
                    )
                    for size in SIZES
                }
                pair["per_size_speedup"] = speedups
                pair["all_size_geomean_speedup"] = (
                    scout_monitor.geometric_mean(list(speedups.values()))
                )
                pair["correctness"] = "exact_enqueue_count_both_arms"
                block_value[str(batch)] = pair
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
