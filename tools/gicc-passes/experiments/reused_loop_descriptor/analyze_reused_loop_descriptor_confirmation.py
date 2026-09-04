#!/usr/bin/env python3
"""Audit three independent reused-loop-descriptor confirmation allocations."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any

import monitor_reused_loop_descriptor_scout as scout_monitor
import prepare_reused_loop_descriptor_confirmation as transition_base


BATCHES = tuple(transition_base.BATCHES)
SIZES = tuple(transition_base.SIZES)
SMALL_SIZES = tuple(transition_base.SMALL_SIZES)
ALLOCATIONS = (1, 2, 3)
BLOCKS = (1, 2)
ORDERS = {1: ("baseline", "reused"), 2: ("reused", "baseline")}
MONITOR_SCHEMA = "gicc-reused-loop-descriptor-confirmation-monitor-v1"
ConfirmError = transition_base.TransitionError
common = scout_monitor.common
sha256 = transition_base.sha256_file
fingerprint = transition_base.bridge._fingerprint


def geomean(values: list[float]) -> float:
    try:
        return scout_monitor.geometric_mean(values)
    except common.MonitorError as exc:
        raise ConfirmError(str(exc)) from exc


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def allocation_bootstrap(speedups: list[float]) -> dict[str, Any]:
    if len(speedups) != 3 or any(
            value <= 0 or not math.isfinite(value) for value in speedups):
        raise ConfirmError("confirmation requires three allocation speedups")
    samples = [
        geomean([speedups[index] for index in sample])
        for sample in itertools.product(range(3), repeat=3)
    ]
    return {
        "estimate": geomean(speedups),
        "allocation_speedups": speedups,
        "allocation_wins": sum(value > 1.0 for value in speedups),
        "method": (
            "exact 3-out-of-3 paired bootstrap clustered by independent "
            "pdebug allocation"
        ),
        "bootstrap_samples": len(samples),
        "lower_2_5_percent": percentile(samples, 0.025),
        "upper_97_5_percent": percentile(samples, 0.975),
    }


def analyze_speedups(
    rows: dict[int, dict[int, dict[int, dict[str, float]]]],
) -> dict[str, Any]:
    if set(rows) != set(ALLOCATIONS):
        raise ConfirmError("confirmation requires allocations 1, 2, and 3")
    allocation_values = []
    per_allocation = {}
    batch_clusters = {batch: [] for batch in BATCHES}
    for allocation in ALLOCATIONS:
        blocks = rows[allocation]
        if set(blocks) != set(BLOCKS):
            raise ConfirmError(f"allocation {allocation} requires AB and BA")
        allocation_cells = []
        for block in BLOCKS:
            batches = blocks[block]
            if set(batches) != set(BATCHES):
                raise ConfirmError("confirmation batch coverage changed")
            for batch in BATCHES:
                size_values = batches[batch]
                if set(size_values) != set(SMALL_SIZES):
                    raise ConfirmError("confirmation target sizes changed")
                values = [float(size_values[size]) for size in SMALL_SIZES]
                if any(value <= 0 or not math.isfinite(value)
                       for value in values):
                    raise ConfirmError("confirmation speedup is invalid")
                allocation_cells.extend(values)
                batch_clusters[batch].append(geomean(values))
        aggregate = geomean(allocation_cells)
        allocation_values.append(aggregate)
        per_allocation[str(allocation)] = {
            "target_speedup_cells": allocation_cells,
            "geometric_mean_speedup": aggregate,
            "win": aggregate > 1.0,
        }
    primary = allocation_bootstrap(allocation_values)
    per_batch = {}
    for batch in BATCHES:
        values = batch_clusters[batch]
        median = statistics.median(values)
        per_batch[str(batch)] = {
            "block_allocation_geomeans": values,
            "median_speedup": median,
            "wins": sum(value > 1.0 for value in values),
            "no_regression_gate_passed": median >= 0.98,
        }
    criteria = {
        "aggregate_point_estimate": {
            "observed": primary["estimate"], "threshold": 1.01,
            "comparison": "greater_than_or_equal",
            "passed": primary["estimate"] >= 1.01,
        },
        "allocation_cluster_bootstrap_lower_95": {
            "observed": primary["lower_2_5_percent"], "threshold": 1.0,
            "comparison": "strictly_greater",
            "passed": primary["lower_2_5_percent"] > 1.0,
        },
        "allocation_level_wins": {
            "observed": primary["allocation_wins"], "threshold": 2,
            "comparison": "greater_than_or_equal",
            "passed": primary["allocation_wins"] >= 2,
        },
        "both_batch_no_regression_medians": {
            "observed": {
                str(batch): per_batch[str(batch)]["median_speedup"]
                for batch in BATCHES
            },
            "threshold": 0.98,
            "comparison": "each_greater_than_or_equal",
            "passed": all(
                value["no_regression_gate_passed"]
                for value in per_batch.values()
            ),
        },
        "all_enqueue_and_ir_shape_audits": {
            "observed": 24, "threshold": 24, "passed": True,
        },
    }
    return {
        "per_allocation": per_allocation,
        "per_batch": per_batch,
        "primary_speedup": primary,
        "correctness_gate": {
            "passed": True,
            "paired_batch_blocks": 12,
            "network_operation_count_audits": 24,
        },
        "confirmation_gate": {
            "passed": all(item["passed"] for item in criteria.values()),
            "criteria": criteria,
        },
    }


def _transition_paths(value: dict[str, Any]) -> dict[str, Path]:
    paths = {}
    for record in value["files"]:
        raw = Path(record["path"])
        paths[record["role"]] = (
            raw.resolve() if raw.is_absolute()
            else (transition_base.ROOT / raw).resolve()
        )
    return paths


def validate_transition(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    value = transition_base.verify_contained_report(path)
    expected_boundary = {
        "compiler_lto_decisions_only": True,
        "application_source_hash_verified": True,
        "application_source_visible_to_model": False,
        "application_source_modified": False,
        "model_invoked": False,
        "provider_call_authorized": False,
        "scheduler_job_submitted": False,
        "candidate_model_visible_before_confirmation": False,
    }
    if value.get("boundary") != expected_boundary:
        raise ConfirmError("reused-loop compiler-only boundary changed")
    expected = {
        "queue": "pdebug", "nodes": 2, "ranks": 2,
        "ranks_per_node": 1, "cpu_cores_per_rank": 64,
        "gpus_per_rank": 1, "independent_allocations": 3,
        "maximum_active_or_queued_jobs": 1,
        "balanced_blocks_per_allocation": ["AB", "BA"],
        "batches": list(BATCHES), "all_message_sizes": list(SIZES),
        "target_message_sizes": list(SMALL_SIZES),
        "warmup_iterations": 10, "measured_iterations": 21,
        "scout_or_confirmation_labels_visible_to_model": False,
    }
    contract = value.get("confirmation_contract")
    if not isinstance(contract, dict) or any(
            contract.get(key) != wanted for key, wanted in expected.items()):
        raise ConfirmError("reused-loop confirmation contract changed")
    return value, _transition_paths(value)


def _artifact_records(value: dict[str, Any]) -> dict[Path, str]:
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise ConfirmError("confirmation monitor lacks frozen artifacts")
    records: dict[Path, str] = {}
    for record in artifacts:
        if (not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ConfirmError("confirmation monitor has invalid artifact")
        path = Path(record["path"]).resolve()
        if (path in records or not path.is_file()
                or sha256(path) != record["sha256"]):
            raise ConfirmError(f"confirmation artifact changed: {path}")
        records[path] = record["sha256"]
    return records


def _expected_driver_lines(allocation: int) -> list[str]:
    lines = [
        "REUSED_DESCRIPTOR_CONFIRM_CONFIG "
        f"allocation={allocation} nodes=2 ranks=2 ppn=1 cores=64 "
        "batches=4,64 sizes=canonical16 blocks=AB,BA",
    ]
    for block in BLOCKS:
        order = ORDERS[block]
        lines.append(
            "REUSED_DESCRIPTOR_CONFIRM_BLOCK_START "
            f"allocation={allocation} block={block} order={' '.join(order)}"
        )
        for batch in BATCHES:
            for arm in order:
                lines.extend([
                    "REUSED_DESCRIPTOR_CONFIRM_RUN_START "
                    f"allocation={allocation} block={block} batch={batch} "
                    f"arm={arm}",
                    "REUSED_DESCRIPTOR_CONFIRM_RUN_DONE "
                    f"allocation={allocation} block={block} batch={batch} "
                    f"arm={arm}",
                ])
        lines.append(
            "REUSED_DESCRIPTOR_CONFIRM_BLOCK_DONE "
            f"allocation={allocation} block={block}"
        )
    lines.append(
        f"REUSED_DESCRIPTOR_CONFIRM_DONE allocation={allocation} "
        "pairs=4 runs=8"
    )
    return lines


def _validate_monitor(
    path: Path, transition_path: Path, transition_files: dict[str, Path],
) -> tuple[int, dict[int, dict[int, dict[str, float]]], dict[str, Any]]:
    value = transition_base.read_json(path)
    if (not isinstance(value, dict)
            or value.get("schema_version") != MONITOR_SCHEMA
            or value.get("state") != "passed"):
        raise ConfirmError(f"confirmation monitor did not pass: {path}")
    allocation = value.get("allocation")
    if allocation not in ALLOCATIONS:
        raise ConfirmError("confirmation allocation number is invalid")
    expected = {
        "queue": "pdebug", "nodes": 2, "ranks": 2, "ppn": 1,
        "blocks": 2, "orders": ["AB", "BA"],
        "batches": list(BATCHES), "message_sizes": list(SIZES),
        "warmup": 10, "measured": 21,
    }
    jobspec = value.get("jobspec", {})
    scheduler = value.get("scheduler", {})
    resource_set = value.get("resource_set", {})
    nodelist = resource_set.get("nodelist")
    if (value.get("expected") != expected
            or scheduler.get("exit_code") != 0
            or scheduler.get("exception_types") != []
            or jobspec.get("queue") != "pdebug"
            or jobspec.get("duration_seconds") != 1200.0
            or not isinstance(nodelist, list) or len(nodelist) != 2
            or len(set(nodelist)) != 2):
        raise ConfirmError(f"allocation {allocation} Flux contract changed")
    try:
        scout_monitor.validate_allocation_shape(jobspec, resource_set)
    except common.MonitorError as exc:
        raise ConfirmError(str(exc)) from exc
    records = _artifact_records(value)
    here = Path(__file__).resolve().parent
    baseline = transition_files["frozen_baseline_binary"]
    reused = transition_files["frozen_reused_binary"]
    required = {
        transition_path.resolve(), baseline, reused,
        transition_files["frozen_build_provenance"],
        transition_files["preregistered_transition_protocol"],
        transition_files["transition_preparer"],
        (here / "run_reused_loop_descriptor_confirmation.sh").resolve(),
        (here / "continue_reused_loop_descriptor_confirmation.sh").resolve(),
        (here / "monitor_reused_loop_descriptor_confirmation.py").resolve(),
        Path(__file__).resolve(),
    }
    if set(records) != required:
        raise ConfirmError("reused-loop confirmation artifact set changed")
    runner = (here / "run_reused_loop_descriptor_confirmation.sh").resolve()
    if jobspec.get("embedded_script_sha256") != records[runner]:
        raise ConfirmError("Flux did not execute the frozen confirmation runner")
    command = jobspec.get("command")
    try:
        script_index = command.index("{{tmpdir}}/script")
    except (AttributeError, ValueError):
        raise ConfirmError("confirmation lacks its Flux batch command") from None
    arguments = command[script_index + 1:]
    output_root = path.resolve().parent.parent
    if (len(arguments) != 4
            or Path(arguments[0]).resolve() != baseline
            or Path(arguments[1]).resolve() != reused
            or Path(arguments[2]).resolve() != output_root
            or arguments[3] != str(allocation)):
        raise ConfirmError("confirmation batch arguments changed")
    driver_record = value.get("driver_stdout")
    driver_stderr = value.get("driver_stderr")
    if not isinstance(driver_record, dict) or not isinstance(driver_stderr, dict):
        raise ConfirmError(f"allocation {allocation} lacks driver logs")
    driver = Path(driver_record.get("path", ""))
    driver_err = Path(driver_stderr.get("path", ""))
    if (not driver.is_file() or sha256(driver) != driver_record.get("sha256")
            or driver.stat().st_size != driver_record.get("bytes")
            or not driver_err.is_file()
            or sha256(driver_err) != driver_stderr.get("sha256")
            or driver_err.stat().st_size != driver_stderr.get("bytes")
            or driver_err.stat().st_size != 0
            or driver.read_text(encoding="utf-8").splitlines()
            != _expected_driver_lines(allocation)):
        raise ConfirmError(f"allocation {allocation} driver logs changed")
    blocks = value.get("blocks")
    if not isinstance(blocks, dict) or set(blocks) != {"1", "2"}:
        raise ConfirmError(f"allocation {allocation} lacks both blocks")
    speedups: dict[int, dict[int, dict[str, float]]] = {}
    for block in BLOCKS:
        block_value = blocks[str(block)]
        if not isinstance(block_value, dict) or set(block_value) != {
                str(batch) for batch in BATCHES}:
            raise ConfirmError("confirmation block batch coverage changed")
        speedups[block] = {}
        for batch in BATCHES:
            pair = block_value[str(batch)]
            if not isinstance(pair, dict) or set(pair) != {
                    "baseline", "reused", "per_size_speedup",
                    "all_size_geomean_speedup", "correctness"}:
                raise ConfirmError("confirmation pair is incomplete")
            run_dir = (
                output_root / f"allocation{allocation}"
                / f"block{block}" / f"batch{batch}"
            )
            reparsed = {
                arm: scout_monitor.parse_run(
                    run_dir / f"{arm}.out", run_dir / f"{arm}.err", batch,
                )
                for arm in ("baseline", "reused")
            }
            if reparsed != {arm: pair[arm] for arm in reparsed}:
                raise ConfirmError("confirmation monitor disagrees with raw logs")
            regenerated = {
                size: (
                    reparsed["baseline"]["rows"][size][
                        "median_us_per_message"
                    ]
                    / reparsed["reused"]["rows"][size][
                        "median_us_per_message"
                    ]
                )
                for size in SIZES
            }
            recorded = pair["per_size_speedup"]
            if (not isinstance(recorded, dict) or set(recorded) != set(SIZES)
                    or any(not math.isclose(
                        float(recorded[size]), regenerated[size],
                        rel_tol=1e-12, abs_tol=1e-12,
                    ) for size in SIZES)
                    or not math.isclose(
                        float(pair["all_size_geomean_speedup"]),
                        geomean(list(regenerated.values())),
                        rel_tol=1e-12, abs_tol=1e-12,
                    )
                    or pair.get("correctness")
                    != "exact_enqueue_count_both_arms"):
                raise ConfirmError("reused-loop speedups do not regenerate")
            speedups[block][batch] = {
                size: regenerated[size] for size in SMALL_SIZES
            }
    summary = {
        "allocation": allocation, "job_id": value.get("job_id"),
        "nodelist": nodelist, "monitor": str(path.resolve()),
        "monitor_sha256": sha256(path),
        "artifact_set_id": fingerprint([
            {"path": str(item), "sha256": records[item]}
            for item in sorted(records, key=str)
        ]),
    }
    return allocation, speedups, summary


def analyze_monitors(
    transition_path: Path, monitor_paths: list[Path],
) -> dict[str, Any]:
    transition, transition_files = validate_transition(transition_path)
    if len(monitor_paths) != 3:
        raise ConfirmError("confirmation requires exactly three monitors")
    rows = {}
    summaries = []
    for path in monitor_paths:
        allocation, speedups, summary = _validate_monitor(
            path.resolve(), transition_path.resolve(), transition_files,
        )
        if allocation in rows:
            raise ConfirmError("confirmation repeats an allocation number")
        rows[allocation] = speedups
        summaries.append(summary)
    if len({item["job_id"] for item in summaries}) != 3:
        raise ConfirmError("confirmation did not use three independent jobs")
    if len({item["artifact_set_id"] for item in summaries}) != 1:
        raise ConfirmError("confirmation allocations used different artifacts")
    analysis = analyze_speedups(rows)
    payload = {
        "schema_version": "gicc-reused-loop-descriptor-confirmation-v1",
        "scope": (
            "Three sequential independent N2 pdebug allocations with AB/BA "
            "compiler/LTO schedules; no model or source modification."
        ),
        "model_invoked": False,
        "application_source_visible_to_model": False,
        "application_source_modified": False,
        "provider_call_authorized": False,
        "transition_id": transition["transition_id"],
        "dormant_compiler_candidate_id": transition[
            "dormant_compiler_candidate"
        ]["candidate_id"],
        "transition": str(transition_path.resolve()),
        "transition_sha256": sha256(transition_path),
        "allocation_monitors": sorted(
            summaries, key=lambda item: item["allocation"]
        ),
        **analysis,
    }
    return {**payload, "result_id": fingerprint(payload)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze_monitors(
            args.transition.resolve(), [path.resolve() for path in args.monitor],
        )
        if args.out.exists():
            raise ConfirmError(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result["confirmation_gate"], sort_keys=True))
        return 0
    except (
        ConfirmError, common.MonitorError, OSError, KeyError,
        TypeError, ValueError,
    ) as exc:
        print(f"reused-loop-confirm: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
