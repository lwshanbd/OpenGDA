#!/usr/bin/env python3
"""Audit three independent order-balanced producer-fission allocations."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import monitor_producer_fission_oracle_scout as scout_monitor
import prepare_producer_fission_confirmation as transition_base


SIZES = tuple(transition_base.SIZES)
ALLOCATIONS = (1, 2, 3)
BLOCKS = (1, 2)
ORDERS = {1: ("baseline", "fission"), 2: ("fission", "baseline")}
MONITOR_SCHEMA = "gicc-producer-fission-confirmation-monitor-v1"
ConfirmError = transition_base.TransitionError
common = scout_monitor.common
sha256 = transition_base.sha256_file
fingerprint = transition_base.bridge._fingerprint


def geomean(values: list[float]) -> float:
    return transition_base.geomean(values)


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
        raise ConfirmError(
            "confirmation requires three finite positive allocation speedups"
        )
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
    rows: dict[int, dict[int, dict[str, float]]],
) -> dict[str, Any]:
    if set(rows) != set(ALLOCATIONS):
        raise ConfirmError("confirmation requires allocations 1, 2, and 3")
    wanted_sizes = {str(size) for size in SIZES}
    allocation_values = []
    per_allocation = {}
    per_size = {str(size): [] for size in SIZES}
    for allocation in ALLOCATIONS:
        blocks = rows[allocation]
        if set(blocks) != set(BLOCKS):
            raise ConfirmError(
                f"allocation {allocation} requires AB and BA blocks"
            )
        values = []
        for block in BLOCKS:
            if set(blocks[block]) != wanted_sizes:
                raise ConfirmError(
                    f"allocation {allocation}/block {block} changed sizes"
                )
            for size in SIZES:
                value = blocks[block][str(size)]
                if value <= 0 or not math.isfinite(value):
                    raise ConfirmError(
                        "producer-fission speedup is not finite positive"
                    )
                values.append(value)
                per_size[str(size)].append(value)
        aggregate = geomean(values)
        allocation_values.append(aggregate)
        per_allocation[str(allocation)] = {
            "paired_speedups": values,
            "geometric_mean_speedup": aggregate,
            "win": aggregate > 1.0,
        }
    primary = allocation_bootstrap(allocation_values)
    size_summaries = {
        key: {
            "paired_speedups": values,
            "geometric_mean_speedup": geomean(values),
            "wins": sum(value > 1.0 for value in values),
        }
        for key, values in per_size.items()
    }
    criteria = {
        "aggregate_point_estimate": {
            "observed": primary["estimate"],
            "threshold": 1.03,
            "comparison": "greater_than_or_equal",
            "passed": primary["estimate"] >= 1.03,
        },
        "allocation_cluster_bootstrap_lower_95": {
            "observed": primary["lower_2_5_percent"],
            "threshold": 1.0,
            "comparison": "strictly_greater",
            "passed": primary["lower_2_5_percent"] > 1.0,
        },
        "allocation_level_wins": {
            "observed": primary["allocation_wins"],
            "threshold": 2,
            "comparison": "greater_than_or_equal",
            "passed": primary["allocation_wins"] >= 2,
        },
        "all_correctness_pairs": {
            "observed": 12,
            "threshold": 12,
            "passed": True,
        },
    }
    return {
        "per_allocation": per_allocation,
        "per_size": size_summaries,
        "primary_speedup": primary,
        "correctness_gate": {"passed": True, "paired_checks": 12},
        "confirmation_gate": {
            "passed": all(item["passed"] for item in criteria.values()),
            "criteria": criteria,
        },
    }


def _transition_paths(value: dict[str, Any]) -> dict[str, Path]:
    paths = {}
    for record in value["files"]:
        raw = Path(record["path"])
        path = (
            raw.resolve() if raw.is_absolute()
            else (transition_base.ROOT / raw).resolve()
        )
        paths[record["role"]] = path
    return paths


def validate_transition(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    value = transition_base.verify_contained_report(path)
    boundary = value.get("boundary")
    if boundary != {
        "compiler_lto_decisions_only": True,
        "application_source_modified": False,
        "model_invoked": False,
        "provider_call_authorized": False,
        "scheduler_job_submitted": False,
        "candidate_model_visible_before_confirmation": False,
    }:
        raise ConfirmError("producer-fission compiler-only boundary changed")
    contract = value.get("confirmation_contract")
    expected = {
        "queue": "pdebug",
        "nodes": 2,
        "ranks": 16,
        "ranks_per_node": 8,
        "cpu_cores_per_rank": 8,
        "gpus_per_rank": 1,
        "independent_allocations": 3,
        "maximum_active_or_queued_jobs": 1,
        "balanced_blocks_per_allocation": ["AB", "BA"],
        "sizes": list(SIZES),
        "iterations": 200,
        "nccheck": 10,
        "scout_or_confirmation_labels_visible_to_model": False,
    }
    if not isinstance(contract, dict) or any(
            contract.get(key) != wanted for key, wanted in expected.items()):
        raise ConfirmError("producer-fission confirmation contract changed")
    return value, _transition_paths(value)


def _artifact_records(value: dict[str, Any]) -> dict[Path, str]:
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise ConfirmError("confirmation monitor lacks frozen artifacts")
    records = {}
    for record in artifacts:
        if (not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ConfirmError("confirmation monitor has an invalid artifact")
        path = Path(record["path"]).resolve()
        if path in records or not path.is_file() or sha256(path) != record["sha256"]:
            raise ConfirmError(f"confirmation artifact changed: {path}")
        records[path] = record["sha256"]
    return records


def _expected_driver_lines(allocation: int) -> list[str]:
    lines = [
        "PFISSION_CONFIRM_CONFIG "
        f"allocation={allocation} nodes=2 ranks=16 ppn=8 cores=8 "
        "sizes=1024,4096 iterations=200 nccheck=10 blocks=AB,BA",
    ]
    for block in BLOCKS:
        order = ORDERS[block]
        lines.append(
            "PFISSION_CONFIRM_BLOCK_START "
            f"allocation={allocation} block={block} order={' '.join(order)}"
        )
        for size in SIZES:
            for arm in order:
                lines.extend([
                    "PFISSION_CONFIRM_RUN_START "
                    f"allocation={allocation} block={block} size={size} arm={arm}",
                    "PFISSION_CONFIRM_RUN_DONE "
                    f"allocation={allocation} block={block} size={size} arm={arm}",
                ])
        lines.append(
            "PFISSION_CONFIRM_BLOCK_DONE "
            f"allocation={allocation} block={block}"
        )
    lines.append(
        f"PFISSION_CONFIRM_DONE allocation={allocation} pairs=4 runs=8"
    )
    return lines


def _validate_monitor(
    path: Path, transition_path: Path, transition_files: dict[str, Path],
) -> tuple[int, dict[int, dict[str, float]], dict[str, Any]]:
    value = transition_base.read_json(path)
    if (not isinstance(value, dict)
            or value.get("schema_version") != MONITOR_SCHEMA
            or value.get("state") != "passed"):
        raise ConfirmError(f"confirmation monitor did not pass: {path}")
    allocation = value.get("allocation")
    if allocation not in ALLOCATIONS:
        raise ConfirmError("confirmation allocation number is invalid")
    expected = {
        "queue": "pdebug", "nodes": 2, "ranks": 16, "ppn": 8,
        "blocks": 2, "orders": ["AB", "BA"], "sizes": list(SIZES),
        "iterations": 200, "nccheck": 10,
    }
    resources = [{
        "type": "node", "count": 2,
        "with": [{
            "type": "slot", "count": 8, "label": "task",
            "with": [
                {"type": "core", "count": 8},
                {"type": "gpu", "count": 1},
            ],
        }],
    }]
    jobspec = value.get("jobspec", {})
    scheduler = value.get("scheduler", {})
    if (value.get("expected") != expected
            or scheduler.get("exit_code") != 0
            or scheduler.get("exception_types") != []
            or jobspec.get("queue") != "pdebug"
            or jobspec.get("duration_seconds") != 1200.0
            or jobspec.get("resources") != resources):
        raise ConfirmError(f"allocation {allocation} Flux contract changed")
    nodelist = value.get("resource_set", {}).get("nodelist")
    if (not isinstance(nodelist, list) or len(nodelist) != 2
            or len(set(nodelist)) != 2):
        raise ConfirmError(f"allocation {allocation} lacks two exact nodes")

    records = _artifact_records(value)
    here = Path(__file__).resolve().parent
    baseline = transition_files["frozen_baseline_binary"]
    fission = transition_files["frozen_fission_binary"]
    required = {
        transition_path.resolve(), baseline, fission,
        transition_files["frozen_build_provenance"],
        transition_files["preregistered_transition_protocol"],
        transition_files["transition_preparer"],
        (here / "run_producer_fission_confirmation.sh").resolve(),
        (here / "continue_producer_fission_confirmation.sh").resolve(),
        (here / "monitor_producer_fission_confirmation.py").resolve(),
        Path(__file__).resolve(),
    }
    if set(records) != required:
        raise ConfirmError("producer-fission confirmation artifact set changed")

    runner = (here / "run_producer_fission_confirmation.sh").resolve()
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
            or Path(arguments[1]).resolve() != fission
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
            or driver_err.stat().st_size != 0):
        raise ConfirmError(f"allocation {allocation} driver logs changed")
    if driver.read_text(encoding="utf-8").splitlines() != _expected_driver_lines(
            allocation):
        raise ConfirmError(f"allocation {allocation} execution order changed")

    blocks = value.get("blocks")
    if not isinstance(blocks, dict) or set(blocks) != {"1", "2"}:
        raise ConfirmError(f"allocation {allocation} lacks both blocks")
    speedups = {}
    for block in BLOCKS:
        block_value = blocks[str(block)]
        if not isinstance(block_value, dict) or set(block_value) != {
                str(size) for size in SIZES}:
            raise ConfirmError(
                f"allocation {allocation}/block {block} changed sizes"
            )
        speedups[block] = {}
        for size in SIZES:
            pair = block_value[str(size)]
            if not isinstance(pair, dict) or set(pair) != {
                    "baseline", "fission", "speedup"}:
                raise ConfirmError("confirmation pair is incomplete")
            reparsed = {}
            run_dir = (
                output_root / f"allocation{allocation}"
                / f"block{block}" / f"size{size}"
            )
            for arm in ("baseline", "fission"):
                parsed = scout_monitor.parse_run(run_dir / f"{arm}.out", size)
                stderr = run_dir / f"{arm}.err"
                parsed["stderr"] = {
                    **common.summarize_stderr(stderr),
                    "path": str(stderr.resolve()),
                    "sha256": common.sha256(stderr),
                }
                if parsed != pair[arm]:
                    raise ConfirmError(
                        f"allocation {allocation}/block {block}/{size}/{arm} "
                        "monitor disagrees with raw logs"
                    )
                reparsed[arm] = parsed
            baseline_seconds = reparsed["baseline"]["seconds"]
            fission_seconds = reparsed["fission"]["seconds"]
            if (reparsed["baseline"]["iterations"]
                    != reparsed["fission"]["iterations"]
                    or not math.isclose(
                        reparsed["baseline"]["final_l2"],
                        reparsed["fission"]["final_l2"],
                        rel_tol=1e-6, abs_tol=1e-7,
                    )):
                raise ConfirmError("producer-fission correctness pair changed")
            speedup = baseline_seconds / fission_seconds
            if not math.isclose(
                    float(pair["speedup"]), speedup,
                    rel_tol=1e-12, abs_tol=1e-12):
                raise ConfirmError("producer-fission speedup does not regenerate")
            speedups[block][str(size)] = speedup
    summary = {
        "allocation": allocation,
        "job_id": value.get("job_id"),
        "nodelist": nodelist,
        "monitor": str(path.resolve()),
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
        "schema_version": "gicc-producer-fission-confirmation-v1",
        "scope": (
            "Three sequential independent N2 pdebug allocations with AB/BA "
            "compiler/LTO schedule blocks; no model or source modification."
        ),
        "model_invoked": False,
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
            args.transition.resolve(),
            [path.resolve() for path in args.monitor],
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
    except (ConfirmError, common.MonitorError, OSError, KeyError,
            TypeError, ValueError) as exc:
        print(f"producer-fission-confirm: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
