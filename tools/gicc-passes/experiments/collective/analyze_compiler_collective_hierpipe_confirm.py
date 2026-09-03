#!/usr/bin/env python3
"""Audit the one-allocation rotated hierarchy-pipeline confirmation."""

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

import analyze_compiler_collective_topology_scout as base
import monitor_compiler_collective_job as monitor_base


ARMS = (
    "hierarchical_double_tree",
    "hierarchical_double_tree_pipe4",
    "hierarchical_double_tree_pipe8",
)
ORDERS = {
    1: list(ARMS),
    2: [ARMS[1], ARMS[2], ARMS[0]],
    3: [ARMS[2], ARMS[0], ARMS[1]],
}
SIZES = base.SIZES
ConfirmError = base.ScoutError
sha256 = base.sha256
read_json = base.read_json
geomean = base.geomean
fingerprint = base.fingerprint


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_block_bootstrap(ratios: list[float]) -> dict[str, Any]:
    if len(ratios) != 3 or any(
            value <= 0 or not math.isfinite(value) for value in ratios):
        raise ConfirmError("paired confirmation requires three finite ratios")
    estimates = [
        geomean([ratios[index] for index in sample])
        for sample in itertools.product(range(3), repeat=3)
    ]
    return {
        "estimate": geomean(ratios),
        "replicate_block_ratios": ratios,
        "method": (
            "exact 3-out-of-3 paired bootstrap over rotated blocks in one "
            "allocation"
        ),
        "bootstrap_samples": len(estimates),
        "lower_2_5_percent": percentile(estimates, 0.025),
        "upper_97_5_percent": percentile(estimates, 0.975),
    }


def _validate_scout(value: Any) -> dict[str, Any]:
    if (not isinstance(value, dict)
            or value.get("schema_version")
            != "gicc-collective-hierpipe-scout-analysis-v1"
            or value.get("model_invoked") is not False
            or value.get("application_source_modified") is not False
            or value.get("hierarchical_pipeline_gate", {}).get("passed")
            is not True):
        raise ConfirmError("confirmation requires the passed compiler-only scout")
    payload = dict(value)
    result_id = payload.pop("result_id", None)
    if result_id != fingerprint(payload):
        raise ConfirmError("scout analysis is not content-addressed")
    per_size = value.get("per_size")
    if not isinstance(per_size, dict) or set(per_size) != {
            str(size) for size in SIZES}:
        raise ConfirmError("scout analysis has incomplete size coverage")
    uniform = value.get("aggregate", {}).get("best_uniform_algorithm")
    if uniform not in ARMS:
        raise ConfirmError("scout best-uniform arm is invalid")
    for size in SIZES:
        if per_size[str(size)].get("winner") not in ARMS:
            raise ConfirmError("scout pointwise policy is invalid")
    return value


def analyze_rows(
    rows: dict[int, dict[str, dict[str, float]]], scout_value: Any,
) -> dict[str, Any]:
    scout = _validate_scout(scout_value)
    if set(rows) != {1, 2, 3}:
        raise ConfirmError("confirmation requires blocks 1, 2, and 3")
    wanted_sizes = {str(size) for size in SIZES}
    for replicate, arms in rows.items():
        if set(arms) != set(ARMS):
            raise ConfirmError(f"block {replicate} has incomplete arm coverage")
        for arm, values in arms.items():
            if set(values) != wanted_sizes:
                raise ConfirmError(
                    f"block {replicate}/{arm} has incomplete size coverage"
                )
            if any(value <= 0 or not math.isfinite(value)
                   for value in values.values()):
                raise ConfirmError("confirmation latency is not finite positive")

    pooled = {
        arm: {
            str(size): statistics.median(
                rows[replicate][arm][str(size)] for replicate in (1, 2, 3)
            )
            for size in SIZES
        }
        for arm in ARMS
    }
    arm_geomeans = {
        arm: geomean(list(pooled[arm].values())) for arm in ARMS
    }
    best_uniform = min(ARMS, key=lambda arm: (arm_geomeans[arm], arm))
    per_size = {}
    pointwise_values = []
    uniform_ratios = []
    winners = set()
    for size in SIZES:
        key = str(size)
        winner = min(ARMS, key=lambda arm: (pooled[arm][key], arm))
        winners.add(winner)
        pointwise_values.append(pooled[winner][key])
        ratio = pooled[best_uniform][key] / pooled[winner][key]
        uniform_ratios.append(ratio)
        per_size[key] = {
            "algorithm_median_us": {
                arm: pooled[arm][key] for arm in ARMS
            },
            "winner": winner,
            "winner_us": pooled[winner][key],
            "best_uniform_over_winner": ratio,
        }
    pointwise_geomean = geomean(pointwise_values)
    uniform_over_pointwise = arm_geomeans[best_uniform] / pointwise_geomean
    maximum_headroom = max(uniform_ratios)

    scout_uniform = scout["aggregate"]["best_uniform_algorithm"]
    frozen_policy = {
        str(size): scout["per_size"][str(size)]["winner"] for size in SIZES
    }
    nonuniform_sizes = [
        str(size) for size in SIZES
        if frozen_policy[str(size)] != scout_uniform
    ]
    replicate_speedups = []
    per_size_speedups = {str(size): [] for size in SIZES}
    for replicate in (1, 2, 3):
        uniform_values = [
            rows[replicate][scout_uniform][str(size)] for size in SIZES
        ]
        selected_values = [
            rows[replicate][frozen_policy[str(size)]][str(size)]
            for size in SIZES
        ]
        replicate_speedups.append(
            geomean(uniform_values) / geomean(selected_values)
        )
        for size in SIZES:
            key = str(size)
            per_size_speedups[key].append(
                rows[replicate][scout_uniform][key]
                / rows[replicate][frozen_policy[key]][key]
            )
    primary = paired_block_bootstrap(replicate_speedups)
    persistence = {
        key: {
            "scout_selected_algorithm": frozen_policy[key],
            "scout_uniform_algorithm": scout_uniform,
            "wins_over_scout_uniform": sum(value > 1.0 for value in ratios),
            "replicate_speedups": ratios,
            "passed": sum(value > 1.0 for value in ratios) >= 2,
        }
        for key, ratios in per_size_speedups.items()
        if key in nonuniform_sizes
    }
    persistence_passed = bool(persistence) and all(
        item["passed"] for item in persistence.values()
    )
    criteria = {
        "scout_frozen_policy_speedup": {
            "observed": primary["estimate"],
            "threshold": 1.05,
            "passed": primary["estimate"] >= 1.05,
        },
        "scout_frozen_policy_paired_lower_95": {
            "observed": primary["lower_2_5_percent"],
            "threshold": 1.0,
            "comparison": "strictly_greater",
            "passed": primary["lower_2_5_percent"] > 1.0,
        },
        "scout_nonuniform_winner_persistence": {
            "observed": {
                key: item["wins_over_scout_uniform"]
                for key, item in persistence.items()
            },
            "threshold": "at least 2 of 3 blocks for every nonuniform size",
            "passed": persistence_passed,
        },
        "two_distinct_pooled_size_winners": {
            "observed": len(winners),
            "threshold": 2,
            "passed": len(winners) >= 2,
        },
        "best_uniform_over_pooled_pointwise_geomean": {
            "observed": uniform_over_pointwise,
            "threshold": 1.05,
            "passed": uniform_over_pointwise >= 1.05,
        },
        "maximum_pooled_single_size_headroom": {
            "observed": maximum_headroom,
            "threshold": 1.10,
            "passed": maximum_headroom >= 1.10,
        },
    }
    return {
        "per_size_pooled_median": per_size,
        "aggregate": {
            "algorithm_geomean_us": arm_geomeans,
            "best_uniform_algorithm": best_uniform,
            "best_uniform_geomean_us": arm_geomeans[best_uniform],
            "pointwise_oracle_geomean_us": pointwise_geomean,
            "best_uniform_over_pointwise_geomean": uniform_over_pointwise,
            "maximum_single_size_headroom": maximum_headroom,
            "distinct_size_winners": sorted(winners),
        },
        "scout_frozen_policy": {
            "uniform_algorithm": scout_uniform,
            "size_algorithms": frozen_policy,
            "nonuniform_sizes": nonuniform_sizes,
            "paired_speedup": primary,
            "nonuniform_winner_persistence": persistence,
        },
        "confirmation_gate": {
            "passed": all(item["passed"] for item in criteria.values()),
            "criteria": criteria,
        },
    }


def _artifact_records(
    value: dict[str, Any], replicate: int, scout_path: Path,
) -> tuple[dict[Path, str], Path]:
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise ConfirmError(f"block {replicate} lacks frozen artifacts")
    records: dict[Path, str] = {}
    for record in artifacts:
        if (not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ConfirmError(f"block {replicate} has an invalid artifact")
        path = Path(record["path"]).resolve()
        if path in records or not path.is_file() or sha256(path) != record["sha256"]:
            raise ConfirmError(f"block {replicate} artifact changed: {path}")
        records[path] = record["sha256"]
    freezes = [path for path in records if path.name == "FROZEN_V3_MANIFEST.json"]
    if len(freezes) != 1:
        raise ConfirmError(f"block {replicate} lacks one offline freeze")
    bundle = freezes[0].parent
    script_dir = Path(__file__).resolve().parent
    required = {
        freezes[0],
        (bundle / "discovery/graph.json").resolve(),
        (bundle / "inputs/platform.json").resolve(),
        scout_path.resolve(),
        (scout_path.parent / "monitor.json").resolve(),
        (script_dir / "PROTOCOL_DRAFT.md").resolve(),
        (script_dir / "run_compiler_collective_hierpipe_confirm.sh").resolve(),
        (script_dir / "continue_compiler_collective_hierpipe_confirm.sh").resolve(),
        (script_dir / "monitor_compiler_collective_replicate.py").resolve(),
        Path(__file__).resolve(),
    }
    for arm in ARMS:
        required.update({
            (bundle / f"binaries/{arm}/compiler_collective_eval").resolve(),
            (bundle / f"binaries/{arm}/build-provenance.json").resolve(),
            (bundle / f"controls/{arm}-hint.json").resolve(),
        })
    if set(records) != required:
        raise ConfirmError(f"block {replicate} artifact set changed")
    return records, bundle


def _validate_monitor(
    path: Path, scout_path: Path,
) -> tuple[int, dict[str, dict[str, float]], dict[str, Any]]:
    value = read_json(path)
    if (not isinstance(value, dict)
            or value.get("schema_version")
            != "gicc-collective-replicate-job-monitor-v1"
            or value.get("state") != "passed"):
        raise ConfirmError(f"confirmation monitor did not pass: {path}")
    replicate = value.get("replicate")
    if replicate not in ORDERS:
        raise ConfirmError("confirmation replicate is invalid")
    expected = {
        "nodes": 4, "ranks": 32, "ppn": 8, "runs": 7, "warmup": 2,
        "sizes": list(SIZES),
    }
    jobspec = value.get("jobspec", {})
    scheduler = value.get("scheduler", {})
    resources = [{
        "type": "node", "count": 4,
        "with": [{
            "type": "slot", "count": 8, "label": "task",
            "with": [
                {"type": "core", "count": 8},
                {"type": "gpu", "count": 1},
            ],
        }],
    }]
    if (value.get("expected") != expected
            or scheduler.get("exit_code") != 0
            or scheduler.get("exception_types") != []
            or jobspec.get("queue") != "pdebug"
            or jobspec.get("duration_seconds") != 1800.0
            or jobspec.get("resources") != resources):
        raise ConfirmError(f"block {replicate} runtime contract changed")
    allocation = value.get("resource_set", {}).get("nodelist")
    if not isinstance(allocation, list) or not allocation:
        raise ConfirmError(f"block {replicate} lacks an exact node list")
    records, bundle = _artifact_records(value, replicate, scout_path)
    runner = (
        Path(__file__).resolve().parent
        / "run_compiler_collective_hierpipe_confirm.sh"
    ).resolve()
    if jobspec.get("embedded_script_sha256") != records[runner]:
        raise ConfirmError("Flux did not execute the frozen confirmation runner")

    driver_record = value.get("driver_stdout")
    driver_stderr = value.get("driver_stderr")
    if not isinstance(driver_record, dict) or not isinstance(driver_stderr, dict):
        raise ConfirmError(f"block {replicate} lacks a driver log")
    driver = Path(driver_record.get("path", ""))
    driver_err = Path(driver_stderr.get("path", ""))
    if (not driver.is_file() or sha256(driver) != driver_record.get("sha256")
            or driver.stat().st_size != driver_record.get("bytes")
            or not driver_err.is_file()
            or sha256(driver_err) != driver_stderr.get("sha256")
            or driver_err.stat().st_size != driver_stderr.get("bytes")
            or driver_err.stat().st_size != 0):
        raise ConfirmError(f"block {replicate} driver log changed")
    expected_lines = [
        f"HIERPIPE_CONFIRM_CONFIG replicate={replicate} "
        f"arms={' '.join(ORDERS[replicate])}",
    ]
    for arm in ORDERS[replicate]:
        expected_lines.extend([
            f"HIERPIPE_CONFIRM_ARM_START replicate={replicate} arm={arm}",
            f"HIERPIPE_CONFIRM_ARM_DONE replicate={replicate} arm={arm}",
        ])
    expected_lines.append(f"HIERPIPE_CONFIRM_DONE replicate={replicate}")
    if driver.read_text().splitlines() != expected_lines:
        raise ConfirmError(f"block {replicate} arm order changed")
    command = jobspec.get("command")
    try:
        script_index = command.index("{{tmpdir}}/script")
    except (AttributeError, ValueError):
        raise ConfirmError("confirmation lacks its Flux batch command") from None
    arguments = command[script_index + 1:]
    output_root = driver.parent.parent.resolve()
    if (len(arguments) != 2
            or Path(arguments[0]).resolve() != bundle
            or Path(arguments[1]).resolve() != output_root):
        raise ConfirmError("confirmation batch arguments changed")

    benchmarks = value.get("benchmarks")
    if not isinstance(benchmarks, dict) or set(benchmarks) != set(ARMS):
        raise ConfirmError(f"block {replicate} lacks complete arm coverage")
    rows = {}
    for arm in ARMS:
        record = benchmarks[arm]
        if not isinstance(record, dict):
            raise ConfirmError(f"block {replicate}/{arm} is invalid")
        benchmark = record.get("benchmark")
        stdout = record.get("stdout")
        stderr = record.get("stderr")
        if (not isinstance(benchmark, dict) or not isinstance(stdout, dict)
                or not isinstance(stderr, dict)):
            raise ConfirmError(f"block {replicate}/{arm} is incomplete")
        log = Path(stdout.get("path", ""))
        err = Path(stderr.get("path", ""))
        if (not log.is_file() or sha256(log) != stdout.get("sha256")
                or log.stat().st_size != stdout.get("bytes")
                or not err.is_file() or sha256(err) != stderr.get("sha256")
                or err.stat().st_size != stderr.get("bytes")):
            raise ConfirmError(f"block {replicate}/{arm} log changed")
        reparsed = monitor_base.validate_output(
            log, arm, list(SIZES), 4, 32, 8, 7, 2,
        )
        if reparsed != benchmark:
            raise ConfirmError(
                f"block {replicate}/{arm} monitor disagrees with raw log"
            )
        results = benchmark.get("results")
        if not isinstance(results, dict):
            raise ConfirmError(f"block {replicate}/{arm} lacks results")
        rows[arm] = {key: float(number) for key, number in results.items()}
    summary = {
        "replicate": replicate,
        "job_id": value.get("job_id"),
        "nodelist": allocation,
        "arm_order": ORDERS[replicate],
        "monitor": str(path.resolve()),
        "monitor_sha256": sha256(path),
        "artifact_set_id": fingerprint([
            {"path": str(item), "sha256": records[item]}
            for item in sorted(records, key=str)
        ]),
    }
    return replicate, rows, summary


def analyze_monitors(scout_path: Path, monitor_paths: list[Path]) -> dict[str, Any]:
    scout = _validate_scout(read_json(scout_path))
    scout_monitor = (scout_path.parent / "monitor.json").resolve()
    if (Path(scout.get("monitor", "")).resolve() != scout_monitor
            or not scout_monitor.is_file()
            or sha256(scout_monitor) != scout.get("monitor_sha256")):
        raise ConfirmError("scout analysis is not bound to its raw monitor")
    if len(monitor_paths) != 3:
        raise ConfirmError("confirmation requires exactly three block monitors")
    rows = {}
    summaries = []
    for path in monitor_paths:
        replicate, block, summary = _validate_monitor(path.resolve(), scout_path)
        if replicate in rows:
            raise ConfirmError("confirmation has a duplicate block")
        rows[replicate] = block
        summaries.append(summary)
    if set(rows) != {1, 2, 3}:
        raise ConfirmError("confirmation blocks are incomplete")
    if len({item["job_id"] for item in summaries}) != 1:
        raise ConfirmError("confirmation blocks did not share one allocation")
    if len({json.dumps(item["nodelist"]) for item in summaries}) != 1:
        raise ConfirmError("confirmation blocks did not share exact nodes")
    if len({item["artifact_set_id"] for item in summaries}) != 1:
        raise ConfirmError("confirmation blocks used different artifacts")
    analysis = analyze_rows(rows, scout)
    payload = {
        "schema_version": "gicc-collective-hierpipe-confirmation-v1",
        "scope": (
            "One pdebug allocation, three paired rotated-order blocks, frozen "
            "compiler controls only; no model or application-source change."
        ),
        "model_invoked": False,
        "application_source_modified": False,
        "scout_result_id": scout["result_id"],
        "scout_analysis": str(scout_path.resolve()),
        "scout_analysis_sha256": sha256(scout_path),
        "job_id": summaries[0]["job_id"],
        "nodelist": summaries[0]["nodelist"],
        "block_monitors": sorted(summaries, key=lambda item: item["replicate"]),
        **analysis,
    }
    result = dict(payload)
    result["result_id"] = fingerprint(payload)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scout-analysis", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze_monitors(
            args.scout_analysis.resolve(), [path.resolve() for path in args.monitor]
        )
        if args.out.exists():
            raise ConfirmError(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result["confirmation_gate"], sort_keys=True))
        return 0
    except (ConfirmError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"compiler-collective-hierpipe-confirm: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
