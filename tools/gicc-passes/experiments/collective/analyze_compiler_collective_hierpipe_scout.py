#!/usr/bin/env python3
"""Audit the topology-matched n4 hierarchical-tree pipeline scout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import analyze_compiler_collective_topology_scout as base


ARMS = (
    "hierarchical_double_tree",
    "hierarchical_double_tree_pipe4",
    "hierarchical_double_tree_pipe8",
)
SIZES = base.SIZES
ScoutError = base.ScoutError
sha256 = base.sha256
read_json = base.read_json
geomean = base.geomean
fingerprint = base.fingerprint


def analyze_rows(rows: dict[str, dict[str, float]]) -> dict[str, Any]:
    if set(rows) != set(ARMS):
        raise ScoutError("scout does not contain exactly the three declared arms")
    wanted_sizes = {str(size) for size in SIZES}
    if any(set(values) != wanted_sizes for values in rows.values()):
        raise ScoutError("scout size coverage is incomplete")
    algorithm_geomeans = {
        arm: geomean(list(rows[arm].values())) for arm in ARMS
    }
    best_uniform = min(
        ARMS, key=lambda arm: (algorithm_geomeans[arm], arm),
    )
    per_size = {}
    pointwise_times = []
    best_uniform_ratios = []
    winners = set()
    for size in SIZES:
        key = str(size)
        winner = min(ARMS, key=lambda arm: (rows[arm][key], arm))
        winners.add(winner)
        pointwise_times.append(rows[winner][key])
        ratio = rows[best_uniform][key] / rows[winner][key]
        best_uniform_ratios.append(ratio)
        per_size[key] = {
            "algorithm_us": {arm: rows[arm][key] for arm in ARMS},
            "winner": winner,
            "winner_us": rows[winner][key],
            "best_uniform_over_winner": ratio,
        }
    pointwise_geomean = geomean(pointwise_times)
    uniform_over_pointwise = (
        algorithm_geomeans[best_uniform] / pointwise_geomean
    )
    maximum = max(best_uniform_ratios)
    criteria = {
        "two_distinct_size_winners": {
            "observed": len(winners), "threshold": 2,
            "passed": len(winners) >= 2,
        },
        "best_uniform_over_pointwise_geomean": {
            "observed": uniform_over_pointwise, "threshold": 1.05,
            "passed": uniform_over_pointwise >= 1.05,
        },
        "maximum_single_size_headroom": {
            "observed": maximum, "threshold": 1.10,
            "passed": maximum >= 1.10,
        },
    }
    return {
        "per_size": per_size,
        "aggregate": {
            "algorithm_geomean_us": algorithm_geomeans,
            "best_uniform_algorithm": best_uniform,
            "best_uniform_geomean_us": algorithm_geomeans[best_uniform],
            "pointwise_oracle_geomean_us": pointwise_geomean,
            "best_uniform_over_pointwise_geomean": uniform_over_pointwise,
            "maximum_single_size_headroom": maximum,
            "distinct_size_winners": sorted(winners),
        },
        "hierarchical_pipeline_gate": {
            "passed": all(row["passed"] for row in criteria.values()),
            "criteria": criteria,
        },
    }


def analyze_monitor(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if (not isinstance(value, dict)
            or value.get("schema_version")
            != "gicc-collective-replicate-job-monitor-v1"
            or value.get("state") != "passed"):
        raise ScoutError("hierarchy-pipeline scout monitor did not pass")
    expected = value.get("expected")
    if expected != {
        "nodes": 4, "ranks": 32, "ppn": 8, "runs": 3, "warmup": 1,
        "sizes": list(SIZES),
    }:
        raise ScoutError("hierarchy-pipeline runtime contract changed")
    scheduler = value.get("scheduler", {})
    jobspec = value.get("jobspec", {})
    if (scheduler.get("exit_code") != 0
            or scheduler.get("exception_types") != []
            or jobspec.get("queue") != "pdebug"):
        raise ScoutError("scout is not a clean pdebug completion")
    expected_resources = [{
        "type": "node", "count": 4,
        "with": [{
            "type": "slot", "count": 8, "label": "task",
            "with": [
                {"type": "core", "count": 8},
                {"type": "gpu", "count": 1},
            ],
        }],
    }]
    if (jobspec.get("resources") != expected_resources
            or jobspec.get("duration_seconds") != 1200.0):
        raise ScoutError("scout did not request exactly four nodes")
    resource_set = value.get("resource_set")
    if (not isinstance(resource_set, dict)
            or not isinstance(resource_set.get("nodelist"), list)
            or not resource_set["nodelist"]):
        raise ScoutError("scout lacks its exact allocated nodes")

    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ScoutError("scout lacks frozen artifacts")
    runner_sha = None
    freeze_paths = []
    seen_artifacts = set()
    for record in artifacts:
        if (not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ScoutError("scout has an invalid artifact record")
        artifact = Path(record["path"])
        if artifact in seen_artifacts:
            raise ScoutError("scout has duplicate artifacts")
        seen_artifacts.add(artifact)
        if not artifact.is_file() or sha256(artifact) != record["sha256"]:
            raise ScoutError(f"scout artifact changed: {artifact}")
        if artifact.name == "run_compiler_collective_hierpipe_scout.sh":
            runner_sha = record["sha256"]
        if artifact.name == "FROZEN_V3_MANIFEST.json":
            freeze_paths.append(artifact.resolve())
    if (runner_sha is None or len(freeze_paths) != 1
            or jobspec.get("embedded_script_sha256") != runner_sha):
        raise ScoutError("Flux did not execute the frozen scout runner")

    driver_record = value.get("driver_stdout")
    if not isinstance(driver_record, dict):
        raise ScoutError("scout lacks a driver log")
    driver = Path(driver_record.get("path", ""))
    if (not driver.is_file() or sha256(driver) != driver_record.get("sha256")
            or driver.stat().st_size != driver_record.get("bytes")):
        raise ScoutError("scout driver log changed")
    lines = [
        "HIERPIPE_CONFIG nodes=4 ranks=32 ppn=8 runs=3 warmup=1 "
        "arms=" + " ".join(ARMS),
    ]
    for arm in ARMS:
        lines.extend([
            f"HIERPIPE_ARM_START arm={arm}",
            f"HIERPIPE_ARM_DONE arm={arm}",
        ])
    lines.append("HIERPIPE_DONE")
    if driver.read_text().splitlines() != lines:
        raise ScoutError("scout driver order is invalid")
    command = jobspec.get("command")
    if (not isinstance(command, list)
            or any(not isinstance(item, str) for item in command)):
        raise ScoutError("scout lacks its Flux batch command")
    try:
        script_index = command.index("{{tmpdir}}/script")
    except ValueError:
        raise ScoutError("scout lacks its Flux batch command") from None
    arguments = command[script_index + 1:]
    if (len(arguments) != 2
            or Path(arguments[0]).resolve() != freeze_paths[0].parent
            or Path(arguments[1]).resolve() != driver.parent.resolve()):
        raise ScoutError("scout batch arguments changed")

    benchmarks = value.get("benchmarks")
    if not isinstance(benchmarks, dict) or set(benchmarks) != set(ARMS):
        raise ScoutError("scout benchmark coverage is incomplete")
    rows = {}
    for arm in ARMS:
        benchmark = benchmarks[arm].get("benchmark")
        stdout = benchmarks[arm].get("stdout")
        if not isinstance(benchmark, dict) or not isinstance(stdout, dict):
            raise ScoutError(f"scout arm is incomplete: {arm}")
        log = Path(stdout.get("path", ""))
        if (not log.is_file() or sha256(log) != stdout.get("sha256")
                or log.stat().st_size != stdout.get("bytes")):
            raise ScoutError(f"scout log changed: {arm}")
        results = benchmark.get("results")
        if not isinstance(results, dict):
            raise ScoutError(f"scout lacks results: {arm}")
        rows[arm] = {key: float(number) for key, number in results.items()}
    analysis = analyze_rows(rows)
    payload = {
        "schema_version": "gicc-collective-hierpipe-scout-analysis-v1",
        "scope": (
            "Topology-matched n4 compiler-catalog control; no source change "
            "or model result. It gates any later provider experiment."
        ),
        "model_invoked": False,
        "application_source_modified": False,
        "job_id": value.get("job_id"),
        "nodelist": resource_set["nodelist"],
        "monitor": str(path.resolve()),
        "monitor_sha256": sha256(path),
        "artifact_count": len(artifacts),
        **analysis,
    }
    result = dict(payload)
    result["result_id"] = fingerprint(payload)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze_monitor(args.monitor.resolve())
        if args.out.exists():
            raise ScoutError(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result["hierarchical_pipeline_gate"], sort_keys=True))
        return 0
    except (ScoutError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"compiler-collective-hierpipe-scout: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
