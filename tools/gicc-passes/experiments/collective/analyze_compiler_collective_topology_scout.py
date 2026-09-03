#!/usr/bin/env python3
"""Audit the bounded 4-node topology scout and quantify compiler headroom."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any


ARMS = ("baseline_auto", "hierarchical_double_tree")
SIZES = (
    1024, 4096, 8192, 65536, 262144,
    1048576, 4194304, 8388608, 16777216,
)


class ScoutError(RuntimeError):
    """The scout archive is incomplete, changed, or violates its contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ScoutError(f"cannot read JSON {path}: {exc}") from exc


def geomean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value)
                         for value in values):
        raise ScoutError("latencies and ratios must be positive and finite")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def analyze_rows(rows: dict[str, dict[str, float]]) -> dict[str, Any]:
    if set(rows) != set(ARMS):
        raise ScoutError("scout does not contain exactly the two declared arms")
    wanted_sizes = {str(size) for size in SIZES}
    if any(set(values) != wanted_sizes for values in rows.values()):
        raise ScoutError("scout size coverage is incomplete")
    baseline = rows["baseline_auto"]
    tree = rows["hierarchical_double_tree"]
    per_size = {}
    ratios = []
    winners = set()
    for size in SIZES:
        key = str(size)
        winner = min(ARMS, key=lambda name: (rows[name][key], name))
        winners.add(winner)
        ratio = baseline[key] / rows[winner][key]
        ratios.append(ratio)
        per_size[key] = {
            "baseline_us": baseline[key],
            "hierarchical_double_tree_us": tree[key],
            "winner": winner,
            "baseline_over_winner": ratio,
        }
    pointwise = geomean(ratios)
    maximum = max(ratios)
    criteria = {
        "two_distinct_size_winners": {
            "observed": len(winners), "threshold": 2,
            "passed": len(winners) >= 2,
        },
        "baseline_over_pointwise_geomean": {
            "observed": pointwise, "threshold": 1.05,
            "passed": pointwise >= 1.05,
        },
        "maximum_single_size_headroom": {
            "observed": maximum, "threshold": 1.10,
            "passed": maximum >= 1.10,
        },
    }
    return {
        "per_size": per_size,
        "aggregate": {
            "baseline_geomean_us": geomean(list(baseline.values())),
            "hierarchical_double_tree_geomean_us": geomean(list(tree.values())),
            "baseline_over_pointwise_geomean": pointwise,
            "maximum_single_size_headroom": maximum,
            "distinct_size_winners": sorted(winners),
        },
        "v4_topology_hypothesis": {
            "promising": all(row["passed"] for row in criteria.values()),
            "criteria": criteria,
        },
    }


def analyze_monitor(path: Path) -> dict[str, Any]:
    value = read_json(path)
    if (not isinstance(value, dict)
            or value.get("schema_version")
            != "gicc-collective-replicate-job-monitor-v1"
            or value.get("state") != "passed"):
        raise ScoutError("topology scout monitor did not pass")
    expected = value.get("expected")
    if expected != {
        "nodes": 4, "ranks": 32, "ppn": 8, "runs": 3, "warmup": 1,
        "sizes": list(SIZES),
    }:
        raise ScoutError("topology scout runtime contract changed")
    scheduler = value.get("scheduler", {})
    jobspec = value.get("jobspec", {})
    if (scheduler.get("exit_code") != 0
            or scheduler.get("exception_types") != []
            or jobspec.get("queue") != "pdebug"):
        raise ScoutError("topology scout is not a clean pdebug completion")
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
            or jobspec.get("duration_seconds") != 900.0):
        raise ScoutError("topology scout did not request exactly four nodes")
    resource_set = value.get("resource_set")
    if (not isinstance(resource_set, dict)
            or not isinstance(resource_set.get("nodelist"), list)
            or not resource_set["nodelist"]):
        raise ScoutError("topology scout lacks its exact allocated nodes")

    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise ScoutError("topology scout lacks frozen artifacts")
    artifact_rows = []
    runner_sha = None
    freeze_paths = []
    seen_artifacts = set()
    for record in artifacts:
        if (not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ScoutError("topology scout has an invalid artifact record")
        artifact = Path(record["path"])
        if artifact in seen_artifacts:
            raise ScoutError("topology scout has duplicate artifacts")
        seen_artifacts.add(artifact)
        if not artifact.is_file() or sha256(artifact) != record["sha256"]:
            raise ScoutError(f"topology scout artifact changed: {artifact}")
        if artifact.name == "run_compiler_collective_topology_scout.sh":
            runner_sha = record["sha256"]
        if artifact.name == "FROZEN_V3_MANIFEST.json":
            freeze_paths.append(artifact.resolve())
        artifact_rows.append(record)
    if (runner_sha is None or len(freeze_paths) != 1
            or jobspec.get("embedded_script_sha256") != runner_sha):
        raise ScoutError("Flux did not execute the frozen scout runner")

    driver_record = value.get("driver_stdout")
    if not isinstance(driver_record, dict):
        raise ScoutError("topology scout lacks a driver log")
    driver = Path(driver_record.get("path", ""))
    if (not driver.is_file() or sha256(driver) != driver_record.get("sha256")
            or driver.stat().st_size != driver_record.get("bytes")):
        raise ScoutError("topology scout driver log changed")
    if driver.read_text().splitlines() != [
        "SCOUT_CONFIG nodes=4 ranks=32 ppn=8 runs=3 warmup=1 "
        "arms=baseline_auto hierarchical_double_tree",
        "SCOUT_ARM_START arm=baseline_auto",
        "SCOUT_ARM_DONE arm=baseline_auto",
        "SCOUT_ARM_START arm=hierarchical_double_tree",
        "SCOUT_ARM_DONE arm=hierarchical_double_tree",
        "SCOUT_DONE",
    ]:
        raise ScoutError("topology scout driver order is invalid")
    command = jobspec.get("command")
    if (not isinstance(command, list)
            or any(not isinstance(item, str) for item in command)):
        raise ScoutError("topology scout lacks its Flux batch command")
    try:
        script_index = command.index("{{tmpdir}}/script")
    except ValueError:
        raise ScoutError("topology scout lacks its Flux batch command") from None
    arguments = command[script_index + 1:]
    if (len(arguments) != 2
            or Path(arguments[0]).resolve() != freeze_paths[0].parent
            or Path(arguments[1]).resolve() != driver.parent.resolve()):
        raise ScoutError("topology scout batch arguments changed")

    benchmarks = value.get("benchmarks")
    if not isinstance(benchmarks, dict) or set(benchmarks) != set(ARMS):
        raise ScoutError("topology scout benchmark coverage is incomplete")
    rows = {}
    for arm in ARMS:
        benchmark = benchmarks[arm].get("benchmark")
        stdout = benchmarks[arm].get("stdout")
        if (not isinstance(benchmark, dict) or not isinstance(stdout, dict)):
            raise ScoutError(f"topology scout arm is incomplete: {arm}")
        log = Path(stdout.get("path", ""))
        if (not log.is_file() or sha256(log) != stdout.get("sha256")
                or log.stat().st_size != stdout.get("bytes")):
            raise ScoutError(f"topology scout log changed: {arm}")
        results = benchmark.get("results")
        if not isinstance(results, dict):
            raise ScoutError(f"topology scout lacks results: {arm}")
        rows[arm] = {key: float(number) for key, number in results.items()}
    analysis = analyze_rows(rows)
    payload = {
        "schema_version": "gicc-collective-topology-scout-analysis-v1",
        "scope": (
            "Post-v3 exploratory topology hypothesis only; existing two-node "
            "binaries ran at four nodes, so this is not paper or model evidence."
        ),
        "model_invoked": False,
        "application_source_modified": False,
        "job_id": value.get("job_id"),
        "nodelist": resource_set["nodelist"],
        "monitor": str(path.resolve()),
        "monitor_sha256": sha256(path),
        "artifact_count": len(artifact_rows),
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
        print(json.dumps(result["v4_topology_hypothesis"], sort_keys=True))
        return 0
    except (ScoutError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"compiler-collective-topology-scout: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
