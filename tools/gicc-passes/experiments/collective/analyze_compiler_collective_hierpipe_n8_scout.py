#!/usr/bin/env python3
"""Audit the topology-matched n8 rotated hierarchy-pipeline scout."""

from __future__ import annotations

import argparse
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
GRAPH_ID = "sha256:ce569e2575cfdc924004631dcf96a2d81077520c3198c8f78edde3a4405cf806"
BUNDLE_ID = "sha256:4ed11ecb6a6049b07e18ccf4e88cc7f54f7c1247c119ed1d37a133d6b017f361"
TOPOLOGY_LABEL = "n8"
NODES = 8
RANKS = 64
PPN = 8
RUNS = 7
WARMUP = 2
BATCH_DURATION_SECONDS = 1200.0
LOG_PREFIX = "HIERPIPE_N8"
PROTOCOL_FILENAME = "HIERPIPE_N8_PROTOCOL.md"
RUNNER_FILENAME = "run_compiler_collective_hierpipe_n8_scout.sh"
CONTROLLER_FILENAME = "continue_compiler_collective_hierpipe_n8_scout.sh"
ANALYZER_PATH = Path(__file__).resolve()
SUPPORT_FILENAMES: tuple[str, ...] = ()
RESULT_SCHEMA = "gicc-collective-hierpipe-n8-scout-v1"
CAPACITY_GATE_KEY = "n8_capacity_gate"
RESULT_SCOPE = (
    "One eight-node pdebug allocation with three rotated compiler-control "
    "blocks; no model or application-source change."
)
PROGRAM_NAME = "compiler-collective-hierpipe-n8-scout"
ScoutError = base.ScoutError
sha256 = base.sha256
read_json = base.read_json
geomean = base.geomean
fingerprint = base.fingerprint


def analyze_rows(rows: dict[int, dict[str, dict[str, float]]]) -> dict[str, Any]:
    if set(rows) != {1, 2, 3}:
        raise ScoutError("n8 scout requires blocks 1, 2, and 3")
    wanted_sizes = {str(size) for size in SIZES}
    for replicate, arms in rows.items():
        if set(arms) != set(ARMS):
            raise ScoutError(f"block {replicate} has incomplete arm coverage")
        for arm, values in arms.items():
            if set(values) != wanted_sizes:
                raise ScoutError(
                    f"block {replicate}/{arm} has incomplete size coverage"
                )
            if any(value <= 0 or not math.isfinite(value)
                   for value in values.values()):
                raise ScoutError("n8 scout latency is not finite positive")

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

    small_unpipelined = [
        str(size) for size in SIZES if size <= 262144
        and per_size[str(size)]["winner"] == ARMS[0]
    ]
    large_pipeline = [
        str(size) for size in SIZES if size >= 4194304
        and per_size[str(size)]["winner"] in ARMS[1:]
    ]
    nonuniform_pipeline_sizes = [
        str(size) for size in SIZES
        if per_size[str(size)]["winner"] in ARMS[1:]
        and per_size[str(size)]["winner"] != best_uniform
    ]
    persistence = {}
    for key in nonuniform_pipeline_sizes:
        winner = per_size[key]["winner"]
        ratios = [
            rows[replicate][best_uniform][key]
            / rows[replicate][winner][key]
            for replicate in (1, 2, 3)
        ]
        wins = sum(value > 1.0 for value in ratios)
        persistence[key] = {
            "pipeline_algorithm": winner,
            "best_uniform_algorithm": best_uniform,
            "replicate_speedups": ratios,
            "wins_over_best_uniform": wins,
            "passed": wins >= 2,
        }
    persistence_passed = bool(persistence) and all(
        item["passed"] for item in persistence.values()
    )
    criteria = {
        "small_unpipelined_winner": {
            "observed_sizes": small_unpipelined,
            "threshold": "at least one size <=262144 bytes",
            "passed": bool(small_unpipelined),
        },
        "large_pipeline_winner": {
            "observed_sizes": large_pipeline,
            "threshold": "at least one size >=4194304 bytes",
            "passed": bool(large_pipeline),
        },
        "two_distinct_pooled_size_winners": {
            "observed": len(winners), "threshold": 2,
            "passed": len(winners) >= 2,
        },
        "best_uniform_over_pointwise_geomean": {
            "observed": uniform_over_pointwise, "threshold": 1.05,
            "passed": uniform_over_pointwise >= 1.05,
        },
        "maximum_single_size_headroom": {
            "observed": maximum_headroom, "threshold": 1.10,
            "passed": maximum_headroom >= 1.10,
        },
        "pipeline_winner_rotated_persistence": {
            "observed": {
                key: item["wins_over_best_uniform"]
                for key, item in persistence.items()
            },
            "threshold": "at least 2 of 3 blocks for every nonuniform pipeline winner",
            "passed": persistence_passed,
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
        "pipeline_winner_persistence": persistence,
        CAPACITY_GATE_KEY: {
            "passed": all(item["passed"] for item in criteria.values()),
            "criteria": criteria,
        },
    }


def _validate_bundle(records: dict[Path, str], bundle: Path) -> None:
    freeze_path = (bundle / "FROZEN_V3_MANIFEST.json").resolve()
    graph_path = (bundle / "discovery/graph.json").resolve()
    platform_path = (bundle / "inputs/platform.json").resolve()
    freeze = read_json(freeze_path)
    if not isinstance(freeze, dict):
        raise ScoutError(f"{TOPOLOGY_LABEL} offline freeze is invalid")
    payload = dict(freeze)
    manifest_id = payload.pop("manifest_id", None)
    if manifest_id != BUNDLE_ID or manifest_id != fingerprint(payload):
        raise ScoutError(f"{TOPOLOGY_LABEL} offline freeze ID changed")
    graph = read_json(graph_path)
    platform = read_json(platform_path)
    if (not isinstance(graph, dict) or graph.get("graph_id") != GRAPH_ID
            or freeze.get("graph", {}).get("graph_id") != GRAPH_ID
            or not isinstance(platform, dict)
            or platform.get("topology", {}).get("nodes") != NODES
            or platform.get("topology", {}).get("ranks_per_node") != PPN
            or platform.get("disabled_algorithms")
            != ["hierarchical_direct", "hierarchical_ring"]):
        raise ScoutError(f"{TOPOLOGY_LABEL} graph/profile topology changed")
    for path in (freeze_path, graph_path, platform_path):
        if path not in records:
            raise ScoutError(
                f"{TOPOLOGY_LABEL} bundle artifacts are incomplete"
            )


def _artifact_records(value: dict[str, Any], replicate: int) -> tuple[dict[Path, str], Path]:
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise ScoutError(f"block {replicate} lacks frozen artifacts")
    records: dict[Path, str] = {}
    for record in artifacts:
        if (not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ScoutError(f"block {replicate} has an invalid artifact")
        path = Path(record["path"]).resolve()
        if path in records or not path.is_file() or sha256(path) != record["sha256"]:
            raise ScoutError(f"block {replicate} artifact changed: {path}")
        records[path] = record["sha256"]
    freezes = [path for path in records if path.name == "FROZEN_V3_MANIFEST.json"]
    if len(freezes) != 1:
        raise ScoutError(f"block {replicate} lacks one offline freeze")
    bundle = freezes[0].parent
    script_dir = ANALYZER_PATH.parent
    required = {
        freezes[0],
        (bundle / "discovery/graph.json").resolve(),
        (bundle / "inputs/platform.json").resolve(),
        (script_dir / PROTOCOL_FILENAME).resolve(),
        (script_dir / RUNNER_FILENAME).resolve(),
        (script_dir / CONTROLLER_FILENAME).resolve(),
        (script_dir / "monitor_compiler_collective_replicate.py").resolve(),
        ANALYZER_PATH,
    }
    required.update(
        (script_dir / filename).resolve() for filename in SUPPORT_FILENAMES
    )
    for arm in ARMS:
        required.update({
            (bundle / f"binaries/{arm}/compiler_collective_eval").resolve(),
            (bundle / f"binaries/{arm}/build-provenance.json").resolve(),
            (bundle / f"controls/{arm}-hint.json").resolve(),
        })
    if set(records) != required:
        raise ScoutError(f"block {replicate} artifact set changed")
    _validate_bundle(records, bundle)
    return records, bundle


def _validate_monitor(path: Path) -> tuple[int, dict[str, dict[str, float]], dict[str, Any]]:
    value = read_json(path)
    if (not isinstance(value, dict)
            or value.get("schema_version")
            != "gicc-collective-replicate-job-monitor-v1"
            or value.get("state") != "passed"):
        raise ScoutError(
            f"{TOPOLOGY_LABEL} scout monitor did not pass: {path}"
        )
    replicate = value.get("replicate")
    if replicate not in ORDERS:
        raise ScoutError(f"{TOPOLOGY_LABEL} scout replicate is invalid")
    expected = {
        "nodes": NODES, "ranks": RANKS, "ppn": PPN,
        "runs": RUNS, "warmup": WARMUP,
        "sizes": list(SIZES),
    }
    jobspec = value.get("jobspec", {})
    scheduler = value.get("scheduler", {})
    resources = [{
        "type": "node", "count": NODES,
        "with": [{
            "type": "slot", "count": PPN, "label": "task",
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
            or jobspec.get("duration_seconds") != BATCH_DURATION_SECONDS
            or jobspec.get("resources") != resources):
        raise ScoutError(f"block {replicate} runtime contract changed")
    allocation = value.get("resource_set", {}).get("nodelist")
    if not isinstance(allocation, list) or not allocation:
        raise ScoutError(f"block {replicate} lacks an exact node list")
    records, bundle = _artifact_records(value, replicate)
    runner = (ANALYZER_PATH.parent / RUNNER_FILENAME).resolve()
    if jobspec.get("embedded_script_sha256") != records[runner]:
        raise ScoutError(
            f"Flux did not execute the frozen {TOPOLOGY_LABEL} runner"
        )

    driver_record = value.get("driver_stdout")
    driver_stderr = value.get("driver_stderr")
    if not isinstance(driver_record, dict) or not isinstance(driver_stderr, dict):
        raise ScoutError(f"block {replicate} lacks a driver log")
    driver = Path(driver_record.get("path", ""))
    driver_err = Path(driver_stderr.get("path", ""))
    if (not driver.is_file() or sha256(driver) != driver_record.get("sha256")
            or driver.stat().st_size != driver_record.get("bytes")
            or not driver_err.is_file()
            or sha256(driver_err) != driver_stderr.get("sha256")
            or driver_err.stat().st_size != driver_stderr.get("bytes")
            or driver_err.stat().st_size != 0):
        raise ScoutError(f"block {replicate} driver log changed")
    expected_lines = [
        f"{LOG_PREFIX}_CONFIG replicate={replicate} "
        f"arms={' '.join(ORDERS[replicate])}",
    ]
    for arm in ORDERS[replicate]:
        expected_lines.extend([
            f"{LOG_PREFIX}_ARM_START replicate={replicate} arm={arm}",
            f"{LOG_PREFIX}_ARM_DONE replicate={replicate} arm={arm}",
        ])
    expected_lines.append(f"{LOG_PREFIX}_DONE replicate={replicate}")
    if driver.read_text().splitlines() != expected_lines:
        raise ScoutError(f"block {replicate} arm order changed")
    command = jobspec.get("command")
    try:
        script_index = command.index("{{tmpdir}}/script")
    except (AttributeError, ValueError):
        raise ScoutError(
            f"{TOPOLOGY_LABEL} scout lacks its Flux batch command"
        ) from None
    arguments = command[script_index + 1:]
    output_root = driver.parent.parent.resolve()
    if (len(arguments) != 2
            or Path(arguments[0]).resolve() != bundle
            or Path(arguments[1]).resolve() != output_root):
        raise ScoutError(f"{TOPOLOGY_LABEL} scout batch arguments changed")

    benchmarks = value.get("benchmarks")
    if not isinstance(benchmarks, dict) or set(benchmarks) != set(ARMS):
        raise ScoutError(f"block {replicate} lacks complete arm coverage")
    rows = {}
    for arm in ARMS:
        record = benchmarks[arm]
        if not isinstance(record, dict):
            raise ScoutError(f"block {replicate}/{arm} is invalid")
        benchmark = record.get("benchmark")
        stdout = record.get("stdout")
        stderr = record.get("stderr")
        if (not isinstance(benchmark, dict) or not isinstance(stdout, dict)
                or not isinstance(stderr, dict)):
            raise ScoutError(f"block {replicate}/{arm} is incomplete")
        log = Path(stdout.get("path", ""))
        err = Path(stderr.get("path", ""))
        if (not log.is_file() or sha256(log) != stdout.get("sha256")
                or log.stat().st_size != stdout.get("bytes")
                or not err.is_file() or sha256(err) != stderr.get("sha256")
                or err.stat().st_size != stderr.get("bytes")):
            raise ScoutError(f"block {replicate}/{arm} log changed")
        reparsed = monitor_base.validate_output(
            log, arm, list(SIZES), NODES, RANKS, PPN, RUNS, WARMUP,
        )
        if reparsed != benchmark:
            raise ScoutError(f"block {replicate}/{arm} monitor/raw log mismatch")
        rows[arm] = {
            key: float(number) for key, number in benchmark["results"].items()
        }
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


def analyze_monitors(monitor_paths: list[Path]) -> dict[str, Any]:
    if len(monitor_paths) != 3:
        raise ScoutError("n8 scout requires exactly three block monitors")
    rows = {}
    summaries = []
    for path in monitor_paths:
        replicate, block, summary = _validate_monitor(path.resolve())
        if replicate in rows:
            raise ScoutError("n8 scout has a duplicate block")
        rows[replicate] = block
        summaries.append(summary)
    if set(rows) != {1, 2, 3}:
        raise ScoutError("n8 scout blocks are incomplete")
    if len({item["job_id"] for item in summaries}) != 1:
        raise ScoutError("n8 scout blocks did not share one allocation")
    if len({json.dumps(item["nodelist"]) for item in summaries}) != 1:
        raise ScoutError("n8 scout blocks did not share exact nodes")
    if len({item["artifact_set_id"] for item in summaries}) != 1:
        raise ScoutError("n8 scout blocks used different artifacts")
    analysis = analyze_rows(rows)
    payload = {
        "schema_version": RESULT_SCHEMA,
        "scope": RESULT_SCOPE,
        "model_invoked": False,
        "application_source_modified": False,
        "graph_id": GRAPH_ID,
        "bundle_id": BUNDLE_ID,
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
    parser.add_argument("--monitor", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze_monitors([path.resolve() for path in args.monitor])
        if args.out.exists():
            raise ScoutError(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result[CAPACITY_GATE_KEY], sort_keys=True))
        return 0
    except (ScoutError, monitor_base.MonitorError, OSError, KeyError,
            TypeError, ValueError) as exc:
        print(f"{PROGRAM_NAME}: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
