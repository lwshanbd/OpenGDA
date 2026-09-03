#!/usr/bin/env python3
"""Quantify the exploratory n=2/n=4 collective topology crossover."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


ALGORITHMS = ("baseline_auto", "hierarchical_double_tree")
SIZES = (
    1024, 4096, 8192, 65536, 262144,
    1048576, 4194304, 8388608, 16777216,
)


class CrossTopologyError(RuntimeError):
    """An input is incomplete or is outside the exploratory contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def geomean(values: list[float]) -> float:
    if not values or any(
        value <= 0 or not math.isfinite(value) for value in values
    ):
        raise CrossTopologyError("latencies must be positive and finite")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def n2_rows(value: Any) -> dict[str, dict[str, float]]:
    if (not isinstance(value, dict)
            or value.get("schema_version")
            != "gicc-collective-control-analysis-v1"
            or value.get("gate_c", {}).get("passed") is not False):
        raise CrossTopologyError("n=2 input is not the closed negative v3 screen")
    per_size = value.get("per_size")
    if not isinstance(per_size, dict) or set(per_size) != {
        str(size) for size in SIZES
    }:
        raise CrossTopologyError("n=2 size coverage is incomplete")
    rows = {algorithm: {} for algorithm in ALGORITHMS}
    for size in SIZES:
        algorithms = per_size[str(size)].get("algorithm_median_us")
        if not isinstance(algorithms, dict):
            raise CrossTopologyError(f"n=2 algorithms missing at {size} B")
        for algorithm in ALGORITHMS:
            try:
                rows[algorithm][str(size)] = float(algorithms[algorithm])
            except (KeyError, TypeError, ValueError) as exc:
                raise CrossTopologyError(
                    f"n=2 algorithm missing at {size} B: {algorithm}"
                ) from exc
    return rows


def n4_rows(value: Any) -> dict[str, dict[str, float]]:
    if (not isinstance(value, dict)
            or value.get("schema_version")
            != "gicc-collective-topology-scout-analysis-v1"
            or value.get("model_invoked") is not False
            or value.get("application_source_modified") is not False
            or value.get("v4_topology_hypothesis", {}).get("promising")
            is not False):
        raise CrossTopologyError("n=4 input is not the closed exploratory scout")
    per_size = value.get("per_size")
    if not isinstance(per_size, dict) or set(per_size) != {
        str(size) for size in SIZES
    }:
        raise CrossTopologyError("n=4 size coverage is incomplete")
    rows = {algorithm: {} for algorithm in ALGORITHMS}
    keys = {
        "baseline_auto": "baseline_us",
        "hierarchical_double_tree": "hierarchical_double_tree_us",
    }
    for size in SIZES:
        record = per_size[str(size)]
        if not isinstance(record, dict):
            raise CrossTopologyError(f"n=4 result missing at {size} B")
        for algorithm, key in keys.items():
            try:
                rows[algorithm][str(size)] = float(record[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise CrossTopologyError(
                    f"n=4 algorithm missing at {size} B: {algorithm}"
                ) from exc
    return rows


def analyze_values(n2_value: Any, n4_value: Any) -> dict[str, Any]:
    by_nodes = {2: n2_rows(n2_value), 4: n4_rows(n4_value)}
    topology_results = {}
    all_samples = {algorithm: [] for algorithm in ALGORITHMS}
    conditional_samples = []
    conditional_rule = {}
    every_point_matches_rule = True
    for nodes, rows in by_nodes.items():
        aggregates = {
            algorithm: geomean(list(rows[algorithm].values()))
            for algorithm in ALGORITHMS
        }
        winner = min(ALGORITHMS, key=lambda name: (aggregates[name], name))
        loser = next(name for name in ALGORITHMS if name != winner)
        point_winners = {}
        for size in SIZES:
            key = str(size)
            point_winner = min(
                ALGORITHMS, key=lambda name: (rows[name][key], name),
            )
            point_winners[key] = point_winner
            every_point_matches_rule &= point_winner == winner
        topology_results[str(nodes)] = {
            "algorithm_geomean_us": aggregates,
            "uniform_winner": winner,
            "loser_over_winner": aggregates[loser] / aggregates[winner],
            "per_size_winners": point_winners,
        }
        conditional_rule[str(nodes)] = winner
        conditional_samples.extend(rows[winner].values())
        for algorithm in ALGORITHMS:
            all_samples[algorithm].extend(rows[algorithm].values())

    uniform = {
        algorithm: geomean(all_samples[algorithm])
        for algorithm in ALGORITHMS
    }
    best_uniform = min(ALGORITHMS, key=lambda name: (uniform[name], name))
    topology_conditional = geomean(conditional_samples)
    headroom = uniform[best_uniform] / topology_conditional
    winner_changes = len(set(conditional_rule.values())) > 1
    return {
        "topologies": topology_results,
        "equal_topology_weight_aggregate": {
            "algorithm_geomean_us": uniform,
            "best_uniform_algorithm": best_uniform,
            "best_uniform_geomean_us": uniform[best_uniform],
            "topology_conditional_geomean_us": topology_conditional,
            "best_uniform_over_topology_conditional": headroom,
        },
        "decision_diagnosis": {
            "observed_topology_rule": conditional_rule,
            "winner_changes_with_nodes": winner_changes,
            "all_18_point_winners_match_topology_rule": (
                every_point_matches_rule
            ),
            "compiler_level_performance_space_observed": headroom > 1.0,
            "llm_unique_value_demonstrated": False,
            "reason": (
                "A two-case deterministic topology rule fits every observed "
                "baseline-versus-tree winner; richer choices or held-out "
                "generalization are required to establish value unique to "
                "language reasoning."
            ),
        },
    }


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CrossTopologyError(f"cannot read JSON {path}: {exc}") from exc


def analyze_files(n2_path: Path, n4_path: Path) -> dict[str, Any]:
    n2_path = n2_path.resolve()
    n4_path = n4_path.resolve()
    analysis = analyze_values(read_json(n2_path), read_json(n4_path))
    payload = {
        "schema_version": "gicc-collective-cross-topology-exploration-v1",
        "scope": (
            "Post-hoc exploratory synthesis only; the n=4 input used n=2 "
            "binaries and is not paper, model, or confirmatory evidence."
        ),
        "model_invoked": False,
        "application_source_modified": False,
        "inputs": {
            "n2_control_screen": {
                "path": str(n2_path), "sha256": sha256(n2_path),
            },
            "n4_topology_scout": {
                "path": str(n4_path), "sha256": sha256(n4_path),
            },
        },
        **analysis,
    }
    result = dict(payload)
    result["result_id"] = fingerprint(payload)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n2", type=Path, required=True)
    parser.add_argument("--n4", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze_files(args.n2, args.n4)
        if args.out.exists():
            raise CrossTopologyError(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result["decision_diagnosis"], sort_keys=True))
        return 0
    except (CrossTopologyError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"compiler-collective-cross-topology: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
