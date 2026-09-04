#!/usr/bin/env python3
"""Apply the frozen descriptive headroom gate to a passed loop scout."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import statistics
from typing import Any


BATCHES = (4, 64)
REPLICATES = tuple(range(1, 7))
ARMS = ("baseline", "reused")
SIZES = (
    "1B", "2B", "4B", "8B", "64B", "256B", "1KB", "4KB",
    "16KB", "64KB", "256KB", "512KB", "1MB", "2MB", "4MB",
    "16MB",
)
SMALL_SIZES = (
    "1B", "2B", "4B", "8B", "64B", "256B", "1KB", "4KB",
    "16KB", "64KB",
)


def geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def analyze(monitor: dict[str, Any]) -> dict[str, Any]:
    if monitor.get("state") != "passed":
        raise ValueError("reused-loop-descriptor monitor did not pass")
    batches: dict[str, Any] = {}
    all_clusters = []
    operation_count_audits = 0
    for batch in BATCHES:
        clusters = []
        full_clusters = []
        for replicate in REPLICATES:
            pair = monitor["runs"][str(replicate)][str(batch)]
            expected_enqueues = 31 * batch * len(SIZES)
            for arm in ARMS:
                actual = int(pair[arm]["enqueue_actual"])
                expected = int(pair[arm]["enqueue_expected"])
                if actual != expected_enqueues or expected != expected_enqueues:
                    raise ValueError(
                        f"enqueue audit mismatch for rep{replicate} "
                        f"batch{batch} {arm}"
                    )
                operation_count_audits += 1
            per_size = pair["per_size_speedup"]
            clusters.append(geometric_mean([
                float(per_size[size]) for size in SMALL_SIZES
            ]))
            full_clusters.append(geometric_mean([
                float(per_size[size]) for size in SIZES
            ]))
        all_clusters.extend(clusters)
        median = statistics.median(clusters)
        wins = sum(value > 1.0 for value in clusters)
        batches[str(batch)] = {
            "small_message_pair_geomeans": clusters,
            "small_message_median_speedup": median,
            "small_message_wins": wins,
            "all_size_pair_geomeans": full_clusters,
            "promising": wins >= 4 and median >= 1.02,
        }
    overall = geometric_mean(all_clusters)
    no_regression = all(
        batches[str(batch)]["small_message_median_speedup"] >= 0.98
        for batch in BATCHES
    )
    passed = (
        any(batches[str(batch)]["promising"] for batch in BATCHES)
        and overall >= 1.01
        and no_regression
    )
    return {
        "schema_version": "gicc-reused-loop-descriptor-analysis-v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "correctness_gate": {
            "passed": True,
            "paired_clusters": len(all_clusters),
            "network_operation_count_audits": operation_count_audits,
        },
        "target_stratum": {
            "sizes": list(SMALL_SIZES),
            "reason": "host descriptor preparation is mechanism-relevant",
        },
        "batches": batches,
        "all_cluster_small_message_geomean_speedup": overall,
        "oracle_headroom_gate": {
            "passed": passed,
            "criterion": (
                "one batch >=4/6 wins with median >=1.02; all-cluster "
                "geomean >=1.01; no batch median <0.98"
            ),
            "interpretation": (
                "eligible for an independently frozen multi-allocation confirmation"
                if passed
                else "keep reused-loop-descriptor candidate model-invisible"
            ),
            "paper_claim": False,
            "provider_protocol_permitted": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = analyze(json.loads(args.monitor.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    result["monitor"] = str(args.monitor.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
