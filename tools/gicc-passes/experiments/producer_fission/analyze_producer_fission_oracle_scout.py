#!/usr/bin/env python3
"""Summarize paired runtimes from a passed producer-fission monitor."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    monitor = json.loads(args.monitor.read_text(encoding="utf-8"))
    if monitor.get("state") != "passed":
        raise SystemExit("producer-fission monitor did not pass")

    sizes: dict[str, Any] = {}
    promising_sizes = []
    all_speedups = []
    for size in (1024, 4096):
        speedups = [
            float(monitor["runs"][str(rep)][str(size)]["speedup"])
            for rep in range(1, 5)
        ]
        all_speedups.extend(speedups)
        median = statistics.median(speedups)
        wins = sum(value > 1.0 for value in speedups)
        passed = wins >= 3 and median >= 1.03
        if passed:
            promising_sizes.append(size)
        sizes[str(size)] = {
            "paired_speedups": speedups,
            "median_speedup": median,
            "min_speedup": min(speedups),
            "max_speedup": max(speedups),
            "wins": wins,
            "gate_passed": passed,
        }

    geometric_mean = math.exp(
        sum(math.log(value) for value in all_speedups) / len(all_speedups)
    )
    result = {
        "schema_version": "gicc-producer-fission-oracle-analysis-v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "monitor": str(args.monitor.resolve()),
        "correctness_gate": {"passed": True, "paired_checks": 8},
        "sizes": sizes,
        "all_pairs_geometric_mean_speedup": geometric_mean,
        "oracle_headroom_gate": {
            "passed": bool(promising_sizes),
            "promising_sizes": promising_sizes,
            "criterion": "at least one size has >=3/4 wins and median speedup >=1.03",
            "interpretation": (
                "eligible for a larger compiler-oracle confirmation"
                if promising_sizes
                else "mask producer-frontier fission from model evaluation"
            ),
            "paper_claim": False,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
