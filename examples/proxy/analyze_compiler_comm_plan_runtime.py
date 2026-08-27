#!/usr/bin/env python3
"""Validate structural-plan runtime logs and quantify the exact action oracle."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "proxy"))

from analyze_compiler_lto_eval import OFFSETS, SCENARIOS, expected_hash  # noqa: E402


STRUCTURAL = ("adjacent-k16-grid1", "far-k64-grid8")
KERNEL_SCENARIO = {
    "eval_adjacent_batch": "adjacent-k16-grid1",
    "eval_far_batch": "far-k64-grid8",
}
FIXED_ROUTES = {
    "tiny-k1-grid1": (1, 0),
    "reuse-k32-grid1": (32, 0),
    "large-k1-grid8": (1, 0),
    "dynamic-k1-grid1": (0, 2),
    "static4-k4-grid4": (4, 0),
}
ACTION_ROUTES = {
    "eval_adjacent_batch": {
        "proxy_device": (0, 17),
        "trigger_descriptor_batch": (16, 0),
        "trigger_coalesced_loop": (1, 0),
    },
    "eval_far_batch": {
        "proxy_device": (0, 65),
        "trigger_descriptor_batch": (64, 0),
        "trigger_coalesced_loop": (1, 0),
    },
}
RESULT = re.compile(r"([a-z0-9_]+)=([^ ]+)")


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def bootstrap_ratio_ci(
    numerator: list[float], denominator: list[float], seed: int = 20260827,
) -> tuple[float, float]:
    if len(numerator) != len(denominator) or not numerator:
        raise ValueError("paired bootstrap requires equal non-empty vectors")
    ratios = [left / right for left, right in zip(numerator, denominator)]
    rng = random.Random(seed)
    estimates = []
    for _ in range(10000):
        estimates.append(geomean([
            ratios[rng.randrange(len(ratios))] for _ in ratios
        ]))
    estimates.sort()
    return estimates[249], estimates[9749]


def load_contract(path: Path) -> tuple[list[str], dict[str, dict[str, Any]]]:
    value = json.loads(path.read_text())
    arms = value.get("arms")
    if not isinstance(arms, list) or len(arms) != 9:
        raise ValueError("control manifest must contain nine arms")
    names = [arm["name"] for arm in arms]
    if len(set(names)) != 9:
        raise ValueError("control manifest has duplicate arms")
    return names, {arm["name"]: arm for arm in arms}


def routes_for(arm: dict[str, Any]) -> dict[str, tuple[int, int]]:
    routes = dict(FIXED_ROUTES)
    for selection in arm["selections"]:
        kernel = selection["kernel"]
        routes[KERNEL_SCENARIO[kernel]] = ACTION_ROUTES[kernel][selection["kind"]]
    if set(routes) != set(SCENARIOS):
        raise ValueError(f"{arm['name']}: incomplete route contract")
    return routes


def parse_log(
    path: Path, names: list[str], contract: dict[str, dict[str, Any]],
) -> tuple[str, int, dict[str, dict[str, Any]]]:
    arm = next((name for name in names if path.name.endswith(f"-{name}.log")), None)
    if arm is None:
        raise ValueError(f"cannot infer arm from {path}")
    match = re.search(r"rep([0-9]+)-", path.name)
    if match is None:
        raise ValueError(f"cannot infer replicate from {path}")
    rep = int(match.group(1))
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text().splitlines():
        if not line.startswith("COMPILER_LTO_EVAL "):
            continue
        fields = dict(RESULT.findall(line))
        scenario = fields["scenario"]
        if scenario in rows:
            raise ValueError(f"{path}: duplicate scenario {scenario}")
        rows[scenario] = {
            "runs": int(fields["runs"]),
            "median_us": float(fields["median_us"]),
            "p25_us": float(fields["p25_us"]),
            "p75_us": float(fields["p75_us"]),
            "staged": int(fields["staged"]),
            "pushed": int(fields["pushed"]),
            "hash": fields["hash"],
            "data": fields["data"],
        }
    if set(rows) != set(SCENARIOS):
        raise ValueError(f"{path}: missing or unknown scenarios")
    expected_routes = routes_for(contract[arm])
    for scenario, row in rows.items():
        if row["data"] != "OK":
            raise ValueError(f"{path}: {scenario} data={row['data']}")
        base, size = OFFSETS[scenario]
        if row["hash"] != expected_hash(base, size):
            raise ValueError(f"{path}: {scenario} data hash mismatch")
        staged, pushed = expected_routes[scenario]
        wanted = (staged * row["runs"], pushed * row["runs"])
        observed = (row["staged"], row["pushed"])
        if observed != wanted:
            raise ValueError(
                f"{path}: {scenario} routes={observed}, expected={wanted}"
            )
    return arm, rep, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    names, contract = load_contract(args.manifest)
    parsed: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    for path in args.logs:
        arm, rep, rows = parse_log(path, names, contract)
        if arm in parsed.setdefault(rep, {}):
            raise SystemExit(f"duplicate rep={rep} arm={arm}")
        parsed[rep][arm] = rows
    for rep, arms in parsed.items():
        if set(arms) != set(names):
            raise SystemExit(f"rep={rep}: expected all nine arms")

    per_rep_scores: dict[str, list[float]] = {name: [] for name in names}
    aggregate: dict[str, Any] = {}
    for name in names:
        scenario_values = {
            scenario: [
                parsed[rep][name][scenario]["median_us"] for rep in sorted(parsed)
            ]
            for scenario in SCENARIOS
        }
        medians = {
            scenario: geomean(values)
            for scenario, values in scenario_values.items()
        }
        for rep in sorted(parsed):
            per_rep_scores[name].append(geomean([
                parsed[rep][name][scenario]["median_us"]
                for scenario in STRUCTURAL
            ]))
        aggregate[name] = {
            "geomean_median_us": medians,
            "structural_score_us": geomean([
                medians[scenario] for scenario in STRUCTURAL
            ]),
            "selections": contract[name]["selections"],
        }

    oracle = min(names, key=lambda name: aggregate[name]["structural_score_us"])
    baseline = "plan_tt"
    ratio = aggregate[baseline]["structural_score_us"] / aggregate[oracle][
        "structural_score_us"
    ]
    ci_low, ci_high = bootstrap_ratio_ci(
        per_rep_scores[baseline], per_rep_scores[oracle]
    )
    per_rep_oracle = [
        min(names, key=lambda name: per_rep_scores[name][index])
        for index in range(len(parsed))
    ]
    summary = {
        "schema_version": "gicc-communication-plan-runtime-summary-v1",
        "replicate_count": len(parsed),
        "structural_scenarios": list(STRUCTURAL),
        "aggregate": aggregate,
        "exact_oracle": {
            "arm": oracle,
            "stable_across_replicates": len(set(per_rep_oracle)) == 1,
            "per_replicate_arm": per_rep_oracle,
            "trigger_batch_baseline": baseline,
            "baseline_over_oracle": ratio,
            "paired_bootstrap_95_ci": [ci_low, ci_high],
        },
        "scope": (
            "Exact oracle over the two compiler-proved structural opportunities "
            "and their 3x3 candidate product; no model result."
        ),
    }
    print("arm      adjacent_us    far_us   structural_score_us")
    for name in sorted(names):
        row = aggregate[name]
        print(
            f"{name:<8} "
            f"{row['geomean_median_us']['adjacent-k16-grid1']:11.3f} "
            f"{row['geomean_median_us']['far-k64-grid8']:9.3f} "
            f"{row['structural_score_us']:21.3f}"
        )
    print(
        f"oracle={oracle} baseline_over_oracle={ratio:.6f} "
        f"ci95=[{ci_low:.6f},{ci_high:.6f}]"
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
