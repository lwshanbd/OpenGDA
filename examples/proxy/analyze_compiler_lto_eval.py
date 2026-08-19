#!/usr/bin/env python3
"""Validate compiler_lto_eval control logs and summarize paired medians."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


ARMS = ("default", "proxy", "trigger", "hand")
SCENARIOS = (
    "tiny-k1-grid1",
    "reuse-k32-grid1",
    "adjacent-k16-grid1",
    "far-k64-grid8",
    "large-k1-grid8",
    "dynamic-k1-grid1",
    "static4-k4-grid4",
)
FORCED_ACTIONS = {
    # LTO proves that the device-loaded offsets are not host-knowable, so the
    # bridge exposes no competing lowering for this site.  Separate binaries
    # may still have layout noise, but that is not decision regret.
    "dynamic-k1-grid1": "proxy",
}
OFFSETS = {
    "tiny-k1-grid1": (0, 256),
    "reuse-k32-grid1": (1 << 20, 4096),
    "adjacent-k16-grid1": (2 << 20, 16 * 4096),
    "far-k64-grid8": (4 << 20, 64 * 4096),
    "large-k1-grid8": (8 << 20, 1 << 20),
    "dynamic-k1-grid1": (12 << 20, 4096),
    "static4-k4-grid4": (14 << 20, 4 * 4096),
}

# Per timed iteration: (host-staged descriptors, proxy-ring commands).
# A proxy kernel adds one quiet per active lane; static4 drains four lanes.
ROUTES = {
    "default": {
        "tiny-k1-grid1": (1, 0), "reuse-k32-grid1": (32, 0),
        "adjacent-k16-grid1": (16, 0), "far-k64-grid8": (64, 0),
        "large-k1-grid8": (1, 0), "dynamic-k1-grid1": (0, 2),
        "static4-k4-grid4": (4, 0),
    },
    "trigger": {
        "tiny-k1-grid1": (1, 0), "reuse-k32-grid1": (32, 0),
        "adjacent-k16-grid1": (16, 0), "far-k64-grid8": (64, 0),
        "large-k1-grid8": (1, 0), "dynamic-k1-grid1": (0, 2),
        "static4-k4-grid4": (4, 0),
    },
    "proxy": {
        "tiny-k1-grid1": (0, 2), "reuse-k32-grid1": (0, 33),
        "adjacent-k16-grid1": (0, 17), "far-k64-grid8": (0, 65),
        "large-k1-grid8": (0, 2), "dynamic-k1-grid1": (0, 2),
        "static4-k4-grid4": (0, 8),
    },
    "hand": {
        "tiny-k1-grid1": (0, 2), "reuse-k32-grid1": (32, 0),
        "adjacent-k16-grid1": (16, 0), "far-k64-grid8": (64, 0),
        "large-k1-grid8": (0, 2), "dynamic-k1-grid1": (0, 2),
        "static4-k4-grid4": (0, 8),
    },
}

RESULT = re.compile(r"([a-z0-9_]+)=([^ ]+)")


def expected_hash(base: int, size: int) -> str:
    value = 1469598103934665603
    for index in range(base, base + size):
        byte = 1 + ((index // 256) * 17 + index) % 251
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def parse_log(path: Path) -> tuple[str, int, dict[str, dict[str, Any]]]:
    arm = next((candidate for candidate in ARMS
                if path.name.endswith(f"-{candidate}.log")), None)
    if arm is None:
        raise ValueError(f"cannot infer arm from {path}")
    rep_match = re.search(r"rep([0-9]+)-", path.name)
    if rep_match is None:
        raise ValueError(f"cannot infer replicate from {path}")
    rep = int(rep_match.group(1))
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
        raise ValueError(
            f"{path}: scenarios={sorted(rows)}, expected={sorted(SCENARIOS)}"
        )
    for scenario, row in rows.items():
        if row["data"] != "OK":
            raise ValueError(f"{path}: {scenario} data={row['data']}")
        base, size = OFFSETS[scenario]
        wanted_hash = expected_hash(base, size)
        if row["hash"] != wanted_hash:
            raise ValueError(
                f"{path}: {scenario} hash={row['hash']} expected={wanted_hash}"
            )
        staged, pushed = ROUTES[arm][scenario]
        staged *= row["runs"]
        pushed *= row["runs"]
        if (row["staged"], row["pushed"]) != (staged, pushed):
            raise ValueError(
                f"{path}: {scenario} routes={(row['staged'], row['pushed'])} "
                f"expected={(staged, pushed)}"
            )
    return arm, rep, rows


def materialized_action(arm: str, scenario: str) -> str:
    """Name the route an arm actually materializes for this deployment."""
    route = ROUTES[arm][scenario]
    matches = [
        action for action in ("proxy", "trigger")
        if ROUTES[action][scenario] == route
    ]
    if len(matches) == 1:
        return matches[0]
    # The proxy-only legality case gives every arm the same route.
    if scenario in FORCED_ACTIONS:
        return FORCED_ACTIONS[scenario]
    raise ValueError(f"cannot identify action for arm={arm} scenario={scenario}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    parsed: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
    for path in args.logs:
        arm, rep, rows = parse_log(path)
        if arm in parsed.setdefault(rep, {}):
            raise SystemExit(f"duplicate rep={rep} arm={arm}")
        parsed[rep][arm] = rows
    for rep, arms in parsed.items():
        if set(arms) != set(ARMS):
            raise SystemExit(f"rep={rep}: arms={sorted(arms)}, expected={list(ARMS)}")

    summary: dict[str, Any] = {"replicates": {}, "aggregate": {}}
    decision_ratios: dict[str, list[float]] = {"default": [], "hand": []}
    decision_matches: dict[str, int] = {"default": 0, "hand": 0}
    print("scenario                  default    proxy  trigger     hand  best-action")
    for scenario in SCENARIOS:
        per_arm: dict[str, list[float]] = {arm: [] for arm in ARMS}
        for rep in sorted(parsed):
            for arm in ARMS:
                per_arm[arm].append(parsed[rep][arm][scenario]["median_us"])
        geometric = {
            arm: math.exp(sum(math.log(x) for x in values) / len(values))
            for arm, values in per_arm.items()
        }
        forced = FORCED_ACTIONS.get(scenario)
        if forced is not None:
            best = f"{forced}-only"
            scenario_summary = {
                "geomean_median_us": geometric,
                "legal_actions": [forced],
                "decision_bearing": False,
                "best_uniform_legal_control": None,
                "default_over_best": None,
                "hand_over_best": None,
                "note": (
                    "Every arm materializes the forced proxy action; timing "
                    "differences between separately linked binaries are not "
                    "optimization regret."
                ),
            }
        else:
            best = min(("proxy", "trigger"), key=lambda arm: geometric[arm])
            default_ratio = geometric["default"] / geometric[best]
            hand_ratio = geometric["hand"] / geometric[best]
            decision_ratios["default"].append(default_ratio)
            decision_ratios["hand"].append(hand_ratio)
            per_rep_best = [
                min(
                    ("proxy", "trigger"),
                    key=lambda arm: parsed[rep][arm][scenario]["median_us"],
                )
                for rep in sorted(parsed)
            ]
            default_action = materialized_action("default", scenario)
            hand_action = materialized_action("hand", scenario)
            decision_matches["default"] += int(default_action == best)
            decision_matches["hand"] += int(hand_action == best)
            scenario_summary = {
                "geomean_median_us": geometric,
                "legal_actions": ["proxy", "trigger"],
                "decision_bearing": True,
                "best_uniform_legal_control": best,
                "per_replicate_best_action": per_rep_best,
                "stable_winner": len(set(per_rep_best)) == 1,
                "default_materialized_action": default_action,
                "default_matches_best": default_action == best,
                "hand_materialized_action": hand_action,
                "hand_matches_best": hand_action == best,
                "default_over_best": default_ratio,
                "hand_over_best": hand_ratio,
            }
        summary["aggregate"][scenario] = scenario_summary
        print(f"{scenario:<25} " + " ".join(
            f"{geometric[arm]:8.3f}" for arm in ARMS
        ) + f"  {best}")

    summary["decision_summary"] = {
        "scenario_count": len(SCENARIOS) - len(FORCED_ACTIONS),
        "excluded_forced_scenarios": sorted(FORCED_ACTIONS),
        "stable_winner_scenarios": sum(
            value.get("stable_winner", False)
            for value in summary["aggregate"].values()
        ),
        "default_matches_best": decision_matches["default"],
        "hand_matches_best": decision_matches["hand"],
        "geomean_default_over_best_uniform": math.exp(
            sum(math.log(x) for x in decision_ratios["default"])
            / len(decision_ratios["default"])
        ),
        "geomean_hand_over_best_uniform": math.exp(
            sum(math.log(x) for x in decision_ratios["hand"])
            / len(decision_ratios["hand"])
        ),
    }

    for rep, arms in sorted(parsed.items()):
        summary["replicates"][str(rep)] = {
            arm: {scenario: row["median_us"] for scenario, row in rows.items()}
            for arm, rows in arms.items()
        }
    summary["scope"] = (
        "Control pairing only; no model result. Forced-action scenarios are "
        "excluded from decision regret. best_uniform_legal_control is not "
        "the full mixed-action oracle for the four-site static group."
    )
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
