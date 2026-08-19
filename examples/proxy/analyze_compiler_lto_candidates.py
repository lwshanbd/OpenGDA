#!/usr/bin/env python3
"""Validate paired compiler-policy binaries and score them against oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "gicc-passes" / "python"))

import gicc_llm_bridge as bridge  # noqa: E402
from analyze_compiler_lto_eval import (  # noqa: E402
    FORCED_ACTIONS,
    OFFSETS,
    RESULT,
    ROUTES,
    SCENARIOS,
    expected_hash,
    materialized_action,
)
from compiler_lto_eval import SCENARIO_FOR_KERNEL  # noqa: E402


KERNEL_FOR_SCENARIO = {value: key for key, value in SCENARIO_FOR_KERNEL.items()}
ARM_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]+$")


class AnalysisError(ValueError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_binding(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("response must be ARM=PATH")
    arm, raw_path = value.split("=", 1)
    if not ARM_PATTERN.fullmatch(arm):
        raise argparse.ArgumentTypeError(f"invalid arm name {arm!r}")
    return arm, Path(raw_path)


def parse_replicates(value: str) -> list[int]:
    try:
        replicates = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "replicates must be comma-separated integers"
        ) from exc
    if not replicates or len(set(replicates)) != len(replicates):
        raise argparse.ArgumentTypeError("replicates must be nonempty and unique")
    return replicates


def validated_response(dossier: dict[str, Any], path: Path) -> dict[str, Any]:
    response = read_json(path)
    _, accepted, errors = bridge.decision_to_hint(dossier, response)
    if not accepted or errors:
        raise AnalysisError(f"bridge rejects {path}: {errors}")
    return response


def response_actions(dossier: dict[str, Any], response: dict[str, Any]) \
        -> dict[str, list[str]]:
    by_kernel: dict[str, list[dict[str, Any]]] = {}
    for site in dossier["sites"]:
        by_kernel.setdefault(site["kernel"], []).append(site)
    output: dict[str, list[str]] = {}
    for kernel, sites in by_kernel.items():
        sites.sort(key=lambda site: site["site_id"])
        output[kernel] = [
            response["decisions"][site["site_id"]]["action"] for site in sites
        ]
    return output


def route_for_actions(sites: list[dict[str, Any]], actions: list[str]) \
        -> tuple[int, int]:
    if len(sites) != len(actions):
        raise AnalysisError("site/action cardinality mismatch")
    physical = ["trigger" if action == "default" else action
                for action in actions]
    if any(action not in {"proxy", "trigger"} for action in physical):
        raise AnalysisError(f"unsupported route action(s) {physical}")

    if len(sites) == 1 and sites[0].get("in_loop") is True:
        batch = sites[0].get("batch_size")
        if not isinstance(batch, int) or batch <= 0:
            raise AnalysisError("modeled loop lacks batch_size")
        return (batch, 0) if physical[0] == "trigger" else (0, batch + 1)

    staged = sum(action == "trigger" for action in physical)
    proxy_indices = [index for index, action in enumerate(physical)
                     if action == "proxy"]
    # Static independent sites are assigned sorted block slots. A proxy quiet
    # drains lanes 0..highest-used-slot; a one-site kernel drains one lane.
    active_lanes = max(proxy_indices) + 1 if proxy_indices else 0
    pushed = len(proxy_indices) + active_lanes
    return staged, pushed


def routes_for_response(dossier: dict[str, Any], response: dict[str, Any]) \
        -> dict[str, tuple[int, int]]:
    actions = response_actions(dossier, response)
    by_kernel: dict[str, list[dict[str, Any]]] = {}
    for site in dossier["sites"]:
        by_kernel.setdefault(site["kernel"], []).append(site)
    routes: dict[str, tuple[int, int]] = {}
    for scenario, kernel in KERNEL_FOR_SCENARIO.items():
        sites = sorted(by_kernel[kernel], key=lambda site: site["site_id"])
        routes[scenario] = route_for_actions(sites, actions[kernel])
    return routes


def routes_for_builtin(arm: str) -> dict[str, tuple[int, int]]:
    if arm not in ROUTES:
        raise AnalysisError(f"no route contract for built-in arm {arm!r}")
    return ROUTES[arm]


def parse_log(path: Path, routes: dict[str, tuple[int, int]]) \
        -> tuple[str, int, str, dict[str, dict[str, Any]]]:
    match = re.fullmatch(r"rep([0-9]+)-([a-zA-Z0-9_.-]+)\.log", path.name)
    if match is None:
        raise AnalysisError(f"cannot infer replicate/arm from {path}")
    rep, arm = int(match.group(1)), match.group(2)
    lines = path.read_text().splitlines()
    provenance = [line for line in lines if line.startswith("RUN rep=")]
    if len(provenance) != 1:
        raise AnalysisError(f"{path}: expected one binary provenance line")
    provenance_fields = dict(RESULT.findall(provenance[0]))
    if provenance_fields.get("arm") != arm:
        raise AnalysisError(f"{path}: provenance arm mismatch")
    try:
        provenance_rep = int(provenance_fields.get("rep", ""))
    except ValueError as exc:
        raise AnalysisError(f"{path}: invalid provenance replicate") from exc
    if provenance_rep != rep:
        raise AnalysisError(f"{path}: provenance replicate mismatch")
    binary = Path(provenance_fields.get("binary", ""))
    if binary.name != f"compiler_lto_eval_{arm}":
        raise AnalysisError(f"{path}: provenance binary/arm mismatch")
    binary_sha256 = provenance_fields.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", binary_sha256):
        raise AnalysisError(f"{path}: invalid binary sha256")

    rows: dict[str, dict[str, Any]] = {}
    for line in lines:
        if not line.startswith("COMPILER_LTO_EVAL "):
            continue
        fields = dict(RESULT.findall(line))
        scenario = fields["scenario"]
        if scenario in rows:
            raise AnalysisError(f"{path}: duplicate scenario {scenario}")
        rows[scenario] = {
            "runs": int(fields["runs"]),
            "median_us": float(fields["median_us"]),
            "p25_us": float(fields["p25_us"]),
            "p75_us": float(fields["p75_us"]),
            "staged": int(fields["staged"]),
            "pushed": int(fields["pushed"]),
            "hash": fields["hash"],
            "data": fields["data"],
            "binary_sha256": binary_sha256,
        }
    if set(rows) != set(SCENARIOS):
        raise AnalysisError(f"{path}: incomplete scenario set")
    for scenario, row in rows.items():
        if row["data"] != "OK":
            raise AnalysisError(f"{path}: {scenario} data={row['data']}")
        base, size = OFFSETS[scenario]
        if row["hash"] != expected_hash(base, size):
            raise AnalysisError(f"{path}: {scenario} payload hash mismatch")
        staged, pushed = routes[scenario]
        wanted = staged * row["runs"], pushed * row["runs"]
        actual = row["staged"], row["pushed"]
        if actual != wanted:
            raise AnalysisError(
                f"{path}: {scenario} routes={actual}, expected={wanted}"
            )
    return arm, rep, binary_sha256, rows


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def builtin_actions(arm: str) -> dict[str, list[str]]:
    actions: dict[str, list[str]] = {}
    for scenario, kernel in KERNEL_FOR_SCENARIO.items():
        if scenario in FORCED_ACTIONS:
            action = FORCED_ACTIONS[scenario]
        else:
            action = materialized_action(arm, scenario)
        count = 4 if kernel == "eval_static4_parallel" else 1
        actions[kernel] = [action] * count
    return actions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--dossier", required=True, type=Path)
    ap.add_argument("--response", action="append", default=[],
                    type=parse_binding, metavar="ARM=PATH")
    ap.add_argument("--base-arms", default="default,hand")
    ap.add_argument("--oracle-arm", default="measured-oracle")
    ap.add_argument("--expected-reps", type=parse_replicates,
                    default=parse_replicates("1,2,3,4"))
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    try:
        dossier = bridge._verified_dossier(read_json(args.dossier))
        base_arms = [arm for arm in args.base_arms.split(",") if arm]
        if any(not ARM_PATTERN.fullmatch(arm) for arm in base_arms):
            raise AnalysisError("invalid --base-arms")
        responses: dict[str, tuple[Path, dict[str, Any]]] = {}
        for arm, path in args.response:
            if arm in responses or arm in base_arms:
                raise AnalysisError(f"duplicate arm {arm}")
            responses[arm] = (path, validated_response(dossier, path))
        expected_arms = base_arms + list(responses)
        if args.oracle_arm not in expected_arms:
            raise AnalysisError("--oracle-arm is absent from evaluated arms")
        if args.oracle_arm in responses:
            oracle_producer = responses[args.oracle_arm][1].get("producer", {})
            if (
                oracle_producer.get("kind") != "measured_oracle_control"
                or oracle_producer.get("oracle_or_runtime_results_read") is not True
            ):
                raise AnalysisError(
                    "--oracle-arm must be an explicitly labeled measured oracle"
                )

        route_contracts = {arm: routes_for_builtin(arm) for arm in base_arms}
        route_contracts.update({
            arm: routes_for_response(dossier, response)
            for arm, (_, response) in responses.items()
        })
        action_contracts = {arm: builtin_actions(arm) for arm in base_arms}
        action_contracts.update({
            arm: response_actions(dossier, response)
            for arm, (_, response) in responses.items()
        })

        parsed: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
        binary_hashes: dict[str, set[str]] = {
            arm: set() for arm in expected_arms
        }
        for path in args.logs:
            name_match = re.fullmatch(
                r"rep[0-9]+-([a-zA-Z0-9_.-]+)\.log", path.name
            )
            if name_match is None or name_match.group(1) not in route_contracts:
                raise AnalysisError(f"{path}: arm has no route contract")
            arm, rep, binary_sha256, rows = parse_log(
                path, route_contracts[name_match.group(1)]
            )
            if arm in parsed.setdefault(rep, {}):
                raise AnalysisError(f"duplicate rep={rep} arm={arm}")
            parsed[rep][arm] = rows
            binary_hashes[arm].add(binary_sha256)
        for rep, arms in parsed.items():
            if set(arms) != set(expected_arms):
                raise AnalysisError(
                    f"rep={rep}: arms={sorted(arms)}, expected={expected_arms}"
                )
        if set(parsed) != set(args.expected_reps):
            raise AnalysisError(
                f"replicates={sorted(parsed)}, expected={sorted(args.expected_reps)}"
            )
        inconsistent = {
            arm: sorted(hashes) for arm, hashes in binary_hashes.items()
            if len(hashes) != 1
        }
        if inconsistent:
            raise AnalysisError(
                f"policy arm used multiple binaries across replicates: {inconsistent}"
            )

        aggregate: dict[str, Any] = {}
        print("scenario                  " + " ".join(
            f"{arm:>15}" for arm in expected_arms
        ))
        for scenario in SCENARIOS:
            medians: dict[str, float] = {}
            for arm in expected_arms:
                medians[arm] = geomean([
                    parsed[rep][arm][scenario]["median_us"]
                    for rep in sorted(parsed)
                ])
            oracle_us = medians[args.oracle_arm]
            aggregate[scenario] = {
                "geomean_median_us": medians,
                "slowdown_vs_measured_oracle": {
                    arm: value / oracle_us for arm, value in medians.items()
                },
                "decision_bearing": scenario not in FORCED_ACTIONS,
            }
            print(f"{scenario:<25} " + " ".join(
                f"{medians[arm]:15.3f}" for arm in expected_arms
            ))

        scored = [scenario for scenario in SCENARIOS
                  if scenario not in FORCED_ACTIONS]
        policies: dict[str, Any] = {}
        oracle_actions = action_contracts[args.oracle_arm]
        for arm in expected_arms:
            primary = geomean([
                aggregate[scenario]["slowdown_vs_measured_oracle"][arm]
                for scenario in scored
            ])
            secondary = sum(
                aggregate[scenario]["geomean_median_us"][arm]
                for scenario in SCENARIOS
            )
            matches = sum(
                action_contracts[arm][KERNEL_FOR_SCENARIO[scenario]]
                == oracle_actions[KERNEL_FOR_SCENARIO[scenario]]
                for scenario in scored
            )
            paired_scores = {
                str(rep): geomean([
                    parsed[rep][arm][scenario]["median_us"]
                    / parsed[rep][args.oracle_arm][scenario]["median_us"]
                    for scenario in scored
                ])
                for rep in sorted(parsed)
            }
            policies[arm] = {
                "primary_geomean_regret": primary,
                "secondary_sum_geomean_medians_us": secondary,
                "scenario_action_matches": matches,
                "scenario_action_count": len(scored),
                "paired_primary_regret_by_replicate": paired_scores,
                "geomean_paired_primary_regret": geomean(
                    list(paired_scores.values())
                ),
                "actions_by_kernel": action_contracts[arm],
            }

        provenance = {
            arm: {
                "response": str(path),
                "response_sha256": sha256_file(path),
                "producer": response.get("producer"),
            }
            for arm, (path, response) in responses.items()
        }
        summary = {
            "scope": (
                "paired real-LTO policy binaries; forced proxy-only scenario "
                "excluded from primary decision regret"
            ),
            "dossier_id": dossier["dossier_id"],
            "replicates": sorted(parsed),
            "oracle_arm": args.oracle_arm,
            "binary_sha256_by_arm": {
                arm: next(iter(binary_hashes[arm])) for arm in expected_arms
            },
            "aggregate": aggregate,
            "policies": policies,
            "response_provenance": provenance,
        }
        print("\npolicy                         regret matches sum_us")
        for arm in expected_arms:
            row = policies[arm]
            print(f"{arm:<30} {row['primary_geomean_regret']:>7.4f}x "
                  f"{row['scenario_action_matches']}/{row['scenario_action_count']} "
                  f"{row['secondary_sum_geomean_medians_us']:>9.3f}")
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return 0
    except (AnalysisError, bridge.BridgeError, OSError, ValueError) as exc:
        print(f"compiler-lto-candidates: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
