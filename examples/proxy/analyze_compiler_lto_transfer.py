#!/usr/bin/env python3
"""Validate and score source-free compiler-policy transfer without an oracle runtime denominator."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "gicc-passes" / "python"))

import gicc_llm_bridge as bridge  # noqa: E402
from analyze_compiler_lto_candidates import (  # noqa: E402
    ARM_PATTERN,
    AnalysisError,
    builtin_actions,
    geomean,
    parse_binding,
    parse_log,
    parse_replicates,
    read_json,
    response_actions,
    routes_for_builtin,
    routes_for_response,
    sha256_file,
    validated_response,
)
from analyze_compiler_lto_eval import FORCED_ACTIONS, SCENARIOS  # noqa: E402
from compiler_lto_eval import SCENARIO_FOR_KERNEL  # noqa: E402


KERNEL_FOR_SCENARIO = {
    scenario: kernel for kernel, scenario in SCENARIO_FOR_KERNEL.items()
}


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_bootstrap(values: list[float]) -> dict[str, Any]:
    """Bootstrap complete allocation replicates, exactly when tractable."""
    if len(values) <= 8:
        samples = itertools.product(range(len(values)), repeat=len(values))
        estimates = [
            geomean([values[index] for index in sample]) for sample in samples
        ]
        method = "exact n-out-of-n paired bootstrap over allocation replicates"
        seed = None
    else:
        seed = 0
        rng = random.Random(seed)
        count = 100_000
        estimates = [
            geomean([
                values[rng.randrange(len(values))]
                for _ in range(len(values))
            ])
            for _ in range(count)
        ]
        method = "deterministic Monte Carlo paired bootstrap over allocations"
    return {
        "method": method,
        "samples": len(estimates),
        "random_seed": seed,
        "lower_2_5_percent": percentile(estimates, 0.025),
        "upper_97_5_percent": percentile(estimates, 0.975),
    }


def exact_two_sided_sign_test(values: list[float]) -> dict[str, Any]:
    above = sum(value > 1.0 for value in values)
    below = sum(value < 1.0 for value in values)
    count = above + below
    if count == 0:
        p_value = 1.0
    else:
        extreme = max(above, below)
        tail = sum(math.comb(count, index)
                   for index in range(extreme, count + 1)) / (2 ** count)
        p_value = min(1.0, 2.0 * tail)
    return {
        "null_ratio": 1.0,
        "above": above,
        "below": below,
        "ties_excluded": len(values) - count,
        "p_value": p_value,
    }


def require_oracle_label(response: dict[str, Any]) -> None:
    producer = response.get("producer", {})
    if (
        producer.get("kind") != "measured_oracle_control"
        or producer.get("oracle_or_runtime_results_read") is not True
    ):
        raise AnalysisError(
            "--oracle-response must be an explicitly labeled measured oracle"
        )


def require_source_free_candidate(response: dict[str, Any]) -> None:
    producer = response.get("producer", {})
    required_false = (
        "source_read",
        "historical_runtime_grid_read",
        "frozen_evaluation_results_read",
    )
    wrong = [key for key in required_false if producer.get(key) is not False]
    if wrong:
        raise AnalysisError(
            "candidate provenance does not explicitly exclude: "
            + ", ".join(wrong)
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--dossier", required=True, type=Path)
    ap.add_argument("--response", action="append", default=[],
                    type=parse_binding, metavar="ARM=PATH")
    ap.add_argument("--candidate-arm", required=True)
    ap.add_argument("--oracle-response", required=True, type=Path)
    ap.add_argument("--runtime-duplicate-arm", default="measured-oracle")
    ap.add_argument("--base-arms", default="default,hand")
    ap.add_argument("--expected-reps", type=parse_replicates,
                    default=parse_replicates("1,2,3,4,5"))
    ap.add_argument("--json", required=True, type=Path)
    args = ap.parse_args()

    try:
        dossier = bridge._verified_dossier(read_json(args.dossier))
        base_arms = [arm for arm in args.base_arms.split(",") if arm]
        if any(not ARM_PATTERN.fullmatch(arm) for arm in base_arms):
            raise AnalysisError("invalid --base-arms")
        if not ARM_PATTERN.fullmatch(args.runtime_duplicate_arm):
            raise AnalysisError("invalid --runtime-duplicate-arm")

        responses: dict[str, tuple[Path, dict[str, Any]]] = {}
        for arm, path in args.response:
            if arm in responses or arm in base_arms:
                raise AnalysisError(f"duplicate arm {arm}")
            responses[arm] = (path, validated_response(dossier, path))
        if args.candidate_arm not in responses:
            raise AnalysisError("--candidate-arm has no --response binding")
        if args.runtime_duplicate_arm in base_arms or args.runtime_duplicate_arm in responses:
            raise AnalysisError("runtime duplicate arm collides with another arm")

        candidate = responses[args.candidate_arm][1]
        require_source_free_candidate(candidate)
        oracle = validated_response(dossier, args.oracle_response)
        require_oracle_label(oracle)

        expected_arms = base_arms + list(responses) + [args.runtime_duplicate_arm]
        route_contracts = {arm: routes_for_builtin(arm) for arm in base_arms}
        route_contracts.update({
            arm: routes_for_response(dossier, response)
            for arm, (_, response) in responses.items()
        })
        route_contracts[args.runtime_duplicate_arm] = routes_for_response(
            dossier, oracle
        )
        action_contracts = {arm: builtin_actions(arm) for arm in base_arms}
        action_contracts.update({
            arm: response_actions(dossier, response)
            for arm, (_, response) in responses.items()
        })
        oracle_actions = response_actions(dossier, oracle)
        action_contracts[args.runtime_duplicate_arm] = oracle_actions

        parsed: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
        binary_hashes: dict[str, set[str]] = {arm: set() for arm in expected_arms}
        for path in args.logs:
            match = re.fullmatch(
                r"rep[0-9]+-([a-zA-Z0-9_.-]+)\.log", path.name
            )
            if match is None or match.group(1) not in route_contracts:
                raise AnalysisError(f"{path}: arm has no route contract")
            arm, rep, binary_sha, rows = parse_log(
                path, route_contracts[match.group(1)]
            )
            if arm in parsed.setdefault(rep, {}):
                raise AnalysisError(f"duplicate rep={rep} arm={arm}")
            parsed[rep][arm] = rows
            binary_hashes[arm].add(binary_sha)
        if set(parsed) != set(args.expected_reps):
            raise AnalysisError(
                f"replicates={sorted(parsed)}, expected={sorted(args.expected_reps)}"
            )
        for rep, arms in parsed.items():
            if set(arms) != set(expected_arms):
                raise AnalysisError(
                    f"rep={rep}: arms={sorted(arms)}, expected={expected_arms}"
                )
        inconsistent = {
            arm: sorted(values) for arm, values in binary_hashes.items()
            if len(values) != 1
        }
        if inconsistent:
            raise AnalysisError(
                f"arm used multiple binaries across replicates: {inconsistent}"
            )
        resolved_hashes = {
            arm: next(iter(binary_hashes[arm])) for arm in expected_arms
        }

        scored = [scenario for scenario in SCENARIOS
                  if scenario not in FORCED_ACTIONS]
        candidate_actions = action_contracts[args.candidate_arm]
        scenario_matches = {
            scenario: (
                candidate_actions[KERNEL_FOR_SCENARIO[scenario]]
                == oracle_actions[KERNEL_FOR_SCENARIO[scenario]]
            )
            for scenario in scored
        }
        site_matches = sum(
            candidate_actions[KERNEL_FOR_SCENARIO[scenario]][index]
            == oracle_actions[KERNEL_FOR_SCENARIO[scenario]][index]
            for scenario in scored
            for index in range(len(oracle_actions[KERNEL_FOR_SCENARIO[scenario]]))
        )
        site_count = sum(
            len(oracle_actions[KERNEL_FOR_SCENARIO[scenario]])
            for scenario in scored
        )
        all_actions_match = all(scenario_matches.values())
        if not all_actions_match:
            raise AnalysisError(
                "runtime duplicate requires candidate/oracle action identity; "
                "use the general candidate analyzer for a partial match"
            )
        if (
            resolved_hashes[args.candidate_arm]
            != resolved_hashes[args.runtime_duplicate_arm]
        ):
            raise AnalysisError(
                "candidate and oracle duplicate have identical actions but "
                "different binaries; stabilize the HIP compilation-unit ID"
            )

        aggregate: dict[str, Any] = {}
        for scenario in SCENARIOS:
            medians = {
                arm: geomean([
                    parsed[rep][arm][scenario]["median_us"]
                    for rep in sorted(parsed)
                ])
                for arm in expected_arms
            }
            aggregate[scenario] = {
                "decision_bearing": scenario in scored,
                "geomean_median_us": medians,
                "reference_over_candidate": {
                    arm: medians[arm] / medians[args.candidate_arm]
                    for arm in expected_arms
                    if arm not in {args.candidate_arm, args.runtime_duplicate_arm}
                },
            }

        reference_arms = [
            arm for arm in expected_arms
            if arm not in {args.candidate_arm, args.runtime_duplicate_arm}
        ]
        paired: dict[str, Any] = {}
        for arm in reference_arms:
            by_rep = {
                str(rep): geomean([
                    parsed[rep][arm][scenario]["median_us"]
                    / parsed[rep][args.candidate_arm][scenario]["median_us"]
                    for scenario in scored
                ])
                for rep in sorted(parsed)
            }
            values = list(by_rep.values())
            paired[arm] = {
                "speedup_reference_over_candidate_by_replicate": by_rep,
                "geomean_speedup_reference_over_candidate": geomean(values),
                "candidate_faster_replicates": sum(value > 1.0 for value in values),
                "replicate_count": len(values),
                "paired_bootstrap_95_percentile_ci": paired_bootstrap(values),
                "two_sided_exact_sign_test": exact_two_sided_sign_test(values),
            }

        duplicate_by_rep = {
            str(rep): geomean([
                parsed[rep][args.runtime_duplicate_arm][scenario]["median_us"]
                / parsed[rep][args.candidate_arm][scenario]["median_us"]
                for scenario in scored
            ])
            for rep in sorted(parsed)
        }
        duplicate_values = list(duplicate_by_rep.values())
        summary = {
            "schema_version": "gicc-compiler-lto-transfer-v1",
            "scope": (
                "paired real-LTO transfer; measured oracle is used only for "
                "action agreement, never as the performance denominator"
            ),
            "dossier_id": dossier["dossier_id"],
            "replicates": sorted(parsed),
            "candidate_arm": args.candidate_arm,
            "reference_arms": reference_arms,
            "forced_scenarios_excluded": sorted(FORCED_ACTIONS),
            "runtime_oracle_denominator_used": False,
            "binary_sha256_by_arm": resolved_hashes,
            "oracle_action_agreement": {
                "scenario_matches": sum(scenario_matches.values()),
                "scenario_count": len(scenario_matches),
                "site_matches": site_matches,
                "site_count": site_count,
                "by_scenario": scenario_matches,
            },
            "same_binary_runtime_duplicate": {
                "arm": args.runtime_duplicate_arm,
                "binary_sha256_equal": (
                    resolved_hashes[args.candidate_arm]
                    == resolved_hashes[args.runtime_duplicate_arm]
                ),
                "duplicate_over_candidate_by_replicate": duplicate_by_rep,
                "geomean_duplicate_over_candidate": geomean(duplicate_values),
                "paired_bootstrap_95_percentile_ci":
                    paired_bootstrap(duplicate_values),
                "two_sided_exact_sign_test":
                    exact_two_sided_sign_test(duplicate_values),
            },
            "paired_transfer": paired,
            "aggregate": aggregate,
            "actions_by_arm": action_contracts,
            "response_provenance": {
                arm: {
                    "response": str(path),
                    "response_sha256": sha256_file(path),
                    "producer": response.get("producer"),
                }
                for arm, (path, response) in responses.items()
            },
            "oracle_response_provenance": {
                "response": str(args.oracle_response),
                "response_sha256": sha256_file(args.oracle_response),
                "producer": oracle.get("producer"),
            },
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        print(
            f"oracle_action_match={sum(scenario_matches.values())}/"
            f"{len(scenario_matches)} scenarios, {site_matches}/{site_count} sites"
        )
        for arm in reference_arms:
            result = paired[arm]
            ci = result["paired_bootstrap_95_percentile_ci"]
            print(
                f"{arm}/candidate="
                f"{result['geomean_speedup_reference_over_candidate']:.6f}x "
                f"paired-CI=[{ci['lower_2_5_percent']:.6f}, "
                f"{ci['upper_97_5_percent']:.6f}] "
                f"wins={result['candidate_faster_replicates']}/"
                f"{result['replicate_count']} "
                f"sign-p={result['two_sided_exact_sign_test']['p_value']:.6f}"
            )
        print(
            "same-binary-duplicate/candidate="
            f"{geomean(duplicate_values):.6f}x"
        )
        return 0
    except (AnalysisError, bridge.BridgeError, OSError, ValueError) as exc:
        print(f"compiler-lto-transfer: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
