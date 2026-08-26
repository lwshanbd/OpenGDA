#!/usr/bin/env python3
"""Validate one LLM policy's balanced real-LTO runtime comparison."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/gicc-passes/python"))
sys.path.insert(0, str(HERE))

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
from analyze_compiler_lto_transfer import (  # noqa: E402
    exact_two_sided_sign_test,
    paired_bootstrap,
    require_oracle_label,
)
from compiler_lto_eval import SCENARIO_FOR_KERNEL  # noqa: E402


KERNEL_FOR_SCENARIO = {
    scenario: kernel for kernel, scenario in SCENARIO_FOR_KERNEL.items()
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--dossier", required=True, type=Path)
    ap.add_argument("--response", action="append", default=[],
                    type=parse_binding, metavar="ARM=PATH")
    ap.add_argument("--candidate-arm", required=True)
    ap.add_argument("--trial-analysis", required=True, type=Path)
    ap.add_argument("--oracle-response", required=True, type=Path)
    ap.add_argument("--base-arms", default="default,hand")
    ap.add_argument("--expected-reps", type=parse_replicates,
                    default=parse_replicates("1,2,3,4,5,6,7,8,9,10"))
    ap.add_argument("--json", required=True, type=Path)
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
        if args.candidate_arm not in responses:
            raise AnalysisError("candidate arm has no response binding")
        expected_arms = base_arms + list(responses)
        if len(expected_arms) != 5:
            raise AnalysisError("balanced protocol requires exactly five runtime arms")

        trial_analysis = read_json(args.trial_analysis)
        if trial_analysis.get("schema_version") != "gicc-compiler-lto-llm-trial-analysis-v1":
            raise AnalysisError("wrong trial-analysis schema")
        policy = trial_analysis.get("policies", {}).get(args.candidate_arm)
        if not isinstance(policy, dict):
            raise AnalysisError("candidate absent from trial analysis")
        candidate_path, candidate = responses[args.candidate_arm]
        if sha256_file(candidate_path) != policy.get("representative_response_sha256"):
            raise AnalysisError("candidate is not the frozen representative response")

        oracle = validated_response(dossier, args.oracle_response)
        require_oracle_label(oracle)
        oracle_actions = response_actions(dossier, oracle)
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
        binary_hashes: dict[str, set[str]] = {arm: set() for arm in expected_arms}
        for path in args.logs:
            match = re.fullmatch(r"rep[0-9]+-([a-zA-Z0-9_.-]+)\.log", path.name)
            if match is None or match.group(1) not in route_contracts:
                raise AnalysisError(f"{path}: arm has no route contract")
            arm, rep, binary_sha, rows = parse_log(path, route_contracts[match.group(1)])
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
            raise AnalysisError(f"arm used multiple binaries: {inconsistent}")

        scored = [scenario for scenario in SCENARIOS if scenario not in FORCED_ACTIONS]
        aggregate: dict[str, Any] = {}
        for scenario in SCENARIOS:
            aggregate[scenario] = {
                "decision_bearing": scenario in scored,
                "geomean_median_us": {
                    arm: geomean([
                        parsed[rep][arm][scenario]["median_us"]
                        for rep in sorted(parsed)
                    ])
                    for arm in expected_arms
                },
            }
        candidate_actions = action_contracts[args.candidate_arm]
        by_scenario = {
            scenario: candidate_actions[KERNEL_FOR_SCENARIO[scenario]]
            == oracle_actions[KERNEL_FOR_SCENARIO[scenario]]
            for scenario in scored
        }
        site_matches = sum(
            actual == expected
            for scenario in scored
            for actual, expected in zip(
                candidate_actions[KERNEL_FOR_SCENARIO[scenario]],
                oracle_actions[KERNEL_FOR_SCENARIO[scenario]],
            )
        )
        site_count = sum(
            len(oracle_actions[KERNEL_FOR_SCENARIO[scenario]])
            for scenario in scored
        )

        references = [arm for arm in expected_arms if arm != args.candidate_arm]
        paired: dict[str, Any] = {}
        for arm in references:
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

        summary = {
            "schema_version": "gicc-compiler-lto-llm-runtime-v1",
            "scope": (
                "paired real-LTO runtime; measured oracle used only for action "
                "agreement and never as a runtime arm or denominator"
            ),
            "dossier_id": dossier["dossier_id"],
            "candidate_arm": args.candidate_arm,
            "candidate_population_frequency": policy["frequency"],
            "candidate_population_trials": policy["trials"],
            "replicates": sorted(parsed),
            "reference_arms": references,
            "runtime_oracle_denominator_used": False,
            "forced_scenarios_excluded": sorted(FORCED_ACTIONS),
            "binary_sha256_by_arm": {
                arm: next(iter(binary_hashes[arm])) for arm in expected_arms
            },
            "oracle_action_agreement": {
                "scenario_matches": sum(by_scenario.values()),
                "scenario_count": len(by_scenario),
                "site_matches": site_matches,
                "site_count": site_count,
                "by_scenario": by_scenario,
            },
            "paired_runtime": paired,
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
            "trial_analysis": str(args.trial_analysis),
            "trial_analysis_sha256": sha256_file(args.trial_analysis),
            "oracle_response": str(args.oracle_response),
            "oracle_response_sha256": sha256_file(args.oracle_response),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        agreement = summary["oracle_action_agreement"]
        print(
            f"{args.candidate_arm} oracle={agreement['scenario_matches']}/"
            f"{agreement['scenario_count']} scenarios, "
            f"{agreement['site_matches']}/{agreement['site_count']} sites"
        )
        for arm, row in paired.items():
            ci = row["paired_bootstrap_95_percentile_ci"]
            print(
                f"{arm}/candidate="
                f"{row['geomean_speedup_reference_over_candidate']:.6f}x "
                f"CI=[{ci['lower_2_5_percent']:.6f},"
                f"{ci['upper_97_5_percent']:.6f}] "
                f"wins={row['candidate_faster_replicates']}/"
                f"{row['replicate_count']} "
                f"sign-p={row['two_sided_exact_sign_test']['p_value']:.6f}"
            )
        return 0
    except (AnalysisError, bridge.BridgeError, OSError, ValueError) as exc:
        print(f"compiler-lto-llm-runtime: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
