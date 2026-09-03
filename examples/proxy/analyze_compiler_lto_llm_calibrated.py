#!/usr/bin/env python3
"""Analyze completed calibrated LTO LLM trials under the frozen protocol."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/gicc-passes/python"))
sys.path.insert(0, str(HERE))

import analyze_compiler_lto_llm_trials as zero_analysis  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import prepare_compiler_lto_llm_calibrated as prepare  # noqa: E402
import run_compiler_lto_llm_calibrated_trials as runner  # noqa: E402
from analyze_compiler_lto_candidates import (  # noqa: E402
    AnalysisError,
    read_json,
    response_actions,
    sha256_file,
    validated_response,
)
from analyze_compiler_lto_eval import FORCED_ACTIONS, SCENARIOS  # noqa: E402
from compiler_lto_eval import SCENARIO_FOR_KERNEL  # noqa: E402


ANALYSIS_SCHEMA = "gicc-compiler-lto-llm-calibrated-analysis-v1"
KERNEL_FOR_SCENARIO = {
    scenario: kernel for kernel, scenario in SCENARIO_FOR_KERNEL.items()
}


def policy_analysis(dossier: dict[str, Any],
                    observed: list[dict[str, Any]],
                    oracle: dict[str, Any],
                    gbt: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Group accepted decisions and score only against post-call controls."""
    sites = [site["site_id"] for site in dossier["sites"]]
    site_by_id = {site["site_id"]: site for site in dossier["sites"]}
    accepted = [row for row in observed if row["bridge_accepted"]]
    unique_vectors: list[tuple[str, ...]] = []
    for row in accepted:
        vector = tuple(row["actions"][site] for site in sites)
        row["action_vector"] = vector
        if vector not in unique_vectors:
            unique_vectors.append(vector)
    policy_for = {
        vector: f"llm-calibrated-policy{index:02d}"
        for index, vector in enumerate(unique_vectors, 1)
    }
    oracle_by_kernel = response_actions(dossier, oracle)
    gbt_by_kernel = response_actions(dossier, gbt)
    scored = [scenario for scenario in SCENARIOS if scenario not in FORCED_ACTIONS]
    policies: dict[str, Any] = {}
    for vector in unique_vectors:
        name = policy_for[vector]
        members = [
            row for row in accepted if row["action_vector"] == vector
        ]
        action_map = dict(zip(sites, vector))
        by_kernel: dict[str, list[str]] = {}
        for site in dossier["sites"]:
            by_kernel.setdefault(site["kernel"], []).append(
                action_map[site["site_id"]]
            )
        scenario_matches = {
            scenario: by_kernel[KERNEL_FOR_SCENARIO[scenario]]
            == oracle_by_kernel[KERNEL_FOR_SCENARIO[scenario]]
            for scenario in scored
        }
        site_matches = sum(
            actual == expected
            for scenario in scored
            for actual, expected in zip(
                by_kernel[KERNEL_FOR_SCENARIO[scenario]],
                oracle_by_kernel[KERNEL_FOR_SCENARIO[scenario]],
            )
        )
        site_count = sum(
            len(oracle_by_kernel[KERNEL_FOR_SCENARIO[scenario]])
            for scenario in scored
        )
        policies[name] = {
            "frequency": len(members),
            "trials": [row["trial"] for row in members],
            "representative_response": members[0]["response"],
            "representative_response_sha256": members[0]["response_sha256"],
            "actions_by_site": action_map,
            "actions_by_kernel": by_kernel,
            "oracle_action_agreement": {
                "scenario_matches": sum(scenario_matches.values()),
                "scenario_count": len(scenario_matches),
                "site_matches": site_matches,
                "site_count": site_count,
                "by_scenario": scenario_matches,
            },
            "matches_calibrated_gbt": by_kernel == gbt_by_kernel,
        }
        for row in members:
            row["policy"] = name
            row["exact_scenario_oracle_match"] = all(
                scenario_matches.values()
            )

    distributions: dict[str, Any] = {}
    for site in sites:
        counts = collections.Counter(
            row["actions"][site] for row in accepted
        )
        distributions[site] = {
            "kernel": site_by_id[site]["kernel"],
            "accepted_response_counts": dict(sorted(counts.items())),
        }
    return policies, distributions


def analyze(*, request_path: Path, trials_dir: Path,
            calibration_dossier_path: Path,
            calibration_results_path: Path,
            evaluation_dossier_path: Path, oracle_response_path: Path,
            gbt_response_path: Path) -> dict[str, Any]:
    request = prepare.verify(
        request_path=request_path,
        calibration_dossier_path=calibration_dossier_path,
        calibration_results_path=calibration_results_path,
        evaluation_dossier_path=evaluation_dossier_path,
    )
    dossier = bridge._verified_dossier(read_json(evaluation_dossier_path))
    oracle = validated_response(dossier, oracle_response_path)
    zero_analysis.require_measured_oracle(oracle)
    gbt = validated_response(dossier, gbt_response_path)
    index = read_json(trials_dir / "run-index.json")
    total = request["provider_delivery"]["independent_responses"]
    runs = index.get("runs") if isinstance(index, dict) else None
    authorization_id = index.get("authorization_id") if isinstance(index, dict) else None
    if (index.get("schema_version") != runner.INDEX_SCHEMA
            or index.get("status") != "complete"
            or index.get("request_id") != request["request_id"]
            or not isinstance(authorization_id, str)
            or not isinstance(runs, list)
            or len(runs) != total):
        raise AnalysisError("run index is not a complete frozen-request result")
    authorization_value = read_json(trials_dir / "authorization.json")
    runner.verify_authorization(authorization_value, request)
    if authorization_value.get("authorization_id") != authorization_id:
        raise AnalysisError("run index/authorization ID mismatch")

    prompt_hash = runner.delivery_hash(request, "user_prompt")
    system_hash = runner.delivery_hash(request, "system_prompt")
    schema_hash = runner.delivery_hash(request, "response_schema")
    sites = [site["site_id"] for site in dossier["sites"]]
    observed: list[dict[str, Any]] = []
    retry_count = 0
    decision_models: set[str] = set()
    all_models: set[str] = set()
    total_cost = 0.0
    for wanted, record in enumerate(runs, 1):
        trial = runner.verify_archived_run(
            record, trials_dir, request, authorization_id,
        )
        if trial != wanted:
            raise AnalysisError("run index trial order is not 1..N")
        if (record.get("prompt_sha256") != prompt_hash
                or record.get("system_prompt_sha256") != system_hash
                or record.get("response_schema_sha256") != schema_hash
                or record.get("provider_call_succeeded") is not True):
            raise AnalysisError(f"trial{trial:02d}: request binding failed")
        attempts = record["attempts"]
        if (attempts[-1].get("returncode") != 0
                or attempts[-1].get("transport_error") is not None):
            raise AnalysisError(f"trial{trial:02d}: final transport failed")
        retry_count += len(attempts) - 1
        envelope = read_json(
            trials_dir / f"trial{trial:02d}"
            / f"attempt{len(attempts)}.stdout.json"
        )
        primary_model, models, cost = zero_analysis.decision_model(envelope)
        decision_models.add(primary_model)
        all_models.update(models)
        total_cost += cost
        response_path = trials_dir / f"trial{trial:02d}" / "response.json"
        row: dict[str, Any] = {
            "trial": trial,
            "bridge_accepted": record.get("bridge_accepted") is True,
            "fallback_applied": record.get("fallback_applied") is True,
            "bridge_errors": record.get("bridge_errors"),
            "response": str(response_path),
            "response_sha256": sha256_file(response_path),
            "response_sha256_canonical": record.get(
                "response_sha256_canonical"
            ),
            "decision_model": primary_model,
            "models_observed": models,
            "provider_cost_usd_reported": cost,
        }
        if row["bridge_accepted"]:
            if row["fallback_applied"]:
                raise AnalysisError(f"trial{trial:02d}: accepted but fell back")
            response = validated_response(dossier, response_path)
            if (zero_analysis.canonical_sha256(response)
                    != record.get("response_sha256_canonical")):
                raise AnalysisError(f"trial{trial:02d}: response hash mismatch")
            row["actions"] = {
                site: response["decisions"][site]["action"] for site in sites
            }
        elif not row["fallback_applied"]:
            raise AnalysisError(f"trial{trial:02d}: invalid response did not fall back")
        observed.append(row)

    policies, distributions = policy_analysis(dossier, observed, oracle, gbt)
    accepted = [row for row in observed if row["bridge_accepted"]]
    exact_oracle = sum(
        row.get("exact_scenario_oracle_match", False) for row in observed
    )
    matched_gbt_trials = sum(
        row["frequency"] for row in policies.values()
        if row["matches_calibrated_gbt"]
    )
    return {
        "schema_version": ANALYSIS_SCHEMA,
        "scope": (
            "post-freeze analysis; provider saw compiler evidence but no "
            "source, IR, evaluation runtime labels, or oracle labels"
        ),
        "request": str(request_path),
        "request_sha256": sha256_file(request_path),
        "request_id": request["request_id"],
        "authorization_id": authorization_id,
        "authorization_sha256": sha256_file(
            trials_dir / "authorization.json"
        ),
        "dossier_id": dossier["dossier_id"],
        "trial_count": len(observed),
        "accepted_count": len(accepted),
        "fallback_count": len(observed) - len(accepted),
        "retry_count": retry_count,
        "unique_accepted_response_count": len({
            row["response_sha256_canonical"] for row in accepted
        }),
        "unique_accepted_policy_count": len(policies),
        "preregistered_primary": {
            "metric": "exact evaluation action-oracle agreement",
            "exact_oracle_trials": exact_oracle,
            "trial_count": len(observed),
            "rate": exact_oracle / len(observed),
            "fallback_trials_count_as_nonmatches": True,
        },
        "agreement_with_calibrated_gbt": {
            "matching_trials": matched_gbt_trials,
            "trial_count": len(observed),
            "rate": matched_gbt_trials / len(observed),
        },
        "decision_models": sorted(decision_models),
        "all_provider_models_observed": sorted(all_models),
        "provider_cost_usd_reported_total": total_cost,
        "actions_by_site_distribution": distributions,
        "policies": policies,
        "trials": observed,
        "oracle_response": str(oracle_response_path),
        "oracle_response_sha256": sha256_file(oracle_response_path),
        "calibrated_gbt_response": str(gbt_response_path),
        "calibrated_gbt_response_sha256": sha256_file(gbt_response_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--trials", required=True, type=Path)
    parser.add_argument("--calibration-dossier", required=True, type=Path)
    parser.add_argument("--calibration-results", required=True, type=Path)
    parser.add_argument("--evaluation-dossier", required=True, type=Path)
    parser.add_argument("--oracle-response", required=True, type=Path)
    parser.add_argument("--gbt-response", required=True, type=Path)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = analyze(
            request_path=args.request.resolve(),
            trials_dir=args.trials.resolve(),
            calibration_dossier_path=args.calibration_dossier.resolve(),
            calibration_results_path=args.calibration_results.resolve(),
            evaluation_dossier_path=args.evaluation_dossier.resolve(),
            oracle_response_path=args.oracle_response.resolve(),
            gbt_response_path=args.gbt_response.resolve(),
        )
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        primary = result["preregistered_primary"]
        print(
            f"trials={result['trial_count']} accepted={result['accepted_count']} "
            f"fallback={result['fallback_count']} "
            f"unique_policies={result['unique_accepted_policy_count']} "
            f"exact_oracle={primary['exact_oracle_trials']}/{primary['trial_count']}"
        )
        return 0
    except (AnalysisError, runner.TrialError, prepare.PrepareError,
            bridge.BridgeError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"analyze-compiler-lto-llm-calibrated: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
