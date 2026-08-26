#!/usr/bin/env python3
"""Validate frozen LLM trials and reveal action-oracle agreement afterward."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/gicc-passes/python"))
sys.path.insert(0, str(HERE))

import gicc_llm_bridge as bridge  # noqa: E402
from analyze_compiler_lto_candidates import (  # noqa: E402
    AnalysisError,
    read_json,
    response_actions,
    sha256_file,
    validated_response,
)
from analyze_compiler_lto_eval import FORCED_ACTIONS, SCENARIOS  # noqa: E402
from compiler_lto_eval import SCENARIO_FOR_KERNEL  # noqa: E402


KERNEL_FOR_SCENARIO = {
    scenario: kernel for kernel, scenario in SCENARIO_FOR_KERNEL.items()
}


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def require_measured_oracle(response: dict[str, Any]) -> None:
    producer = response.get("producer", {})
    if (
        producer.get("kind") != "measured_oracle_control"
        or producer.get("oracle_or_runtime_results_read") is not True
    ):
        raise AnalysisError("oracle response is not explicitly labeled")


def decision_model(envelope: dict[str, Any]) -> tuple[str, list[str], float]:
    usage = envelope.get("modelUsage")
    if not isinstance(usage, dict) or not usage:
        raise AnalysisError("provider envelope lacks modelUsage")
    rows: list[tuple[str, int, float]] = []
    for model, values in usage.items():
        if not isinstance(model, str) or not isinstance(values, dict):
            raise AnalysisError("malformed modelUsage")
        rows.append((
            model,
            int(values.get("outputTokens", 0)),
            float(values.get("costUSD", 0.0)),
        ))
    rows.sort(key=lambda row: (-row[1], row[0]))
    return rows[0][0], sorted(model for model, _, _ in rows), sum(
        cost for _, _, cost in rows
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--protocol", required=True, type=Path)
    ap.add_argument("--trials", required=True, type=Path)
    ap.add_argument("--dossier", required=True, type=Path)
    ap.add_argument("--oracle-response", required=True, type=Path)
    ap.add_argument("--gbt-response", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    args = ap.parse_args()

    try:
        protocol = read_json(args.protocol)
        if protocol.get("schema_version") != "gicc-compiler-lto-llm-protocol-v1":
            raise AnalysisError("wrong protocol schema")
        boundary = protocol.get("data_boundary", {})
        required_false = (
            "candidate_may_generate_code",
            "calibration_labels_visible",
            "evaluation_results_visible",
            "source_visible",
        )
        if any(boundary.get(key) is not False for key in required_false):
            raise AnalysisError("protocol does not enforce the source/label boundary")
        if boundary.get("model_tools") != []:
            raise AnalysisError("protocol permitted model tools")
        dossier = bridge._verified_dossier(read_json(args.dossier))
        if dossier["dossier_id"] != protocol.get("dossier_id"):
            raise AnalysisError("protocol/dossier ID mismatch")
        oracle = validated_response(dossier, args.oracle_response)
        require_measured_oracle(oracle)
        gbt = validated_response(dossier, args.gbt_response)

        index = read_json(args.trials / "run-index.json")
        runs = index.get("trials")
        if not isinstance(runs, list) or len(runs) != protocol.get("trials"):
            raise AnalysisError("run index does not contain every preregistered trial")
        protocol_sha = sha256_file(args.protocol)
        sites = [site["site_id"] for site in dossier["sites"]]
        site_by_id = {site["site_id"]: site for site in dossier["sites"]}
        oracle_by_kernel = response_actions(dossier, oracle)
        gbt_by_kernel = response_actions(dossier, gbt)

        observed: list[dict[str, Any]] = []
        vectors: list[tuple[str, ...]] = []
        decision_models: set[str] = set()
        all_models: set[str] = set()
        total_cost = 0.0
        for wanted, indexed in enumerate(runs, 1):
            if indexed.get("trial") != wanted:
                raise AnalysisError("run index trial order is not 1..N")
            trial_dir = args.trials / f"trial{wanted:02d}"
            run = read_json(trial_dir / "run.json")
            if run != indexed:
                raise AnalysisError(f"trial{wanted:02d}: run/index mismatch")
            if run.get("protocol_sha256") != protocol_sha:
                raise AnalysisError(f"trial{wanted:02d}: protocol hash mismatch")
            if not run.get("provider_call_succeeded"):
                raise AnalysisError(f"trial{wanted:02d}: provider call failed")
            if not run.get("bridge_accepted") or run.get("fallback_applied"):
                raise AnalysisError(f"trial{wanted:02d}: response was not accepted")
            attempts = run.get("attempts")
            if (
                not isinstance(attempts, list)
                or len(attempts) != 1
                or attempts[0].get("returncode") != 0
            ):
                raise AnalysisError(f"trial{wanted:02d}: unexpected retry/failure")

            response_path = trial_dir / "response.json"
            response = validated_response(dossier, response_path)
            if canonical_sha256(response) != run.get("response_sha256_canonical"):
                raise AnalysisError(f"trial{wanted:02d}: response hash mismatch")
            envelope = read_json(trial_dir / "attempt1.stdout.json")
            if sha256_file(trial_dir / "attempt1.stdout.json") != attempts[0].get(
                "stdout_sha256"
            ):
                raise AnalysisError(f"trial{wanted:02d}: envelope hash mismatch")
            primary_model, models, cost = decision_model(envelope)
            decision_models.add(primary_model)
            all_models.update(models)
            total_cost += cost
            vector = tuple(response["decisions"][site]["action"] for site in sites)
            vectors.append(vector)
            observed.append({
                "trial": wanted,
                "response": str(response_path),
                "response_sha256": sha256_file(response_path),
                "response_sha256_canonical": canonical_sha256(response),
                "decision_model": primary_model,
                "models_observed": models,
                "provider_cost_usd_reported": cost,
                "actions": dict(zip(sites, vector)),
            })

        unique_vectors: list[tuple[str, ...]] = []
        for vector in vectors:
            if vector not in unique_vectors:
                unique_vectors.append(vector)
        policy_for = {
            vector: f"llm-policy{index:02d}"
            for index, vector in enumerate(unique_vectors, 1)
        }
        scored = [scenario for scenario in SCENARIOS if scenario not in FORCED_ACTIONS]
        policies: dict[str, Any] = {}
        for vector in unique_vectors:
            name = policy_for[vector]
            members = [index + 1 for index, value in enumerate(vectors) if value == vector]
            representative = observed[members[0] - 1]
            action_map = dict(zip(sites, vector))
            by_kernel: dict[str, list[str]] = {}
            for site in dossier["sites"]:
                by_kernel.setdefault(site["kernel"], []).append(action_map[site["site_id"]])
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
                "trials": members,
                "representative_response": representative["response"],
                "representative_response_sha256": representative["response_sha256"],
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

        distributions = {}
        for site in sites:
            counts = collections.Counter(
                observed[index]["actions"][site] for index in range(len(observed))
            )
            distributions[site] = {
                "kernel": site_by_id[site]["kernel"],
                "counts": dict(sorted(counts.items())),
            }
        summary = {
            "schema_version": "gicc-compiler-lto-llm-trial-analysis-v1",
            "scope": "post-freeze action analysis; no source or labels entered provider calls",
            "protocol": str(args.protocol),
            "protocol_sha256": protocol_sha,
            "dossier_id": dossier["dossier_id"],
            "trial_count": len(observed),
            "accepted_count": len(observed),
            "fallback_count": 0,
            "retry_count": 0,
            "unique_response_count": len({row["response_sha256_canonical"] for row in observed}),
            "unique_policy_count": len(policies),
            "decision_models": sorted(decision_models),
            "all_provider_models_observed": sorted(all_models),
            "provider_cost_usd_reported_total": total_cost,
            "actions_by_site_distribution": distributions,
            "policies": policies,
            "trials": observed,
            "oracle_response": str(args.oracle_response),
            "oracle_response_sha256": sha256_file(args.oracle_response),
            "calibrated_gbt_response": str(args.gbt_response),
            "calibrated_gbt_response_sha256": sha256_file(args.gbt_response),
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

        print(
            f"trials={len(observed)} accepted={len(observed)} "
            f"unique_policies={len(policies)} decision_models={sorted(decision_models)}"
        )
        for name, row in policies.items():
            agreement = row["oracle_action_agreement"]
            print(
                f"{name} frequency={row['frequency']} "
                f"oracle={agreement['scenario_matches']}/{agreement['scenario_count']} scenarios, "
                f"{agreement['site_matches']}/{agreement['site_count']} sites "
                f"representative={row['representative_response']}"
            )
        return 0
    except (AnalysisError, bridge.BridgeError, OSError, ValueError) as exc:
        print(f"compiler-lto-llm-trials: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
