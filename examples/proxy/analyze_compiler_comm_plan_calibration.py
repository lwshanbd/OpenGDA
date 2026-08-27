#!/usr/bin/env python3
"""Validate six-opportunity controls and derive a factorized 3^6/4^6 oracle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "proxy"))

import compiler_comm_plan_eval as controls  # noqa: E402
from analyze_compiler_lto_calibration import (  # noqa: E402
    REGION,
    expected_hash,
)
from compiler_lto_calibration import SCENARIO_FOR_KERNEL  # noqa: E402


KINDS = (
    "proxy_device",
    "trigger_descriptor_batch",
    "trigger_coalesced_loop",
    "trigger_coalesced_early",
)
ARM_FOR_KIND = {
    "proxy_device": "uniform_p",
    "trigger_descriptor_batch": "uniform_t",
    "trigger_coalesced_loop": "uniform_c",
    "trigger_coalesced_early": "uniform_e",
}
FIXED_ROUTES = {
    "single-512-g1": (1, 0),
    "single-2k-g4": (1, 0),
    "single-16k-g1": (1, 0),
    "single-64k-g8": (1, 0),
    "single-256k-g4": (1, 0),
    "single-512k-g8": (1, 0),
    "reuse-1k-k8-g1": (8, 0),
    "reuse-2k-k24-g1": (24, 0),
    "reuse-8k-k48-g4": (48, 0),
    "static2-2k-g2": (2, 0),
    "static3-8k-g4": (3, 0),
    "static6-16k-g8": (6, 0),
}
FIELDS = re.compile(r"([a-z0-9_]+)=([^ ]+)")


class AnalysisError(ValueError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def geomean(values: list[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise AnalysisError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def bootstrap_ratio_ci(
    numerator: list[float], denominator: list[float], seed: int = 20260827,
) -> tuple[float, float]:
    if len(numerator) != len(denominator) or not numerator:
        raise AnalysisError("paired bootstrap requires equal non-empty vectors")
    ratios = [left / right for left, right in zip(numerator, denominator)]
    rng = random.Random(seed)
    estimates = []
    for _ in range(10000):
        estimates.append(geomean([
            ratios[rng.randrange(len(ratios))] for _ in ratios
        ]))
    estimates.sort()
    return estimates[249], estimates[9749]


def routes_for(arm: dict[str, Any]) -> dict[str, tuple[int, int]]:
    routes = dict(FIXED_ROUTES)
    for selection in arm.get("selections", []):
        kernel = selection.get("kernel")
        scenario = SCENARIO_FOR_KERNEL.get(kernel)
        if scenario is None or scenario in routes:
            raise AnalysisError(f"{arm.get('name')}: invalid opportunity kernel")
        effects = selection.get("effects")
        if not isinstance(effects, dict):
            raise AnalysisError(f"{arm.get('name')}: missing candidate effects")
        kind = selection.get("kind")
        if kind == "proxy_device":
            operations = effects.get("network_operations")
            if not isinstance(operations, int) or operations <= 0:
                raise AnalysisError(f"{arm.get('name')}: invalid proxy operations")
            # The loop sends N commands and one completion/quiet command.
            routes[scenario] = (0, operations + 1)
        elif kind in (
            "trigger_descriptor_batch",
            "trigger_coalesced_loop",
            "trigger_coalesced_early",
        ):
            descriptors = effects.get("host_descriptors")
            if not isinstance(descriptors, int) or descriptors <= 0:
                raise AnalysisError(
                    f"{arm.get('name')}: invalid host descriptor count"
                )
            routes[scenario] = (descriptors, 0)
        else:
            raise AnalysisError(f"{arm.get('name')}: unknown action kind {kind}")
    if set(routes) != set(SCENARIO_FOR_KERNEL.values()):
        missing = sorted(set(SCENARIO_FOR_KERNEL.values()) - set(routes))
        raise AnalysisError(f"{arm.get('name')}: incomplete routes: {missing}")
    return routes


def load_contract(
    graph_path: Path, manifest_path: Path,
) -> tuple[list[str], dict[str, dict[str, Any]], list[str], tuple[str, ...]]:
    graph = read_json(graph_path)
    manifest = read_json(manifest_path)
    try:
        controls.verify_manifest(graph, manifest, manifest_path.parent)
    except (controls.ControlError, OSError, ValueError) as exc:
        raise AnalysisError(f"invalid compiler control manifest: {exc}") from exc
    schema = manifest.get("schema_version")
    if schema == controls.UNIFORM_MANIFEST_SCHEMA:
        kinds = KINDS[:3]
    elif schema == controls.PLACEMENT_MANIFEST_SCHEMA:
        kinds = KINDS
    else:
        raise AnalysisError("expected a uniform or placement-control manifest")
    if manifest.get("model_invoked") is not False:
        raise AnalysisError("capacity controls must not contain model output")
    arms = manifest["arms"]
    names = [arm["name"] for arm in arms]
    expected_names = [ARM_FOR_KIND[kind] for kind in kinds]
    if names != expected_names:
        raise AnalysisError(
            f"uniform controls must be ordered {expected_names}"
        )
    contract = {arm["name"]: arm for arm in arms}
    for kind in kinds:
        name = ARM_FOR_KIND[kind]
        observed = {row["kind"] for row in contract[name]["selections"]}
        if observed != {kind}:
            raise AnalysisError(f"{name}: does not uniformly select {kind}")
    structural = [
        SCENARIO_FOR_KERNEL[item["kernel"]]
        for item in manifest["opportunity_order"]
    ]
    if len(structural) != 6 or len(set(structural)) != 6:
        raise AnalysisError("calibration contract must expose six opportunities")
    return names, contract, structural, kinds


def parse_log(
    path: Path,
    names: list[str],
    routes_by_arm: dict[str, dict[str, tuple[int, int]]],
) -> tuple[str, int, str, dict[str, dict[str, Any]]]:
    arm = next((name for name in names if path.name.endswith(f"-{name}.log")), None)
    if arm is None:
        raise AnalysisError(f"cannot infer arm from {path}")
    match = re.search(r"rep([0-9]+)-", path.name)
    if match is None:
        raise AnalysisError(f"cannot infer replicate from {path}")
    rep = int(match.group(1))
    lines = path.read_text().splitlines()
    run_lines = [line for line in lines if line.startswith("RUN rep=")]
    if len(run_lines) != 1:
        raise AnalysisError(f"{path}: expected exactly one RUN provenance line")
    provenance = dict(FIELDS.findall(run_lines[0]))
    if provenance.get("rep") != str(rep) or provenance.get("arm") != arm:
        raise AnalysisError(f"{path}: RUN provenance mismatch")
    binary = Path(provenance.get("binary", ""))
    if binary.name not in {
        f"compiler_comm_plan_calibration_{arm}",
        f"compiler_comm_plan_placement_{arm}",
    }:
        raise AnalysisError(f"{path}: binary/arm mismatch")
    binary_sha = provenance.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", binary_sha):
        raise AnalysisError(f"{path}: invalid binary SHA-256")

    rows: dict[str, dict[str, Any]] = {}
    kernel_for_scenario = {
        scenario: kernel for kernel, scenario in SCENARIO_FOR_KERNEL.items()
    }
    for line in lines:
        if not line.startswith("COMPILER_LTO_CALIBRATION "):
            continue
        fields = dict(FIELDS.findall(line))
        scenario = fields.get("scenario", "")
        if scenario in rows:
            raise AnalysisError(f"{path}: duplicate scenario {scenario}")
        if scenario not in kernel_for_scenario:
            raise AnalysisError(f"{path}: unknown scenario {scenario}")
        if fields.get("kernel") != kernel_for_scenario[scenario]:
            raise AnalysisError(f"{path}: scenario/kernel mismatch")
        row = {
            "runs": int(fields["runs"]),
            "median_us": float(fields["median_us"]),
            "p25_us": float(fields["p25_us"]),
            "p75_us": float(fields["p75_us"]),
            "staged": int(fields["staged"]),
            "pushed": int(fields["pushed"]),
            "hash": fields["hash"],
            "data": fields["data"],
        }
        if (
            row["runs"] <= 0
            or row["p25_us"] <= 0
            or not row["p25_us"] <= row["median_us"] <= row["p75_us"]
        ):
            raise AnalysisError(f"{path}: {scenario} invalid timing fields")
        base, size = REGION[scenario]
        if row["data"] != "OK" or row["hash"] != expected_hash(base, size):
            raise AnalysisError(f"{path}: {scenario} payload validation failed")
        staged, pushed = routes_by_arm[arm][scenario]
        wanted = staged * row["runs"], pushed * row["runs"]
        observed = row["staged"], row["pushed"]
        if observed != wanted:
            raise AnalysisError(
                f"{path}: {scenario} routes={observed}, expected={wanted}"
            )
        rows[scenario] = row
    if set(rows) != set(kernel_for_scenario):
        raise AnalysisError(f"{path}: incomplete scenario set")
    return arm, rep, binary_sha, rows


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--graph", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument(
        "--expected-reps", type=parse_replicates,
        default=parse_replicates("1,2,3,4"),
    )
    args = parser.parse_args()
    try:
        names, contract, structural, kinds = load_contract(
            args.graph, args.manifest
        )
        placement_minimum_ratio = 1.01
        placement_minimum_late = 1
        placement_minimum_early = 1
        routes_by_arm = {name: routes_for(contract[name]) for name in names}
        parsed: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
        binary_hashes = {name: set() for name in names}
        for path in args.logs:
            arm, rep, binary_sha, rows = parse_log(path, names, routes_by_arm)
            if arm in parsed.setdefault(rep, {}):
                raise AnalysisError(f"duplicate rep={rep} arm={arm}")
            parsed[rep][arm] = rows
            binary_hashes[arm].add(binary_sha)
        if set(parsed) != set(args.expected_reps):
            raise AnalysisError(
                f"replicates={sorted(parsed)}, expected={sorted(args.expected_reps)}"
            )
        for rep, arms in parsed.items():
            if set(arms) != set(names):
                raise AnalysisError(f"rep={rep}: incomplete uniform controls")
        if any(len(values) != 1 for values in binary_hashes.values()):
            raise AnalysisError("an arm used multiple binaries across replicates")
        protocol_sha256 = None
        if len(kinds) == 4:
            if args.protocol is None:
                raise AnalysisError(
                    "placement analysis requires the frozen --protocol"
                )
            protocol = read_json(args.protocol)
            if protocol.get("schema_version") != (
                "gicc-communication-plan-placement-protocol-v1"
            ):
                raise AnalysisError("unexpected placement protocol schema")
            gate_contract = protocol.get("llm_gate", {})
            placement_minimum_ratio = gate_contract.get(
                "minimum_best_uniform_over_fixed_oracle_ratio"
            )
            placement_minimum_late = gate_contract.get(
                "requires_stable_late_winner_count_at_least"
            )
            placement_minimum_early = gate_contract.get(
                "requires_stable_early_winner_count_at_least"
            )
            if (
                not isinstance(placement_minimum_ratio, (int, float))
                or placement_minimum_ratio <= 1.0
                or not isinstance(placement_minimum_late, int)
                or placement_minimum_late < 1
                or not isinstance(placement_minimum_early, int)
                or placement_minimum_early < 1
            ):
                raise AnalysisError("invalid placement gate in frozen protocol")
            frozen = protocol.get("artifacts", {})
            if (
                frozen.get("opportunity_graph", {}).get("sha256")
                != sha256_file(args.graph)
                or frozen.get("control_manifest", {}).get("sha256")
                != sha256_file(args.manifest)
                or frozen.get("source_sha256")
                != read_json(args.manifest).get("source_sha256")
            ):
                raise AnalysisError(
                    "graph, manifest, or source does not match frozen protocol"
                )
            observed_hashes = {
                name: next(iter(values))
                for name, values in binary_hashes.items()
            }
            if observed_hashes != frozen.get("binaries"):
                raise AnalysisError(
                    "runtime binary hashes do not match frozen protocol"
                )
            protocol_sha256 = sha256_file(args.protocol)

        aggregate: dict[str, Any] = {}
        per_rep_scores: dict[str, list[float]] = {name: [] for name in names}
        for name in names:
            medians = {
                scenario: geomean([
                    parsed[rep][name][scenario]["median_us"]
                    for rep in sorted(parsed)
                ])
                for scenario in SCENARIO_FOR_KERNEL.values()
            }
            for rep in sorted(parsed):
                per_rep_scores[name].append(geomean([
                    parsed[rep][name][scenario]["median_us"]
                    for scenario in structural
                ]))
            aggregate[name] = {
                "geomean_median_us": medians,
                "structural_score_us": geomean([
                    medians[scenario] for scenario in structural
                ]),
                "selections": contract[name]["selections"],
            }

        selection_for = {
            (row["kernel"], row["kind"]): row
            for arm in contract.values() for row in arm["selections"]
        }
        site_oracles = []
        for scenario in structural:
            kernel = next(
                kernel for kernel, value in SCENARIO_FOR_KERNEL.items()
                if value == scenario
            )
            times = {
                kind: aggregate[ARM_FOR_KIND[kind]]["geomean_median_us"][scenario]
                for kind in kinds
            }
            best = min(kinds, key=lambda kind: times[kind])
            per_rep_best = [
                min(
                    kinds,
                    key=lambda kind: parsed[rep][ARM_FOR_KIND[kind]][scenario][
                        "median_us"
                    ],
                )
                for rep in sorted(parsed)
            ]
            selected = selection_for[(kernel, best)]
            site_oracles.append({
                "opportunity_id": selected["opportunity_id"],
                "kernel": kernel,
                "scenario": scenario,
                "geomean_median_us_by_action": times,
                "best_action": best,
                "candidate_id": selected["candidate_id"],
                "per_replicate_best_action": per_rep_best,
                "stable_winner": len(set(per_rep_best)) == 1,
            })

        factorized_score = geomean([
            row["geomean_median_us_by_action"][row["best_action"]]
            for row in site_oracles
        ])
        # Evaluate one frozen factorized policy selected from the aggregate
        # medians.  Re-selecting the minimum independently in every replicate
        # would turn timing noise into optimistic oracle performance.
        factorized_per_rep = [
            geomean([
                parsed[rep][ARM_FOR_KIND[row["best_action"]]][row["scenario"]][
                    "median_us"
                ]
                for row in site_oracles
            ])
            for rep in sorted(parsed)
        ]
        baseline = "uniform_t"
        ratio = aggregate[baseline]["structural_score_us"] / factorized_score
        ci_low, ci_high = bootstrap_ratio_ci(
            per_rep_scores[baseline], factorized_per_rep
        )
        uniform_oracle = min(
            names, key=lambda name: aggregate[name]["structural_score_us"]
        )
        uniform_ratio = (
            aggregate[uniform_oracle]["structural_score_us"]
            / factorized_score
        )
        uniform_ci_low, uniform_ci_high = bootstrap_ratio_ci(
            per_rep_scores[uniform_oracle], factorized_per_rep, seed=20260828
        )
        stable_winners = [
            row["best_action"] for row in site_oracles if row["stable_winner"]
        ]
        stable_winner_counts = {
            kind: stable_winners.count(kind) for kind in kinds
        }
        placement_gate = None
        if len(kinds) == 4:
            late_count = stable_winner_counts["trigger_coalesced_loop"]
            early_count = stable_winner_counts["trigger_coalesced_early"]
            placement_gate = {
                "definition": (
                    "The preregistered minimum number of stable late and early "
                    "winners must be met, and the "
                    "paired-bootstrap lower bound for best-uniform/fixed-oracle "
                    "must exceed the preregistered effect ratio."
                ),
                "stable_late_winner_count": late_count,
                "stable_early_winner_count": early_count,
                "required_stable_late_winner_count": placement_minimum_late,
                "required_stable_early_winner_count": placement_minimum_early,
                "mixed_stable_placement": (
                    late_count >= placement_minimum_late
                    and early_count >= placement_minimum_early
                ),
                "minimum_effect_ratio": placement_minimum_ratio,
                "passed": (
                    late_count >= placement_minimum_late
                    and early_count >= placement_minimum_early
                    and uniform_ci_low > placement_minimum_ratio
                ),
            }
        summary = {
            "schema_version": (
                "gicc-communication-plan-placement-runtime-v1"
                if len(kinds) == 4
                else "gicc-communication-plan-calibration-runtime-v1"
            ),
            "graph_id": read_json(args.graph)["graph_id"],
            "graph_sha256": sha256_file(args.graph),
            "manifest_id": read_json(args.manifest)["manifest_id"],
            "manifest_sha256": sha256_file(args.manifest),
            "source_sha256": read_json(args.manifest)["source_sha256"],
            "protocol_sha256": protocol_sha256,
            "replicates": sorted(parsed),
            "binary_sha256_by_arm": {
                name: next(iter(binary_hashes[name])) for name in names
            },
            "structural_scenarios": structural,
            "aggregate": aggregate,
            "site_oracles": site_oracles,
            "factorized_oracle": {
                "policy_space_size": len(kinds) ** len(structural),
                "structural_score_us": factorized_score,
                "stable_site_count": sum(row["stable_winner"] for row in site_oracles),
                "site_count": len(site_oracles),
                "trigger_batch_baseline": baseline,
                "trigger_batch_over_oracle": ratio,
                "paired_bootstrap_95_ci": [ci_low, ci_high],
                "best_uniform_arm": uniform_oracle,
                "best_uniform_score_us": aggregate[uniform_oracle][
                    "structural_score_us"
                ],
                "best_uniform_over_oracle": uniform_ratio,
                "best_uniform_over_oracle_paired_bootstrap_95_ci": [
                    uniform_ci_low, uniform_ci_high,
                ],
                "stable_winner_count_by_action": stable_winner_counts,
                "scope": (
                    f"Exact factorized oracle over {len(kinds)}^6 compiler "
                    "candidate-ID plans because the six named scenarios are timed "
                    "independently; it does not measure cross-site interaction."
                ),
            },
            "placement_llm_gate": placement_gate,
            "model_invoked": False,
            "source_visible_to_model": False,
            "scope": (
                "Compiler-generated uniform controls and factorized capacity "
                "labels; no LLM result and no source modification."
            ),
        }

        print(
            "scenario                         proxy_us trigger_us "
            "coalesce_us early_us best stable"
        )
        for row in site_oracles:
            times = row["geomean_median_us_by_action"]
            print(
                f"{row['scenario']:<32} "
                f"{times['proxy_device']:8.3f} "
                f"{times['trigger_descriptor_batch']:10.3f} "
                f"{times['trigger_coalesced_loop']:11.3f} "
                + (
                    f"{times['trigger_coalesced_early']:8.3f} "
                    if 'trigger_coalesced_early' in times else "       - "
                )
                +
                f"{row['best_action']:>29} {str(row['stable_winner']):>6}"
            )
        print(
            f"factorized_oracle={factorized_score:.3f}us "
            f"uniform_trigger_over_oracle={ratio:.6f} "
            f"ci95=[{ci_low:.6f},{ci_high:.6f}] "
            f"stable={summary['factorized_oracle']['stable_site_count']}/"
            f"{len(site_oracles)}"
        )
        if placement_gate is not None:
            print(
                f"best_uniform_over_oracle={uniform_ratio:.6f} "
                f"ci95=[{uniform_ci_low:.6f},{uniform_ci_high:.6f}] "
                f"stable_late={placement_gate['stable_late_winner_count']} "
                f"stable_early={placement_gate['stable_early_winner_count']} "
                f"llm_gate={placement_gate['passed']}"
            )
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return 0
    except (AnalysisError, controls.ControlError, OSError, ValueError) as exc:
        print(f"compiler-comm-plan-calibration-analysis: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
