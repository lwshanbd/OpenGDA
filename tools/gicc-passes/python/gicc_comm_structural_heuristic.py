#!/usr/bin/env python3
"""Emit a deterministic source-free communication compiler control.

The control is deliberately narrower than the relational decision graph.  It
uses independent deployment calibration plus compiler-derived group/launch
facts to compare only uniform proxy and uniform trigger issue overhead.  It
never chooses a schedule transform or a mixed route.  Missing facts, missing
physical candidates, or a cost tie fall back atomically to the semantic
compiler anchor for that opportunity.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import gicc_comm_group_plan_bridge as plans
import gicc_llm_bridge as bridge


CONTROL_SCHEMA = "gicc-communication-structural-heuristic-control-v1"
HEURISTIC_VERSION = "calibrated-uniform-issue-overhead-v1"
MEASUREMENT_KEYS = (
    "proxy_fixed_us",
    "proxy_per_op_issue_us",
    "trigger_fixed_us",
    "trigger_per_op_stage_us",
)


class HeuristicError(ValueError):
    """The graph cannot support a deterministic compiler control safely."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeuristicError(f"cannot read JSON {path}: {exc}") from exc


def _positive_real(value: Any) -> float | None:
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(float(value)) or float(value) <= 0):
        return None
    return float(value)


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _anchor(opportunity: dict[str, Any]) -> dict[str, Any]:
    preference = (
        "group_uniform_default", "site_default",
        "group_uniform_trigger", "site_trigger",
        "group_uniform_proxy", "site_proxy",
    )
    for kind in preference:
        matches = [
            candidate for candidate in opportunity["candidates"]
            if candidate.get("kind") == kind
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise HeuristicError(
                f"opportunity contains duplicate semantic anchor kind {kind}"
            )
    if not opportunity["candidates"]:
        raise HeuristicError("opportunity has no compiler candidate")
    return opportunity["candidates"][0]


def _uniform_candidate(
    opportunity: dict[str, Any], action: str,
) -> dict[str, Any] | None:
    if action not in {"proxy", "trigger"}:
        raise HeuristicError(f"unsupported physical action {action!r}")
    kinds = {f"group_uniform_{action}", f"site_{action}"}
    matches = []
    site_ids = opportunity["site_ids"]
    expected_materializer = plans.ACTION_TO_MATERIALIZER[action]
    for candidate in opportunity["candidates"]:
        if candidate.get("kind") not in kinds:
            continue
        actions = candidate.get("effects", {}).get("site_actions")
        materializers = candidate.get("materializer", {}).get("sites")
        if (
            isinstance(actions, dict)
            and set(actions) == set(site_ids)
            and all(actions[site_id] == action for site_id in site_ids)
            and isinstance(materializers, dict)
            and set(materializers) == set(site_ids)
            and all(materializers[site_id] == expected_materializer
                    for site_id in site_ids)
        ):
            matches.append(candidate)
    if len(matches) > 1:
        raise HeuristicError(
            f"opportunity has duplicate uniform {action} candidates"
        )
    return matches[0] if matches else None


def _opportunity_shape(
    opportunity: dict[str, Any], worker_lanes: int,
) -> tuple[int, int, int] | None:
    facts = opportunity.get("compiler_facts", {})
    site_ids = opportunity.get("site_ids", [])
    group_size = _positive_int(facts.get("group_size"))
    batches = facts.get("batch_size")
    semantics = facts.get("site_semantic_facts")
    if (
        group_size != len(site_ids)
        or not isinstance(batches, list)
        or len(batches) != group_size
        or not isinstance(semantics, dict)
        or set(semantics) != set(site_ids)
    ):
        return None
    parsed_batches = [_positive_int(value) for value in batches]
    if (
        any(value is None for value in parsed_batches)
        or len(set(parsed_batches)) != 1
        or parsed_batches[0] != group_size
    ):
        return None
    grids = [
        _positive_int(semantics[site_id].get("grid_blocks"))
        if isinstance(semantics[site_id], dict) else None
        for site_id in site_ids
    ]
    if any(value is None for value in grids) or len(set(grids)) != 1:
        return None
    operations = parsed_batches[0]
    grid_blocks = grids[0]
    proxy_concurrency = min(operations, grid_blocks, worker_lanes)
    return operations, grid_blocks, proxy_concurrency


def _select_opportunity(
    graph: dict[str, Any], opportunity: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    anchor = _anchor(opportunity)
    measurements = graph.get("platform_profile", {}).get("measurements", {})
    deployment = graph.get("platform_profile", {}).get(
        "deployment_constraints", {}
    )
    values = {
        key: _positive_real(measurements.get(key))
        if isinstance(measurements, dict) else None
        for key in MEASUREMENT_KEYS
    }
    worker_lanes = (
        _positive_int(deployment.get("proxy_worker_lanes"))
        if isinstance(deployment, dict) else None
    )
    diagnostic: dict[str, Any] = {
        "status": "semantic_anchor_fallback",
        "selected_candidate_id": anchor["candidate_id"],
    }
    if worker_lanes is None or any(value is None for value in values.values()):
        diagnostic["reason"] = "incomplete_independent_platform_calibration"
        return anchor, diagnostic
    shape = _opportunity_shape(opportunity, worker_lanes)
    if shape is None:
        diagnostic["reason"] = "incomplete_or_nonuniform_compiler_shape"
        return anchor, diagnostic
    proxy = _uniform_candidate(opportunity, "proxy")
    trigger = _uniform_candidate(opportunity, "trigger")
    if proxy is None or trigger is None:
        diagnostic["reason"] = "uniform_physical_candidate_missing"
        return anchor, diagnostic

    operations, grid_blocks, proxy_concurrency = shape
    proxy_us = (
        values["proxy_fixed_us"]
        + values["proxy_per_op_issue_us"] * operations / proxy_concurrency
    )
    trigger_us = (
        values["trigger_fixed_us"]
        + values["trigger_per_op_stage_us"] * operations
    )
    diagnostic.update({
        "operations": operations,
        "grid_blocks": grid_blocks,
        "proxy_worker_lanes": worker_lanes,
        "proxy_concurrency": proxy_concurrency,
        "estimated_issue_us": {
            "proxy": proxy_us,
            "trigger": trigger_us,
        },
    })
    if math.isclose(proxy_us, trigger_us, rel_tol=1e-12, abs_tol=1e-12):
        diagnostic["reason"] = "estimated_issue_cost_tie"
        return anchor, diagnostic
    selected = proxy if proxy_us < trigger_us else trigger
    diagnostic.update({
        "status": "uniform_physical_route_selected",
        "reason": "lower_calibrated_issue_overhead",
        "selected_candidate_id": selected["candidate_id"],
        "selected_action": "proxy" if selected is proxy else "trigger",
    })
    return selected, diagnostic


def make_decision(
    graph_value: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = plans.verified_graph(graph_value)
    selections: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for opportunity in graph["opportunities"]:
        opportunity_id = opportunity["opportunity_id"]
        candidate, diagnostic = _select_opportunity(graph, opportunity)
        selections[opportunity_id] = {
            "candidate_id": candidate["candidate_id"],
            "confidence": 1.0,
            "rationale": (
                "deterministic source-free calibrated issue-overhead rule"
                if diagnostic["status"] == "uniform_physical_route_selected"
                else "deterministic semantic-anchor fallback"
            ),
        }
        diagnostics[opportunity_id] = diagnostic
    return {
        "schema_version": plans.DECISION_SCHEMA,
        "graph_id": graph["graph_id"],
        "selections": selections,
    }, diagnostics


def make_control(
    graph_value: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    graph = plans.verified_graph(graph_value)
    decision, diagnostics = make_decision(graph)
    hint, accepted, errors = plans.plan_to_hint(graph, decision)
    if not accepted:
        raise HeuristicError(
            "compiler bridge rejected its deterministic control: "
            + "; ".join(errors)
        )
    payload = {
        "schema_version": CONTROL_SCHEMA,
        "heuristic_version": HEURISTIC_VERSION,
        "graph_id": graph["graph_id"],
        "boundary": {
            "source_visible": False,
            "evaluation_runtime_results_visible": False,
            "independent_platform_calibration_visible": True,
            "provider_output_visible": False,
            "output_is_graph_bound_candidate_ids": True,
            "compiler_revalidates_before_materialization": True,
            "schedule_transforms_selectable_by_rule": False,
            "mixed_routes_selectable_by_rule": False,
        },
        "rule": {
            "scope": "uniform proxy versus uniform trigger issue overhead",
            "proxy_estimate": (
                "proxy_fixed_us + proxy_per_op_issue_us * operations / "
                "min(operations, grid_blocks, proxy_worker_lanes)"
            ),
            "trigger_estimate": (
                "trigger_fixed_us + trigger_per_op_stage_us * operations"
            ),
            "required_compiler_facts": [
                "group_size", "uniform batch_size equal to group_size",
                "uniform positive grid_blocks",
            ],
            "fallback": "semantic_anchor_for_each_opportunity",
        },
        "opportunity_diagnostics": diagnostics,
        "decision_id": bridge._fingerprint(decision),
        "hint_id": bridge._fingerprint(hint),
    }
    control = {"control_id": bridge._fingerprint(payload), **payload}
    return decision, hint, control


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--hint", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        decision, hint, control = make_control(_read_json(args.graph))
        bridge._write_json_atomic(args.decision, decision)
        bridge._write_json_atomic(args.hint, hint)
        bridge._write_json_atomic(args.control, control)
        print(
            "gicc-communication-structural-heuristic: wrote compiler control; "
            f"control_id={control['control_id']}",
            file=sys.stderr,
        )
        return 0
    except (HeuristicError, plans.GroupPlanError, OSError, ValueError) as exc:
        print(
            f"gicc-communication-structural-heuristic: ERROR: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
