"""Pure aggregation for compiler-only LLM policy screens.

Family-specific adapters must first verify provider archives, bridge decisions,
compiler graphs, and held-out runtime-control labels.  This module then applies
the preregistered intention-to-treat, modal-policy, and best-of-N summaries.  It
does no I/O and has no provider, compiler, scheduler, or source-edit path.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from typing import Any


VIEWS = ("relational", "descriptors", "opaque")


class CapabilityMetricsError(RuntimeError):
    """A policy screen is incomplete or violates the metric contract."""


def fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _positive_map(value: Any, label: str) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise CapabilityMetricsError(f"{label} must be a non-empty object")
    result: dict[str, float] = {}
    for key, number in value.items():
        if (not isinstance(key, str) or not key
                or not isinstance(number, (int, float))
                or isinstance(number, bool)
                or not math.isfinite(number) or number <= 0):
            raise CapabilityMetricsError(f"{label} has an invalid value")
        result[key] = float(number)
    return result


def _policy(value: Any, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise CapabilityMetricsError(f"{label} must be a non-empty object")
    if any(not isinstance(slot, str) or not slot
           or not isinstance(option, str) or not option
           for slot, option in value.items()):
        raise CapabilityMetricsError(f"{label} has an invalid slot or option")
    return dict(value)


def policy_id(policy: dict[str, str]) -> str:
    return fingerprint({"selected_ids_by_slot": _policy(policy, "policy")})


def weighted_geomean_ratio(numerator: dict[str, float],
                           denominator: dict[str, float],
                           weights: dict[str, float]) -> float:
    top = _positive_map(numerator, "numerator")
    bottom = _positive_map(denominator, "denominator")
    weight = _positive_map(weights, "weights")
    if set(top) != set(bottom) or set(top) != set(weight):
        raise CapabilityMetricsError("cost and weight unit sets differ")
    total = sum(weight.values())
    return math.exp(sum(
        weight[unit] * math.log(top[unit] / bottom[unit])
        for unit in sorted(top)
    ) / total)


def _control(value: Any, label: str) -> tuple[dict[str, str], dict[str, float]]:
    if not isinstance(value, dict) or set(value) != {
        "selected_ids_by_slot", "cost_by_unit",
    }:
        raise CapabilityMetricsError(f"{label} has an invalid shape")
    return (
        _policy(value["selected_ids_by_slot"], f"{label} policy"),
        _positive_map(value["cost_by_unit"], f"{label} costs"),
    )


def score_trial(trial: Any, *, oracle: dict[str, Any],
                anchor: dict[str, Any], deterministic: dict[str, Any],
                unit_weights: dict[str, float]) -> dict[str, Any]:
    """Score one already validated response against held-out compiler controls."""
    if not isinstance(trial, dict):
        raise CapabilityMetricsError("trial must be an object")
    required = {
        "view", "trial", "bridge_accepted", "selected_ids_by_slot",
        "cost_by_unit",
    }
    if set(trial) != required:
        raise CapabilityMetricsError("trial fields do not match metric schema")
    view = trial["view"]
    index = trial["trial"]
    accepted = trial["bridge_accepted"]
    if view not in VIEWS:
        raise CapabilityMetricsError("trial has an unknown view")
    if not isinstance(index, int) or isinstance(index, bool) or index <= 0:
        raise CapabilityMetricsError("trial index must be a positive integer")
    if not isinstance(accepted, bool):
        raise CapabilityMetricsError("bridge_accepted must be Boolean")

    selected = _policy(trial["selected_ids_by_slot"], "selected policy")
    cost = _positive_map(trial["cost_by_unit"], "selected costs")
    oracle_policy, oracle_cost = _control(oracle, "oracle")
    anchor_policy, anchor_cost = _control(anchor, "anchor")
    deterministic_policy, deterministic_cost = _control(
        deterministic, "deterministic"
    )
    weights = _positive_map(unit_weights, "unit weights")
    slot_set = set(oracle_policy)
    if (set(selected) != slot_set or set(anchor_policy) != slot_set
            or set(deterministic_policy) != slot_set):
        raise CapabilityMetricsError("policy slot sets differ")
    unit_set = set(oracle_cost)
    if (set(cost) != unit_set or set(anchor_cost) != unit_set
            or set(deterministic_cost) != unit_set or set(weights) != unit_set):
        raise CapabilityMetricsError("control cost unit sets differ")
    if not accepted and (selected != anchor_policy or cost != anchor_cost):
        raise CapabilityMetricsError(
            "invalid response was not scored with atomic anchor fallback"
        )

    matches = sum(
        selected[slot] == oracle_policy[slot] for slot in sorted(slot_set)
    )
    return {
        "view": view,
        "trial": index,
        "bridge_accepted": accepted,
        "fallback_applied": not accepted,
        "policy_id": policy_id(selected),
        "selected_ids_by_slot": selected,
        "anchor_policy_id": policy_id(anchor_policy),
        "exact_oracle_policy": selected == oracle_policy,
        "decision_slot_accuracy": matches / len(slot_set),
        "oracle_normalized_geomean_regret": weighted_geomean_ratio(
            cost, oracle_cost, weights,
        ),
        "deterministic_control_normalized_geomean_regret": (
            weighted_geomean_ratio(cost, deterministic_cost, weights)
        ),
        "geomean_speedup_over_anchor": weighted_geomean_ratio(
            anchor_cost, cost, weights,
        ),
    }


def _geomean(values: list[float]) -> float:
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise CapabilityMetricsError("geomean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def _mean(values: list[float]) -> float:
    if not values:
        raise CapabilityMetricsError("mean requires at least one value")
    return sum(values) / len(values)


def _representative(rows: list[dict[str, Any]], wanted_id: str) -> dict[str, Any]:
    matching = [row for row in rows if row["policy_id"] == wanted_id]
    if not matching:
        raise CapabilityMetricsError("representative policy is absent")
    first = matching[0]
    metric_keys = (
        "selected_ids_by_slot", "oracle_normalized_geomean_regret",
        "deterministic_control_normalized_geomean_regret",
        "geomean_speedup_over_anchor", "exact_oracle_policy",
        "decision_slot_accuracy",
    )
    if any(any(row[key] != first[key] for key in metric_keys)
           for row in matching[1:]):
        raise CapabilityMetricsError("one policy has inconsistent screen costs")
    return {
        "policy_id": wanted_id,
        "selected_ids_by_slot": first["selected_ids_by_slot"],
        "oracle_normalized_geomean_regret": first[
            "oracle_normalized_geomean_regret"
        ],
        "deterministic_control_normalized_geomean_regret": first[
            "deterministic_control_normalized_geomean_regret"
        ],
        "geomean_speedup_over_anchor": first["geomean_speedup_over_anchor"],
        "exact_oracle_policy": first["exact_oracle_policy"],
        "decision_slot_accuracy": first["decision_slot_accuracy"],
    }


def summarize_view(rows: list[dict[str, Any]], expected_trials: int = 20) -> dict[str, Any]:
    """Summarize one view without oracle-informed modal-policy tie breaking."""
    if (not isinstance(expected_trials, int) or isinstance(expected_trials, bool)
            or expected_trials <= 0):
        raise CapabilityMetricsError("expected_trials must be positive")
    if len(rows) != expected_trials or not rows:
        raise CapabilityMetricsError("view does not have the expected trial count")
    views = {row.get("view") for row in rows if isinstance(row, dict)}
    if len(views) != 1 or next(iter(views)) not in VIEWS:
        raise CapabilityMetricsError("view summary mixes or omits views")
    if sorted(row.get("trial") for row in rows) != list(
        range(1, expected_trials + 1)
    ):
        raise CapabilityMetricsError("trials are not exactly 1..N")
    accepted = [row for row in rows if row.get("bridge_accepted") is True]
    if any(not isinstance(row.get("bridge_accepted"), bool) for row in rows):
        raise CapabilityMetricsError("scored row lacks bridge outcome")
    anchor_ids = {row.get("anchor_policy_id") for row in rows}
    if len(anchor_ids) != 1:
        raise CapabilityMetricsError("trials disagree on anchor policy")

    counts = Counter(row["policy_id"] for row in accepted)
    for screened_policy_id in {row["policy_id"] for row in rows}:
        _representative(rows, screened_policy_id)
    if counts:
        modal_count = max(counts.values())
        # Never use held-out regret to resolve a modal tie.
        modal_id = min(policy for policy, count in counts.items()
                       if count == modal_count)
        modal_source = "modal_accepted_policy"
        best = min(
            accepted,
            key=lambda row: (
                row["oracle_normalized_geomean_regret"], row["policy_id"],
            ),
        )
        best_id = best["policy_id"]
        entropy = -sum(
            (count / len(accepted)) * math.log(count / len(accepted))
            for count in counts.values()
        )
    else:
        modal_count = 0
        modal_id = next(iter(anchor_ids))
        modal_source = "anchor_fallback_no_accepted_response"
        best_id = modal_id
        entropy = 0.0

    exact_accepted = [row for row in accepted if row["exact_oracle_policy"]]
    result = {
        "view": next(iter(views)),
        "trial_count": len(rows),
        "accepted_count": len(accepted),
        "invalid_output_rate": 1.0 - len(accepted) / len(rows),
        "intention_to_treat": {
            "oracle_normalized_geomean_regret": _geomean([
                row["oracle_normalized_geomean_regret"] for row in rows
            ]),
            "deterministic_control_normalized_geomean_regret": _geomean([
                row["deterministic_control_normalized_geomean_regret"]
                for row in rows
            ]),
            "geomean_speedup_over_anchor": _geomean([
                row["geomean_speedup_over_anchor"] for row in rows
            ]),
            "exact_oracle_policy_rate": _mean([
                float(row["exact_oracle_policy"]) for row in rows
            ]),
            "mean_decision_slot_accuracy": _mean([
                row["decision_slot_accuracy"] for row in rows
            ]),
        },
        "accepted_only_diagnostic": {
            "exact_oracle_policy_rate": (
                len(exact_accepted) / len(accepted) if accepted else None
            ),
            "mean_decision_slot_accuracy": (
                _mean([row["decision_slot_accuracy"] for row in accepted])
                if accepted else None
            ),
        },
        "stability": {
            "unique_accepted_policy_count": len(counts),
            "accepted_policy_entropy_nats": entropy,
            "modal_count": modal_count,
            "modal_rate_among_accepted": (
                modal_count / len(accepted) if accepted else None
            ),
            "modal_rate_intention_to_treat": modal_count / len(rows),
        },
        "primary_modal_representative": {
            "selection_rule": modal_source,
            **_representative(rows, modal_id),
        },
        "posthoc_capability_upper_bound": {
            "selection_rule": "minimum_screen_regret_among_accepted_best_of_N",
            "best_of": expected_trials,
            "posthoc": True,
            **_representative(rows, best_id),
        },
        "offline_screen_is_runtime_speedup_evidence": False,
    }
    return result


def summarize_views(rows_by_view: dict[str, list[dict[str, Any]]],
                    expected_trials: int = 20) -> dict[str, Any]:
    if set(rows_by_view) != set(VIEWS):
        raise CapabilityMetricsError("exactly three information views are required")
    views = {
        view: summarize_view(rows_by_view[view], expected_trials)
        for view in VIEWS
    }
    relational = views["relational"]
    effects = {}
    for comparator in ("descriptors", "opaque"):
        effects[f"{comparator}_to_relational_itt_regret_ratio"] = (
            views[comparator]["intention_to_treat"][
                "oracle_normalized_geomean_regret"
            ] / relational["intention_to_treat"][
                "oracle_normalized_geomean_regret"
            ]
        )
        effects[f"{comparator}_to_relational_modal_regret_ratio"] = (
            views[comparator]["primary_modal_representative"][
                "oracle_normalized_geomean_regret"
            ] / relational["primary_modal_representative"][
                "oracle_normalized_geomean_regret"
            ]
        )
    return {
        "views": views,
        "semantic_context_effect": {
            **effects,
            "ratio_above_one_favors_relational": True,
            "same_action_set_required_upstream": True,
            "performance_superiority_claimed": False,
        },
    }
