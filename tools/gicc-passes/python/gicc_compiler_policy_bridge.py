"""Uniform strict bridge for all compiler decision-suite graph families.

The family bridges remain authoritative for validation and hint construction.
This module dispatches to them, revalidates their selected graph-bound IDs, and
normalizes a policy for capability metrics.  It never exposes the compiler hint
or its private materializer identities to the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import gicc_collective_plan_bridge as collective
import gicc_comm_group_plan_bridge as communication
import gicc_comm_plan_bridge as structural
import gicc_llm_bridge as fingerprinting
import gicc_llm_capability_metrics as metrics


class CompilerPolicyBridgeError(ValueError):
    """A graph family, family hint, or normalized policy is inconsistent."""


@dataclass(frozen=True)
class _Adapter:
    family: str
    verify: Callable[[Any], dict[str, Any]]
    response_schema: Callable[[Any], dict[str, Any]]
    accept: Callable[[Any, Any], tuple[dict[str, Any], bool, list[str]]]


_ADAPTERS = {
    structural.GRAPH_SCHEMA: _Adapter(
        "communication_coalescing_and_trigger_placement",
        structural.verified_graph,
        structural.decision_response_schema,
        structural.plan_to_hint,
    ),
    communication.GRAPH_SCHEMA: _Adapter(
        "communication_route_or_schedule",
        communication.verified_graph,
        communication.decision_response_schema,
        communication.plan_to_hint,
    ),
    collective.GRAPH_SCHEMA: _Adapter(
        "collective_algorithm_and_size_policy",
        collective.verified_graph,
        collective.decision_response_schema,
        collective.decision_to_hint,
    ),
}


def _adapter(graph_value: Any) -> _Adapter:
    if not isinstance(graph_value, dict):
        raise CompilerPolicyBridgeError("compiler graph must be an object")
    adapter = _ADAPTERS.get(graph_value.get("schema_version"))
    if adapter is None:
        raise CompilerPolicyBridgeError("unsupported compiler graph schema")
    return adapter


def _fallback_selection(family: str,
                        graph: dict[str, Any]) -> dict[str, str]:
    selected: dict[str, str] = {}
    for opportunity in graph["opportunities"]:
        opportunity_id = opportunity["opportunity_id"]
        if family == "communication_coalescing_and_trigger_placement":
            matches = [
                candidate for candidate in opportunity["candidates"]
                if candidate["kind"] == "trigger_descriptor_batch"
            ]
            if len(matches) != 1:
                raise CompilerPolicyBridgeError(
                    "structural graph lacks one compiler fallback"
                )
            selected[opportunity_id] = matches[0]["candidate_id"]
        elif family == "communication_route_or_schedule":
            preference = (
                "group_uniform_default", "site_default",
                "group_uniform_trigger", "site_trigger",
                "group_uniform_proxy", "site_proxy",
            )
            candidate = next(
                (item for kind in preference
                 for item in opportunity["candidates"]
                 if item["kind"] == kind),
                opportunity["candidates"][0],
            )
            selected[opportunity_id] = candidate["candidate_id"]
        else:
            for slot in opportunity["decision_slots"]:
                matches = [
                    option for option in slot["options"]
                    if option["role"] == "anchor"
                ]
                if len(matches) != 1:
                    raise CompilerPolicyBridgeError(
                        "collective slot lacks one semantic anchor"
                    )
                selected[f"{opportunity_id}/{slot['slot_id']}"] = matches[0][
                    "option_id"
                ]
    return selected


def _hint_selection(family: str, graph: dict[str, Any], hint: Any,
                    accepted: bool, errors: list[str]) -> dict[str, str]:
    if not isinstance(hint, dict):
        raise CompilerPolicyBridgeError("family bridge returned a non-object hint")
    metadata = hint.get("llm_metadata")
    metadata_errors = (
        metadata.get("errors", [] if accepted else None)
        if isinstance(metadata, dict) else None
    )
    if (not isinstance(metadata, dict)
            or metadata.get("accepted") is not accepted
            or metadata.get("graph_id") != graph["graph_id"]
            or metadata_errors != errors):
        raise CompilerPolicyBridgeError("family hint metadata is inconsistent")
    if accepted and metadata.get("compiler_only_output") is not True:
        raise CompilerPolicyBridgeError("accepted hint lacks compiler-only boundary")

    fallback = _fallback_selection(family, graph)
    if family in {
        "communication_coalescing_and_trigger_placement",
        "communication_route_or_schedule",
    }:
        nested = metadata.get("selected_candidates")
        if nested is None and not accepted and family == (
            "communication_coalescing_and_trigger_placement"
        ):
            selected = fallback
        elif isinstance(nested, dict):
            selected = dict(nested)
        else:
            raise CompilerPolicyBridgeError("family hint lacks selected candidates")
    else:
        nested = metadata.get("selected_option_ids")
        if not isinstance(nested, dict):
            raise CompilerPolicyBridgeError("collective hint lacks selected options")
        selected = {}
        for opportunity_id, slots in nested.items():
            if not isinstance(slots, dict):
                raise CompilerPolicyBridgeError(
                    "collective hint has invalid selected slots"
                )
            for slot_id, option_id in slots.items():
                selected[f"{opportunity_id}/{slot_id}"] = option_id

    if not accepted and selected != fallback:
        raise CompilerPolicyBridgeError("rejected response did not fall back atomically")
    return selected


def _validate_selection(family: str, graph: dict[str, Any],
                        selected: dict[str, str]) -> None:
    allowed: dict[str, set[str]] = {}
    for opportunity in graph["opportunities"]:
        opportunity_id = opportunity["opportunity_id"]
        if family == "collective_algorithm_and_size_policy":
            for slot in opportunity["decision_slots"]:
                allowed[f"{opportunity_id}/{slot['slot_id']}"] = {
                    option["option_id"] for option in slot["options"]
                }
        else:
            allowed[opportunity_id] = {
                candidate["candidate_id"]
                for candidate in opportunity["candidates"]
            }
    if set(selected) != set(allowed):
        raise CompilerPolicyBridgeError("normalized policy slot set is incomplete")
    if any(not isinstance(selected[slot], str)
           or selected[slot] not in allowed[slot] for slot in allowed):
        raise CompilerPolicyBridgeError(
            "normalized policy contains a non-compiler-generated ID"
        )


def decision_to_policy(graph_value: Any, decision: Any) -> dict[str, Any]:
    """Strictly bridge one response and return its uniform compiler policy."""
    adapter = _adapter(graph_value)
    graph = adapter.verify(graph_value)
    schema = adapter.response_schema(graph)
    hint, accepted, errors = adapter.accept(graph, decision)
    if not isinstance(accepted, bool) or not isinstance(errors, list):
        raise CompilerPolicyBridgeError("family bridge returned an invalid outcome")
    selected = _hint_selection(
        adapter.family, graph, hint, accepted, errors,
    )
    _validate_selection(adapter.family, graph, selected)
    return {
        "decision_family": adapter.family,
        "graph_schema": graph["schema_version"],
        "graph_id": graph["graph_id"],
        "response_schema_id": fingerprinting._fingerprint(schema),
        "bridge_accepted": accepted,
        "bridge_errors": errors,
        "fallback_applied": not accepted,
        "selected_ids_by_slot": selected,
        "policy_id": metrics.policy_id(selected),
        "compiler_hint_id": fingerprinting._fingerprint(hint),
        "compiler_hint": hint,
        "compiler_hint_is_private_and_never_provider_input": True,
    }
