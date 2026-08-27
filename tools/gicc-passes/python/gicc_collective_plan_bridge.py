#!/usr/bin/env python3
"""Build and enforce compiler-only collective algorithm plans.

The LLVM pass emits a source-free inventory of one semantic collective anchor,
ABI-identical compiler catalog targets, call-graph facts, and compiler-owned
message-size thresholds.  This bridge turns that inventory into relational
decision slots.  A model may return only existing opaque option IDs; it cannot
name functions, write code/IR, add an algorithm, alter a threshold, or assert
legality.  ``accept`` expands valid IDs into a narrow hint, and the LTO pass
independently repeats catalog, contract, ABI, threshold, and content-ID checks
before rewriting a call.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

import gicc_llm_bridge as bridge


INVENTORY_SCHEMA = "gicc-collective-inventory-v1"
PROFILE_SCHEMA = "gicc-collective-platform-profile-v1"
GRAPH_SCHEMA = "gicc-collective-opportunity-graph-v1"
DECISION_SCHEMA = "gicc-collective-plan-decision-v1"
HINT_SCHEMA = "gicc-collective-hint-v1"
ID_RE = re.compile(r"^(anchor|catalog|opportunity|option|candidate):[0-9a-f]{24}$")
MODEL_VIEW_KINDS = ("relational", "descriptors", "opaque")


class CollectivePlanError(ValueError):
    """The compiler inventory or external decision violates the boundary."""


def _digest_id(kind: str, parts: Iterable[str]) -> str:
    payload = kind.encode()
    for part in parts:
        payload += b"\0" + part.encode()
    return f"{kind}:{hashlib.sha256(payload).hexdigest()[:24]}"


def _fields_parts(fields: dict[str, str]) -> list[str]:
    return [f"{key}={fields[key]}" for key in sorted(fields)]


def _target_id(role: str, descriptor: dict[str, str]) -> str:
    kind = "anchor" if role == "anchor" else "catalog"
    return _digest_id(kind, _fields_parts(descriptor))


def _option_id(opportunity_id: str, slot_id: str, target_id: str) -> str:
    return _digest_id("option", [opportunity_id, slot_id, target_id])


def _uniform_plan_id(opportunity_id: str, target_id: str) -> str:
    return _digest_id("candidate", [opportunity_id, "uniform", target_id])


def _policy_plan_id(
    opportunity_id: str,
    element_bytes: int,
    rules: list[dict[str, Any]],
) -> str:
    parts = [opportunity_id, "size_policy", str(element_bytes)]
    for rule in rules:
        bound = "*" if rule["max_bytes"] is None else str(rule["max_bytes"])
        parts.append(f"{bound}:{rule['target_id']}")
    return _digest_id("candidate", parts)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectivePlanError(f"cannot read JSON {path}: {exc}") from exc


def _string_fields(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, dict) or not value:
        raise CollectivePlanError(f"{label} descriptor must be a nonempty object")
    if any(not isinstance(key, str) or not isinstance(item, str) or not item
           for key, item in value.items()):
        raise CollectivePlanError(f"{label} descriptor fields must be strings")
    if not isinstance(value.get("family"), str) or not isinstance(
            value.get("contract"), str):
        raise CollectivePlanError(f"{label} descriptor lacks family/contract")
    return dict(value)


def _verified_inventory(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != INVENTORY_SCHEMA:
        raise CollectivePlanError(f"expected inventory schema {INVENTORY_SCHEMA}")
    if value.get("compiler_only") is not True or value.get("source_visible") is not False:
        raise CollectivePlanError("inventory boundary is not compiler-only/source-free")
    opportunities = value.get("opportunities")
    if not isinstance(opportunities, list) or not opportunities:
        raise CollectivePlanError("inventory has no collective opportunities")
    seen_opportunities: set[str] = set()
    normalized = []
    for opportunity in opportunities:
        if not isinstance(opportunity, dict):
            raise CollectivePlanError("opportunity must be an object")
        opportunity_id = opportunity.get("opportunity_id")
        if (not isinstance(opportunity_id, str)
                or not ID_RE.fullmatch(opportunity_id)
                or not opportunity_id.startswith("opportunity:")
                or opportunity_id in seen_opportunities):
            raise CollectivePlanError("invalid or duplicate opportunity_id")
        seen_opportunities.add(opportunity_id)
        family = opportunity.get("family")
        contract = opportunity.get("contract")
        anchor = _string_fields(
            opportunity.get("anchor_descriptor"), label="anchor"
        )
        if anchor.get("family") != family or anchor.get("contract") != contract:
            raise CollectivePlanError(f"{opportunity_id}: anchor contract mismatch")
        anchor_id = opportunity.get("anchor_id")
        if anchor_id != _target_id("anchor", anchor):
            raise CollectivePlanError(f"{opportunity_id}: invalid anchor_id")
        call_facts = opportunity.get("call_facts")
        if not isinstance(call_facts, dict):
            raise CollectivePlanError(f"{opportunity_id}: missing call facts")
        element_bytes = call_facts.get("element_bytes")
        if (isinstance(element_bytes, bool) or not isinstance(element_bytes, int)
                or element_bytes <= 0):
            raise CollectivePlanError(f"{opportunity_id}: invalid element width")
        thresholds = opportunity.get("compiler_policy_thresholds_bytes")
        if (not isinstance(thresholds, list)
                or any(isinstance(item, bool) or not isinstance(item, int)
                       or item <= 0 for item in thresholds)
                or thresholds != sorted(set(thresholds))):
            raise CollectivePlanError(f"{opportunity_id}: invalid thresholds")

        targets = [{
            "target_id": anchor_id,
            "role": "anchor",
            "descriptor": anchor,
            "compiler_legality": {
                "semantic_anchor": True,
                "exact_function_type": True,
                "void_call_materializer": True,
            },
        }]
        seen_targets = {anchor_id}
        candidates = opportunity.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            raise CollectivePlanError(f"{opportunity_id}: no catalog candidates")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise CollectivePlanError("catalog candidate must be an object")
            descriptor = _string_fields(
                candidate.get("descriptor"), label="catalog"
            )
            target_id = candidate.get("catalog_id")
            legality = candidate.get("compiler_legality")
            if (target_id != _target_id("catalog", descriptor)
                    or target_id in seen_targets):
                raise CollectivePlanError(f"{opportunity_id}: invalid catalog_id")
            if descriptor.get("family") != family or descriptor.get(
                    "contract") != contract:
                raise CollectivePlanError(f"{opportunity_id}: candidate contract mismatch")
            expected_legality = {
                "same_family": True,
                "same_semantic_contract": True,
                "exact_function_type": True,
                "void_call_materializer": True,
            }
            if legality != expected_legality:
                raise CollectivePlanError(f"{opportunity_id}: incomplete compiler legality")
            seen_targets.add(target_id)
            targets.append({
                "target_id": target_id,
                "role": "candidate",
                "descriptor": descriptor,
                "compiler_legality": legality,
            })
        normalized.append({
            "opportunity_id": opportunity_id,
            "family": family,
            "contract": contract,
            "call_facts": call_facts,
            "element_bytes": element_bytes,
            "thresholds": thresholds,
            "targets": targets,
        })
    return {
        "inventory_id": bridge._fingerprint(value),
        "target_triple": value.get("target_triple"),
        "opportunities": normalized,
    }


def _verified_profile(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != PROFILE_SCHEMA:
        raise CollectivePlanError(f"expected platform schema {PROFILE_SCHEMA}")
    profile_id = value.get("platform_id")
    topology = value.get("topology")
    if not isinstance(profile_id, str) or not profile_id:
        raise CollectivePlanError("platform_id is missing")
    if not isinstance(topology, dict):
        raise CollectivePlanError("platform topology is missing")
    for key in ("nodes", "ranks_per_node"):
        item = topology.get(key)
        if isinstance(item, bool) or not isinstance(item, int) or item <= 0:
            raise CollectivePlanError(f"topology.{key} must be positive")
    disabled = value.get("disabled_algorithms", [])
    if not isinstance(disabled, list) or any(not isinstance(item, str) for item in disabled):
        raise CollectivePlanError("disabled_algorithms must be strings")
    return value


def _slot_bounds(thresholds: list[int]) -> list[tuple[int | None, int | None]]:
    bounds: list[tuple[int | None, int | None]] = []
    lower = None
    for threshold in thresholds:
        bounds.append((lower, threshold))
        lower = threshold + 1
    bounds.append((lower, None))
    return bounds


def _algorithm(target: dict[str, Any]) -> str:
    descriptor = target["descriptor"]
    return descriptor.get("algorithm", "compiler_anchor")


def make_graph(inventory_value: Any, profile_value: Any) -> dict[str, Any]:
    inventory = _verified_inventory(inventory_value)
    profile = _verified_profile(profile_value)
    disabled = set(profile.get("disabled_algorithms", []))
    opportunities = []
    for source in inventory["opportunities"]:
        targets = [
            target for target in source["targets"]
            if target["role"] == "anchor" or _algorithm(target) not in disabled
        ]
        if len(targets) < 2:
            raise CollectivePlanError(
                f"{source['opportunity_id']}: platform leaves no optimization target"
            )
        decision_slots = []
        for index, (lower, upper) in enumerate(_slot_bounds(source["thresholds"])):
            slot_id = f"message-bin-{index}"
            options = []
            for target in targets:
                option_payload = {
                    "option_id": _option_id(
                        source["opportunity_id"], slot_id, target["target_id"]
                    ),
                    "target_id": target["target_id"],
                    "role": target["role"],
                    "algorithm": _algorithm(target),
                    "compiler_descriptor": target["descriptor"],
                    "compiler_legality": target["compiler_legality"],
                }
                options.append(option_payload)
            decision_slots.append({
                "slot_id": slot_id,
                "message_bytes": {"min": lower, "max": upper},
                "options": options,
            })
        opportunities.append({
            "opportunity_id": source["opportunity_id"],
            "kind": "compiler_collective_algorithm_and_size_policy",
            "semantic_contract": {
                "family": source["family"],
                "contract": source["contract"],
            },
            "compiler_facts": {
                "call": source["call_facts"],
                "target_triple": inventory["target_triple"],
                "topology": profile["topology"],
                "hardware": profile.get("hardware", {}),
                "transport": profile.get("transport", {}),
                "resource_constraints": profile.get("resource_constraints", {}),
                "message_distribution": profile.get("message_distribution", {}),
            },
            "decision_slots": decision_slots,
            "joint_action_space_size": len(targets) ** len(decision_slots),
        })
    payload = {
        "schema_version": GRAPH_SCHEMA,
        "compiler_inputs": {
            "inventory_id": inventory["inventory_id"],
            "platform_id": profile["platform_id"],
        },
        "boundary": {
            "source_visible": False,
            "model_may_generate_code": False,
            "model_may_generate_ir": False,
            "model_may_name_functions": False,
            "model_may_change_thresholds": False,
            "model_may_assert_legality": False,
            "model_output": "compiler-generated option IDs only",
            "compiler_revalidates_before_materialization": True,
        },
        "objective": {
            "metric": "end_to_end_collective_wall_time",
            "instruction": (
                "Choose exactly one compiler-generated option ID for every "
                "message-size slot. Optimize the joint size policy without "
                "inventing algorithms, thresholds, code, or legality."
            ),
        },
        "platform_profile": profile,
        "opportunities": opportunities,
    }
    graph = dict(payload)
    graph["graph_id"] = bridge._fingerprint(payload)
    return graph


def verified_graph(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != GRAPH_SCHEMA:
        raise CollectivePlanError(f"expected graph schema {GRAPH_SCHEMA}")
    graph_id = value.get("graph_id")
    if not isinstance(graph_id, str):
        raise CollectivePlanError("graph_id is missing")
    payload = dict(value)
    del payload["graph_id"]
    if graph_id != bridge._fingerprint(payload):
        raise CollectivePlanError("graph_id does not match graph content")
    expected_boundary = {
        "source_visible": False,
        "model_may_generate_code": False,
        "model_may_generate_ir": False,
        "model_may_name_functions": False,
        "model_may_change_thresholds": False,
        "model_may_assert_legality": False,
        "model_output": "compiler-generated option IDs only",
        "compiler_revalidates_before_materialization": True,
    }
    if value.get("boundary") != expected_boundary:
        raise CollectivePlanError("graph boundary is not compiler-only")
    opportunities = value.get("opportunities")
    if not isinstance(opportunities, list) or not opportunities:
        raise CollectivePlanError("graph has no opportunities")
    seen_opportunities: set[str] = set()
    seen_options: set[str] = set()
    for opportunity in opportunities:
        if not isinstance(opportunity, dict):
            raise CollectivePlanError("opportunity must be an object")
        opportunity_id = opportunity.get("opportunity_id")
        if (not isinstance(opportunity_id, str)
                or opportunity_id in seen_opportunities):
            raise CollectivePlanError("invalid or duplicate opportunity")
        seen_opportunities.add(opportunity_id)
        slots = opportunity.get("decision_slots")
        if not isinstance(slots, list) or not slots:
            raise CollectivePlanError(f"{opportunity_id}: missing decision slots")
        for slot in slots:
            if not isinstance(slot, dict) or not isinstance(slot.get("slot_id"), str):
                raise CollectivePlanError("invalid decision slot")
            options = slot.get("options")
            if not isinstance(options, list) or len(options) < 2:
                raise CollectivePlanError("decision slot has fewer than two options")
            for option in options:
                if not isinstance(option, dict):
                    raise CollectivePlanError("option must be an object")
                option_id = option.get("option_id")
                expected = _option_id(
                    opportunity_id, slot["slot_id"], option.get("target_id", "")
                )
                if option_id != expected or option_id in seen_options:
                    raise CollectivePlanError("invalid, stale, or duplicate option_id")
                seen_options.add(option_id)
        if opportunity.get("joint_action_space_size") != len(
                slots[0]["options"]) ** len(slots):
            raise CollectivePlanError("joint action space size is inconsistent")
    return value


def _model_candidate_class_id(
    opportunity_id: str, visible_candidate: dict[str, Any],
) -> str:
    payload = json.dumps(
        visible_candidate, sort_keys=True, separators=(",", ":")
    )
    digest = hashlib.sha256(
        (opportunity_id + "\0" + payload).encode()
    ).hexdigest()[:24]
    return f"candidate-class:{digest}"


def _model_view(
    graph: dict[str, Any], view_kind: str = "relational",
) -> dict[str, Any]:
    if view_kind not in MODEL_VIEW_KINDS:
        raise CollectivePlanError(f"unknown model view {view_kind}")
    view = {
        "schema_version": "gicc-collective-model-view-v2",
        "view_kind": view_kind,
        "compiler_graph_schema": graph["schema_version"],
        "graph_id": graph["graph_id"],
        "boundary": graph["boundary"],
        "objective": graph["objective"],
        "opportunities": [],
    }
    for opportunity in graph["opportunities"]:
        candidate_classes: dict[str, dict[str, Any]] = {}
        target_to_class: dict[str, str] = {}
        for slot in opportunity["decision_slots"]:
            for option in slot["options"]:
                visible = {
                    "role": option["role"],
                    "compiler_descriptor": option["compiler_descriptor"],
                    "compiler_legality": option["compiler_legality"],
                }
                class_id = _model_candidate_class_id(
                    opportunity["opportunity_id"], visible
                )
                previous = target_to_class.setdefault(option["target_id"], class_id)
                if previous != class_id:
                    raise CollectivePlanError(
                        "one compiler target has inconsistent visible descriptors"
                    )
                candidate = {"candidate_class_id": class_id, **visible}
                existing = candidate_classes.setdefault(class_id, candidate)
                if existing != candidate:
                    raise CollectivePlanError("model candidate-class ID collision")

        shown = {
            "opportunity_id": opportunity["opportunity_id"],
            "kind": opportunity["kind"],
            "semantic_contract": opportunity["semantic_contract"],
            "compiler_facts": opportunity["compiler_facts"],
            "joint_action_space_size": opportunity["joint_action_space_size"],
            "candidate_classes": (
                sorted(
                    candidate_classes.values(),
                    key=lambda item: item["candidate_class_id"],
                )
                if view_kind != "opaque" else []
            ),
            "decision_slots": [],
            "relations": [],
        }
        for slot in opportunity["decision_slots"]:
            shown["decision_slots"].append({
                "slot_id": slot["slot_id"],
                "message_bytes": slot["message_bytes"],
                "allowed_options": [
                    ({
                        "option_id": option["option_id"],
                        "candidate_class_id": target_to_class[option["target_id"]],
                    } if view_kind != "opaque" else {
                        "option_id": option["option_id"],
                    })
                    for option in slot["options"]
                ],
            })
        if view_kind == "relational":
            shown["relations"].append({
                "kind": "increasing_message_size",
                "ordered_slot_ids": [
                    slot["slot_id"] for slot in opportunity["decision_slots"]
                ],
            })
        relation_fields = (
            "communication_graph", "topology", "cross_node_pattern",
            "resource_model", "synchronization",
        )
        for field in relation_fields:
            groups: dict[str, list[str]] = {}
            for candidate in candidate_classes.values():
                value = candidate["compiler_descriptor"].get(field)
                if isinstance(value, str):
                    groups.setdefault(value, []).append(
                        candidate["candidate_class_id"]
                    )
            for value, members in sorted(groups.items()):
                if view_kind == "relational" and len(members) > 1:
                    shown["relations"].append({
                        "kind": "shared_compiler_property",
                        "property": field,
                        "value": value,
                        "candidate_class_ids": sorted(members),
                    })
        view["opportunities"].append(shown)
    return view


def render_prompt(
    graph_value: Any, view_kind: str = "relational",
) -> str:
    graph = verified_graph(graph_value)
    skeleton = {
        "schema_version": DECISION_SCHEMA,
        "graph_id": graph["graph_id"],
        "selections": {
            opportunity["opportunity_id"]: {
                "slot_candidate_ids": {
                    slot["slot_id"]: "<one existing option_id from this slot>"
                    for slot in opportunity["decision_slots"]
                },
                "confidence": 0.0,
                "rationale": "optional short compiler-fact rationale",
            }
            for opportunity in graph["opportunities"]
        },
    }
    return (
        "You are selecting a compiler-stage collective communication plan. "
        "You receive no application source. Return JSON only, using exactly "
        "one existing option_id for every decision slot. Do not emit source, "
        "code, LLVM IR, function names, algorithms, thresholds, or new "
        "legality claims. LLVM independently validates and materializes the "
        "selection.\n\nCompiler opportunity graph:\n"
        + json.dumps(_model_view(graph, view_kind), indent=2, sort_keys=True)
        + "\n\nRequired response shape:\n"
        + json.dumps(skeleton, indent=2, sort_keys=True)
        + "\n"
    )


def _baseline_options(opportunity: dict[str, Any]) -> dict[str, dict[str, Any]]:
    selected = {}
    for slot in opportunity["decision_slots"]:
        anchor = next(option for option in slot["options"]
                      if option["role"] == "anchor")
        selected[slot["slot_id"]] = anchor
    return selected


def _hint_from_options(
    graph: dict[str, Any], chosen: dict[str, dict[str, dict[str, Any]]],
    *, accepted: bool, errors: list[str], rationales: dict[str, str] | None = None,
) -> dict[str, Any]:
    selections = {}
    for opportunity in graph["opportunities"]:
        opportunity_id = opportunity["opportunity_id"]
        by_slot = chosen[opportunity_id]
        thresholds = [
            slot["message_bytes"]["max"]
            for slot in opportunity["decision_slots"][:-1]
        ]
        target_ids = [
            by_slot[slot["slot_id"]]["target_id"]
            for slot in opportunity["decision_slots"]
        ]
        if len(set(target_ids)) == 1:
            target_id = target_ids[0]
            materializer = {
                "kind": "uniform",
                "candidate_id": _uniform_plan_id(opportunity_id, target_id),
                "target_id": target_id,
            }
        else:
            rules = [
                {"max_bytes": maximum, "target_id": target_id}
                for maximum, target_id in zip([*thresholds, None], target_ids)
            ]
            element_bytes = opportunity["compiler_facts"]["call"]["element_bytes"]
            materializer = {
                "kind": "size_policy",
                "candidate_id": _policy_plan_id(
                    opportunity_id, element_bytes, rules
                ),
                "rules": rules,
            }
        selections[opportunity_id] = materializer
    return {
        "version": 1,
        "schema_version": HINT_SCHEMA,
        "selections": selections,
        "llm_metadata": {
            "accepted": accepted,
            "compiler_only_output": True,
            "graph_id": graph["graph_id"],
            "selected_option_ids": {
                opportunity_id: {
                    slot_id: option["option_id"]
                    for slot_id, option in slots.items()
                }
                for opportunity_id, slots in chosen.items()
            },
            "rationales": rationales or {},
            "errors": errors,
        },
    }


def decision_to_hint(
    graph_value: Any, decision: Any,
) -> tuple[dict[str, Any], bool, list[str]]:
    graph = verified_graph(graph_value)
    errors: list[str] = []
    if not isinstance(decision, dict):
        errors.append("decision must be a JSON object")
        decision = {}
    unknown_top = set(decision) - {"schema_version", "graph_id", "selections"}
    if unknown_top:
        errors.append(f"unknown top-level field(s): {sorted(unknown_top)}")
    if decision.get("schema_version") != DECISION_SCHEMA:
        errors.append(f"expected decision schema {DECISION_SCHEMA}")
    if decision.get("graph_id") != graph["graph_id"]:
        errors.append("decision graph_id is stale or does not match")
    values = decision.get("selections")
    if not isinstance(values, dict):
        errors.append("selections must be a JSON object")
        values = {}
    expected = {
        opportunity["opportunity_id"]: opportunity
        for opportunity in graph["opportunities"]
    }
    if set(values) != set(expected):
        missing = sorted(set(expected) - set(values))
        unknown = sorted(set(values) - set(expected))
        if missing:
            errors.append(f"missing opportunity selection(s): {missing}")
        if unknown:
            errors.append(f"unknown opportunity selection(s): {unknown}")

    chosen: dict[str, dict[str, dict[str, Any]]] = {}
    rationales: dict[str, str] = {}
    for opportunity_id in sorted(set(values) & set(expected)):
        entry = values[opportunity_id]
        if not isinstance(entry, dict):
            errors.append(f"{opportunity_id}: selection must be an object")
            continue
        unknown = set(entry) - {"slot_candidate_ids", "confidence", "rationale"}
        if unknown:
            errors.append(f"{opportunity_id}: unknown field(s) {sorted(unknown)}")
        confidence = entry.get("confidence")
        if (isinstance(confidence, bool) or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0 <= float(confidence) <= 1):
            errors.append(f"{opportunity_id}: confidence must be finite in [0, 1]")
        rationale = entry.get("rationale", "")
        if not isinstance(rationale, str) or len(rationale) > 512:
            errors.append(f"{opportunity_id}: rationale must be at most 512 characters")
            rationale = ""
        slots_value = entry.get("slot_candidate_ids")
        slots = {slot["slot_id"]: slot
                 for slot in expected[opportunity_id]["decision_slots"]}
        if not isinstance(slots_value, dict) or set(slots_value) != set(slots):
            errors.append(f"{opportunity_id}: slot IDs must match compiler graph")
            continue
        selected_slots = {}
        for slot_id, option_id in slots_value.items():
            if not isinstance(option_id, str):
                errors.append(f"{opportunity_id}/{slot_id}: option_id must be a string")
                continue
            option = next(
                (item for item in slots[slot_id]["options"]
                 if item["option_id"] == option_id), None
            )
            if option is None:
                errors.append(
                    f"{opportunity_id}/{slot_id}: option_id is not compiler-generated"
                )
                continue
            selected_slots[slot_id] = option
        if len(selected_slots) == len(slots):
            chosen[opportunity_id] = selected_slots
            rationales[opportunity_id] = (
                f"LLM confidence={float(confidence):.3f}: {rationale}"
                if isinstance(confidence, (int, float))
                and not isinstance(confidence, bool) else rationale
            )

    if errors:
        fallback = {
            opportunity_id: _baseline_options(opportunity)
            for opportunity_id, opportunity in expected.items()
        }
        return _hint_from_options(
            graph, fallback, accepted=False, errors=errors
        ), False, errors
    return _hint_from_options(
        graph, chosen, accepted=True, errors=[], rationales=rationales
    ), True, []


def _emit(args: argparse.Namespace) -> int:
    graph = make_graph(_read_json(args.inventory), _read_json(args.platform))
    bridge._write_json_atomic(args.graph, graph)
    bridge._write_text_atomic(args.prompt, render_prompt(graph, args.prompt_view))
    capacity = sum(item["joint_action_space_size"]
                   for item in graph["opportunities"])
    print(
        f"gicc-collective-plan-bridge: wrote {len(graph['opportunities'])} "
        f"opportunity(s), aggregate per-opportunity capacity={capacity}; "
        f"graph_id={graph['graph_id']}", file=sys.stderr,
    )
    return 0


def _accept(args: argparse.Namespace) -> int:
    graph = _read_json(args.graph)
    try:
        response = json.loads(args.response.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        if args.strict:
            print(f"gicc-collective-plan-bridge: REJECTED: {exc}", file=sys.stderr)
            return 2
        response = None
    hint, accepted, errors = decision_to_hint(graph, response)
    if not accepted and args.strict:
        print(
            "gicc-collective-plan-bridge: REJECTED: " + "; ".join(errors),
            file=sys.stderr,
        )
        return 2
    bridge._write_json_atomic(args.hint, hint)
    print(
        f"gicc-collective-plan-bridge: {'accepted' if accepted else 'fallback'}; "
        f"wrote {args.hint}", file=sys.stderr,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    emit = sub.add_parser("emit")
    emit.add_argument("--inventory", type=Path, required=True)
    emit.add_argument("--platform", type=Path, required=True)
    emit.add_argument("--graph", type=Path, required=True)
    emit.add_argument("--prompt", type=Path, required=True)
    emit.add_argument(
        "--prompt-view", choices=MODEL_VIEW_KINDS, default="relational"
    )
    emit.set_defaults(run=_emit)
    accept = sub.add_parser("accept")
    accept.add_argument("--graph", type=Path, required=True)
    accept.add_argument("--response", type=Path, required=True)
    accept.add_argument("--hint", type=Path, required=True)
    accept.add_argument("--strict", action="store_true")
    accept.set_defaults(run=_accept)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.run(args)
    except (CollectivePlanError, OSError, ValueError) as exc:
        print(f"gicc-collective-plan-bridge: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
