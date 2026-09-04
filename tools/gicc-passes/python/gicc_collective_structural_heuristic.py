#!/usr/bin/env python3
"""Emit a deterministic source-free collective compiler control.

Version 1 is preregistered for the held-out n8 hierarchy/pipeline question.
It consumes only a verified compiler graph.  It never reads runtime results,
source, or provider output, and it returns the same opaque option-ID decision
accepted and independently revalidated by the collective LTO bridge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import gicc_collective_plan_bridge as plans
import gicc_llm_bridge as bridge


CONTROL_SCHEMA = "gicc-collective-structural-heuristic-control-v1"
HEURISTIC_VERSION = "topology-hierarchy-pipeline-v1"
PIPELINE_CHUNKS_BY_SLOT = (1, 1, 4, 8)


class HeuristicError(ValueError):
    """The graph cannot support the preregistered heuristic safely."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeuristicError(f"cannot read JSON {path}: {exc}") from exc


def _anchor(slot: dict[str, Any]) -> dict[str, Any]:
    anchors = [option for option in slot["options"] if option["role"] == "anchor"]
    if len(anchors) != 1:
        raise HeuristicError("each collective slot must have one semantic anchor")
    return anchors[0]


def _pipeline_chunks(option: dict[str, Any]) -> int | None:
    value = option.get("compiler_descriptor", {}).get("pipeline_chunks")
    try:
        chunks = int(value)
    except (TypeError, ValueError):
        return None
    return chunks if chunks > 0 else None


def _hierarchy_tree(option: dict[str, Any]) -> bool:
    descriptor = option.get("compiler_descriptor", {})
    return (
        option.get("role") == "candidate"
        and descriptor.get("topology") == "two_level_node_hierarchy"
        and descriptor.get("communication_graph") == "node_double_tree"
        and descriptor.get("step_complexity") == "O_log_nodes_plus_ppn"
        and descriptor.get("synchronization") == "device_cooperative"
        and descriptor.get("dynamic_guarded") == "true"
    )


def make_decision(graph_value: Any) -> dict[str, Any]:
    graph = plans.verified_graph(graph_value)
    topology = graph.get("platform_profile", {}).get("topology", {})
    nodes = topology.get("nodes")
    ranks_per_node = topology.get("ranks_per_node")
    hierarchy_applicable = (
        isinstance(nodes, int) and not isinstance(nodes, bool) and nodes >= 4
        and isinstance(ranks_per_node, int)
        and not isinstance(ranks_per_node, bool) and ranks_per_node > 1
    )
    selections = {}
    for opportunity in graph["opportunities"]:
        slots = opportunity["decision_slots"]
        use_structural_rule = (
            hierarchy_applicable
            and len(slots) == len(PIPELINE_CHUNKS_BY_SLOT)
        )
        selected = {}
        structural_matches = 0
        for index, slot in enumerate(slots):
            option = _anchor(slot)
            desired = PIPELINE_CHUNKS_BY_SLOT[index]
            if use_structural_rule:
                matches = [
                    candidate for candidate in slot["options"]
                    if _hierarchy_tree(candidate)
                    and _pipeline_chunks(candidate) == desired
                ]
                if len(matches) == 1:
                    option = matches[0]
                    structural_matches += 1
            selected[slot["slot_id"]] = option["option_id"]
        complete_structural_policy = structural_matches == len(slots)
        if not complete_structural_policy:
            selected = {
                slot["slot_id"]: _anchor(slot)["option_id"] for slot in slots
            }
        selections[opportunity["opportunity_id"]] = {
            "slot_candidate_ids": selected,
            "confidence": 1.0,
            "rationale": (
                "deterministic source-free topology/pipeline heuristic v1"
                if complete_structural_policy else
                "deterministic semantic-anchor fallback"
            ),
        }
    return {
        "schema_version": plans.DECISION_SCHEMA,
        "graph_id": graph["graph_id"],
        "selections": selections,
    }


def make_control(
    graph_value: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    graph = plans.verified_graph(graph_value)
    decision = make_decision(graph)
    hint, accepted, errors = plans.decision_to_hint(graph, decision)
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
            "runtime_results_visible": False,
            "provider_output_visible": False,
            "output_is_graph_bound_option_ids": True,
            "compiler_revalidates_before_materialization": True,
        },
        "rule": {
            "minimum_nodes_for_hierarchy": 4,
            "minimum_ranks_per_node": 2,
            "required_slot_count": 4,
            "pipeline_chunks_by_ordered_slot": list(PIPELINE_CHUNKS_BY_SLOT),
            "required_compiler_descriptors": {
                "topology": "two_level_node_hierarchy",
                "communication_graph": "node_double_tree",
                "step_complexity": "O_log_nodes_plus_ppn",
                "synchronization": "device_cooperative",
                "dynamic_guarded": "true",
            },
            "fallback": "semantic_anchor_for_every_slot",
        },
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
            "gicc-collective-structural-heuristic: wrote compiler control; "
            f"control_id={control['control_id']}",
            file=sys.stderr,
        )
        return 0
    except (HeuristicError, plans.CollectivePlanError, OSError, ValueError) as exc:
        print(
            f"gicc-collective-structural-heuristic: ERROR: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    sys.exit(main())
