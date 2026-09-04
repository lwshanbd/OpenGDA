#!/usr/bin/env python3
"""Build and enforce compiler-only communication plans.

This is the relational successor to ``gicc_comm_plan_bridge.py``. It joins a
content-addressed LTO dossier with the compiler's per-kernel operation
template, groups communication sites by their compiler-discovered completion
point, represents eligible singleton transfers, and exposes complete
materializable route combinations plus any compiler-proved trigger-placement
transform.

The model never sees source and cannot author code, IR, site IDs, dispatch
names, or legality.  Its entire output is one existing opaque candidate ID per
compiler opportunity.  ``accept`` revalidates every hash and translates that
selection into the narrow hint consumed and re-proved by the LTO passes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import gicc_llm_bridge as bridge


GRAPH_SCHEMA = "gicc-communication-group-opportunity-graph-v1"
MODEL_VIEW_SCHEMA = "gicc-communication-opportunity-model-view-v1"
DECISION_SCHEMA = "gicc-communication-group-plan-decision-v1"
HINT_SCHEMA = "gicc-hint-v1"

ACTION_TO_MATERIALIZER = {
    "default": {"dispatch": "IPC_OR_DWQ", "transform": "NONE"},
    "proxy": {"dispatch": "CPU_PROXY_ENQUEUE", "transform": "NONE"},
    "trigger": {"dispatch": "DWQ_TRIGGER", "transform": "NONE"},
    "ipc": {"dispatch": "IPC_PUSH", "transform": "NONE"},
}
MATERIALIZABLE_ACTION_ORDER = ("default", "proxy", "trigger", "ipc")


class GroupPlanError(ValueError):
    """Compiler facts or a group plan violate the compiler-only boundary."""


def _fingerprint(value: Any) -> str:
    return bridge._fingerprint(value)


def _candidate_id(payload: dict[str, Any]) -> str:
    return "candidate:" + _fingerprint(payload).removeprefix("sha256:")[:24]


def _opportunity_id(site_ids: list[str], completion_site_id: str) -> str:
    payload = "\0".join([*site_ids, completion_site_id]).encode()
    return "group-opportunity:" + hashlib.sha256(payload).hexdigest()[:24]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GroupPlanError(f"cannot read JSON {path}: {exc}") from exc


def _safe_op(op: Any) -> dict[str, Any]:
    """Validate and retain compiler semantics, never source/debug text."""
    if not isinstance(op, dict):
        raise GroupPlanError("kernel operation must be an object")
    site_id = op.get("site_id")
    kind = op.get("kind")
    if not isinstance(site_id, str) or not site_id:
        raise GroupPlanError("kernel operation is missing site_id")
    if kind not in {"put_no_db", "get_no_db", "flush", "quiet"}:
        raise GroupPlanError(f"{site_id}: unknown operation kind {kind!r}")
    args = op.get("args", {})
    if not isinstance(args, dict):
        raise GroupPlanError(f"{site_id}: args must be an object")
    # ArgRef trees contain only IR operation tags, formal indices, constants,
    # and structural offsets.  Parameter/debug names are deliberately absent.
    return {
        "site_id": site_id,
        "kind": kind,
        "args": args,
        "guard": op.get("guard"),
        "loop": op.get("loop"),
        "hk_capable": op.get("hk_capable"),
        "compute_after": op.get("compute_after"),
        "distance_exact": op.get("distance_exact"),
        "batch_size": op.get("batch_size"),
        "completion_site_id": op.get("completion_site_id"),
        "group_early_trigger_legal": op.get("group_early_trigger_legal"),
        "group_early_trigger_reason": op.get("group_early_trigger_reason"),
    }


def verified_templates(values: Iterable[Any]) -> dict[str, Any]:
    templates: list[dict[str, Any]] = []
    sites: dict[str, dict[str, Any]] = {}
    for value in values:
        if not isinstance(value, dict) or value.get("version") != 1:
            raise GroupPlanError("expected compiler kernel-template version 1")
        mangled = value.get("kernel_mangled")
        simple = value.get("kernel_simple")
        ops = value.get("ops")
        if not isinstance(mangled, str) or not isinstance(simple, str):
            raise GroupPlanError("kernel template is missing compiler names")
        if not isinstance(ops, list):
            raise GroupPlanError(f"kernel {simple}: ops must be an array")
        safe_ops = [_safe_op(op) for op in ops]
        safe = {
            "kernel_mangled": mangled,
            "kernel_simple": simple,
            "ops": safe_ops,
        }
        safe["template_id"] = _fingerprint(safe)
        templates.append(safe)
        for order, op in enumerate(safe_ops):
            site_id = op["site_id"]
            if site_id in sites:
                raise GroupPlanError(f"duplicate template site_id {site_id!r}")
            sites[site_id] = {
                **op,
                "kernel": simple,
                "kernel_mangled": mangled,
                "template_id": safe["template_id"],
                "operation_order": order,
            }
    if not templates:
        raise GroupPlanError("no compiler kernel templates supplied")
    templates.sort(key=lambda item: item["kernel_mangled"])
    return {"templates": templates, "sites": sites}


def _expression_relation(expressions: list[Any]) -> str:
    if all(expr == expressions[0] for expr in expressions[1:]):
        return "same_expression"
    if all(isinstance(expr, dict) and expr.get("kind") == "param"
           for expr in expressions):
        formals = [expr.get("param") for expr in expressions]
        if len(set(formals)) == len(formals):
            return "distinct_formals"
    if all(isinstance(expr, dict) and expr.get("kind") == "const_i64"
           for expr in expressions):
        constants = [expr.get("value") for expr in expressions]
        if len(set(constants)) == len(constants):
            return "distinct_constants"
    return "different_or_dynamic_expressions"


def _materializer_for_actions(
    site_ids: list[str], actions: tuple[str, ...], *, early: bool = False,
) -> dict[str, Any]:
    sites: dict[str, Any] = {}
    for site_id, action in zip(site_ids, actions):
        materializer = dict(ACTION_TO_MATERIALIZER[action])
        if early:
            materializer["dispatch"] = "DWQ_TRIGGER"
            materializer["transform"] = "TRIGGER_GROUP_EARLY"
        sites[site_id] = materializer
    return {"sites": sites}


def _candidate(
    kind: str, site_ids: list[str], actions: tuple[str, ...], summary: str,
    proof: list[str], effects: dict[str, Any], *, early: bool = False,
) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "site_ids": site_ids,
        "materializer": _materializer_for_actions(
            site_ids, actions, early=early,
        ),
        "summary": summary,
        "compiler_proof": proof,
        "effects": effects,
    }
    return {"candidate_id": _candidate_id(payload), **payload}


def _fixed_materializer(site: dict[str, Any]) -> dict[str, str]:
    legal = site.get("legal_actions", [])
    for action in ("default", "trigger", "proxy", "ipc"):
        if action in legal:
            return dict(ACTION_TO_MATERIALIZER[action])
    raise GroupPlanError(f"site {site.get('site_id')!r} has no materializable action")


_SEMANTIC_FACT_FIELDS = (
    "op_kind",
    "hk_capable",
    "size_kind",
    "size_bytes",
    "size_log2",
    "transfer_interval",
    "peer_kind",
    "peer_locality",
    "in_loop",
    "loop",
    "guard_kind",
    "guard_density",
    "fan_out",
    "static_launch_sites",
    "launch_contexts",
    "launch_grid",
    "launch_block",
    "grid_blocks",
    "threads_per_block",
    "phase_launch_supported",
    "phase_launch_stream",
    "phase_launch_materialization",
    "phase_launch_reason",
    "kernel_argument_slots_exact",
    "kernel_argument_slot_count",
    "kernel_argument_slot_reason",
    "compute_before_flops",
    "flops_to_first_use",
    "trip_count",
    "distance_exact",
    "iter_estimate",
    "descriptor_reusable",
    "buffer_reusable",
    "coalescable",
    "max_vector_bytes",
    "batch_size",
    "producer_frontier",
)


def _semantic_facts(site: dict[str, Any]) -> dict[str, Any]:
    """Retain rich compiler facts while excluding compiler/source identity."""
    return {key: site.get(key) for key in _SEMANTIC_FACT_FIELDS}


def make_group_graph(dossier_value: Any, template_values: Iterable[Any]) \
        -> dict[str, Any]:
    dossier = bridge._verified_dossier(dossier_value)
    inventory = verified_templates(template_values)
    meta_by_site = inventory["sites"]
    dossier_by_site = {site["site_id"]: site for site in dossier["sites"]}
    missing = sorted(set(dossier_by_site) - set(meta_by_site))
    if missing:
        raise GroupPlanError(f"dossier sites missing from kernel templates: {missing}")

    grouped: dict[tuple[str, str], list[str]] = {}
    for site_id, site in dossier_by_site.items():
        meta = meta_by_site[site_id]
        completion = meta.get("completion_site_id")
        if isinstance(completion, str) and completion:
            grouped.setdefault((meta["kernel_mangled"], completion), []).append(site_id)

    transform_profile = dossier["platform_profile"].get(
        "compiler_transforms", {}
    )
    early_enabled = (
        isinstance(transform_profile, dict)
        and transform_profile.get("group_early_trigger") is True
    )
    opportunities: list[dict[str, Any]] = []
    opportunity_sites: set[str] = set()
    for (_, completion), unordered_ids in sorted(grouped.items()):
        site_ids = sorted(
            unordered_ids, key=lambda site_id: meta_by_site[site_id]["operation_order"]
        )
        if len(site_ids) < 2:
            continue
        metas = [meta_by_site[site_id] for site_id in site_ids]
        if any(meta["kind"] != "put_no_db" for meta in metas):
            continue
        legal_by_site = {
            site_id: tuple(
                action for action in MATERIALIZABLE_ACTION_ORDER
                if action in dossier_by_site[site_id]["legal_actions"]
            )
            for site_id in site_ids
        }
        if any(not actions for actions in legal_by_site.values()):
            continue

        relation_names = (
            "target_rank", "dst_buf", "dst_off", "src_buf", "src_off", "size"
        )
        relations = {}
        for name in relation_names:
            expressions = [meta["args"].get(name) for meta in metas]
            relations[name] = _expression_relation(expressions)

        candidates: list[dict[str, Any]] = []
        action_products = itertools.product(
            *(legal_by_site[site_id] for site_id in site_ids)
        )
        for actions in action_products:
            uniform = len(set(actions)) == 1
            kind = (f"group_uniform_{actions[0]}" if uniform
                    else "group_mixed_route")
            action_map = dict(zip(site_ids, actions))
            candidates.append(_candidate(
                kind,
                site_ids,
                actions,
                "Materialize the compiler-legal per-site routes for one shared completion group.",
                [
                    "every route is advertised legal for its bound compiler site",
                    "the compiler identified one shared completion group",
                ],
                {
                    "site_actions": action_map,
                    "trigger_placement": "original_completion",
                    "host_descriptor_sites": sum(
                        action in {"default", "trigger", "ipc"}
                        for action in actions
                    ),
                    "device_proxy_sites": actions.count("proxy"),
                },
            ))

        early_reasons = {
            meta.get("group_early_trigger_reason") for meta in metas
        }
        kernel_ops = [
            meta for meta in meta_by_site.values()
            if meta["kernel_mangled"] == metas[0]["kernel_mangled"]
        ]
        materializer_shape_legal = (
            sum(meta["kind"] in {"put_no_db", "get_no_db"}
                for meta in kernel_ops) == len(site_ids)
            and sum(meta["kind"] == "flush" for meta in kernel_ops) == 1
        )
        compiler_early_legal = all(
            meta.get("group_early_trigger_legal") is True for meta in metas
        ) and materializer_shape_legal
        early_legal = (
            early_enabled
            and compiler_early_legal
            and all("trigger" in legal_by_site[site_id] for site_id in site_ids)
        )
        if early_legal:
            actions = tuple("trigger" for _ in site_ids)
            candidates.append(_candidate(
                "group_trigger_early",
                site_ids,
                actions,
                "Stage every descriptor and move the compiler-owned shared trigger to the proven post-issue frontier.",
                [
                    "all members are unconditional non-loop host-knowable PUTs",
                    "one flush is mandatory after every group member",
                    "no intervening instruction may write memory",
                    "the final device pass must independently repeat the proof",
                ],
                {
                    "site_actions": dict(zip(site_ids, actions)),
                    "trigger_placement": "post_issue_frontier",
                    "overlap_flops_after_last_site": metas[-1].get("compute_after"),
                    "host_descriptor_sites": len(site_ids),
                    "device_proxy_sites": 0,
                },
                early=True,
            ))

        completion_meta = meta_by_site.get(completion)
        masked = []
        if early_enabled and not early_legal:
            reasons = sorted(
                reason for reason in early_reasons if isinstance(reason, str)
            )
            if not materializer_shape_legal:
                reasons.append(
                    "final materializer requires one all-data-site group and one flush"
                )
            masked.append({
                "kind": "group_trigger_early",
                "reason": reasons or [
                    "compiler did not prove group early-trigger legality"
                ],
            })
        compiler_facts = {
            "kernel": metas[0]["kernel"],
            "operation_order": site_ids,
            "group_size": len(site_ids),
            "batch_size": [dossier_by_site[s].get("batch_size") for s in site_ids],
            "fan_out": [dossier_by_site[s].get("fan_out") for s in site_ids],
            "completion": {
                "site_id": completion,
                "kind": completion_meta.get("kind") if completion_meta else None,
            },
            "argument_relations": relations,
            "site_argument_expressions": {
                site_id: meta_by_site[site_id]["args"] for site_id in site_ids
            },
            "site_legal_actions": legal_by_site,
            "site_semantic_facts": {
                site_id: _semantic_facts(dossier_by_site[site_id])
                for site_id in site_ids
            },
            "launch_contexts": {
                site_id: dossier_by_site[site_id].get("launch_contexts")
                for site_id in site_ids
            },
            "compute_region": {
                "site_flops_to_completion": {
                    site_id: dossier_by_site[site_id].get("flops_to_first_use")
                    for site_id in site_ids
                },
                "distance_exact": all(
                    dossier_by_site[site_id].get("distance_exact") is True
                    for site_id in site_ids
                ),
                "flops_after_last_site": metas[-1].get("compute_after"),
            },
            "dependence_legality": {
                "local_no_intervening_write_proof": all(
                    meta.get("group_early_trigger_legal") is True
                    for meta in metas
                ),
                "final_materializer_shape_legal": materializer_shape_legal,
                "group_early_trigger_legal": compiler_early_legal,
                "compiler_reasons": sorted(
                    reason for reason in early_reasons if isinstance(reason, str)
                ),
            },
        }
        opportunity_id = _opportunity_id(site_ids, completion)
        opportunities.append({
            "opportunity_id": opportunity_id,
            "kind": "shared_completion_communication_group",
            "site_ids": site_ids,
            "compiler_facts": compiler_facts,
            "masked_candidates": masked,
            "candidates": candidates,
        })
        opportunity_sites.update(site_ids)

    # A single transfer still has a real compiler-level decision whenever at
    # least two independently materializable routes exist. Representing it as
    # a one-member opportunity lets the same opaque-ID protocol cover both
    # relational groups and singleton sites without exposing site IDs to the
    # model.
    for site_id, site in sorted(dossier_by_site.items()):
        if site_id in opportunity_sites:
            continue
        actions = tuple(
            action for action in MATERIALIZABLE_ACTION_ORDER
            if action in site.get("legal_actions", [])
        )
        if len(actions) < 2:
            continue
        meta = meta_by_site[site_id]
        completion = meta.get("completion_site_id")
        completion_id = completion if isinstance(completion, str) else ""
        candidates = [
            _candidate(
                f"site_{action}",
                [site_id],
                (action,),
                "Materialize one compiler-advertised route for this transfer.",
                ["the route is advertised legal for the bound compiler site"],
                {
                    "site_actions": {site_id: action},
                    "trigger_placement": "original_completion",
                    "host_descriptor_sites": int(
                        action in {"default", "trigger", "ipc"}
                    ),
                    "device_proxy_sites": int(action == "proxy"),
                },
            )
            for action in actions
        ]
        compiler_facts = {
            "kernel": meta["kernel"],
            "operation_order": [site_id],
            "group_size": 1,
            "batch_size": [site.get("batch_size")],
            "fan_out": [site.get("fan_out")],
            "completion": {
                "site_id": completion_id or None,
                "kind": meta_by_site.get(completion_id, {}).get("kind"),
            },
            "argument_relations": {
                name: "single_expression" for name in (
                    "target_rank", "dst_buf", "dst_off", "src_buf",
                    "src_off", "size",
                )
            },
            "site_argument_expressions": {site_id: meta["args"]},
            "site_legal_actions": {site_id: actions},
            "site_semantic_facts": {site_id: _semantic_facts(site)},
            "launch_contexts": {site_id: site.get("launch_contexts")},
            "compute_region": {
                "site_flops_to_completion": {
                    site_id: site.get("flops_to_first_use")
                },
                "distance_exact": site.get("distance_exact") is True,
                "flops_after_last_site": meta.get("compute_after"),
            },
            "dependence_legality": {
                "local_no_intervening_write_proof": False,
                "final_materializer_shape_legal": True,
                "group_early_trigger_legal": False,
                "compiler_reasons": [
                    "singleton route choice does not relocate communication"
                ],
            },
        }
        opportunity_id = _opportunity_id([site_id], completion_id)
        opportunities.append({
            "opportunity_id": opportunity_id,
            "kind": "single_site_communication_route",
            "site_ids": [site_id],
            "compiler_facts": compiler_facts,
            "masked_candidates": [],
            "candidates": candidates,
        })
        opportunity_sites.add(site_id)

    if not opportunities:
        raise GroupPlanError(
            "no compiler communication opportunity with multiple routes"
        )
    fixed_sites = []
    for site_id, site in sorted(dossier_by_site.items()):
        if site_id in opportunity_sites:
            continue
        fixed_sites.append({
            "site_id": site_id,
            "materializer": _fixed_materializer(site),
            "reason": "not part of a compiler-proved multi-site group",
        })
    payload = {
        "schema_version": GRAPH_SCHEMA,
        "compiler_inputs": {
            "dossier_id": dossier["dossier_id"],
            "kernel_template_ids": [
                template["template_id"] for template in inventory["templates"]
            ],
        },
        "boundary": {
            "source_visible": False,
            "model_may_generate_code": False,
            "model_may_generate_ir": False,
            "model_may_assert_legality": False,
            "model_output": "candidate IDs only",
            "compiler_revalidates_before_materialization": True,
        },
        "objective": {
            "metric": "end_to_end_wall_time",
            "instruction": (
                "Choose one compiler-generated communication plan per opportunity. "
                "Do not invent transformations or override masked candidates."
            ),
        },
        "platform_profile": dossier["platform_profile"],
        "fixed_sites": fixed_sites,
        "opportunities": sorted(
            opportunities, key=lambda item: item["opportunity_id"]
        ),
    }
    graph = dict(payload)
    graph["graph_id"] = _fingerprint(payload)
    return graph


def verified_graph(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != GRAPH_SCHEMA:
        raise GroupPlanError(f"expected graph schema {GRAPH_SCHEMA}")
    graph_id = value.get("graph_id")
    if not isinstance(graph_id, str):
        raise GroupPlanError("graph_id is missing")
    payload = dict(value)
    del payload["graph_id"]
    if graph_id != _fingerprint(payload):
        raise GroupPlanError("graph_id does not match graph content")
    expected_boundary = {
        "source_visible": False,
        "model_may_generate_code": False,
        "model_may_generate_ir": False,
        "model_may_assert_legality": False,
        "model_output": "candidate IDs only",
        "compiler_revalidates_before_materialization": True,
    }
    if value.get("boundary") != expected_boundary:
        raise GroupPlanError("graph boundary is not compiler-only")
    opportunities = value.get("opportunities")
    if not isinstance(opportunities, list) or not opportunities:
        raise GroupPlanError("graph has no opportunities")
    seen_opportunities: set[str] = set()
    seen_sites: set[str] = set()
    seen_candidates: set[str] = set()
    for opportunity in opportunities:
        if not isinstance(opportunity, dict):
            raise GroupPlanError("opportunity must be an object")
        opportunity_id = opportunity.get("opportunity_id")
        site_ids = opportunity.get("site_ids")
        if (not isinstance(opportunity_id, str)
                or opportunity_id in seen_opportunities):
            raise GroupPlanError("invalid or duplicate opportunity_id")
        if (not isinstance(site_ids, list) or not site_ids
                or any(not isinstance(site_id, str) for site_id in site_ids)
                or seen_sites.intersection(site_ids)):
            raise GroupPlanError("invalid or overlapping opportunity sites")
        seen_opportunities.add(opportunity_id)
        seen_sites.update(site_ids)
        legal_by_site = opportunity.get("compiler_facts", {}).get(
            "site_legal_actions", {}
        )
        candidates = opportunity.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise GroupPlanError(f"{opportunity_id}: fewer than two candidates")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise GroupPlanError("candidate must be an object")
            candidate_id = candidate.get("candidate_id")
            candidate_payload = dict(candidate)
            candidate_payload.pop("candidate_id", None)
            if (not isinstance(candidate_id, str)
                    or candidate_id != _candidate_id(candidate_payload)
                    or candidate_id in seen_candidates
                    or candidate.get("site_ids") != site_ids):
                raise GroupPlanError("invalid, stale, or duplicate candidate_id")
            seen_candidates.add(candidate_id)
            materializer = candidate.get("materializer", {}).get("sites")
            if not isinstance(materializer, dict) or set(materializer) != set(site_ids):
                raise GroupPlanError("candidate materializer does not bind its group")
            for site_id, request in materializer.items():
                if not isinstance(request, dict):
                    raise GroupPlanError("site materializer must be an object")
                dispatch = request.get("dispatch")
                transform = request.get("transform")
                if transform == "TRIGGER_GROUP_EARLY":
                    legal = (
                        dispatch == "DWQ_TRIGGER"
                        and candidate.get("kind") == "group_trigger_early"
                        and opportunity["compiler_facts"]["dependence_legality"].get(
                            "group_early_trigger_legal"
                        ) is True
                    )
                else:
                    action = next(
                        (name for name, mat in ACTION_TO_MATERIALIZER.items()
                         if mat == request), None,
                    )
                    legal = action in legal_by_site.get(site_id, [])
                if not legal:
                    raise GroupPlanError("candidate contains an illegal materializer")
    return value


def model_view(graph_value: Any) -> dict[str, Any]:
    """Build the rich, identity-free graph that is eligible for a model."""
    graph = verified_graph(graph_value)
    forbidden_keys = {
        "site_id", "site_ids", "kernel", "kernel_mangled",
        "operation_order", "materializer", "path", "source_path",
        "completion_site_id",
    }

    def without_identity_keys(value: Any) -> Any:
        if isinstance(value, dict):
            return {
                key: without_identity_keys(child)
                for key, child in value.items()
                if key not in forbidden_keys
            }
        if isinstance(value, list):
            return [without_identity_keys(child) for child in value]
        return value

    private_fragments: set[str] = set()
    opportunity_views = []
    for opportunity in graph["opportunities"]:
        site_ids = opportunity["site_ids"]
        facts = opportunity["compiler_facts"]
        private_fragments.update(site_ids)
        for site_id in site_ids:
            source_identity = site_id.split(":", 1)[0]
            if len(source_identity) >= 3 and source_identity != "?":
                private_fragments.add(source_identity)
        kernel = facts.get("kernel")
        if isinstance(kernel, str) and len(kernel) >= 3:
            private_fragments.add(kernel)
        completion = facts.get("completion", {})
        completion_id = completion.get("site_id")
        if isinstance(completion_id, str):
            private_fragments.add(completion_id)

        arguments = facts.get("site_argument_expressions", {})
        legal_actions = facts.get("site_legal_actions", {})
        semantic_facts = facts.get("site_semantic_facts", {})
        launch_contexts = facts.get("launch_contexts", {})
        compute = facts.get("compute_region", {})
        flops = compute.get("site_flops_to_completion", {})
        candidate_views = []
        for candidate in opportunity["candidates"]:
            effects = dict(candidate.get("effects", {}))
            site_actions = effects.pop("site_actions", {})
            effects["route_actions"] = [
                site_actions.get(site_id) for site_id in site_ids
            ]
            candidate_views.append({
                "candidate_id": candidate["candidate_id"],
                "kind": candidate["kind"],
                "summary": candidate["summary"],
                "compiler_proof": candidate["compiler_proof"],
                "effects": effects,
            })
        opportunity_views.append({
            "opportunity_id": opportunity["opportunity_id"],
            "kind": opportunity["kind"],
            "compiler_facts": {
                "transfer_count": len(site_ids),
                "batch_size": facts.get("batch_size"),
                "fan_out": facts.get("fan_out"),
                "completion_kind": completion.get("kind"),
                "argument_relations": facts.get("argument_relations"),
                "transfer_argument_expressions": [
                    arguments.get(site_id) for site_id in site_ids
                ],
                "transfer_legal_actions": [
                    legal_actions.get(site_id) for site_id in site_ids
                ],
                "transfer_semantic_facts": [
                    without_identity_keys(semantic_facts.get(site_id))
                    for site_id in site_ids
                ],
                "launch_contexts": [
                    launch_contexts.get(site_id) for site_id in site_ids
                ],
                "compute_region": {
                    "transfer_flops_to_completion": [
                        flops.get(site_id) for site_id in site_ids
                    ],
                    "distance_exact": compute.get("distance_exact"),
                    "flops_after_last_transfer":
                        compute.get("flops_after_last_site"),
                },
                "dependence_legality": facts.get("dependence_legality"),
            },
            "masked_candidates": opportunity.get("masked_candidates", []),
            "candidates": candidate_views,
        })

    platform = {
        key: value for key, value in graph["platform_profile"].items()
        if key != "provenance"
    }
    view = {
        "schema_version": MODEL_VIEW_SCHEMA,
        "compiler_graph_id": graph["graph_id"],
        "compiler_input_ids": graph["compiler_inputs"],
        "boundary": graph["boundary"],
        "objective": graph["objective"],
        "platform_profile": platform,
        "fixed_site_count": len(graph.get("fixed_sites", [])),
        "opportunities": opportunity_views,
    }
    def validate(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if key in forbidden_keys:
                    raise GroupPlanError(
                        f"identity-bearing key {key!r} escaped into model view"
                    )
                validate(child)
        elif isinstance(value, list):
            for child in value:
                validate(child)
        elif isinstance(value, str):
            if (value.startswith("/") or value.startswith("./")
                    or value.startswith("../")):
                raise GroupPlanError("filesystem path escaped into model view")
            for fragment in private_fragments:
                if len(fragment) >= 3 and fragment in value:
                    raise GroupPlanError(
                        "compiler/source identity escaped into model view"
                    )

    validate(view)
    return view


def render_prompt(graph_value: Any) -> str:
    graph = verified_graph(graph_value)
    public_graph = model_view(graph)
    example = {
        "schema_version": DECISION_SCHEMA,
        "graph_id": graph["graph_id"],
        "selections": {
            "<opportunity_id>": {
                "candidate_id": "<one existing candidate_id>",
                "confidence": 0.0,
                "rationale": "<brief compiler/platform-fact rationale>",
            }
        },
    }
    return (
        "You are a constrained relational communication planner inside LTO. "
        "You receive compiler IR facts, program-dependence legality, launch "
        "contexts, and compiler-generated plans. You cannot see or modify "
        "source and must not emit code, IR, site IDs, dispatch names, new "
        "transformations, or legality claims. Select exactly one existing "
        "candidate_id for every opportunity_id.\n\n"
        "Return ONLY one JSON object with this shape:\n"
        + json.dumps(example, indent=2, sort_keys=True)
        + "\n\nCOMPILER-GENERATED COMMUNICATION GRAPH:\n"
        + json.dumps(public_graph, indent=2, sort_keys=True)
        + "\n"
    )


def _baseline_candidate(opportunity: dict[str, Any]) -> dict[str, Any]:
    preference = (
        "group_uniform_default", "site_default",
        "group_uniform_trigger", "site_trigger",
        "group_uniform_proxy", "site_proxy",
    )
    for kind in preference:
        for candidate in opportunity["candidates"]:
            if candidate["kind"] == kind:
                return candidate
    return opportunity["candidates"][0]


def _hint_from_candidates(
    graph: dict[str, Any], selected: dict[str, dict[str, Any]],
    *, accepted: bool, errors: list[str], rationales: dict[str, str] | None = None,
) -> dict[str, Any]:
    sites: dict[str, Any] = {}
    for fixed in graph.get("fixed_sites", []):
        request = fixed["materializer"]
        if request != ACTION_TO_MATERIALIZER["default"]:
            sites[fixed["site_id"]] = {
                **request,
                "reason": "compiler-fixed non-opportunity action",
            }
            if sites[fixed["site_id"]]["transform"] == "NONE":
                del sites[fixed["site_id"]]["transform"]
    chosen_ids = {}
    for opportunity_id, candidate in selected.items():
        chosen_ids[opportunity_id] = candidate["candidate_id"]
        for site_id, request in candidate["materializer"]["sites"].items():
            if request == ACTION_TO_MATERIALIZER["default"]:
                continue
            site_hint = {
                "dispatch": request["dispatch"],
                "reason": (
                    (rationales or {}).get(opportunity_id, "")
                    if accepted else "deterministic compiler plan fallback"
                ),
            }
            if request["transform"] != "NONE":
                site_hint["transform"] = request["transform"]
            sites[site_id] = site_hint
    return {
        "version": 1,
        "schema_version": HINT_SCHEMA,
        "default_dispatch": "IPC_OR_DWQ",
        "sites": sites,
        "llm_metadata": {
            "accepted": accepted,
            "compiler_only_output": True,
            "graph_id": graph["graph_id"],
            "selected_candidates": chosen_ids,
            "errors": errors,
        },
    }


def plan_to_hint(
    graph_value: Any, decision: Any,
) -> tuple[dict[str, Any], bool, list[str]]:
    graph = verified_graph(graph_value)
    errors: list[str] = []
    selections = decision.get("selections") if isinstance(decision, dict) else None
    if not isinstance(decision, dict):
        errors.append("decision must be a JSON object")
    else:
        unknown_top = set(decision) - {"schema_version", "graph_id", "selections"}
        if unknown_top:
            errors.append(f"unknown top-level field(s): {sorted(unknown_top)}")
        if decision.get("schema_version") != DECISION_SCHEMA:
            errors.append(f"expected decision schema {DECISION_SCHEMA}")
        if decision.get("graph_id") != graph["graph_id"]:
            errors.append("decision graph_id is stale or does not match")
        if not isinstance(selections, dict):
            errors.append("selections must be a JSON object")
    if not isinstance(selections, dict):
        selections = {}
    expected = {
        opportunity["opportunity_id"]: opportunity
        for opportunity in graph["opportunities"]
    }
    if set(selections) != set(expected):
        missing = sorted(set(expected) - set(selections))
        unknown = sorted(set(selections) - set(expected))
        if missing:
            errors.append(f"missing opportunity selection(s): {missing}")
        if unknown:
            errors.append(f"unknown opportunity selection(s): {unknown}")

    selected: dict[str, dict[str, Any]] = {}
    rationales: dict[str, str] = {}
    for opportunity_id in sorted(set(expected) & set(selections)):
        entry = selections[opportunity_id]
        if not isinstance(entry, dict):
            errors.append(f"{opportunity_id}: selection must be an object")
            continue
        unknown_fields = set(entry) - {"candidate_id", "confidence", "rationale"}
        if unknown_fields:
            errors.append(
                f"{opportunity_id}: unknown field(s) {sorted(unknown_fields)}"
            )
        candidate = next(
            (item for item in expected[opportunity_id]["candidates"]
             if item["candidate_id"] == entry.get("candidate_id")), None,
        )
        if candidate is None:
            errors.append(f"{opportunity_id}: candidate_id is not compiler-generated")
            continue
        confidence = entry.get("confidence")
        if (isinstance(confidence, bool)
                or not isinstance(confidence, (int, float))
                or not math.isfinite(float(confidence))
                or not 0 <= float(confidence) <= 1):
            errors.append(f"{opportunity_id}: confidence must be finite in [0, 1]")
            continue
        rationale = entry.get("rationale", "")
        if not isinstance(rationale, str) or len(rationale) > 512:
            errors.append(f"{opportunity_id}: rationale must be at most 512 characters")
            continue
        selected[opportunity_id] = candidate
        rationales[opportunity_id] = (
            f"compiler communication plan {candidate['candidate_id']}; "
            f"LLM confidence={float(confidence):.3f}: {rationale}"
        )

    if errors:
        fallback = {
            opportunity_id: _baseline_candidate(opportunity)
            for opportunity_id, opportunity in expected.items()
        }
        return _hint_from_candidates(
            graph, fallback, accepted=False, errors=errors,
        ), False, errors
    return _hint_from_candidates(
        graph, selected, accepted=True, errors=[], rationales=rationales,
    ), True, []


def _load_templates(meta_dir: Path) -> list[Any]:
    # Feature extraction deliberately writes features.json beside the
    # per-kernel version-1 templates. It is a schema-6 array, not a template,
    # and must not be fed to verified_templates(). Keep rejecting every other
    # malformed JSON file so an unexpected artifact cannot be silently
    # treated as compiler metadata.
    paths = sorted(
        path for path in meta_dir.glob("*.json")
        if path.name != "features.json"
    )
    if not paths:
        raise GroupPlanError(f"no kernel templates under {meta_dir}")
    return [_read_json(path) for path in paths]


def _emit(args: argparse.Namespace) -> int:
    graph = make_group_graph(
        _read_json(args.dossier), _load_templates(args.meta_dir)
    )
    bridge._write_json_atomic(args.graph, graph)
    bridge._write_text_atomic(args.prompt, render_prompt(graph))
    print(
        f"gicc-comm-group-plan-bridge: wrote {len(graph['opportunities'])} "
        f"communication opportunity(s); graph_id={graph['graph_id']}",
        file=sys.stderr,
    )
    return 0


def _accept(args: argparse.Namespace) -> int:
    graph = _read_json(args.graph)
    try:
        response = json.loads(args.response.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        if args.strict:
            print(f"gicc-comm-group-plan-bridge: REJECTED: {exc}", file=sys.stderr)
            return 2
        response = None
    hint, accepted, errors = plan_to_hint(graph, response)
    if not accepted and args.strict:
        print(
            "gicc-comm-group-plan-bridge: REJECTED: " + "; ".join(errors),
            file=sys.stderr,
        )
        return 2
    bridge._write_json_atomic(args.hint, hint)
    print(
        f"gicc-comm-group-plan-bridge: {'accepted' if accepted else 'fallback'}; "
        f"wrote {args.hint}", file=sys.stderr,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    emit = sub.add_parser("emit")
    emit.add_argument("--dossier", type=Path, required=True)
    emit.add_argument("--meta-dir", type=Path, required=True)
    emit.add_argument("--graph", type=Path, required=True)
    emit.add_argument("--prompt", type=Path, required=True)
    emit.set_defaults(run=_emit)
    accept = sub.add_parser("accept")
    accept.add_argument("--graph", type=Path, required=True)
    accept.add_argument("--response", type=Path, required=True)
    accept.add_argument("--hint", type=Path, required=True)
    accept.add_argument("--strict", action="store_true")
    accept.set_defaults(run=_accept)
    return parser


def main() -> int:
    try:
        args = _parser().parse_args()
        return int(args.run(args))
    except (bridge.BridgeError, GroupPlanError) as exc:
        print(f"gicc-comm-group-plan-bridge: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
