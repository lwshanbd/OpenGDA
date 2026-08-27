#!/usr/bin/env python3
"""Bridge compiler-proved communication opportunities to an LLM plan.

The model never receives source and never emits code or IR.  It may only
select opaque candidate IDs that were generated from a content-addressed
compiler dossier.  ``accept`` revalidates the graph and translates the
selection into a narrow ``gicc-hint-v1`` request; the LTO pass independently
re-proves the transformation before materializing it.

The v1 structural opportunity is deliberately conservative: a constant-trip
``put_no_db`` loop whose source and destination adjacency was proved by the
compiler may be represented by one larger PUT instead of N descriptors.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import gicc_llm_bridge as bridge


GRAPH_SCHEMA = "gicc-communication-opportunity-graph-v1"
DECISION_SCHEMA = "gicc-communication-plan-decision-v1"
HINT_SCHEMA = "gicc-hint-v1"

_CANDIDATE_KINDS = (
    "proxy_device",
    "trigger_descriptor_batch",
    "trigger_coalesced_loop",
)

_MATERIALIZER_FOR_KIND = {
    "proxy_device": {"dispatch": "CPU_PROXY_ENQUEUE", "transform": "NONE"},
    "trigger_descriptor_batch": {"dispatch": "DWQ_TRIGGER", "transform": "NONE"},
    "trigger_coalesced_loop": {
        "dispatch": "DWQ_TRIGGER", "transform": "COALESCE_LOOP"
    },
}


class PlanBridgeError(ValueError):
    """The compiler graph or model plan violates the plan boundary."""


def _fingerprint(payload: dict[str, Any]) -> str:
    return bridge._fingerprint(payload)


def _candidate_id(payload: dict[str, Any]) -> str:
    return "candidate:" + _fingerprint(payload).removeprefix("sha256:")[:24]


def _opportunity_id(site_id: str) -> str:
    return "opportunity:" + hashlib.sha256(site_id.encode()).hexdigest()[:24]


def _positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _coalesce_proof(site: dict[str, Any]) -> list[str] | None:
    """Return compiler proof obligations recorded by the dossier, or None."""
    loop = site.get("loop")
    if (
        site.get("op_kind") != "put_no_db"
        or site.get("hk_capable") is not True
        or site.get("coalescable") is not True
        or site.get("in_loop") is not True
        or not isinstance(loop, dict)
        or loop.get("bound_known") is not True
        or not isinstance(loop.get("bound_const"), int)
        or not isinstance(loop.get("iv_start"), int)
        or loop.get("iv_step") != 1
        or site.get("size_kind") != "const"
        or not _positive_int(site.get("size_bytes"))
        or not _positive_int(site.get("trip_count"))
        or site.get("guard_kind") == "field_not_null"
        or "trigger" not in site.get("legal_actions", [])
    ):
        return None
    trips = loop["bound_const"] - loop["iv_start"]
    if trips <= 0 or trips != site["trip_count"]:
        return None
    total = site["size_bytes"] * trips
    if total > (1 << 63) - 1:
        return None
    return [
        "put_no_db operation",
        "host-knowable descriptor",
        "constant positive trip count",
        "constant positive transfer size",
        "iv_step equals one",
        "source offsets are adjacent by exactly size_bytes",
        "destination offsets are adjacent by exactly size_bytes",
        "no per-iteration guard",
        "trigger lowering is legal",
        "coalesced byte count fits signed i64",
    ]


def _candidate(
    site: dict[str, Any], kind: str, dispatch: str, transform: str,
    summary: str, effects: dict[str, Any], proof: list[str],
) -> dict[str, Any]:
    payload = {
        "kind": kind,
        "site_ids": [site["site_id"]],
        "materializer": {"dispatch": dispatch, "transform": transform},
        "summary": summary,
        "compiler_proof": proof,
        "effects": effects,
    }
    return {"candidate_id": _candidate_id(payload), **payload}


def _fixed_materializer(site: dict[str, Any]) -> dict[str, str]:
    legal = site.get("legal_actions", [])
    if "default" in legal:
        return {"dispatch": "IPC_OR_DWQ", "transform": "NONE"}
    if "trigger" in legal:
        return {"dispatch": "DWQ_TRIGGER", "transform": "NONE"}
    if "proxy" in legal:
        return {"dispatch": "CPU_PROXY_ENQUEUE", "transform": "NONE"}
    raise PlanBridgeError(f"site {site.get('site_id')!r} has no safe fixed action")


def make_opportunity_graph(dossier_value: Any) -> dict[str, Any]:
    dossier = bridge._verified_dossier(dossier_value)
    opportunities: list[dict[str, Any]] = []
    fixed_sites: list[dict[str, Any]] = []
    for site in dossier["sites"]:
        proof = _coalesce_proof(site)
        if proof is None:
            fixed_sites.append({
                "site_id": site["site_id"],
                "materializer": _fixed_materializer(site),
                "reason": "not part of a compiler-proved structural opportunity",
            })
            continue
        trips = site["trip_count"]
        size = site["size_bytes"]
        candidates: list[dict[str, Any]] = []
        if "proxy" in site["legal_actions"]:
            candidates.append(_candidate(
                site,
                "proxy_device",
                "CPU_PROXY_ENQUEUE",
                "NONE",
                "Keep the loop on the device and enqueue each operation to a proxy worker.",
                {"network_operations": trips, "host_descriptors": 0},
                ["proxy is a compiler-advertised legal action"],
            ))
        candidates.extend([
            _candidate(
                site,
                "trigger_descriptor_batch",
                "DWQ_TRIGGER",
                "NONE",
                "Stage the loop as N descriptors and release them with one trigger.",
                {"network_operations": trips, "host_descriptors": trips},
                ["trigger is legal", "constant loop is host-reconstructable"],
            ),
            _candidate(
                site,
                "trigger_coalesced_loop",
                "DWQ_TRIGGER",
                "COALESCE_LOOP",
                "Replace the proven contiguous loop by one larger transfer.",
                {
                    "network_operations": 1,
                    "host_descriptors": 1,
                    "coalesced_bytes": size * trips,
                },
                proof,
            ),
        ])
        opportunities.append({
            "opportunity_id": _opportunity_id(site["site_id"]),
            "kind": "loop_communication_plan",
            "site_ids": [site["site_id"]],
            "compiler_facts": {
                "kernel": site.get("kernel"),
                "size_bytes": size,
                "trip_count": trips,
                "batch_size": site.get("batch_size"),
                "grid_blocks": site.get("grid_blocks"),
                "descriptor_reusable": site.get("descriptor_reusable"),
                "buffer_reusable": site.get("buffer_reusable"),
                "coalescable": True,
                "guard_kind": site.get("guard_kind"),
                "flops_to_first_use": site.get("flops_to_first_use"),
            },
            "candidates": candidates,
        })

    if not opportunities:
        raise PlanBridgeError("dossier contains no compiler-proved structural opportunity")
    opportunities.sort(key=lambda item: item["opportunity_id"])
    payload = {
        "schema_version": GRAPH_SCHEMA,
        "compiler_dossier_id": dossier["dossier_id"],
        "boundary": {
            "source_visible": False,
            "model_may_generate_code": False,
            "model_may_generate_ir": False,
            "model_output": "candidate IDs only",
            "compiler_revalidates_before_materialization": True,
        },
        "objective": {
            "metric": "end_to_end_wall_time",
            "instruction": (
                "Choose one compiler-generated legal communication plan for "
                "each opportunity. Do not invent transformations."
            ),
        },
        "platform_profile": dossier["platform_profile"],
        "fixed_sites": fixed_sites,
        "opportunities": opportunities,
    }
    graph = dict(payload)
    graph["graph_id"] = _fingerprint(payload)
    return graph


def verified_graph(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != GRAPH_SCHEMA:
        raise PlanBridgeError(f"expected graph schema {GRAPH_SCHEMA}")
    graph_id = value.get("graph_id")
    if not isinstance(graph_id, str):
        raise PlanBridgeError("graph_id is missing")
    payload = dict(value)
    del payload["graph_id"]
    if graph_id != _fingerprint(payload):
        raise PlanBridgeError("graph_id does not match graph content")
    if value.get("boundary") != {
        "source_visible": False,
        "model_may_generate_code": False,
        "model_may_generate_ir": False,
        "model_output": "candidate IDs only",
        "compiler_revalidates_before_materialization": True,
    }:
        raise PlanBridgeError("graph boundary is not compiler-only")
    opportunities = value.get("opportunities")
    if not isinstance(opportunities, list) or not opportunities:
        raise PlanBridgeError("graph has no opportunities")
    seen_opportunities: set[str] = set()
    seen_candidates: set[str] = set()
    opportunity_sites: set[str] = set()
    for opportunity in opportunities:
        if not isinstance(opportunity, dict):
            raise PlanBridgeError("opportunity must be an object")
        opportunity_id = opportunity.get("opportunity_id")
        if not isinstance(opportunity_id, str) or opportunity_id in seen_opportunities:
            raise PlanBridgeError("invalid or duplicate opportunity_id")
        seen_opportunities.add(opportunity_id)
        site_ids = opportunity.get("site_ids")
        if (
            not isinstance(site_ids, list)
            or len(site_ids) != 1
            or not isinstance(site_ids[0], str)
            or site_ids[0] in opportunity_sites
            or opportunity_id != _opportunity_id(site_ids[0])
        ):
            raise PlanBridgeError("invalid or duplicate opportunity site")
        opportunity_sites.add(site_ids[0])
        candidates = opportunity.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise PlanBridgeError(f"{opportunity_id}: fewer than two candidates")
        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise PlanBridgeError("candidate must be an object")
            candidate_id = candidate.get("candidate_id")
            payload = dict(candidate)
            payload.pop("candidate_id", None)
            if (
                not isinstance(candidate_id, str)
                or candidate_id != _candidate_id(payload)
                or candidate_id in seen_candidates
                or candidate.get("kind") not in _CANDIDATE_KINDS
            ):
                raise PlanBridgeError("invalid, stale, or duplicate candidate_id")
            seen_candidates.add(candidate_id)
            materializer = candidate.get("materializer")
            if materializer != _MATERIALIZER_FOR_KIND[candidate["kind"]]:
                raise PlanBridgeError("candidate has invalid materializer contract")
            if candidate.get("site_ids") != site_ids:
                raise PlanBridgeError("candidate does not bind its opportunity site")
    fixed_sites = value.get("fixed_sites")
    if not isinstance(fixed_sites, list):
        raise PlanBridgeError("fixed_sites must be an array")
    seen_fixed: set[str] = set()
    for fixed in fixed_sites:
        if not isinstance(fixed, dict):
            raise PlanBridgeError("fixed site must be an object")
        site_id = fixed.get("site_id")
        materializer = fixed.get("materializer")
        if (
            not isinstance(site_id, str)
            or site_id in seen_fixed
            or site_id in opportunity_sites
            or materializer not in (
                {"dispatch": "IPC_OR_DWQ", "transform": "NONE"},
                {"dispatch": "DWQ_TRIGGER", "transform": "NONE"},
                {"dispatch": "CPU_PROXY_ENQUEUE", "transform": "NONE"},
            )
        ):
            raise PlanBridgeError("invalid, duplicate, or overlapping fixed site")
        seen_fixed.add(site_id)
    return value


def render_prompt(graph_value: Any) -> str:
    graph = verified_graph(graph_value)
    example = {
        "schema_version": DECISION_SCHEMA,
        "graph_id": graph["graph_id"],
        "selections": {
            "<opportunity_id>": {
                "candidate_id": "<one candidate_id from that opportunity>",
                "confidence": 0.0,
                "rationale": "<brief compiler/platform-fact rationale>",
            }
        },
    }
    return (
        "You are a constrained communication-plan component inside a two-phase "
        "LTO workflow. You cannot see or modify source code and must not emit "
        "code, IR, transformations, site IDs, or new legality claims. Select "
        "exactly one existing candidate_id for every opportunity_id. The "
        "compiler will independently verify and materialize the plan.\n\n"
        "Return ONLY one JSON object with this shape:\n"
        + json.dumps(example, indent=2, sort_keys=True)
        + "\n\nCOMPILER-GENERATED OPPORTUNITY GRAPH:\n"
        + json.dumps(graph, indent=2, sort_keys=True)
        + "\n"
    )


def _fallback_hint(graph: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    sites: dict[str, Any] = {}
    for fixed in graph.get("fixed_sites", []):
        materializer = fixed["materializer"]
        if materializer["dispatch"] != "IPC_OR_DWQ":
            sites[fixed["site_id"]] = {
                "dispatch": materializer["dispatch"],
                "reason": "compiler-fixed non-opportunity action",
            }
    for opportunity in graph["opportunities"]:
        baseline = next(
            candidate for candidate in opportunity["candidates"]
            if candidate["kind"] == "trigger_descriptor_batch"
        )
        for site_id in baseline["site_ids"]:
            sites[site_id] = {
                "dispatch": baseline["materializer"]["dispatch"],
                "reason": "deterministic compiler plan fallback",
            }
    return {
        "version": 1,
        "schema_version": HINT_SCHEMA,
        "default_dispatch": "IPC_OR_DWQ",
        "sites": sites,
        "llm_metadata": {
            "accepted": False,
            "graph_id": graph["graph_id"],
            "errors": errors,
            "fallback": "compiler-generated trigger_descriptor_batch",
        },
    }


def plan_to_hint(
    graph_value: Any, decision: Any,
) -> tuple[dict[str, Any], bool, list[str]]:
    graph = verified_graph(graph_value)
    errors: list[str] = []
    if not isinstance(decision, dict):
        errors.append("decision must be a JSON object")
        selections: Any = {}
    else:
        unknown_top = set(decision) - {"schema_version", "graph_id", "selections"}
        if unknown_top:
            errors.append(f"unknown top-level field(s): {sorted(unknown_top)}")
        if decision.get("schema_version") != DECISION_SCHEMA:
            errors.append(f"expected decision schema {DECISION_SCHEMA}")
        if decision.get("graph_id") != graph["graph_id"]:
            errors.append("decision graph_id is stale or does not match")
        selections = decision.get("selections")
        if not isinstance(selections, dict):
            errors.append("selections must be a JSON object")
            selections = {}

    expected = {
        opportunity["opportunity_id"]: opportunity
        for opportunity in graph["opportunities"]
    }
    missing = sorted(set(expected) - set(selections))
    unknown = sorted(set(selections) - set(expected))
    if missing:
        errors.append(f"missing opportunity selection(s): {missing}")
    if unknown:
        errors.append(f"unknown opportunity selection(s): {unknown}")

    accepted_sites: dict[str, Any] = {}
    for fixed in graph.get("fixed_sites", []):
        materializer = fixed["materializer"]
        if materializer["dispatch"] != "IPC_OR_DWQ":
            accepted_sites[fixed["site_id"]] = {
                "dispatch": materializer["dispatch"],
                "reason": "compiler-fixed non-opportunity action",
            }
    selected_ids: dict[str, str] = {}
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
            continue
        by_id = {
            candidate["candidate_id"]: candidate
            for candidate in expected[opportunity_id]["candidates"]
        }
        candidate_id = entry.get("candidate_id")
        candidate = by_id.get(candidate_id)
        if candidate is None:
            errors.append(f"{opportunity_id}: candidate_id is not compiler-generated")
            continue
        confidence = entry.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            errors.append(f"{opportunity_id}: confidence must be finite in [0, 1]")
            continue
        rationale = entry.get("rationale", "")
        if not isinstance(rationale, str) or len(rationale) > 512:
            errors.append(f"{opportunity_id}: rationale must be at most 512 characters")
            continue
        selected_ids[opportunity_id] = candidate_id
        materializer = candidate["materializer"]
        for site_id in candidate["site_ids"]:
            if site_id in accepted_sites:
                errors.append(f"site {site_id!r} selected by multiple opportunities")
                continue
            site_hint = {
                "dispatch": materializer["dispatch"],
                "reason": (
                    f"compiler plan {candidate_id}; LLM confidence="
                    f"{float(confidence):.3f}: {rationale}"
                ),
            }
            if materializer["transform"] != "NONE":
                site_hint["transform"] = materializer["transform"]
            accepted_sites[site_id] = site_hint

    if errors:
        return _fallback_hint(graph, errors), False, errors
    return {
        "version": 1,
        "schema_version": HINT_SCHEMA,
        "default_dispatch": "IPC_OR_DWQ",
        "sites": accepted_sites,
        "llm_metadata": {
            "accepted": True,
            "decision_schema": DECISION_SCHEMA,
            "graph_id": graph["graph_id"],
            "selected_candidates": selected_ids,
            "compiler_only_output": True,
        },
    }, True, []


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanBridgeError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    bridge._write_json_atomic(path, value)


def _emit(args: argparse.Namespace) -> int:
    graph = make_opportunity_graph(_read_json(args.dossier))
    _write_json(args.graph, graph)
    bridge._write_text_atomic(args.prompt, render_prompt(graph))
    print(
        f"gicc-comm-plan-bridge: wrote {len(graph['opportunities'])} "
        f"opportunities; graph_id={graph['graph_id']}",
        file=sys.stderr,
    )
    return 0


def _accept(args: argparse.Namespace) -> int:
    graph = _read_json(args.graph)
    try:
        decision = _read_json(args.response)
    except PlanBridgeError as exc:
        verified = verified_graph(graph)
        errors = [str(exc)]
        if args.strict:
            print(f"gicc-comm-plan-bridge: REJECTED: {errors[0]}", file=sys.stderr)
            return 2
        hint = _fallback_hint(verified, errors)
        _write_json(args.hint, hint)
        return 0
    hint, accepted, errors = plan_to_hint(graph, decision)
    if not accepted and args.strict:
        print(
            "gicc-comm-plan-bridge: REJECTED: " + "; ".join(errors),
            file=sys.stderr,
        )
        return 2
    _write_json(args.hint, hint)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    emit = sub.add_parser("emit")
    emit.add_argument("--dossier", required=True, type=Path)
    emit.add_argument("--graph", required=True, type=Path)
    emit.add_argument("--prompt", required=True, type=Path)
    emit.set_defaults(func=_emit)
    accept = sub.add_parser("accept")
    accept.add_argument("--graph", required=True, type=Path)
    accept.add_argument("--response", required=True, type=Path)
    accept.add_argument("--hint", required=True, type=Path)
    accept.add_argument("--strict", action="store_true")
    accept.set_defaults(func=_accept)
    args = parser.parse_args()
    try:
        return args.func(args)
    except (PlanBridgeError, bridge.BridgeError, OSError, ValueError) as exc:
        print(f"gicc-comm-plan-bridge: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
