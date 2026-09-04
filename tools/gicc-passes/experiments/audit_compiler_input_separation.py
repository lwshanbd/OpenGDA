#!/usr/bin/env python3
"""Audit the factual separation between the scalar GBT and LLM views.

This tool never invokes a model, compiler, scheduler, or source editor.  It
verifies the frozen decision suite, parses its content-addressed prompt views,
and compares their information contracts with the frozen scalar GBT report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "python"))

import gicc_compiler_decision_suite as decision_suite  # noqa: E402


REPORT_SCHEMA = "gicc-compiler-input-separation-v1"
GBT_SCHEMA = "gicc-compiler-lto-gbt-v1"
GBT_FEATURES = (
    "log2(size_bytes)",
    "log2(logical_ops)",
    "log2(total_bytes)",
    "is_proxy",
    "log2(trigger_batch) or -1",
    "log2(proxy_producers) or -1",
    "log2(proxy_workers) or -1",
)
GBT_UNMAPPED_FACTS = {
    "descriptor_reusable", "coalescable", "flops_to_first_use",
}
PROMPT_MARKERS = (
    "COMPILER-GENERATED OPPORTUNITY GRAPH:\n",
    "COMPILER-GENERATED COMMUNICATION GRAPH:\n",
    "Compiler opportunity graph:\n",
)
VIEWS = ("relational", "descriptors", "opaque")
PROVIDER_PRIVATE_KEYS = frozenset({
    "bitcode", "completion_site_id", "debug_location", "debug_locations",
    "debug_loc", "directory", "filename", "kernel", "kernel_mangled",
    "llvm_ir", "materializer", "module_ir", "operation_order", "path",
    "provenance", "site_id", "site_ids", "source_code", "source_file",
    "source_location", "source_locations", "source_path", "target_id",
})
SOURCE_FILENAME_RE = re.compile(
    r"(?i)(?:^|[\\/\s])[^\\/\s]+\."
    r"(?:c|cc|cpp|cxx|cu|hip|h|hh|hpp|hxx|f|f90|f95|ll|bc)"
    r"(?=$|[:\s,;])"
)
ABSOLUTE_PATH_RE = re.compile(
    r"(?:^|\s)(?:/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+"
    r"|[A-Za-z]:[\\/][^\s]+|\\\\[^\s]+)"
)
RAW_LLVM_IR_RES = (
    re.compile(r"source_filename\s*="),
    re.compile(r"(?:^|\n)\s*(?:define|declare)\b[^@\n]*@[-$._A-Za-z0-9]+"),
    re.compile(r"(?:^|\n)\s*%[-$._A-Za-z0-9]+\s*="),
    re.compile(r"!dbg\b"),
    re.compile(r"(?:^|\n)\s*target\s+(?:triple|datalayout)\s*="),
)


class SeparationError(RuntimeError):
    """Frozen evidence does not prove the claimed input separation."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SeparationError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise SeparationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def evidence(path: Path) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def prompt_graph(path: Path) -> dict[str, Any]:
    try:
        prompt = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SeparationError(f"cannot read prompt {path}: {exc}") from exc
    matches = [marker for marker in PROMPT_MARKERS if marker in prompt]
    if len(matches) != 1:
        raise SeparationError(f"{path}: expected exactly one graph marker")
    tail = prompt.split(matches[0], 1)[1].lstrip()
    try:
        value, _ = json.JSONDecoder().raw_decode(tail)
    except json.JSONDecodeError as exc:
        raise SeparationError(f"{path}: cannot parse embedded graph: {exc}") from exc
    if not isinstance(value, dict):
        raise SeparationError(f"{path}: embedded graph is not an object")
    return value


def walk(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk(child)


def all_keys(value: Any) -> set[str]:
    result: set[str] = set()
    for item in walk(value):
        if isinstance(item, dict):
            result.update(item)
    return result


def strings_with_paths(value: Any, path: str = "$"):
    if isinstance(value, dict):
        for key, child in value.items():
            yield from strings_with_paths(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from strings_with_paths(child, f"{path}[{index}]")
    elif isinstance(value, str):
        yield path, value


def verify_provider_prompt_surface(path: Path,
                                   graph: dict[str, Any]) -> dict[str, Any]:
    """Reject source/IR/private compiler payloads from one provider prompt."""
    boundary = graph.get("boundary")
    if not isinstance(boundary, dict):
        raise SeparationError(f"{path}: embedded graph lacks a boundary")
    expected_boundary = {
        "source_visible": False,
        "model_may_generate_code": False,
        "model_may_generate_ir": False,
        "compiler_revalidates_before_materialization": True,
    }
    for key, expected in expected_boundary.items():
        if boundary.get(key) is not expected:
            raise SeparationError(
                f"{path}: compiler-only boundary {key} is not {expected}"
            )
    if boundary.get("model_output") not in {
        "candidate IDs only", "compiler-generated option IDs only",
    }:
        raise SeparationError(f"{path}: model output is not graph-bound IDs")
    for key, value in boundary.items():
        if key.startswith("model_may_") and value is not False:
            raise SeparationError(f"{path}: permissive boundary field {key}")

    escaped_keys = sorted(PROVIDER_PRIVATE_KEYS & all_keys(graph))
    if escaped_keys:
        raise SeparationError(
            f"{path}: private/source-bearing key(s) escaped: {escaped_keys}"
        )
    try:
        prompt = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SeparationError(f"cannot read prompt {path}: {exc}") from exc
    surfaces = [("prompt", prompt), *strings_with_paths(graph)]
    for location, text in surfaces:
        if ABSOLUTE_PATH_RE.search(text):
            raise SeparationError(
                f"{path}: filesystem path escaped at {location}"
            )
        if SOURCE_FILENAME_RE.search(text):
            raise SeparationError(
                f"{path}: source/IR filename escaped at {location}"
            )
        if any(pattern.search(text) for pattern in RAW_LLVM_IR_RES):
            raise SeparationError(f"{path}: raw LLVM IR escaped at {location}")
    return {
        "compiler_only_boundary_verified": True,
        "application_source_or_source_locations_present": False,
        "raw_llvm_ir_present": False,
        "filesystem_paths_present": False,
        "private_materializer_or_identity_keys_present": False,
    }


def _closed_object(value: Any, properties: set[str], context: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("type") != "object":
        raise SeparationError(f"{context}: expected object schema")
    if value.get("additionalProperties") is not False:
        raise SeparationError(f"{context}: additionalProperties is not false")
    children = value.get("properties")
    required = value.get("required")
    if (not isinstance(children, dict) or set(children) != properties
            or not isinstance(required, list) or set(required) != properties
            or len(required) != len(properties)):
        raise SeparationError(f"{context}: properties/required are not exact")
    return children


def verify_response_schema(path: Path, entry: dict[str, Any],
                           graph: dict[str, Any],
                           selectable: set[str]) -> dict[str, Any]:
    """Verify that provider output authority is a closed graph-ID choice."""
    schema = read_json(path)
    top = _closed_object(
        schema, {"schema_version", "graph_id", "selections"}, str(path),
    )
    if (not isinstance(top["schema_version"], dict)
            or set(top["schema_version"]) != {"const"}
            or not isinstance(top["schema_version"].get("const"), str)):
        raise SeparationError(f"{path}: decision schema version is not constant")
    if top["graph_id"] != {"const": entry.get("graph_id")}:
        raise SeparationError(f"{path}: response graph_id is not suite-bound")
    opportunities = graph.get("opportunities")
    if not isinstance(opportunities, list) or not opportunities:
        raise SeparationError(f"{path}: prompt graph has no opportunities")
    opportunity_ids = {
        item.get("opportunity_id") for item in opportunities
        if isinstance(item, dict) and isinstance(item.get("opportunity_id"), str)
    }
    if len(opportunity_ids) != len(opportunities):
        raise SeparationError(f"{path}: prompt opportunity IDs are invalid")
    selections = _closed_object(
        top["selections"], opportunity_ids, f"{path}: selections",
    )

    observed: list[str] = []
    family = entry.get("decision_family")
    for opportunity_id, selection in selections.items():
        context = f"{path}: selection {opportunity_id}"
        if family == "collective_algorithm_and_size_policy":
            children = _closed_object(
                selection,
                {"slot_candidate_ids", "confidence", "rationale"}, context,
            )
            slots = children["slot_candidate_ids"]
            if (not isinstance(slots, dict) or slots.get("type") != "object"
                    or slots.get("additionalProperties") is not False
                    or not isinstance(slots.get("properties"), dict)
                    or not isinstance(slots.get("required"), list)
                    or set(slots["properties"]) != set(slots["required"])
                    or len(slots["properties"]) != len(slots["required"])):
                raise SeparationError(f"{context}: slot schema is not closed")
            for slot_id, slot in slots["properties"].items():
                if (not isinstance(slot, dict) or set(slot) != {"type", "enum"}
                        or slot.get("type") != "string"
                        or not isinstance(slot.get("enum"), list)
                        or not slot["enum"]):
                    raise SeparationError(
                        f"{context}/{slot_id}: option enum is not exact"
                    )
                observed.extend(slot["enum"])
        elif family in {
            "communication_route_or_schedule",
            "communication_coalescing_and_trigger_placement",
        }:
            children = _closed_object(
                selection, {"candidate_id", "confidence", "rationale"}, context,
            )
            candidate = children["candidate_id"]
            if (not isinstance(candidate, dict)
                    or set(candidate) != {"type", "enum"}
                    or candidate.get("type") != "string"
                    or not isinstance(candidate.get("enum"), list)
                    or not candidate["enum"]):
                raise SeparationError(f"{context}: candidate enum is not exact")
            observed.extend(candidate["enum"])
        else:
            raise SeparationError(f"{path}: unknown decision family {family!r}")
        if children["confidence"] != {
            "type": "number", "minimum": 0.0, "maximum": 1.0,
        }:
            raise SeparationError(f"{context}: confidence schema changed")
        if children["rationale"] != {"type": "string", "maxLength": 512}:
            raise SeparationError(f"{context}: rationale schema changed")

    if (any(not isinstance(value, str) for value in observed)
            or len(observed) != len(set(observed))
            or set(observed) != selectable):
        raise SeparationError(
            f"{path}: response enums differ from provider-visible graph IDs"
        )
    return {
        "schema": evidence(path),
        "all_object_schemas_closed": True,
        "graph_id_is_constant": True,
        "authoritative_choice_fields": (
            ["slot_candidate_ids"]
            if family == "collective_algorithm_and_size_policy"
            else ["candidate_id"]
        ),
        "bounded_metadata_fields": ["confidence", "rationale"],
        "selectable_id_count": len(observed),
        "selectable_ids_equal_prompt": True,
    }


def relation_kinds(value: Any) -> set[str]:
    result: set[str] = set()
    for item in walk(value):
        if not isinstance(item, dict):
            continue
        relations = item.get("relations")
        if not isinstance(relations, list):
            continue
        for relation in relations:
            if isinstance(relation, dict) and isinstance(relation.get("kind"), str):
                result.add(relation["kind"])
    return result


def selectable_ids(value: Any) -> set[str]:
    result: set[str] = set()
    for item in walk(value):
        if not isinstance(item, dict):
            continue
        for key, prefix in (("candidate_id", "candidate:"),
                            ("option_id", "option:")):
            selected = item.get(key)
            if isinstance(selected, str) and selected.startswith(prefix):
                result.add(selected)
    return result


def semantic_families(value: dict[str, Any]) -> dict[str, bool]:
    keys = all_keys(value)
    relations = relation_kinds(value)
    return {
        "operation_and_cross_opportunity_relations": bool(relations),
        "symbolic_argument_expressions": bool({
            "transfer_argument_expressions",
            "shared_transfer_argument_expressions",
            "argument",
        } & keys),
        "dependence_and_legality_proofs": bool({
            "dependence_legality", "compiler_legality", "compiler_proof",
        } & keys),
        "symbolic_transfer_intervals": bool({
            "transfer_interval", "transfer_intervals", "byte_offset",
            "byte_size",
        } & keys),
        "launch_and_resource_structure": bool({
            "launch_contexts", "launch_grid", "grid_blocks",
            "resource_constraints", "resource_model", "rank_placement",
        } & keys),
        "completion_and_compute_distance": bool({
            "completion_kind", "compute_region", "flops_to_first_use",
            "transfer_flops_to_completion",
        } & keys),
        "compiler_candidate_semantics_and_effects": bool({
            "compiler_descriptor", "compiler_proof", "effects", "summary",
        } & keys),
        "collective_topology_and_size_policy": bool({
            "communication_graph", "decision_slots", "thresholds_bytes",
            "message_distribution",
        } & keys),
        "masked_compiler_transform_reasons": bool({
            "masked_candidates", "masked_candidate_count",
        } & keys),
    }


def expected_selectable_count(entry: dict[str, Any]) -> int:
    decision = entry.get("decision_space", {})
    for key in ("selectable_candidate_id_count", "selectable_option_id_count"):
        value = decision.get(key)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    raise SeparationError(f"{entry.get('label')}: no selectable-ID count")


def verify_gbt_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != GBT_SCHEMA:
        raise SeparationError(f"expected GBT schema {GBT_SCHEMA}")
    if tuple(value.get("model_features", ())) != GBT_FEATURES:
        raise SeparationError("GBT report does not bind the seven scalar features")
    if value.get("source_read") is not False:
        raise SeparationError("GBT report does not prove source_read=false")
    if value.get("oracle_or_runtime_results_read") is not False:
        raise SeparationError("GBT report consumed oracle/runtime labels")
    predictions = value.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise SeparationError("GBT report has no predictions")
    unmapped: set[str] = set()
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise SeparationError("GBT prediction is not an object")
        facts = prediction.get("compiler_facts_not_mapped_to_historical_grid")
        if not isinstance(facts, dict):
            raise SeparationError("GBT prediction lacks its unmapped-fact audit")
        unmapped.update(facts)
    if not GBT_UNMAPPED_FACTS.issubset(unmapped):
        raise SeparationError("GBT report no longer exposes the known omitted facts")
    return {
        "schema_version": value["schema_version"],
        "policy": value.get("policy"),
        "feature_count": len(GBT_FEATURES),
        "features": list(GBT_FEATURES),
        "feature_structure": "one numeric vector per action prediction",
        "explicit_relational_edges": 0,
        "known_compiler_facts_present_but_not_used": sorted(unmapped),
        "source_read": False,
        "oracle_or_runtime_results_read": False,
    }


def build_report(suite_path: Path, prompt_dir: Path,
                 gbt_report_path: Path) -> dict[str, Any]:
    suite_value = read_json(suite_path)
    suite = decision_suite.verified_suite(suite_value, prompt_dir)
    gbt = verify_gbt_report(read_json(gbt_report_path))
    entries: dict[str, Any] = {}
    suite_families: set[str] = set()
    all_authority_equal = True
    for entry in suite["entries"]:
        label = entry["label"]
        views: dict[str, Any] = {}
        ids_by_view: dict[str, set[str]] = {}
        graphs_by_view: dict[str, dict[str, Any]] = {}
        for view_name in VIEWS:
            path = prompt_dir / label / f"{view_name}.txt"
            graph = prompt_graph(path)
            if graph.get("view_kind") != view_name:
                raise SeparationError(f"{path}: wrong embedded view_kind")
            surface = verify_provider_prompt_surface(path, graph)
            ids = selectable_ids(graph)
            if len(ids) != expected_selectable_count(entry):
                raise SeparationError(
                    f"{path}: selectable-ID count disagrees with suite"
                )
            ids_by_view[view_name] = ids
            graphs_by_view[view_name] = graph
            families = semantic_families(graph)
            if view_name == "relational":
                suite_families.update(
                    name for name, present in families.items() if present
                )
            views[view_name] = {
                "prompt": evidence(path),
                "selectable_id_count": len(ids),
                "semantic_key_count": len(all_keys(graph)),
                "relation_kinds": sorted(relation_kinds(graph)),
                "semantic_families": families,
                "provider_surface": surface,
            }
        same_authority = all(
            ids_by_view[name] == ids_by_view["relational"] for name in VIEWS
        )
        all_authority_equal = all_authority_equal and same_authority
        schema_path = prompt_dir / label / "response-schema.json"
        schema_contract = verify_response_schema(
            schema_path, entry, graphs_by_view["relational"],
            ids_by_view["relational"],
        )
        entries[label] = {
            "decision_family": entry["decision_family"],
            "same_selectable_ids_across_llm_views": same_authority,
            "selectable_id_set_id": fingerprint(
                sorted(ids_by_view["relational"])
            ),
            "response_contract": schema_contract,
            "views": views,
        }

    required_llm_only = {
        "operation_and_cross_opportunity_relations",
        "symbolic_argument_expressions",
        "dependence_and_legality_proofs",
        "symbolic_transfer_intervals",
        "launch_and_resource_structure",
        "completion_and_compute_distance",
        "compiler_candidate_semantics_and_effects",
        "collective_topology_and_size_policy",
    }
    missing = required_llm_only - suite_families
    if missing:
        raise SeparationError(
            f"relational suite lacks required semantic families: {sorted(missing)}"
        )
    if not all_authority_equal:
        raise SeparationError("LLM information views do not share one action set")

    payload = {
        "schema_version": REPORT_SCHEMA,
        "scope": (
            "input-contract audit only; not a model-quality or performance result"
        ),
        "boundary": {
            "application_source_visible": False,
            "source_locations_visible": False,
            "llvm_ir_visible": False,
            "private_materializers_visible": False,
            "application_source_modified": False,
            "model_invoked": False,
            "provider_invoked": False,
            "compiler_lto_decisions_only": True,
            "provider_prompt_surface_structurally_audited": True,
            "response_schemas_closed_and_graph_bound": True,
        },
        "suite_id": suite["suite_id"],
        "gbt": gbt,
        "llm_relational_semantic_families": sorted(suite_families),
        "entries": {label: entries[label] for label in sorted(entries)},
        "controlled_interpretation": {
            "gbt_and_llm_inputs_identical": False,
            "llm_has_richer_relational_compiler_context": True,
            "same_selectable_ids_across_llm_information_ablation": True,
            "gbt_vs_llm_is_not_an_action-controlled_model_comparison": True,
            "valid_future_llm_ablation": list(VIEWS),
            "performance_superiority_claimed": False,
            "reason": (
                "The scalar GBT is a route baseline. The three LLM views bind "
                "one compiler-generated action set, while only their semantic "
                "information changes. Runtime-held-out evaluation is still required."
            ),
        },
        "evidence": {
            "suite": evidence(suite_path),
            "gbt_report": evidence(gbt_report_path),
        },
    }
    result = dict(payload)
    result["audit_id"] = fingerprint(payload)
    return result


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("emit", "verify"):
        child = subparsers.add_parser(command)
        child.add_argument("--suite", required=True, type=Path)
        child.add_argument("--prompt-dir", required=True, type=Path)
        child.add_argument("--gbt-report", required=True, type=Path)
        child.add_argument(
            "--out" if command == "emit" else "--report",
            required=True, type=Path,
        )
    args = parser.parse_args()
    try:
        result = build_report(args.suite, args.prompt_dir, args.gbt_report)
        if args.command == "emit":
            write_json_atomic(args.out, result)
            action = "wrote"
        else:
            if read_json(args.report) != result:
                raise SeparationError(
                    "input-separation report does not match current evidence"
                )
            action = "verified"
        print(
            f"compiler-input-separation: {action} {len(result['entries'])} "
            f"entries; gbt_features={result['gbt']['feature_count']}; "
            f"audit_id={result['audit_id']}"
        )
        return 0
    except (SeparationError, decision_suite.SuiteError, OSError,
            KeyError, TypeError, ValueError) as exc:
        print(f"compiler-input-separation: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
