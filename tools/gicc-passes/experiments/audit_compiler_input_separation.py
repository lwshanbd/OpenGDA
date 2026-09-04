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
        for view_name in VIEWS:
            path = prompt_dir / label / f"{view_name}.txt"
            graph = prompt_graph(path)
            if graph.get("view_kind") != view_name:
                raise SeparationError(f"{path}: wrong embedded view_kind")
            ids = selectable_ids(graph)
            if len(ids) != expected_selectable_count(entry):
                raise SeparationError(
                    f"{path}: selectable-ID count disagrees with suite"
                )
            ids_by_view[view_name] = ids
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
            }
        same_authority = all(
            ids_by_view[name] == ids_by_view["relational"] for name in VIEWS
        )
        all_authority_equal = all_authority_equal and same_authority
        entries[label] = {
            "decision_family": entry["decision_family"],
            "same_selectable_ids_across_llm_views": same_authority,
            "selectable_id_set_id": fingerprint(
                sorted(ids_by_view["relational"])
            ),
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
            "application_source_modified": False,
            "model_invoked": False,
            "provider_invoked": False,
            "compiler_lto_decisions_only": True,
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
