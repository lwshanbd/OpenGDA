#!/usr/bin/env python3
"""Build a source-free coverage graph from current compiler fact files."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


class CoverageError(RuntimeError):
    """An input does not satisfy the compiler-fact coverage contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_id(prefix: str, value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(prefix.encode() + b'\0' + payload).hexdigest()}"


def walk_json(value: Any, location: str = "$"):
    """Yield every JSON key/value with a stable diagnostic location."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield location, "key", key
            yield from walk_json(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from walk_json(child, f"{location}[{index}]")
    else:
        yield location, "value", value


def validate_candidate(candidate: dict[str, Any]) -> None:
    """Require candidate IDs to commit to the complete compiler payload."""
    candidate_id = candidate.get("candidate_id")
    payload = {key: value for key, value in candidate.items()
               if key != "candidate_id"}
    expected = canonical_id("gicc-schedule-candidate-v1", payload)
    if candidate_id != expected:
        raise CoverageError(
            f"candidate ID {candidate_id!r} does not match payload {expected}"
        )


def validate_model_case(model_case: dict[str, Any]) -> None:
    """Validate all content-addressed links in one model-visible case."""
    facts = model_case.get("compiler_facts")
    expected_case_id = canonical_id("gicc-schedule-coverage-case-v1", facts)
    if model_case.get("case_id") != expected_case_id:
        raise CoverageError("case ID does not match compiler facts")

    legal = model_case.get("legal_candidates")
    if not isinstance(legal, list) or not legal:
        raise CoverageError("model case must expose a nonempty candidate list")
    for candidate in legal:
        if not isinstance(candidate, dict):
            raise CoverageError("compiler candidate must be an object")
        validate_candidate(candidate)
        if candidate.get("case_id") != expected_case_id:
            raise CoverageError("compiler candidate belongs to a different case")

    visible = model_case.get("model_visible_candidate_ids")
    legal_ids = [candidate["candidate_id"] for candidate in legal]
    if visible != legal_ids:
        raise CoverageError("model-visible IDs must exactly match legal candidates")

    oracle = model_case.get("dormant_compiler_oracle")
    if oracle is not None:
        if not isinstance(oracle, dict):
            raise CoverageError("dormant compiler oracle must be an object")
        validate_candidate(oracle)
        if oracle.get("case_id") != expected_case_id:
            raise CoverageError("compiler oracle belongs to a different case")
        if oracle["candidate_id"] in visible:
            raise CoverageError("dormant compiler oracle is model-visible")


def validate_source_free_model(
    model_case: dict[str, Any], *, label: str, forbidden_fragments: set[str]
) -> None:
    """Fail closed if private compiler identity escapes into the model graph."""
    forbidden_keys = {"case_label", "kernel", "path", "site_id", "source_path"}
    fragments = sorted(
        (fragment for fragment in forbidden_fragments if len(fragment) >= 3),
        key=len,
        reverse=True,
    )
    for location, kind, value in walk_json(model_case):
        if kind == "key" and value in forbidden_keys:
            raise CoverageError(f"identity-bearing key {value!r} at {location}")
        if kind != "value" or not isinstance(value, str):
            continue
        if value == label:
            raise CoverageError(f"case label escaped into model graph at {location}")
        if value.startswith("/") or value.startswith("./") or value.startswith("../"):
            raise CoverageError(f"filesystem path escaped into model graph at {location}")
        for fragment in fragments:
            if fragment in value:
                raise CoverageError(
                    f"private compiler identifier {fragment!r} escaped at {location}"
                )


def parse_case(value: str) -> tuple[str, Path]:
    try:
        label, raw_path = value.split("=", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected LABEL=FEATURES_JSON") from exc
    if not label or not label.replace("_", "").replace("-", "").isalnum():
        raise argparse.ArgumentTypeError("case label must be alphanumeric, '-' or '_'")
    return label, Path(raw_path)


def unique(values: list[Any]) -> list[Any]:
    return sorted(set(values), key=lambda item: str(item))


def load_case(label: str, features_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        feature_value = json.loads(features_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoverageError(f"cannot read {label} features: {exc}") from exc
    if not isinstance(feature_value, list) or not feature_value:
        raise CoverageError(f"{label} features must be a nonempty array")
    if any(not isinstance(row, dict) or row.get("schema_version") != 6
           for row in feature_value):
        raise CoverageError(f"{label} does not use compiler feature schema 6")
    transfers = [
        row for row in feature_value
        if row.get("op_kind") in {"put_no_db", "get_no_db"}
    ]
    if not transfers:
        raise CoverageError(f"{label} has no transfer facts")

    template_paths = sorted(
        path for path in features_path.parent.glob("*.json")
        if path != features_path
    )
    if not template_paths:
        raise CoverageError(f"{label} has no sibling kernel templates")
    template_ops: dict[str, dict[str, Any]] = {}
    template_artifacts = []
    forbidden_fragments = {
        str(features_path),
        str(features_path.resolve()),
        features_path.name,
    }
    for path in template_paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CoverageError(f"cannot read template {path}: {exc}") from exc
        if not isinstance(value, dict) or value.get("version") != 1:
            raise CoverageError(f"{path} does not use kernel-template schema 1")
        for op in value.get("ops", []):
            site_id = op.get("site_id") if isinstance(op, dict) else None
            if not isinstance(site_id, str) or not site_id:
                raise CoverageError(f"{path} contains an invalid operation")
            if site_id in template_ops:
                raise CoverageError(f"duplicate site ID across {label} templates")
            template_ops[site_id] = op
        template_artifacts.append({
            "path": str(path.resolve()),
            "sha256": sha256(path),
        })
        forbidden_fragments.update({str(path), str(path.resolve()), path.name})
    for row in transfers:
        if row.get("site_id") not in template_ops:
            raise CoverageError(f"{label} transfer is absent from templates")

    frontiers = [
        row.get("producer_frontier")
        for row in transfers
        if isinstance(row.get("producer_frontier"), dict)
    ]
    template_transfers = [template_ops[row["site_id"]] for row in transfers]
    guard_kinds = unique([row.get("guard_kind") for row in transfers])
    loop_sites = sum(bool(row.get("in_loop")) for row in transfers)
    exact_intervals = sum(
        row.get("transfer_interval", {}).get("symbolically_exact") is True
        and row.get("transfer_interval", {}).get("host_knowable") is True
        for row in transfers
    )
    phase_supported = all(
        row.get("phase_launch_supported") is True for row in transfers
    )
    ordinary = max(
        [int(frontier.get("ordinary_store_sites", 0)) for frontier in frontiers]
        or [0]
    )
    atomic = max(
        [int(frontier.get("atomic_write_sites", 0)) for frontier in frontiers]
        or [0]
    )
    unknown = max(
        [int(frontier.get("unknown_write_sites", 0)) for frontier in frontiers]
        or [0]
    )
    exact_partition = bool(frontiers) and all(
        frontier.get("overlap_partition", {}).get("exact") is True
        for frontier in frontiers
    )
    atomic_domains_known = bool(frontiers) and all(
        frontier.get("atomic_domains_known") is True
        for frontier in frontiers
    )
    buffer_identity_guardable = bool(frontiers) and all(
        frontier.get("buffer_identity_guardable") is True
        for frontier in frontiers
    )
    write_footprint_known = bool(frontiers) and all(
        frontier.get("write_footprint_known") is True
        for frontier in frontiers
    )
    source_identity_guardable = bool(frontiers) and all(
        frontier.get("source_identity_guardable") is True
        for frontier in frontiers
    )
    relation_map = {}
    for frontier in frontiers:
        if frontier.get("source_identity_guardable") is not True:
            continue
        relation = {
            "source_buffer_formal":
                frontier.get("source_identity_buffer_index_param"),
            "pointer_formals": unique(
                frontier.get("source_pointer_candidates", [])
            ),
            "write_pointer_formals": unique(
                frontier.get("ordinary_store_params", []) +
                frontier.get("atomic_write_params", [])
            ),
            "proof": "runtime_source_identity_candidate",
            "required_proof": "whole_write_allocation_disjointness",
        }
        relation_map[json.dumps(
            relation, sort_keys=True, separators=(",", ":")
        )] = relation
    source_identity_relations = [
        relation_map[key] for key in sorted(relation_map)
    ]
    max_compute = max(
        int(row.get("flops_to_first_use") or 0) for row in transfers
    )
    group_reasons = unique([
        op.get("group_early_trigger_reason")
        for op in template_transfers
        if isinstance(op.get("group_early_trigger_reason"), str)
        and op.get("group_early_trigger_reason")
    ])
    frontier_reasons = unique([
        frontier.get("reason") for frontier in frontiers
        if isinstance(frontier.get("reason"), str) and frontier.get("reason")
    ])
    launch_reasons = unique([
        row.get("phase_launch_reason") for row in transfers
        if row.get("phase_launch_supported") is not True
        and isinstance(row.get("phase_launch_reason"), str)
    ])

    if exact_partition and phase_supported:
        schedule_class = "intra_kernel_exact_partition"
    elif loop_sites:
        schedule_class = "loop_carried_communication"
    elif unknown:
        schedule_class = "unknown_side_effect_frontier"
    elif (source_identity_guardable and
          all(kind == "always" for kind in guard_kinds)):
        schedule_class = "guarded_source_identity_candidate"
    elif (atomic and not ordinary and atomic_domains_known
          and buffer_identity_guardable):
        schedule_class = "atomic_producer_no_store_remainder"
    elif atomic and not ordinary:
        schedule_class = "unresolved_atomic_alias_frontier"
    elif all(kind != "always" for kind in guard_kinds) and max_compute == 0:
        schedule_class = "conditional_communication_only"
    else:
        schedule_class = "insufficient_compiler_proof"

    missing_proof_families = []
    if loop_sites:
        missing_proof_families.append("loop_carried_phase_schedule")
    if any(kind != "always" for kind in guard_kinds):
        missing_proof_families.append("guarded_completion_region")
    if max_compute == 0 and not frontiers:
        missing_proof_families.append("cross_launch_producer_pipeline")
    if (source_identity_guardable and not exact_partition and
            all(kind == "always" for kind in guard_kinds)):
        missing_proof_families.append(
            "write_allocation_disjointness"
        )
        missing_proof_families.append(
            "guarded_early_trigger_materialization"
        )
    elif (atomic and not ordinary and atomic_domains_known
            and buffer_identity_guardable):
        missing_proof_families.append("atomic_producer_partition")
    elif atomic and not ordinary:
        missing_proof_families.append(
            "registered_source_alias_disambiguation"
        )
    if unknown:
        missing_proof_families.append("side_effect_alias_partition")
    if not phase_supported:
        missing_proof_families.append("host_phase_materialization")

    facts = {
        "transfer_count": len(transfers),
        "operation_kinds": unique([row.get("op_kind") for row in transfers]),
        "batch_sizes": unique([row.get("batch_size") for row in transfers]),
        "guard_kinds": guard_kinds,
        "loop_transfer_sites": loop_sites,
        "exact_host_transfer_intervals": exact_intervals,
        "all_transfer_intervals_exact": exact_intervals == len(transfers),
        "phase_launch_supported": phase_supported,
        "frontier_analyzed_sites": len(frontiers),
        "ordinary_store_sites": ordinary,
        "atomic_write_sites": atomic,
        "unknown_write_sites": unknown,
        "atomic_domains_known": atomic_domains_known,
        "buffer_identity_guardable": buffer_identity_guardable,
        "write_footprint_known": write_footprint_known,
        "source_identity_guardable": source_identity_guardable,
        "source_identity_relations": source_identity_relations,
        "exact_overlap_partition": exact_partition,
        "max_flops_to_completion": max_compute,
        "compiler_blockers": {
            "completion_group": group_reasons,
            "producer_frontier": frontier_reasons,
            "phase_launch": launch_reasons,
        },
        "schedule_class": schedule_class,
        "missing_proof_families": unique(missing_proof_families),
    }
    case_id = canonical_id("gicc-schedule-coverage-case-v1", facts)
    original_payload = {
        "case_id": case_id,
        "kind": "original_fused",
        "compiler_materializer": "identity",
        "phase_count": 1,
    }
    original_candidate = {
        "candidate_id": canonical_id(
            "gicc-schedule-candidate-v1", original_payload
        ),
        **original_payload,
    }
    oracle_candidate = None
    if exact_partition and phase_supported:
        oracle_payload = {
            "case_id": case_id,
            "kind": "producer_frontier_two_phase",
            "compiler_materializer": "guarded_host_device_fission",
            "phase_count": 2,
        }
        oracle_candidate = {
            "candidate_id": canonical_id(
                "gicc-schedule-candidate-v1", oracle_payload
            ),
            **oracle_payload,
        }
    internal = {
        "case_id": case_id,
        "features": {
            "path": str(features_path.resolve()),
            "sha256": sha256(features_path),
        },
        "templates": template_artifacts,
        "kernel_count": len(unique([row.get("kernel") for row in transfers])),
        "facts": facts,
    }
    model_case = {
        "case_id": case_id,
        "compiler_facts": facts,
        "legal_candidates": [original_candidate],
        "model_visible_candidate_ids": [original_candidate["candidate_id"]],
        "dormant_compiler_oracle": oracle_candidate,
    }
    forbidden_fragments.update(
        str(row["site_id"]) for row in transfers if row.get("site_id")
    )
    forbidden_fragments.update(
        str(row["site_id"]).split(":", 1)[0]
        for row in transfers if row.get("site_id")
    )
    forbidden_fragments.update(
        str(row["kernel"]) for row in transfers if row.get("kernel")
    )
    validate_model_case(model_case)
    validate_source_free_model(
        model_case, label=label, forbidden_fragments=forbidden_fragments
    )
    return internal, model_case


def atomic_write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Compiler schedule coverage",
        "",
        "All classifications below come from schema-6 compiler facts; no source",
        "text or model output is an analyzer input.",
        "",
        "| Case | Transfers | Exact intervals | Producer writes | Host phase | Class | Missing proof families |",
        "|---|---:|---:|---|---|---|---|",
    ]
    for label, case in report["cases"].items():
        facts = case["facts"]
        writes = (
            f"ordinary={facts['ordinary_store_sites']}, "
            f"atomic={facts['atomic_write_sites']}, "
            f"unknown={facts['unknown_write_sites']}"
        )
        missing = ", ".join(facts["missing_proof_families"]) or "none"
        lines.append(
            f"| {label} | {facts['transfer_count']} | "
            f"{facts['exact_host_transfer_intervals']} | {writes} | "
            f"{str(facts['phase_launch_supported']).lower()} | "
            f"{facts['schedule_class']} | {missing} |"
        )
    lines.extend([
        "",
        "`model_graph` deliberately removes paths and kernel/site names. Its",
        "candidate list contains only the content-addressed original schedule; a",
        "compiler oracle is not made model-visible before runtime headroom is",
        "established.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=parse_case, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()
    labels = [label for label, _ in args.case]
    if len(set(labels)) != len(labels):
        raise SystemExit("duplicate case label")

    cases = {}
    model_cases = []
    for label, path in args.case:
        internal, model_case = load_case(label, path)
        cases[label] = internal
        model_cases.append(model_case)
    model_graph = {
        "schema_version": "gicc-source-free-schedule-coverage-v1",
        "cases": sorted(model_cases, key=lambda case: case["case_id"]),
        "policy": {
            "application_source_present": False,
            "model_output_scope": "existing compiler candidate IDs only",
            "provider_call_authorized": False,
        },
    }
    model_graph["graph_id"] = canonical_id(
        "gicc-source-free-schedule-coverage-v1", model_graph
    )
    expected_graph_id = canonical_id(
        "gicc-source-free-schedule-coverage-v1",
        {key: value for key, value in model_graph.items() if key != "graph_id"},
    )
    if model_graph["graph_id"] != expected_graph_id:
        raise CoverageError("model graph ID does not match graph payload")
    report = {
        "schema_version": "gicc-compiler-schedule-coverage-report-v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "cases": cases,
        "model_graph": model_graph,
    }
    atomic_write(args.out, json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.markdown:
        atomic_write(args.markdown, render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
