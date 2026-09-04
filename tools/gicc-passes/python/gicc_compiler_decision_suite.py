#!/usr/bin/env python3
"""Freeze independent compiler/LTO decision tasks into one audited suite.

The suite is an index, not a joint optimizer.  It content-addresses collective
size-policy graphs and communication route/schedule graphs, renders the same
three information views for each graph, and records their exact prompt hashes.
It has no provider, scheduler, compiler, or source-edit path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Callable

import gicc_collective_plan_bridge as collective
import gicc_comm_group_plan_bridge as communication
import gicc_comm_plan_bridge as structural
import gicc_llm_bridge as bridge


SUITE_SCHEMA = "gicc-compiler-decision-suite-v1"
VIEW_KINDS = ("relational", "descriptors", "opaque")
LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
SUITE_BOUNDARY = {
    "compiler_lto_decisions_only": True,
    "source_visible": False,
    "model_may_generate_code_or_ir": False,
    "model_output_is_graph_bound_ids": True,
    "compiler_revalidates_before_materialization": True,
    "provider_call_supported": False,
}
SUITE_COMPOSITION = {
    "entries_are_independent": True,
    "cross_entry_joint_selection": False,
    "cross_entry_cartesian_product_claimed": False,
    "policy_counts_are_reported_per_entry": True,
}


class SuiteError(ValueError):
    """A graph or suite entry violates the frozen compiler-only contract."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SuiteError(f"cannot read JSON {path}: {exc}") from exc


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _file_sha256(path: Path) -> str:
    try:
        return _sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise SuiteError(f"cannot hash {path}: {exc}") from exc


def _id_set_sha256(values: list[str]) -> str:
    encoded = "\n".join(sorted(values)).encode("utf-8")
    return _sha256_bytes(encoded)


def _prompt_views(
    graph: dict[str, Any],
    *,
    model_view: Callable[[Any, str], dict[str, Any]],
    render_prompt: Callable[[Any, str], str],
    label: str,
    prompt_dir: Path | None,
) -> dict[str, Any]:
    records = {}
    for view_kind in VIEW_KINDS:
        view = model_view(graph, view_kind)
        prompt = render_prompt(graph, view_kind)
        encoded = prompt.encode("utf-8")
        records[view_kind] = {
            "model_view_id": bridge._fingerprint(view),
            "prompt_sha256": _sha256_bytes(encoded),
            "prompt_bytes": len(encoded),
        }
        if prompt_dir is not None:
            bridge._write_text_atomic(
                prompt_dir / label / f"{view_kind}.txt", prompt
            )
    return records


def _response_schema_record(
    schema: dict[str, Any], *, label: str, prompt_dir: Path | None,
) -> dict[str, Any]:
    rendered = json.dumps(schema, indent=2, sort_keys=True) + "\n"
    encoded = rendered.encode("utf-8")
    record = {
        "response_schema_id": bridge._fingerprint(schema),
        "file_sha256": _sha256_bytes(encoded),
        "bytes": len(encoded),
    }
    if prompt_dir is not None:
        bridge._write_text_atomic(
            prompt_dir / label / "response-schema.json", rendered
        )
    return record


def _communication_entry(
    label: str, path: Path, prompt_dir: Path | None,
) -> dict[str, Any]:
    graph = communication.verified_graph(_read_json(path))
    candidate_ids = [
        candidate["candidate_id"]
        for opportunity in graph["opportunities"]
        for candidate in opportunity["candidates"]
    ]
    per_opportunity = [
        {
            "opportunity_id": opportunity["opportunity_id"],
            "kind": opportunity["kind"],
            "selectable_candidate_count": len(opportunity["candidates"]),
            "masked_candidate_count": len(
                opportunity.get("masked_candidates", [])
            ),
        }
        for opportunity in graph["opportunities"]
    ]
    payload = {
        "label": label,
        "decision_family": "communication_route_or_schedule",
        "graph_schema": graph["schema_version"],
        "graph_id": graph["graph_id"],
        "graph_file_sha256": _file_sha256(path),
        "decision_space": {
            "opportunity_count": len(per_opportunity),
            "per_opportunity": per_opportunity,
            "independent_policy_count": math.prod(
                item["selectable_candidate_count"] for item in per_opportunity
            ),
            "selectable_candidate_id_count": len(candidate_ids),
            "selectable_candidate_id_set_sha256": _id_set_sha256(
                candidate_ids
            ),
            "masked_candidate_count": sum(
                item["masked_candidate_count"] for item in per_opportunity
            ),
        },
        "response_schema": _response_schema_record(
            communication.decision_response_schema(graph),
            label=label,
            prompt_dir=prompt_dir,
        ),
        "views": _prompt_views(
            graph,
            model_view=communication.model_view,
            render_prompt=communication.render_prompt,
            label=label,
            prompt_dir=prompt_dir,
        ),
    }
    return {"entry_id": bridge._fingerprint(payload), **payload}


def _collective_entry(
    label: str, path: Path, prompt_dir: Path | None,
) -> dict[str, Any]:
    graph = collective.verified_graph(_read_json(path))
    option_ids = [
        option["option_id"]
        for opportunity in graph["opportunities"]
        for slot in opportunity["decision_slots"]
        for option in slot["options"]
    ]
    target_ids = {
        option["target_id"]
        for opportunity in graph["opportunities"]
        for slot in opportunity["decision_slots"]
        for option in slot["options"]
    }
    per_opportunity = [
        {
            "opportunity_id": opportunity["opportunity_id"],
            "kind": opportunity["kind"],
            "decision_slot_count": len(opportunity["decision_slots"]),
            "target_count": len(opportunity["decision_slots"][0]["options"]),
            "joint_policy_count": opportunity["joint_action_space_size"],
        }
        for opportunity in graph["opportunities"]
    ]
    payload = {
        "label": label,
        "decision_family": "collective_algorithm_and_size_policy",
        "graph_schema": graph["schema_version"],
        "graph_id": graph["graph_id"],
        "graph_file_sha256": _file_sha256(path),
        "decision_space": {
            "opportunity_count": len(per_opportunity),
            "per_opportunity": per_opportunity,
            "independent_policy_count": math.prod(
                item["joint_policy_count"] for item in per_opportunity
            ),
            "decision_slot_count": sum(
                item["decision_slot_count"] for item in per_opportunity
            ),
            "target_id_count": len(target_ids),
            "selectable_option_id_count": len(option_ids),
            "selectable_option_id_set_sha256": _id_set_sha256(option_ids),
        },
        "response_schema": _response_schema_record(
            collective.decision_response_schema(graph),
            label=label,
            prompt_dir=prompt_dir,
        ),
        "views": _prompt_views(
            graph,
            model_view=collective._model_view,
            render_prompt=collective.render_prompt,
            label=label,
            prompt_dir=prompt_dir,
        ),
    }
    return {"entry_id": bridge._fingerprint(payload), **payload}


def _structural_entry(
    label: str, path: Path, prompt_dir: Path | None,
) -> dict[str, Any]:
    graph = structural.verified_graph(_read_json(path))
    candidate_ids = [
        candidate["candidate_id"]
        for opportunity in graph["opportunities"]
        for candidate in opportunity["candidates"]
    ]
    per_opportunity = [
        {
            "opportunity_id": opportunity["opportunity_id"],
            "kind": opportunity["kind"],
            "selectable_candidate_count": len(opportunity["candidates"]),
        }
        for opportunity in graph["opportunities"]
    ]
    payload = {
        "label": label,
        "decision_family": (
            "communication_coalescing_and_trigger_placement"
        ),
        "graph_schema": graph["schema_version"],
        "graph_id": graph["graph_id"],
        "graph_file_sha256": _file_sha256(path),
        "decision_space": {
            "opportunity_count": len(per_opportunity),
            "fixed_site_count": len(graph.get("fixed_sites", [])),
            "per_opportunity": per_opportunity,
            "independent_policy_count": math.prod(
                item["selectable_candidate_count"]
                for item in per_opportunity
            ),
            "selectable_candidate_id_count": len(candidate_ids),
            "selectable_candidate_id_set_sha256": _id_set_sha256(
                candidate_ids
            ),
        },
        "response_schema": _response_schema_record(
            structural.decision_response_schema(graph),
            label=label,
            prompt_dir=prompt_dir,
        ),
        "views": _prompt_views(
            graph,
            model_view=structural.model_view,
            render_prompt=structural.render_prompt,
            label=label,
            prompt_dir=prompt_dir,
        ),
    }
    return {"entry_id": bridge._fingerprint(payload), **payload}


def make_suite(
    communication_graphs: list[tuple[str, Path]],
    collective_graphs: list[tuple[str, Path]],
    structural_graphs: list[tuple[str, Path]] | None = None,
    *,
    prompt_dir: Path | None = None,
) -> dict[str, Any]:
    if (tuple(communication.MODEL_VIEW_KINDS) != VIEW_KINDS
            or tuple(collective.MODEL_VIEW_KINDS) != VIEW_KINDS
            or tuple(structural.MODEL_VIEW_KINDS) != VIEW_KINDS):
        raise SuiteError("decision bridges do not expose the same view kinds")
    specs = [
        *(('communication', label, path)
          for label, path in communication_graphs),
        *(('collective', label, path) for label, path in collective_graphs),
        *(('structural', label, path)
          for label, path in (structural_graphs or [])),
    ]
    if not specs:
        raise SuiteError("suite requires at least one compiler decision graph")
    labels = [label for _, label, _ in specs]
    if any(not LABEL_RE.fullmatch(label) for label in labels):
        raise SuiteError("suite labels must match [a-z0-9][a-z0-9_.-]*")
    if len(set(labels)) != len(labels):
        raise SuiteError("suite labels must be unique across decision families")
    entries = []
    for family, label, path in specs:
        if family == "communication":
            entries.append(_communication_entry(label, path, prompt_dir))
        elif family == "collective":
            entries.append(_collective_entry(label, path, prompt_dir))
        else:
            entries.append(_structural_entry(label, path, prompt_dir))
    entries.sort(key=lambda item: item["label"])
    family_counts: dict[str, int] = {}
    for entry in entries:
        family_counts[entry["decision_family"]] = (
            family_counts.get(entry["decision_family"], 0) + 1
        )
    payload = {
        "schema_version": SUITE_SCHEMA,
        "boundary": dict(SUITE_BOUNDARY),
        "composition": dict(SUITE_COMPOSITION),
        "entry_count": len(entries),
        "decision_family_counts": dict(sorted(family_counts.items())),
        "entries": entries,
    }
    return {"suite_id": bridge._fingerprint(payload), **payload}


def verified_suite(value: Any, prompt_dir: Path | None = None) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != SUITE_SCHEMA:
        raise SuiteError(f"expected suite schema {SUITE_SCHEMA}")
    suite_id = value.get("suite_id")
    payload = dict(value)
    payload.pop("suite_id", None)
    if not isinstance(suite_id, str) or suite_id != bridge._fingerprint(payload):
        raise SuiteError("suite_id does not match suite content")
    if value.get("boundary") != SUITE_BOUNDARY:
        raise SuiteError("suite boundary is not compiler-only")
    if value.get("composition") != SUITE_COMPOSITION:
        raise SuiteError("suite composition does not preserve independence")
    entries = value.get("entries")
    if (not isinstance(entries, list) or not entries
            or value.get("entry_count") != len(entries)):
        raise SuiteError("suite entries are missing or inconsistent")
    labels = [entry.get("label") for entry in entries if isinstance(entry, dict)]
    if len(labels) != len(entries) or len(set(labels)) != len(labels):
        raise SuiteError("suite entries must have unique labels")
    observed_families: dict[str, int] = {}
    for entry in entries:
        entry_id = entry.get("entry_id")
        entry_payload = dict(entry)
        entry_payload.pop("entry_id", None)
        if (not isinstance(entry_id, str)
                or entry_id != bridge._fingerprint(entry_payload)):
            raise SuiteError(f"{entry.get('label')}: invalid entry_id")
        family = entry.get("decision_family")
        expected_schema = {
            "communication_route_or_schedule": communication.GRAPH_SCHEMA,
            "collective_algorithm_and_size_policy": collective.GRAPH_SCHEMA,
            "communication_coalescing_and_trigger_placement":
                structural.GRAPH_SCHEMA,
        }.get(family)
        if expected_schema is None or entry.get("graph_schema") != expected_schema:
            raise SuiteError(
                f"{entry.get('label')}: invalid decision family or graph schema"
            )
        observed_families[family] = observed_families.get(family, 0) + 1
        views = entry.get("views")
        if not isinstance(views, dict) or set(views) != set(VIEW_KINDS):
            raise SuiteError(f"{entry.get('label')}: incomplete prompt views")
        for view_kind, record in views.items():
            digest = record.get("prompt_sha256") if isinstance(record, dict) else None
            size = record.get("prompt_bytes") if isinstance(record, dict) else None
            view_id = record.get("model_view_id") if isinstance(record, dict) else None
            if (not isinstance(digest, str)
                    or not re.fullmatch(r"[0-9a-f]{64}", digest)
                    or not isinstance(view_id, str)
                    or not re.fullmatch(r"sha256:[0-9a-f]{64}", view_id)
                    or isinstance(size, bool) or not isinstance(size, int)
                    or size <= 0):
                raise SuiteError(
                    f"{entry.get('label')}/{view_kind}: invalid prompt record"
                )
            if prompt_dir is not None:
                path = prompt_dir / entry["label"] / f"{view_kind}.txt"
                if _file_sha256(path) != digest or path.stat().st_size != size:
                    raise SuiteError(
                        f"{entry['label']}/{view_kind}: prompt content mismatch"
                    )
        response_schema = entry.get("response_schema")
        schema_id = (
            response_schema.get("response_schema_id")
            if isinstance(response_schema, dict) else None
        )
        digest = (
            response_schema.get("file_sha256")
            if isinstance(response_schema, dict) else None
        )
        size = (
            response_schema.get("bytes")
            if isinstance(response_schema, dict) else None
        )
        if (not isinstance(schema_id, str)
                or not re.fullmatch(r"sha256:[0-9a-f]{64}", schema_id)
                or not isinstance(digest, str)
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or isinstance(size, bool) or not isinstance(size, int)
                or size <= 0):
            raise SuiteError(
                f"{entry.get('label')}: invalid response schema record"
            )
        if prompt_dir is not None:
            path = prompt_dir / entry["label"] / "response-schema.json"
            if _file_sha256(path) != digest or path.stat().st_size != size:
                raise SuiteError(
                    f"{entry['label']}: response schema content mismatch"
                )
    if value.get("decision_family_counts") != dict(
        sorted(observed_families.items())
    ):
        raise SuiteError("decision family counts are inconsistent")
    return value


def _spec(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=GRAPH.json")
    if not LABEL_RE.fullmatch(label):
        raise argparse.ArgumentTypeError(
            "LABEL must match [a-z0-9][a-z0-9_.-]*"
        )
    return label, Path(raw_path)


def _emit(args: argparse.Namespace) -> int:
    value = make_suite(
        args.communication, args.collective, args.structural,
        prompt_dir=args.prompt_dir,
    )
    bridge._write_json_atomic(args.out, value)
    print(
        f"gicc-compiler-decision-suite: wrote {value['entry_count']} "
        f"independent entries; suite_id={value['suite_id']}",
        file=sys.stderr,
    )
    return 0


def _verify(args: argparse.Namespace) -> int:
    value = verified_suite(_read_json(args.suite), args.prompt_dir)
    print(
        f"gicc-compiler-decision-suite: verified {value['entry_count']} "
        f"independent entries; suite_id={value['suite_id']}",
        file=sys.stderr,
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    emit = sub.add_parser("emit")
    emit.add_argument(
        "--communication", type=_spec, action="append", default=[],
        metavar="LABEL=GRAPH.json",
    )
    emit.add_argument(
        "--collective", type=_spec, action="append", default=[],
        metavar="LABEL=GRAPH.json",
    )
    emit.add_argument(
        "--structural", type=_spec, action="append", default=[],
        metavar="LABEL=GRAPH.json",
    )
    emit.add_argument("--prompt-dir", type=Path)
    emit.add_argument("--out", type=Path, required=True)
    emit.set_defaults(run=_emit)
    verify = sub.add_parser("verify")
    verify.add_argument("--suite", type=Path, required=True)
    verify.add_argument("--prompt-dir", type=Path)
    verify.set_defaults(run=_verify)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.run(args))
    except (
        SuiteError, communication.GroupPlanError,
        collective.CollectivePlanError, structural.PlanBridgeError,
        OSError, ValueError,
    ) as exc:
        print(f"gicc-compiler-decision-suite: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
