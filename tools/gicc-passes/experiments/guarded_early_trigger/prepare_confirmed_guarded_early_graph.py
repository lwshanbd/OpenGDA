#!/usr/bin/env python3
"""Expose guarded early trigger after its exact runtime confirmation passes.

The tool replays all three confirmation allocations, binds their dormant
compiler candidate to the frozen baseline/guarded kernel metadata, enriches
the old compiler dossier with only the four confirmed guard facts, and writes
a new content-addressed graph/prompt bundle.  Application source is hash-
checked by the confirmation transition but is never shown to a model or
modified.  This tool has no provider, scheduler, compiler, or source-edit path.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PASS_PYTHON = ROOT / "tools/gicc-passes/python"
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(HERE))

import analyze_guarded_early_trigger_confirmation as confirmation  # noqa: E402
import gicc_comm_group_plan_bridge as groups  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


EXPANSION_SCHEMA = "gicc-guarded-early-trigger-graph-expansion-v1"
CONFIRMATION_SCHEMA = "gicc-guarded-early-trigger-confirmation-v1"
CANDIDATE_KIND = "site_guarded_early_trigger"
TRANSFORM = "GUARDED_EARLY_TRIGGER"
GUARD_FACT_KEYS = {
    "guarded_early_trigger_guardable",
    "guarded_early_trigger_write_params",
    "guarded_early_trigger_unsafe_side_effect_sites",
    "guarded_early_trigger_reason",
}


class ExpansionError(RuntimeError):
    """The guarded confirmation or compiler graph cannot be expanded."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpansionError(f"cannot read JSON {path}: {exc}") from exc


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def json_value(value: Any) -> Any:
    return json.loads(json.dumps(value, sort_keys=True))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise ExpansionError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def recorded_path(value: str) -> Path:
    raw = Path(value)
    return raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()


def input_record(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise ExpansionError(f"missing {role}: {resolved}")
    return {
        "role": role,
        "path": display_path(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def replay_confirmation(
    path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Path]]:
    value = read_json(path)
    if (not isinstance(value, dict)
            or value.get("schema_version") != CONFIRMATION_SCHEMA):
        raise ExpansionError("unexpected guarded-trigger confirmation schema")
    payload = dict(value)
    result_id = payload.pop("result_id", None)
    if result_id != bridge._fingerprint(payload):
        raise ExpansionError("guarded-trigger confirmation result ID changed")
    for key, expected in {
        "model_invoked": False,
        "application_source_modified": False,
        "provider_call_authorized": False,
    }.items():
        if value.get(key) is not expected:
            raise ExpansionError(f"guarded confirmation boundary changed: {key}")
    if value.get("correctness_gate", {}).get("passed") is not True:
        raise ExpansionError("guarded confirmation correctness gate did not pass")
    if value.get("runtime_guard_gate", {}).get("passed") is not True:
        raise ExpansionError("guarded confirmation runtime-guard gate did not pass")
    if value.get("confirmation_gate", {}).get("passed") is not True:
        raise ExpansionError("guarded confirmation performance gate did not pass")
    transition_path = Path(value.get("transition", ""))
    if (not transition_path.is_absolute() or not transition_path.is_file()
            or sha256_file(transition_path) != value.get("transition_sha256")):
        raise ExpansionError("guarded confirmation transition changed")
    transition, transition_paths = confirmation.validate_transition(
        transition_path
    )
    dormant = transition.get("dormant_compiler_candidate", {})
    if (dormant.get("kind") != "guarded_early_trigger"
            or dormant.get("compiler_materializer")
            != "allocation_guarded_phase3_trigger"
            or dormant.get("schedule_phase") != 3
            or dormant.get("model_visible") is not False
            or dormant.get("candidate_id")
            != value.get("dormant_compiler_candidate_id")):
        raise ExpansionError("confirmed guarded compiler candidate changed")
    summaries = value.get("allocation_monitors")
    if (not isinstance(summaries, list) or len(summaries) != 3
            or any(not isinstance(item, dict) for item in summaries)):
        raise ExpansionError("guarded confirmation lacks three monitors")
    monitors = [Path(item.get("monitor", "")) for item in summaries]
    if any(not monitor.is_absolute() for monitor in monitors):
        raise ExpansionError("guarded confirmation monitor path is not absolute")
    regenerated = confirmation.analyze_monitors(transition_path, monitors)
    if regenerated != value:
        raise ExpansionError("guarded confirmation does not replay from raw logs")
    return value, transition, transition_paths


def confirmed_metadata(
    transition: dict[str, Any], paths: dict[str, Path],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    baseline_path = paths.get("baseline_kernel_metadata")
    guarded_path = paths.get("guarded_kernel_metadata")
    audit_path = paths.get("frozen_ir_audit")
    if baseline_path is None or guarded_path is None or audit_path is None:
        raise ExpansionError("guarded transition lacks compiler metadata")
    baseline = read_json(baseline_path)
    guarded = read_json(guarded_path)
    if "guarded_early_trigger_device_materialized" in baseline:
        raise ExpansionError("baseline metadata carries guarded attestation")
    if guarded.get("guarded_early_trigger_device_materialized") is not True:
        raise ExpansionError("guarded metadata lacks device attestation")
    normalized = copy.deepcopy(guarded)
    normalized.pop("guarded_early_trigger_device_materialized")
    if normalized != baseline:
        raise ExpansionError("guarded metadata differs beyond its attestation")
    audit = read_json(audit_path)
    if (audit.get("schema_version")
            != "gicc-guarded-early-trigger-ir-audit-v1"
            or audit.get("passed") is not True):
        raise ExpansionError("guarded final-IR audit did not pass")
    dormant = transition["dormant_compiler_candidate"]
    if baseline.get("kernel_mangled") != dormant.get("kernel_mangled"):
        raise ExpansionError("guarded metadata binds another kernel")
    transfers = [
        op for op in baseline.get("ops", [])
        if isinstance(op, dict) and op.get("kind") == "put_no_db"
    ]
    if len(transfers) != 1:
        raise ExpansionError("guarded metadata must contain one PUT")
    op = transfers[0]
    if op.get("site_id") != dormant.get("transfer_site_id"):
        raise ExpansionError("guarded metadata binds another transfer site")
    frontier = op.get("producer_frontier", {})
    facts = {
        "source_pointer_candidates": frontier.get("source_pointer_candidates"),
        "source_buffer_index_param": frontier.get(
            "source_identity_buffer_index_param"
        ),
        "write_pointer_params": frontier.get(
            "guarded_early_trigger_write_params"
        ),
        "unsafe_side_effect_sites": frontier.get(
            "guarded_early_trigger_unsafe_side_effect_sites"
        ),
        "guardable": frontier.get("guarded_early_trigger_guardable"),
    }
    if facts != dormant.get("guard_facts"):
        raise ExpansionError("guarded metadata and dormant candidate disagree")
    return baseline, guarded, op["site_id"]


def _opportunities(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        opportunity["opportunity_id"]: opportunity
        for opportunity in graph["opportunities"]
    }


def _without_keys(value: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: child for key, child in value.items() if key not in keys}


def expand_graph(
    dossier_value: Any, template_value: Any, graph_value: Any,
    dormant_candidate: dict[str, Any], baseline_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    dossier = bridge._verified_dossier(dossier_value)
    current = json_value(groups.verified_graph(graph_value))
    regenerated = json_value(groups.make_group_graph(dossier, [template_value]))
    if regenerated != current:
        raise ExpansionError(
            "current guarded graph does not regenerate from dossier/template"
        )
    if (groups.verified_templates([template_value])
            != groups.verified_templates([baseline_metadata])):
        raise ExpansionError(
            "confirmed kernel metadata changes the route/template semantics"
        )
    site_id = dormant_candidate.get("transfer_site_id")
    if not isinstance(site_id, str):
        raise ExpansionError("dormant guarded candidate lacks a transfer site")
    sites = [site for site in dossier["sites"] if site.get("site_id") == site_id]
    if len(sites) != 1:
        raise ExpansionError("current dossier does not bind the guarded site")
    metadata_ops = [
        op for op in baseline_metadata.get("ops", [])
        if isinstance(op, dict) and op.get("site_id") == site_id
    ]
    if len(metadata_ops) != 1:
        raise ExpansionError("confirmed metadata does not bind the guarded site")
    old_frontier = sites[0].get("producer_frontier")
    new_frontier = metadata_ops[0].get("producer_frontier")
    if not isinstance(old_frontier, dict) or not isinstance(new_frontier, dict):
        raise ExpansionError("guarded site lacks producer-frontier facts")
    if any(old_frontier[key] != new_frontier[key]
           for key in set(old_frontier).intersection(new_frontier)):
        raise ExpansionError("confirmed metadata changes an existing frontier fact")
    if set(new_frontier) - set(old_frontier) != GUARD_FACT_KEYS:
        raise ExpansionError("confirmed metadata has an unexpected fact delta")
    merged_frontier = copy.deepcopy(old_frontier)
    merged_frontier.update({
        key: copy.deepcopy(new_frontier[key]) for key in GUARD_FACT_KEYS
    })

    dossier_payload = copy.deepcopy(dossier)
    dossier_payload.pop("dossier_id")
    target = next(
        site for site in dossier_payload["sites"] if site["site_id"] == site_id
    )
    target["producer_frontier"] = merged_frontier
    profile = copy.deepcopy(dossier_payload["platform_profile"])
    transforms = profile.get("compiler_transforms", {})
    if not isinstance(transforms, dict):
        raise ExpansionError("compiler_transforms profile must be an object")
    gate = transforms.get("guarded_early_trigger")
    if gate is True:
        raise ExpansionError("guarded early-trigger candidate is already enabled")
    if gate is not None and gate is not False:
        raise ExpansionError("guarded early-trigger profile gate is malformed")
    transforms = copy.deepcopy(transforms)
    transforms["guarded_early_trigger"] = True
    profile["compiler_transforms"] = transforms
    dossier_payload["platform_profile"] = profile
    expanded_dossier = {
        **dossier_payload,
        "dossier_id": bridge._fingerprint(dossier_payload),
    }
    bridge._verified_dossier(expanded_dossier)
    expanded = json_value(
        groups.make_group_graph(expanded_dossier, [template_value])
    )
    groups.verified_graph(expanded)

    old_opportunities = _opportunities(current)
    new_opportunities = _opportunities(expanded)
    if set(old_opportunities) != set(new_opportunities):
        raise ExpansionError("guarded expansion changed compiler opportunities")
    additions = []
    target_opportunity = None
    for opportunity_id, old in old_opportunities.items():
        new = new_opportunities[opportunity_id]
        old_candidates = old["candidates"]
        retained = [
            candidate for candidate in new["candidates"]
            if candidate.get("kind") != CANDIDATE_KIND
        ]
        if retained != old_candidates:
            raise ExpansionError(
                f"guarded expansion changed an old candidate in {opportunity_id}"
            )
        additions.extend(
            candidate for candidate in new["candidates"]
            if candidate.get("kind") == CANDIDATE_KIND
        )
        if site_id in old["site_ids"]:
            target_opportunity = opportunity_id
            normalized = copy.deepcopy(new)
            normalized["candidates"] = retained
            facts = normalized["compiler_facts"]
            facts["site_semantic_facts"][site_id]["producer_frontier"] = (
                copy.deepcopy(old_frontier)
            )
            facts["dependence_legality"].pop(
                "guarded_early_trigger_legal", None
            )
            if normalized != old:
                raise ExpansionError(
                    "guarded expansion changed unrelated compiler facts"
                )
        elif new != old:
            raise ExpansionError(
                f"guarded expansion changed unrelated opportunity {opportunity_id}"
            )
    if target_opportunity is None or len(additions) != 1:
        raise ExpansionError("guarded expansion must expose exactly one candidate")
    candidate = additions[0]
    materializer = candidate.get("materializer", {}).get("sites", {})
    if materializer != {
        site_id: {"dispatch": "DWQ_TRIGGER", "transform": TRANSFORM}
    }:
        raise ExpansionError("guarded candidate materializer changed")
    if current.get("fixed_sites") != expanded.get("fixed_sites"):
        raise ExpansionError("guarded expansion changed fixed sites")
    if _without_keys(
        current, {"graph_id", "compiler_inputs", "platform_profile",
                  "opportunities"},
    ) != _without_keys(
        expanded, {"graph_id", "compiler_inputs", "platform_profile",
                   "opportunities"},
    ):
        raise ExpansionError("guarded expansion changed another graph field")
    change = {
        "opportunity_id": target_opportunity,
        "candidate_id": candidate["candidate_id"],
        "candidate_kind": CANDIDATE_KIND,
        "materializer_transform": TRANSFORM,
        "dormant_schedule_candidate_id": dormant_candidate["candidate_id"],
        "confirmed_guard_fact_keys": sorted(GUARD_FACT_KEYS),
        "prior_selectable_candidate_count": sum(
            len(item["candidates"]) for item in current["opportunities"]
        ),
        "expanded_selectable_candidate_count": sum(
            len(item["candidates"]) for item in expanded["opportunities"]
        ),
        "prior_candidate_ids_preserved": True,
        "only_confirmed_candidate_exposed": True,
    }
    return expanded_dossier, expanded, change


def build_bundle(
    confirmation_path: Path, dossier_path: Path, template_path: Path,
    graph_path: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    confirmation_value, transition, transition_paths = replay_confirmation(
        confirmation_path.resolve()
    )
    baseline_metadata, _, _ = confirmed_metadata(
        transition, transition_paths
    )
    dossier_value = read_json(dossier_path)
    expanded_dossier, expanded_graph, change = expand_graph(
        dossier_value, read_json(template_path), read_json(graph_path),
        transition["dormant_compiler_candidate"], baseline_metadata,
    )
    outputs = {
        "expanded_dossier": (
            "expanded-dossier.json", json_text(expanded_dossier),
        ),
        "expanded_graph": (
            "expanded-group-graph.json", json_text(expanded_graph),
        ),
        "response_schema": (
            "response-schema.json",
            json_text(groups.decision_response_schema(expanded_graph)),
        ),
    }
    for view_kind in groups.MODEL_VIEW_KINDS:
        outputs[f"model_view_{view_kind}"] = (
            f"model-view-{view_kind}.json",
            json_text(groups.model_view(expanded_graph, view_kind)),
        )
        outputs[f"prompt_{view_kind}"] = (
            f"prompt-{view_kind}.txt",
            groups.render_prompt(expanded_graph, view_kind),
        )
    rendered = {name: text for _, (name, text) in outputs.items()}
    output_records = []
    for role, (name, text) in sorted(outputs.items()):
        encoded = text.encode("utf-8")
        output_records.append({
            "role": role, "path": name,
            "sha256": sha256_bytes(encoded), "bytes": len(encoded),
        })
    inputs = [
        input_record(confirmation_path, "passed_confirmation_analysis"),
        input_record(dossier_path, "frozen_current_dossier"),
        input_record(template_path, "frozen_current_template"),
        input_record(graph_path, "frozen_current_graph"),
        input_record(
            transition_paths["baseline_kernel_metadata"],
            "confirmed_baseline_kernel_metadata",
        ),
        input_record(
            transition_paths["guarded_kernel_metadata"],
            "confirmed_guarded_kernel_metadata",
        ),
        input_record(
            transition_paths["frozen_ir_audit"], "confirmed_final_ir_audit",
        ),
    ]
    payload = {
        "schema_version": EXPANSION_SCHEMA,
        "status": "expanded_graph_ready_for_suite_refreeze",
        "confirmation": {
            "result_id": confirmation_value["result_id"],
            "transition_id": transition["transition_id"],
            "dormant_schedule_candidate_id": confirmation_value[
                "dormant_compiler_candidate_id"
            ],
            "confirmation_gate_passed": True,
            "correctness_gate_passed": True,
            "runtime_guard_gate_passed": True,
            "raw_allocation_monitors_replayed": 3,
        },
        "graph_transition": {
            "current_dossier_id": dossier_value["dossier_id"],
            "expanded_dossier_id": expanded_dossier["dossier_id"],
            "current_graph_id": read_json(graph_path)["graph_id"],
            "expanded_graph_id": expanded_graph["graph_id"],
            **change,
        },
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_hash_verified": True,
            "application_source_visible_to_model": False,
            "application_source_modified": False,
            "model_invoked": False,
            "provider_call_authorized": False,
            "scheduler_job_submitted": False,
            "frozen_current_graph_modified": False,
            "current_decision_suite_modified": False,
        },
        "next_stage": (
            "independently refreeze and audit a new compiler decision suite; "
            "then freeze an exact provider request and seek separate authorization"
        ),
        "inputs": sorted(inputs, key=lambda item: item["role"]),
        "outputs": output_records,
    }
    manifest = {**payload, "expansion_id": bridge._fingerprint(payload)}
    rendered["manifest.json"] = json_text(manifest)
    return rendered, manifest


def write_bundle(path: Path, rendered: dict[str, str]) -> None:
    output = path.resolve()
    if output.exists():
        raise ExpansionError(f"refusing to overwrite guarded expansion: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
    ))
    try:
        for name, text in rendered.items():
            (temporary / name).write_text(text, encoding="utf-8")
        if output.exists():
            raise ExpansionError(
                f"refusing to overwrite guarded expansion: {output}"
            )
        os.rename(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def verify_contained(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if (not isinstance(manifest, dict)
            or manifest.get("schema_version") != EXPANSION_SCHEMA):
        raise ExpansionError("unexpected guarded-expansion manifest schema")
    payload = dict(manifest)
    expansion_id = payload.pop("expansion_id", None)
    if expansion_id != bridge._fingerprint(payload):
        raise ExpansionError("guarded-expansion ID changed")
    records = manifest.get("inputs")
    if not isinstance(records, list):
        raise ExpansionError("guarded expansion lacks input records")
    paths = {}
    for record in records:
        if (not isinstance(record, dict)
                or not isinstance(record.get("role"), str)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)):
            raise ExpansionError("guarded expansion has invalid input record")
        role = record["role"]
        path = recorded_path(record["path"])
        if role in paths or not path.is_file():
            raise ExpansionError(f"missing or duplicate guarded input: {role}")
        if (sha256_file(path) != record["sha256"]
                or path.stat().st_size != record["bytes"]):
            raise ExpansionError(f"guarded-expansion input changed: {path}")
        paths[role] = path
    required = {
        "passed_confirmation_analysis", "frozen_current_dossier",
        "frozen_current_template", "frozen_current_graph",
        "confirmed_baseline_kernel_metadata",
        "confirmed_guarded_kernel_metadata", "confirmed_final_ir_audit",
    }
    if set(paths) != required:
        raise ExpansionError("guarded expansion input roles changed")
    rendered, regenerated = build_bundle(
        paths["passed_confirmation_analysis"],
        paths["frozen_current_dossier"], paths["frozen_current_template"],
        paths["frozen_current_graph"],
    )
    if regenerated != manifest:
        raise ExpansionError("guarded-expansion manifest does not regenerate")
    output_dir = manifest_path.resolve().parent
    if {path.name for path in output_dir.iterdir()} != set(rendered):
        raise ExpansionError("guarded-expansion output set changed")
    for name, expected in rendered.items():
        path = output_dir / name
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise ExpansionError(f"guarded-expansion output changed: {path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--confirmation-analysis", type=Path, required=True)
    prepare.add_argument("--dossier", type=Path, required=True)
    prepare.add_argument("--template", type=Path, required=True)
    prepare.add_argument("--graph", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify-contained")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            rendered, manifest = build_bundle(
                args.confirmation_analysis, args.dossier,
                args.template, args.graph,
            )
            write_bundle(args.output_dir, rendered)
            action = "prepared"
        else:
            manifest = verify_contained(args.manifest)
            action = "verified-contained"
        print(
            f"guarded-early-graph-expansion: {action}; model_invoked=false; "
            f"provider_call_authorized=false; scheduler_job_submitted=false; "
            f"expansion_id={manifest['expansion_id']}"
        )
        return 0
    except (
        ExpansionError, confirmation.ConfirmError,
        confirmation.common.MonitorError, groups.GroupPlanError,
        bridge.BridgeError, OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"guarded-early-graph-expansion: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
