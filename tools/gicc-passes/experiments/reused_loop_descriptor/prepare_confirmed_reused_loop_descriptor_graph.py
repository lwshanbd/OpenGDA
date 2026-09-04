#!/usr/bin/env python3
"""Expose descriptor reuse only after its exact runtime confirmation passes.

The tool replays all three confirmation allocations, binds the dormant
compiler candidate to the frozen feature and kernel metadata, enriches the old
compiler dossier with only the confirmed i32 loop-bound type, and writes a new
content-addressed graph/prompt bundle.  The transition rechecks the application
source hash, but this tool has no compiler, scheduler, provider, source-
visibility, or source-edit path.
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

import analyze_reused_loop_descriptor_confirmation as confirmation  # noqa: E402
import gicc_comm_group_plan_bridge as groups  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


EXPANSION_SCHEMA = "gicc-reused-loop-descriptor-graph-expansion-v1"
CONFIRMATION_SCHEMA = "gicc-reused-loop-descriptor-confirmation-v1"
CANDIDATE_KIND = "trigger_reused_descriptor_loop"
TRANSFORM = "REUSE_LOOP_DESCRIPTOR"
LOOP_FACT_KEYS = {"bound_param_type"}


class ExpansionError(RuntimeError):
    """The confirmation or compiler graph cannot support expansion."""


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
        raise ExpansionError("unexpected reused-descriptor confirmation schema")
    payload = dict(value)
    result_id = payload.pop("result_id", None)
    if result_id != bridge._fingerprint(payload):
        raise ExpansionError("reused-descriptor confirmation result ID changed")
    for key, expected in {
        "model_invoked": False,
        "application_source_visible_to_model": False,
        "application_source_modified": False,
        "provider_call_authorized": False,
    }.items():
        if value.get(key) is not expected:
            raise ExpansionError(
                f"reused-descriptor confirmation boundary changed: {key}"
            )
    if value.get("correctness_gate", {}).get("passed") is not True:
        raise ExpansionError("reused-descriptor correctness gate did not pass")
    if value.get("confirmation_gate", {}).get("passed") is not True:
        raise ExpansionError("reused-descriptor performance gate did not pass")
    transition_path = Path(value.get("transition", ""))
    if (not transition_path.is_absolute() or not transition_path.is_file()
            or sha256_file(transition_path) != value.get("transition_sha256")):
        raise ExpansionError("reused-descriptor confirmation transition changed")
    transition, transition_paths = confirmation.validate_transition(
        transition_path
    )
    dormant = transition.get("dormant_compiler_candidate", {})
    if (dormant.get("kind") != CANDIDATE_KIND
            or dormant.get("compiler_materializer") != TRANSFORM
            or dormant.get("model_visible") is not False
            or dormant.get("candidate_id")
            != value.get("dormant_compiler_candidate_id")):
        raise ExpansionError("confirmed reused-descriptor candidate changed")
    summaries = value.get("allocation_monitors")
    if (not isinstance(summaries, list) or len(summaries) != 3
            or any(not isinstance(item, dict) for item in summaries)):
        raise ExpansionError("reused-descriptor confirmation lacks three monitors")
    monitors = [Path(item.get("monitor", "")) for item in summaries]
    if any(not monitor.is_absolute() for monitor in monitors):
        raise ExpansionError("reused-descriptor monitor path is not absolute")
    regenerated = confirmation.analyze_monitors(transition_path, monitors)
    if regenerated != value:
        raise ExpansionError(
            "reused-descriptor confirmation does not replay from raw logs"
        )
    return value, transition, transition_paths


def confirmed_compiler_artifacts(
    transition: dict[str, Any], paths: dict[str, Path],
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    required = {
        "baseline_features", "reused_features",
        "baseline_kernel_metadata", "reused_kernel_metadata",
        "frozen_ir_audit",
    }
    if not required.issubset(paths):
        raise ExpansionError("reused transition lacks compiler artifacts")
    baseline_features = read_json(paths["baseline_features"])
    reused_features = read_json(paths["reused_features"])
    if baseline_features != reused_features:
        raise ExpansionError("reused build changes compiler feature facts")
    if not isinstance(baseline_features, list):
        raise ExpansionError("reused compiler features must be a list")
    baseline_metadata = read_json(paths["baseline_kernel_metadata"])
    reused_metadata = read_json(paths["reused_kernel_metadata"])
    if baseline_metadata != reused_metadata:
        raise ExpansionError("reused build changes kernel metadata")
    audit = read_json(paths["frozen_ir_audit"])
    if (audit.get("schema_version")
            != "gicc-reused-loop-descriptor-ir-audit-v1"
            or audit.get("passed") is not True
            or audit.get("host_only_device_identity") is not True):
        raise ExpansionError("reused-descriptor final-IR audit did not pass")

    dormant = transition["dormant_compiler_candidate"]
    site_id = dormant.get("transfer_site_id")
    if (not isinstance(site_id, str)
            or baseline_metadata.get("kernel_mangled")
            != dormant.get("kernel_mangled")):
        raise ExpansionError("reused compiler artifacts bind another kernel")
    feature_rows = [
        row for row in baseline_features
        if isinstance(row, dict) and row.get("site_id") == site_id
    ]
    metadata_rows = [
        row for row in baseline_metadata.get("ops", [])
        if isinstance(row, dict) and row.get("site_id") == site_id
    ]
    if len(feature_rows) != 1 or len(metadata_rows) != 1:
        raise ExpansionError("reused compiler artifacts do not bind one site")
    proof = dormant.get("compiler_proof", {})
    if (feature_rows[0].get("loop") != proof.get("loop")
            or metadata_rows[0].get("args")
            != proof.get("descriptor_arguments")
            or proof.get("network_operation_order_preserved") is not True
            or proof.get("network_operation_count") != {
                "kind": "runtime_loop_bound", "kernel_param_index": 4,
            }):
        raise ExpansionError("reused compiler proof and artifacts disagree")
    return baseline_features, baseline_metadata, site_id


def _opportunities(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        opportunity["opportunity_id"]: opportunity
        for opportunity in graph["opportunities"]
    }


def _without_keys(value: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: child for key, child in value.items() if key not in keys}


def expand_graph(
    dossier_value: Any, template_value: Any, graph_value: Any,
    dormant_candidate: dict[str, Any], confirmed_features: list[dict[str, Any]],
    confirmed_metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Add one confirmed compiler fact and expose only descriptor reuse."""
    dossier = bridge._verified_dossier(dossier_value)
    current = json_value(groups.verified_graph(graph_value))
    regenerated = json_value(groups.make_group_graph(dossier, [template_value]))
    if regenerated != current:
        raise ExpansionError(
            "current loop graph does not regenerate from dossier/template"
        )
    if (groups.verified_templates([template_value])
            != groups.verified_templates([confirmed_metadata])):
        raise ExpansionError(
            "confirmed kernel metadata changes route/template semantics"
        )
    site_id = dormant_candidate.get("transfer_site_id")
    if not isinstance(site_id, str):
        raise ExpansionError("dormant reused candidate lacks a transfer site")

    confirmed_dossier = bridge.make_dossier(
        confirmed_features, dossier["platform_profile"],
    )
    confirmed_sites = [
        site for site in confirmed_dossier["sites"]
        if site.get("site_id") == site_id
    ]
    old_sites = [
        site for site in dossier["sites"] if site.get("site_id") == site_id
    ]
    if len(confirmed_sites) != 1 or len(old_sites) != 1:
        raise ExpansionError("current and confirmed dossiers do not bind one site")
    confirmed_loop = confirmed_sites[0].get("loop")
    old_loop = old_sites[0].get("loop")
    if not isinstance(confirmed_loop, dict) or not isinstance(old_loop, dict):
        raise ExpansionError("reused-descriptor site lacks loop facts")
    if set(confirmed_loop) - set(old_loop) != LOOP_FACT_KEYS:
        raise ExpansionError("confirmed loop metadata has an unexpected fact delta")
    normalized = copy.deepcopy(confirmed_dossier)
    normalized.pop("dossier_id")
    target = next(
        site for site in normalized["sites"] if site["site_id"] == site_id
    )
    for key in LOOP_FACT_KEYS:
        target["loop"].pop(key)
    old_payload = copy.deepcopy(dossier)
    old_payload.pop("dossier_id")
    if json_value(normalized) != json_value(old_payload):
        raise ExpansionError("confirmed compiler facts change more than loop type")
    if confirmed_loop.get("bound_param_type") != "i32":
        raise ExpansionError("confirmed loop bound is not an i32 formal")

    expanded_payload = copy.deepcopy(confirmed_dossier)
    expanded_payload.pop("dossier_id")
    profile = copy.deepcopy(expanded_payload["platform_profile"])
    transforms = profile.get("compiler_transforms", {})
    if not isinstance(transforms, dict):
        raise ExpansionError("compiler_transforms profile must be an object")
    gate = transforms.get("reused_loop_descriptor")
    if gate is True:
        raise ExpansionError("reused-descriptor candidate is already enabled")
    if gate is not None and gate is not False:
        raise ExpansionError("reused-descriptor profile gate is malformed")
    transforms = copy.deepcopy(transforms)
    transforms["reused_loop_descriptor"] = True
    profile["compiler_transforms"] = transforms
    expanded_payload["platform_profile"] = profile
    expanded_dossier = {
        **expanded_payload,
        "dossier_id": bridge._fingerprint(expanded_payload),
    }
    bridge._verified_dossier(expanded_dossier)
    expanded = json_value(
        groups.make_group_graph(expanded_dossier, [template_value])
    )
    groups.verified_graph(expanded)

    old_opportunities = _opportunities(current)
    new_opportunities = _opportunities(expanded)
    if set(old_opportunities) != set(new_opportunities):
        raise ExpansionError("reused expansion changed compiler opportunities")
    additions = []
    target_opportunity = None
    for opportunity_id, old in old_opportunities.items():
        new = new_opportunities[opportunity_id]
        retained = [
            candidate for candidate in new["candidates"]
            if candidate.get("kind") != CANDIDATE_KIND
        ]
        if retained != old["candidates"]:
            raise ExpansionError(
                f"reused expansion changed an old candidate in {opportunity_id}"
            )
        additions.extend(
            candidate for candidate in new["candidates"]
            if candidate.get("kind") == CANDIDATE_KIND
        )
        if site_id in old["site_ids"]:
            target_opportunity = opportunity_id
            normalized_opportunity = copy.deepcopy(new)
            normalized_opportunity["candidates"] = retained
            facts = normalized_opportunity["compiler_facts"]
            facts["site_semantic_facts"][site_id]["loop"].pop(
                "bound_param_type"
            )
            facts["dependence_legality"].pop(
                "reused_loop_descriptor_legal", None
            )
            if normalized_opportunity != old:
                raise ExpansionError(
                    "reused expansion changed unrelated compiler facts"
                )
        elif new != old:
            raise ExpansionError(
                f"reused expansion changed unrelated opportunity {opportunity_id}"
            )
    if target_opportunity is None or len(additions) != 1:
        raise ExpansionError("reused expansion must expose exactly one candidate")
    candidate = additions[0]
    materializer = candidate.get("materializer", {}).get("sites", {})
    if materializer != {
        site_id: {"dispatch": "DWQ_TRIGGER", "transform": TRANSFORM}
    }:
        raise ExpansionError("reused-descriptor materializer changed")
    expected_count = dormant_candidate.get("compiler_proof", {}).get(
        "network_operation_count"
    )
    if (candidate.get("effects", {}).get("network_operations")
            != expected_count
            or candidate.get("effects", {}).get("host_descriptor_instances")
            != expected_count
            or candidate.get("effects", {}).get("caller_descriptor_arrays")
            != 0):
        raise ExpansionError("reused-descriptor effect contract changed")
    if current.get("fixed_sites") != expanded.get("fixed_sites"):
        raise ExpansionError("reused expansion changed fixed sites")
    if _without_keys(
        current, {"graph_id", "compiler_inputs", "platform_profile",
                  "opportunities"},
    ) != _without_keys(
        expanded, {"graph_id", "compiler_inputs", "platform_profile",
                   "opportunities"},
    ):
        raise ExpansionError("reused expansion changed another graph field")
    change = {
        "opportunity_id": target_opportunity,
        "candidate_id": candidate["candidate_id"],
        "candidate_kind": CANDIDATE_KIND,
        "materializer_transform": TRANSFORM,
        "dormant_schedule_candidate_id": dormant_candidate["candidate_id"],
        "confirmed_loop_fact_keys": sorted(LOOP_FACT_KEYS),
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
    features, metadata, _ = confirmed_compiler_artifacts(
        transition, transition_paths
    )
    dossier_value = read_json(dossier_path)
    expanded_dossier, expanded_graph, change = expand_graph(
        dossier_value, read_json(template_path), read_json(graph_path),
        transition["dormant_compiler_candidate"], features, metadata,
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
            transition_paths["baseline_features"],
            "confirmed_baseline_features",
        ),
        input_record(
            transition_paths["reused_features"],
            "confirmed_reused_features",
        ),
        input_record(
            transition_paths["baseline_kernel_metadata"],
            "confirmed_baseline_kernel_metadata",
        ),
        input_record(
            transition_paths["reused_kernel_metadata"],
            "confirmed_reused_kernel_metadata",
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
        raise ExpansionError(f"refusing to overwrite reused expansion: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
    ))
    try:
        for name, text in rendered.items():
            (temporary / name).write_text(text, encoding="utf-8")
        if output.exists():
            raise ExpansionError(
                f"refusing to overwrite reused expansion: {output}"
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
        raise ExpansionError("unexpected reused-expansion manifest schema")
    payload = dict(manifest)
    expansion_id = payload.pop("expansion_id", None)
    if expansion_id != bridge._fingerprint(payload):
        raise ExpansionError("reused-expansion ID changed")
    records = manifest.get("inputs")
    if not isinstance(records, list):
        raise ExpansionError("reused expansion lacks input records")
    paths = {}
    for record in records:
        if (not isinstance(record, dict)
                or not isinstance(record.get("role"), str)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)):
            raise ExpansionError("reused expansion has invalid input record")
        role = record["role"]
        path = recorded_path(record["path"])
        if role in paths or not path.is_file():
            raise ExpansionError(f"missing or duplicate reused input: {role}")
        if (sha256_file(path) != record["sha256"]
                or path.stat().st_size != record["bytes"]):
            raise ExpansionError(f"reused-expansion input changed: {path}")
        paths[role] = path
    required = {
        "passed_confirmation_analysis", "frozen_current_dossier",
        "frozen_current_template", "frozen_current_graph",
        "confirmed_baseline_features", "confirmed_reused_features",
        "confirmed_baseline_kernel_metadata",
        "confirmed_reused_kernel_metadata", "confirmed_final_ir_audit",
    }
    if set(paths) != required:
        raise ExpansionError("reused expansion input roles changed")
    rendered, regenerated = build_bundle(
        paths["passed_confirmation_analysis"],
        paths["frozen_current_dossier"], paths["frozen_current_template"],
        paths["frozen_current_graph"],
    )
    if regenerated != manifest:
        raise ExpansionError("reused-expansion manifest does not regenerate")
    output_dir = manifest_path.resolve().parent
    if {path.name for path in output_dir.iterdir()} != set(rendered):
        raise ExpansionError("reused-expansion output set changed")
    for name, expected in rendered.items():
        path = output_dir / name
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise ExpansionError(f"reused-expansion output changed: {path}")
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
            f"reused-descriptor-graph-expansion: {action}; "
            f"model_invoked=false; provider_call_authorized=false; "
            f"scheduler_job_submitted=false; "
            f"expansion_id={manifest['expansion_id']}"
        )
        return 0
    except (
        ExpansionError, confirmation.ConfirmError,
        confirmation.common.MonitorError, groups.GroupPlanError,
        bridge.BridgeError, OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"reused-descriptor-graph-expansion: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
