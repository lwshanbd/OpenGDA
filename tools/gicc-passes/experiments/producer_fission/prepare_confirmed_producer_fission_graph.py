#!/usr/bin/env python3
"""Expand the compiler-only Jacobi graph after a passed confirmation.

This tool has no scheduler, compiler, provider, or application-source path.
It replays the three independent producer-fission confirmation allocations,
verifies that the current graph still contains exactly one dormant compiler
candidate, and writes a new content-addressed dossier/graph/prompt bundle.
The frozen input graph is never modified and the resulting bundle does not
authorize a model call.
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

import analyze_producer_fission_confirmation as confirmation  # noqa: E402
import gicc_comm_group_plan_bridge as groups  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


EXPANSION_SCHEMA = "gicc-producer-fission-graph-expansion-v1"
CONFIRMATION_SCHEMA = "gicc-producer-fission-confirmation-v1"
FISSION_KIND = "group_producer_frontier_two_phase"
FISSION_TRANSFORM = "PRODUCER_FRONTIER_TWO_PHASE"


class ExpansionError(RuntimeError):
    """The confirmation or compiler graph cannot support graph expansion."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExpansionError(f"cannot read JSON {path}: {exc}") from exc


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def json_value(value: Any) -> Any:
    """Normalize tuples and other JSON containers to their frozen form."""
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
    """Replay all raw monitors and return the analysis and its transition."""
    value = read_json(path)
    if (not isinstance(value, dict)
            or value.get("schema_version") != CONFIRMATION_SCHEMA):
        raise ExpansionError("unexpected producer-fission confirmation schema")
    payload = dict(value)
    result_id = payload.pop("result_id", None)
    if result_id != bridge._fingerprint(payload):
        raise ExpansionError("producer-fission confirmation result ID changed")
    for key, expected in {
        "model_invoked": False,
        "application_source_modified": False,
        "provider_call_authorized": False,
    }.items():
        if value.get(key) is not expected:
            raise ExpansionError(
                f"producer-fission confirmation boundary changed: {key}"
            )
    if value.get("correctness_gate", {}).get("passed") is not True:
        raise ExpansionError("producer-fission correctness gate did not pass")
    if value.get("confirmation_gate", {}).get("passed") is not True:
        raise ExpansionError("producer-fission confirmation gate did not pass")

    transition_path = Path(value.get("transition", ""))
    if (not transition_path.is_absolute() or not transition_path.is_file()
            or sha256_file(transition_path) != value.get("transition_sha256")):
        raise ExpansionError("producer-fission confirmation transition changed")
    transition, transition_paths = confirmation.validate_transition(
        transition_path
    )
    dormant = transition.get("dormant_compiler_candidate", {})
    if (dormant.get("kind") != "producer_frontier_two_phase"
            or dormant.get("compiler_materializer")
            != "guarded_host_device_fission"
            or dormant.get("phase_count") != 2
            or dormant.get("candidate_id")
            != value.get("dormant_compiler_candidate_id")):
        raise ExpansionError("confirmed dormant compiler candidate changed")

    summaries = value.get("allocation_monitors")
    if (not isinstance(summaries, list) or len(summaries) != 3
            or any(not isinstance(item, dict) for item in summaries)):
        raise ExpansionError("confirmation lacks three allocation monitors")
    monitor_paths = [Path(item.get("monitor", "")) for item in summaries]
    if any(not monitor.is_absolute() for monitor in monitor_paths):
        raise ExpansionError("confirmation monitor path is not absolute")
    regenerated = confirmation.analyze_monitors(
        transition_path, monitor_paths,
    )
    if regenerated != value:
        raise ExpansionError(
            "confirmation does not replay from its raw allocation logs"
        )
    return value, transition, transition_paths


def bind_confirmed_case(
    dossier: dict[str, Any], template_paths: list[Path],
    transition: dict[str, Any], transition_paths: dict[str, Path],
) -> tuple[Path, Path]:
    """Bind the route graph to the exact compiler facts used by the oracle."""
    coverage_path = transition_paths.get("source_free_schedule_coverage")
    if coverage_path is None:
        raise ExpansionError("confirmation transition lacks schedule coverage")
    coverage_path = coverage_path.resolve()
    report = read_json(coverage_path)
    candidate, coverage_graph_id = (
        confirmation.transition_base.validate_dormant_candidate(report)
    )
    if (candidate != transition.get("dormant_compiler_candidate")
            or coverage_graph_id
            != transition.get("schedule_coverage_graph_id")):
        raise ExpansionError("confirmation and schedule coverage disagree")
    cases = report.get("cases")
    if not isinstance(cases, dict):
        raise ExpansionError("schedule coverage lacks private case bindings")
    matches = [
        (label, case) for label, case in cases.items()
        if isinstance(case, dict) and case.get("case_id") == candidate["case_id"]
    ]
    if len(matches) != 1:
        raise ExpansionError("confirmed schedule case is not uniquely bound")
    label, frozen_case = matches[0]
    feature_record = frozen_case.get("features")
    template_records = frozen_case.get("templates")
    if (not isinstance(feature_record, dict)
            or not isinstance(feature_record.get("path"), str)
            or not isinstance(feature_record.get("sha256"), str)
            or not isinstance(template_records, list)
            or not template_records):
        raise ExpansionError("confirmed schedule case lacks compiler artifacts")
    features_path = Path(feature_record["path"]).resolve()
    if (not features_path.is_file()
            or sha256_file(features_path) != feature_record["sha256"]):
        raise ExpansionError("confirmed compiler features changed")
    recorded_templates = []
    for record in template_records:
        if (not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)):
            raise ExpansionError("confirmed compiler template record is invalid")
        path = Path(record["path"]).resolve()
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise ExpansionError(f"confirmed compiler template changed: {path}")
        recorded_templates.append(path)
    if sorted(recorded_templates) != sorted(path.resolve() for path in template_paths):
        raise ExpansionError(
            "graph templates differ from the confirmed compiler artifacts"
        )
    regenerated_private, regenerated_model = (
        confirmation.transition_base.coverage.load_case(label, features_path)
    )
    if regenerated_private != frozen_case:
        raise ExpansionError("private compiler schedule case does not regenerate")
    if regenerated_model.get("dormant_compiler_oracle") != candidate:
        raise ExpansionError("compiler features regenerate another dormant candidate")
    expected_dossier = bridge.make_dossier(
        read_json(features_path), dossier["platform_profile"],
    )
    if json_value(expected_dossier) != json_value(dossier):
        raise ExpansionError(
            "current dossier does not come from the confirmed compiler features"
        )
    return coverage_path, features_path


def _opportunities(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        opportunity["opportunity_id"]: opportunity
        for opportunity in graph["opportunities"]
    }


def _without_keys(value: dict[str, Any], keys: set[str]) -> dict[str, Any]:
    return {key: child for key, child in value.items() if key not in keys}


def expand_graph(
    dossier_value: Any, template_values: list[Any], graph_value: Any,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Expose only the previously masked producer-fission candidate."""
    dossier = bridge._verified_dossier(dossier_value)
    current = json_value(groups.verified_graph(graph_value))
    regenerated = json_value(groups.make_group_graph(dossier, template_values))
    if regenerated != current:
        raise ExpansionError(
            "current compiler graph does not regenerate from dossier/templates"
        )
    transforms = dossier.get("platform_profile", {}).get(
        "compiler_transforms", {}
    )
    if not isinstance(transforms, dict):
        raise ExpansionError("compiler_transforms profile must be an object")
    fission_gate = transforms.get("producer_frontier_fission")
    if fission_gate is True:
        raise ExpansionError("producer-fission candidate is already enabled")
    if fission_gate is not None and fission_gate is not False:
        raise ExpansionError("producer-fission profile gate is malformed")

    current_opportunities = _opportunities(current)
    masked_locations = [
        opportunity_id
        for opportunity_id, opportunity in current_opportunities.items()
        for candidate in opportunity.get("masked_candidates", [])
        if candidate.get("kind") == FISSION_KIND
    ]
    visible_before = [
        candidate
        for opportunity in current["opportunities"]
        for candidate in opportunity["candidates"]
        if candidate.get("kind") == FISSION_KIND
    ]
    if len(masked_locations) != 1 or visible_before:
        raise ExpansionError(
            "current graph must contain exactly one masked producer-fission "
            "candidate and no visible copy"
        )

    dossier_payload = copy.deepcopy(dossier)
    dossier_payload.pop("dossier_id")
    profile = copy.deepcopy(dossier_payload["platform_profile"])
    expanded_transforms = copy.deepcopy(transforms)
    expanded_transforms["producer_frontier_fission"] = True
    profile["compiler_transforms"] = expanded_transforms
    dossier_payload["platform_profile"] = profile
    expanded_dossier = {
        **dossier_payload,
        "dossier_id": bridge._fingerprint(dossier_payload),
    }
    bridge._verified_dossier(expanded_dossier)
    expanded = json_value(
        groups.make_group_graph(expanded_dossier, template_values)
    )
    groups.verified_graph(expanded)

    expanded_opportunities = _opportunities(expanded)
    if set(expanded_opportunities) != set(current_opportunities):
        raise ExpansionError("graph expansion changed compiler opportunities")
    new_candidates = []
    for opportunity_id, old in current_opportunities.items():
        new = expanded_opportunities[opportunity_id]
        if _without_keys(old, {"candidates", "masked_candidates"}) != (
                _without_keys(new, {"candidates", "masked_candidates"})):
            raise ExpansionError(
                f"graph expansion changed compiler facts for {opportunity_id}"
            )
        old_candidates = old["candidates"]
        retained = [
            candidate for candidate in new["candidates"]
            if candidate.get("kind") != FISSION_KIND
        ]
        if retained != old_candidates:
            raise ExpansionError(
                f"graph expansion changed an existing candidate in {opportunity_id}"
            )
        new_candidates.extend(
            candidate for candidate in new["candidates"]
            if candidate.get("kind") == FISSION_KIND
        )
        old_masked = [
            candidate for candidate in old.get("masked_candidates", [])
            if candidate.get("kind") != FISSION_KIND
        ]
        if old_masked != new.get("masked_candidates", []):
            raise ExpansionError(
                f"graph expansion changed another mask in {opportunity_id}"
            )
    if len(new_candidates) != 1:
        raise ExpansionError("expansion must expose exactly one new candidate")
    candidate = new_candidates[0]
    if masked_locations[0] not in {
            opportunity_id
            for opportunity_id, opportunity in expanded_opportunities.items()
            if candidate in opportunity["candidates"]}:
        raise ExpansionError("producer-fission candidate moved opportunities")
    materializers = candidate.get("materializer", {}).get("sites", {})
    if (not isinstance(materializers, dict) or not materializers
            or any(request != {
                "dispatch": "DWQ_TRIGGER", "transform": FISSION_TRANSFORM,
            } for request in materializers.values())):
        raise ExpansionError("producer-fission materializer contract changed")

    if current.get("fixed_sites") != expanded.get("fixed_sites"):
        raise ExpansionError("graph expansion changed compiler-fixed sites")
    if (current["compiler_inputs"]["kernel_template_ids"]
            != expanded["compiler_inputs"]["kernel_template_ids"]):
        raise ExpansionError("graph expansion changed compiler templates")
    if _without_keys(
        current, {"graph_id", "compiler_inputs", "platform_profile",
                  "opportunities"},
    ) != _without_keys(
        expanded, {"graph_id", "compiler_inputs", "platform_profile",
                   "opportunities"},
    ):
        raise ExpansionError("graph expansion changed an unrelated graph field")
    change = {
        "opportunity_id": masked_locations[0],
        "candidate_id": candidate["candidate_id"],
        "candidate_kind": FISSION_KIND,
        "materializer_transform": FISSION_TRANSFORM,
        "prior_selectable_candidate_count": sum(
            len(item["candidates"]) for item in current["opportunities"]
        ),
        "expanded_selectable_candidate_count": sum(
            len(item["candidates"]) for item in expanded["opportunities"]
        ),
        "prior_candidate_ids_preserved": True,
        "only_previously_masked_candidate_exposed": True,
    }
    return expanded_dossier, expanded, change


def build_bundle(
    confirmation_path: Path, dossier_path: Path,
    template_paths: list[Path], graph_path: Path,
) -> tuple[dict[str, str], dict[str, Any]]:
    confirmation_value, transition, transition_paths = replay_confirmation(
        confirmation_path.resolve()
    )
    if not template_paths:
        raise ExpansionError("at least one compiler kernel template is required")
    ordered_templates = sorted(path.resolve() for path in template_paths)
    template_values = [read_json(path) for path in ordered_templates]
    dossier_value = read_json(dossier_path)
    coverage_path, features_path = bind_confirmed_case(
        bridge._verified_dossier(dossier_value), ordered_templates,
        transition, transition_paths,
    )
    expanded_dossier, expanded_graph, change = expand_graph(
        dossier_value, template_values, read_json(graph_path)
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
            "role": role,
            "path": name,
            "sha256": sha256_bytes(encoded),
            "bytes": len(encoded),
        })
    inputs = [
        input_record(confirmation_path, "passed_confirmation_analysis"),
        input_record(coverage_path, "source_free_schedule_coverage"),
        input_record(features_path, "confirmed_compiler_features"),
        input_record(dossier_path, "frozen_current_dossier"),
        input_record(graph_path, "frozen_current_graph"),
    ]
    inputs.extend(
        input_record(path, f"compiler_kernel_template:{index}")
        for index, path in enumerate(ordered_templates)
    )
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
            "application_source_input": False,
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
        raise ExpansionError(f"refusing to overwrite graph expansion: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
    ))
    try:
        for name, text in rendered.items():
            target = temporary / name
            target.write_text(text, encoding="utf-8")
        if output.exists():
            raise ExpansionError(
                f"refusing to overwrite graph expansion: {output}"
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
        raise ExpansionError("unexpected graph-expansion manifest schema")
    payload = dict(manifest)
    expansion_id = payload.pop("expansion_id", None)
    if expansion_id != bridge._fingerprint(payload):
        raise ExpansionError("graph-expansion ID changed")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, list):
        raise ExpansionError("graph expansion lacks input records")
    paths: dict[str, Path] = {}
    for record in inputs:
        if (not isinstance(record, dict)
                or not isinstance(record.get("role"), str)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)):
            raise ExpansionError("graph expansion has an invalid input record")
        role = record["role"]
        path = recorded_path(record["path"])
        if role in paths or not path.is_file():
            raise ExpansionError(f"missing or duplicate graph input: {role}")
        if (sha256_file(path) != record["sha256"]
                or path.stat().st_size != record["bytes"]):
            raise ExpansionError(f"graph-expansion input changed: {path}")
        paths[role] = path
    required = {
        "passed_confirmation_analysis", "source_free_schedule_coverage",
        "confirmed_compiler_features", "frozen_current_dossier",
        "frozen_current_graph",
    }
    if not required.issubset(paths):
        raise ExpansionError("graph expansion lacks required inputs")
    template_roles = sorted(
        (role for role in paths
         if role.startswith("compiler_kernel_template:")),
        key=lambda role: int(role.rsplit(":", 1)[1]),
    )
    if set(paths) != required.union(template_roles) or not template_roles:
        raise ExpansionError("graph expansion has unexpected input roles")
    rendered, regenerated = build_bundle(
        paths["passed_confirmation_analysis"],
        paths["frozen_current_dossier"],
        [paths[role] for role in template_roles],
        paths["frozen_current_graph"],
    )
    if regenerated != manifest:
        raise ExpansionError("graph-expansion manifest does not regenerate")
    output_dir = manifest_path.resolve().parent
    if set(path.name for path in output_dir.iterdir()) != set(rendered):
        raise ExpansionError("graph-expansion output set changed")
    for name, expected in rendered.items():
        path = output_dir / name
        if not path.is_file() or path.read_text(encoding="utf-8") != expected:
            raise ExpansionError(f"graph-expansion output changed: {path}")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--confirmation-analysis", type=Path, required=True)
    prepare.add_argument("--dossier", type=Path, required=True)
    prepare.add_argument("--template", type=Path, action="append", required=True)
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
            f"producer-fission-graph-expansion: {action}; "
            f"model_invoked=false; provider_call_authorized=false; "
            f"scheduler_job_submitted=false; expansion_id="
            f"{manifest['expansion_id']}"
        )
        return 0
    except (
        ExpansionError, confirmation.ConfirmError,
        confirmation.common.MonitorError, groups.GroupPlanError,
        bridge.BridgeError, OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"producer-fission-graph-expansion: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
