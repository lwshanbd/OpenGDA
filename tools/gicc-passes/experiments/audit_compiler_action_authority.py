#!/usr/bin/env python3
"""Audit the compiler-policy authority absent from the frozen route GBT.

This is an interface audit, not a model comparison.  It verifies the exact
historical GBT output vocabulary, the current compiler decision suite, the
input-separation report, and the conditional compiler frontier.  It then
counts compiler actions and independent policies that the frozen scalar GBT
schema cannot name without being redesigned.  It invokes no compiler,
scheduler, model, provider, runtime benchmark, or source reader/editor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE))

import audit_compiler_action_frontier as action_frontier  # noqa: E402
import audit_compiler_input_separation as input_separation  # noqa: E402
import gicc_compiler_decision_suite as decision_suite  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


REPORT_SCHEMA = "gicc-compiler-action-authority-v1"
GBT_ROUTE_ACTIONS = {"default", "proxy", "trigger"}
DISPATCH_TO_ROUTE = {
    "IPC_OR_DWQ": "default",
    "CPU_PROXY_ENQUEUE": "proxy",
    "DWQ_TRIGGER": "trigger",
}
STRUCTURAL_TRANSFORMS = {"COALESCE_LOOP", "COALESCE_LOOP_EARLY"}
COMMUNICATION_LABELS = {
    "jacobi", "minimod", "mixed_lto", "mm_minimal", "loop_lto",
}
BOUNDARY = {
    "application_source_read": False,
    "application_source_modified": False,
    "compiler_invoked": False,
    "scheduler_invoked": False,
    "runtime_benchmark_invoked": False,
    "model_invoked": False,
    "provider_invoked": False,
    "provider_call_authorized": False,
    "compiler_lto_decisions_only": True,
}


class AuthorityError(RuntimeError):
    """Frozen artifacts do not prove the claimed interface authority gap."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthorityError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise AuthorityError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise AuthorityError(f"missing authority evidence: {resolved}")
    return {
        "path": display_path(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def recorded_path(value: str) -> Path:
    raw = Path(value)
    return raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()


def verify_recorded_evidence(records: Any) -> None:
    if not isinstance(records, dict) or not records:
        raise AuthorityError("conditional frontier lacks evidence records")
    for role, record in records.items():
        if (not isinstance(role, str) or not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)):
            raise AuthorityError("conditional frontier has malformed evidence")
        path = recorded_path(record["path"])
        if (not path.is_file() or sha256_file(path) != record["sha256"]
                or path.stat().st_size != record["bytes"]):
            raise AuthorityError(f"conditional frontier evidence changed: {role}")


def gbt_output_contract(value: Any) -> dict[str, Any]:
    checked = input_separation.verify_gbt_report(value)
    predictions = value.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        raise AuthorityError("GBT report has no predictions")
    observed_actions: set[str] = set()
    legal_actions: set[str] = set()
    for prediction in predictions:
        chosen = prediction.get("chosen_action") if isinstance(prediction, dict) else None
        legal = prediction.get("legal_actions") if isinstance(prediction, dict) else None
        if (not isinstance(chosen, str) or not isinstance(legal, list)
                or chosen not in legal
                or any(not isinstance(item, str) for item in legal)):
            raise AuthorityError("GBT prediction does not contain one legal route label")
        observed_actions.add(chosen)
        legal_actions.update(legal)
        if any(key in prediction for key in (
                "candidate_id", "option_id", "decision_slots",
                "size_policy", "compiler_transform")):
            raise AuthorityError("GBT report unexpectedly contains a structured action")
    if legal_actions != GBT_ROUTE_ACTIONS:
        raise AuthorityError("GBT legal route vocabulary changed")
    return {
        **checked,
        "prediction_count": len(predictions),
        "legal_output_vocabulary": sorted(legal_actions),
        "observed_output_vocabulary": sorted(observed_actions),
        "output_shape": "one chosen_action route label per prediction",
        "candidate_id_selection": False,
        "collective_size_policy_selection": False,
        "compiler_transform_selection": False,
    }


def suite_entry(entries: dict[str, Any], label: str, path: Path,
                graph: dict[str, Any]) -> dict[str, Any]:
    entry = entries.get(label)
    if (not isinstance(entry, dict) or entry.get("graph_id") != graph.get("graph_id")
            or entry.get("graph_file_sha256") != sha256_file(path)):
        raise AuthorityError(f"{label}: graph does not match frozen suite")
    return entry


def route_materializer(value: Any) -> str | None:
    if not isinstance(value, dict) or value.get("transform") != "NONE":
        return None
    return DISPATCH_TO_ROUTE.get(value.get("dispatch"))


def analyze_communication(
    graphs: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    per_entry = {}
    candidate_count = 0
    atomic_group_candidates = 0
    mixed_group_candidates = 0
    for label in sorted(graphs):
        graph, entry = graphs[label]
        local_candidates = 0
        local_atomic = 0
        local_mixed = 0
        for opportunity in graph["opportunities"]:
            site_ids = opportunity["site_ids"]
            for candidate in opportunity["candidates"]:
                actions = candidate.get("effects", {}).get("site_actions")
                materializers = candidate.get("materializer", {}).get("sites")
                if (not isinstance(actions, dict) or set(actions) != set(site_ids)
                        or set(actions.values()) - GBT_ROUTE_ACTIONS
                        or not isinstance(materializers, dict)
                        or set(materializers) != set(site_ids)
                        or any(route_materializer(item) != actions[site]
                               for site, item in materializers.items())):
                    raise AuthorityError(
                        f"{label}: current communication candidate is not route-composable"
                    )
                local_candidates += 1
                if len(site_ids) > 1:
                    local_atomic += 1
                    if len(set(actions.values())) > 1:
                        local_mixed += 1
        expected = entry["decision_space"]["selectable_candidate_id_count"]
        if local_candidates != expected:
            raise AuthorityError(f"{label}: communication candidate count changed")
        policy_count = entry["decision_space"]["independent_policy_count"]
        per_entry[label] = {
            "independent_policy_count": policy_count,
            "route_composable_candidate_count": local_candidates,
            "atomic_multi_site_candidate_count": local_atomic,
            "mixed_route_candidate_count": local_mixed,
            "semantic_actions_outside_gbt_route_vocabulary": 0,
            "structured_output_absent_from_frozen_gbt_schema": local_atomic > 0,
        }
        candidate_count += local_candidates
        atomic_group_candidates += local_atomic
        mixed_group_candidates += local_mixed
    if (candidate_count, atomic_group_candidates, mixed_group_candidates) != (32, 27, 18):
        raise AuthorityError("communication authority totals changed")
    return {
        "per_entry": per_entry,
        "route_composable_candidate_count": candidate_count,
        "atomic_multi_site_candidate_count": atomic_group_candidates,
        "mixed_route_candidate_count": mixed_group_candidates,
        "interpretation": (
            "All current communication candidates use the GBT route vocabulary, "
            "but 27 are atomic multi-site candidate-ID decisions absent from the "
            "frozen one-label GBT output schema. This is an interface-granularity "
            "gap, not proof that a redesigned structured ML baseline could not act."
        ),
    }


def analyze_structural(graph: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    route_counts = []
    structural_count = 0
    transforms: set[str] = set()
    for opportunity in graph["opportunities"]:
        route_count = 0
        local_structural = 0
        for candidate in opportunity["candidates"]:
            materializer = candidate.get("materializer")
            route = route_materializer(materializer)
            if route is not None:
                route_count += 1
            elif (isinstance(materializer, dict)
                  and materializer.get("dispatch") == "DWQ_TRIGGER"
                  and materializer.get("transform") in STRUCTURAL_TRANSFORMS):
                local_structural += 1
                transforms.add(materializer["transform"])
            else:
                raise AuthorityError("structural graph contains an unknown materializer")
        if (route_count, local_structural) != (2, 2):
            raise AuthorityError("each structural opportunity must be 2 route + 2 transform")
        route_counts.append(route_count)
        structural_count += local_structural
    total = entry["decision_space"]["independent_policy_count"]
    route_only = math.prod(route_counts)
    beyond_route = total - route_only
    if (len(route_counts), structural_count, total, route_only, beyond_route,
            transforms) != (6, 12, 4096, 64, 4032, STRUCTURAL_TRANSFORMS):
        raise AuthorityError("structural authority totals changed")
    return {
        "opportunity_count": len(route_counts),
        "compiler_transform_candidate_id_count": structural_count,
        "compiler_transforms": sorted(transforms),
        "independent_policy_count": total,
        "route_only_policy_count": route_only,
        "policies_using_at_least_one_structural_transform": beyond_route,
        "structural_policy_fraction": beyond_route / total,
    }


def analyze_collective(graph: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    opportunities = graph["opportunities"]
    if len(opportunities) != 1:
        raise AuthorityError("collective suite entry must contain one opportunity")
    slots = opportunities[0].get("decision_slots")
    if not isinstance(slots, list) or not slots:
        raise AuthorityError("collective graph lacks decision slots")
    nonanchor_options = 0
    anchor_counts = []
    total_counts = []
    algorithms: set[str] = set()
    communication_graphs: set[str] = set()
    pipeline_depths: set[int] = set()
    for slot in slots:
        options = slot.get("options")
        if not isinstance(options, list):
            raise AuthorityError("collective slot lacks options")
        anchors = [item for item in options if item.get("role") == "anchor"]
        candidates = [item for item in options if item.get("role") == "candidate"]
        if (len(anchors) != 1 or anchors[0].get("algorithm") != "baseline_auto"
                or len(candidates) != 7):
            raise AuthorityError("collective anchor/candidate cohort changed")
        if any(item.get("algorithm") in GBT_ROUTE_ACTIONS for item in options):
            raise AuthorityError("collective algorithm collided with GBT route labels")
        for item in candidates:
            descriptor = item.get("compiler_descriptor", {})
            algorithms.add(item["algorithm"])
            communication_graphs.add(descriptor.get("communication_graph"))
            pipeline_depths.add(int(descriptor.get("pipeline_chunks")))
        anchor_counts.append(len(anchors))
        total_counts.append(len(options))
        nonanchor_options += len(candidates)
    total = entry["decision_space"]["independent_policy_count"]
    anchors_only = math.prod(anchor_counts)
    nonanchor_policies = total - anchors_only
    if (len(slots), nonanchor_options, total, nonanchor_policies) != (4, 28, 4096, 4095):
        raise AuthorityError("collective authority totals changed")
    return {
        "decision_slot_count": len(slots),
        "nonanchor_compiler_option_id_count": nonanchor_options,
        "nonanchor_algorithms": sorted(algorithms),
        "communication_graph_families": sorted(communication_graphs),
        "pipeline_depths": sorted(pipeline_depths),
        "independent_policy_count": total,
        "all_anchor_policy_count": anchors_only,
        "policies_using_at_least_one_nonanchor_algorithm": nonanchor_policies,
        "nonanchor_policy_fraction": nonanchor_policies / total,
    }


def analyze_conditional_frontier(value: Any) -> dict[str, Any]:
    report = action_frontier.verify_report(value)
    verify_recorded_evidence(report.get("evidence"))
    records = report["conditional_frontier"]
    transforms = {item.get("compiler_materializer_transform") for item in records}
    if (transforms != {
            "PRODUCER_FRONTIER_TWO_PHASE", "GUARDED_EARLY_TRIGGER",
            "REUSE_LOOP_DESCRIPTOR",
            } or any(item.get("runtime_status") != "unconfirmed"
                     or item.get("performance_claim_supported") is not False
                     or item.get("model_visibility")
                     != "forbidden_until_positive_confirmation_and_refreeze"
                     for item in records)):
        raise AuthorityError("conditional frontier overstates model authority")
    return {
        "frontier_id": report["frontier_id"],
        "runtime_unconfirmed_transform_count": len(records),
        "compiler_transforms": sorted(transforms),
        "model_visible_transform_count": 0,
        "performance_claim_supported": False,
    }


def parse_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, raw = value.split("=", 1)
    if not label or not raw:
        raise argparse.ArgumentTypeError("expected nonempty LABEL=PATH")
    return label, Path(raw)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    suite_path = args.suite.resolve()
    prompt_dir = args.prompt_dir.resolve()
    gbt_path = args.gbt_report.resolve()
    suite = decision_suite.verified_suite(read_json(suite_path), prompt_dir)
    entries = {item["label"]: item for item in suite["entries"]}
    gbt = gbt_output_contract(read_json(gbt_path))

    expected_separation = input_separation.build_report(
        suite_path, prompt_dir, gbt_path,
    )
    separation_path = args.input_separation.resolve()
    separation = read_json(separation_path)
    if separation != expected_separation:
        raise AuthorityError("input-separation report does not regenerate")

    communication_specs = dict(args.communication)
    if set(communication_specs) != COMMUNICATION_LABELS:
        raise AuthorityError("authority audit requires all five communication graphs")
    communication_graphs = {}
    for label, raw_path in communication_specs.items():
        path = raw_path.resolve()
        graph = decision_suite.communication.verified_graph(read_json(path))
        communication_graphs[label] = (
            graph, suite_entry(entries, label, path, graph),
        )

    structural_path = args.structural_graph.resolve()
    structural_graph = decision_suite.structural.verified_graph(
        read_json(structural_path)
    )
    structural_entry = suite_entry(
        entries, "coalescing_placement", structural_path, structural_graph,
    )
    collective_path = args.collective_graph.resolve()
    collective_graph = decision_suite.collective.verified_graph(
        read_json(collective_path)
    )
    collective_entry = suite_entry(
        entries, "collective_n8", collective_path, collective_graph,
    )

    communication = analyze_communication(communication_graphs)
    structural = analyze_structural(structural_graph, structural_entry)
    collective = analyze_collective(collective_graph, collective_entry)
    conditional_path = args.conditional_frontier.resolve()
    conditional = analyze_conditional_frontier(read_json(conditional_path))

    payload = {
        "schema_version": REPORT_SCHEMA,
        "boundary": dict(BOUNDARY),
        "suite_id": suite["suite_id"],
        "input_separation_audit_id": separation["audit_id"],
        "gbt_output_contract": gbt,
        "current_compiler_policy_authority": {
            "communication_route_or_group": communication,
            "coalescing_and_placement": structural,
            "collective_algorithm_and_size_policy": collective,
        },
        "conditional_compiler_policy_authority": conditional,
        "claim_separation": {
            "frozen_gbt_is_a_valid_route_baseline": True,
            "frozen_gbt_and_compiler_policy_interfaces_have_equal_authority": False,
            "current_nonroute_compiler_candidate_or_option_ids": 40,
            "current_nonroute_policy_counts_by_independent_entry": {
                "coalescing_placement": 4032,
                "collective_n8": 4095,
            },
            "conditional_runtime_unconfirmed_nonroute_transforms": 3,
            "wider_authority_comes_from_compiler_interface_not_llm_identity": True,
            "structured_ml_could_use_the_same_interface": True,
            "llm_is_required_for_these_actions": False,
            "valid_model_intelligence_comparison_requires_equal_action_authority": True,
            "llm_performance_superiority_claimed": False,
        },
        "evidence": {
            "auditor": evidence(Path(__file__)),
            "suite": evidence(suite_path),
            "gbt_report": evidence(gbt_path),
            "input_separation": evidence(separation_path),
            "conditional_frontier": evidence(conditional_path),
            "structural_graph": evidence(structural_path),
            "collective_graph": evidence(collective_path),
            **{
                f"communication_graph_{label}": evidence(path.resolve())
                for label, path in sorted(communication_specs.items())
            },
        },
    }
    return {"authority_id": bridge._fingerprint(payload), **payload}


def verify_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != REPORT_SCHEMA:
        raise AuthorityError(f"expected report schema {REPORT_SCHEMA}")
    payload = dict(value)
    authority_id = payload.pop("authority_id", None)
    if authority_id != bridge._fingerprint(payload):
        raise AuthorityError("authority_id does not match report content")
    if value.get("boundary") != BOUNDARY:
        raise AuthorityError("authority report crossed its offline boundary")
    claims = value.get("claim_separation", {})
    required = {
        "wider_authority_comes_from_compiler_interface_not_llm_identity": True,
        "structured_ml_could_use_the_same_interface": True,
        "llm_is_required_for_these_actions": False,
        "valid_model_intelligence_comparison_requires_equal_action_authority": True,
        "llm_performance_superiority_claimed": False,
    }
    if any(claims.get(key) is not expected for key, expected in required.items()):
        raise AuthorityError("authority report confuses interface width with LLM quality")
    return value


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--suite", type=Path, required=True)
    value.add_argument("--prompt-dir", type=Path, required=True)
    value.add_argument("--gbt-report", type=Path, required=True)
    value.add_argument("--input-separation", type=Path, required=True)
    value.add_argument("--communication", type=parse_spec, action="append", required=True)
    value.add_argument("--structural-graph", type=Path, required=True)
    value.add_argument("--collective-graph", type=Path, required=True)
    value.add_argument("--conditional-frontier", type=Path, required=True)
    value.add_argument("--out", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = verify_report(build_report(args))
    atomic_write(args.out.resolve(), json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
