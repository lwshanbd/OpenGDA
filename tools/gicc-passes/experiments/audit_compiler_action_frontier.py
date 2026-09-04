#!/usr/bin/env python3
"""Audit dormant compiler/LTO actions without treating them as model evidence.

The audit replays the pure graph-expansion functions for the three frozen
compiler materializers that are still behind runtime gates.  It reads compiler
metadata and final-IR audits, but never reads application source, invokes a
model/provider, runs a compiler, or touches the scheduler.  The expanded
graphs exist only in memory: they quantify a conditional compiler frontier and
must not be used as model input before confirmation and suite refreezing.
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
PASS_PYTHON = HERE.parent / "python"
PRODUCER_DIR = HERE / "producer_fission"
GUARDED_DIR = HERE / "guarded_early_trigger"
REUSED_DIR = HERE / "reused_loop_descriptor"
for directory in (PASS_PYTHON, PRODUCER_DIR, GUARDED_DIR, REUSED_DIR):
    sys.path.insert(0, str(directory))

import gicc_compiler_decision_suite as decision_suite  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import prepare_confirmed_guarded_early_graph as guarded_expansion  # noqa: E402
import prepare_confirmed_producer_fission_graph as producer_expansion  # noqa: E402
import prepare_confirmed_reused_loop_descriptor_graph as reused_expansion  # noqa: E402
import prepare_guarded_early_trigger_confirmation as guarded_proof  # noqa: E402
import prepare_producer_fission_confirmation as producer_proof  # noqa: E402
import prepare_reused_loop_descriptor_confirmation as reused_proof  # noqa: E402


REPORT_SCHEMA = "gicc-compiler-action-frontier-v1"
EXPECTED_COUNTS = {
    "jacobi": (9, 10),
    "mm_minimal": (3, 4),
    "loop_lto": (2, 3),
}
BOUNDARY = {
    "application_source_read": False,
    "application_source_modified": False,
    "compiler_invoked": False,
    "scheduler_invoked": False,
    "model_invoked": False,
    "provider_invoked": False,
    "provider_call_authorized": False,
    "runtime_evidence_used": False,
    "expanded_graphs_written": False,
    "conditional_candidates_model_visible": False,
}


class FrontierError(RuntimeError):
    """The frozen compiler artifacts do not prove the claimed frontier."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FrontierError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise FrontierError(f"cannot hash {path}: {exc}") from exc
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
        raise FrontierError(f"missing frontier evidence: {resolved}")
    return {
        "path": display_path(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def graph_summary(value: Any) -> dict[str, Any]:
    graph = producer_expansion.groups.verified_graph(value)
    per_opportunity = []
    candidate_ids = []
    for opportunity in graph["opportunities"]:
        ids = [item["candidate_id"] for item in opportunity["candidates"]]
        if len(ids) != len(set(ids)):
            raise FrontierError("an opportunity contains duplicate candidate IDs")
        candidate_ids.extend(ids)
        per_opportunity.append({
            "opportunity_id": opportunity["opportunity_id"],
            "selectable_candidate_count": len(ids),
        })
    if len(candidate_ids) != len(set(candidate_ids)):
        raise FrontierError("a graph contains duplicate selectable candidate IDs")
    return {
        "graph_id": graph["graph_id"],
        "opportunity_count": len(per_opportunity),
        "per_opportunity": per_opportunity,
        "independent_policy_count": math.prod(
            item["selectable_candidate_count"] for item in per_opportunity
        ),
        "candidate_ids": sorted(candidate_ids),
    }


def verify_current_graph(
    label: str, path: Path, suite_entry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    value = read_json(path)
    summary = graph_summary(value)
    space = suite_entry.get("decision_space", {})
    if (suite_entry.get("label") != label
            or suite_entry.get("decision_family")
            != "communication_route_or_schedule"
            or suite_entry.get("graph_id") != summary["graph_id"]
            or suite_entry.get("graph_file_sha256") != sha256_file(path)
            or space.get("independent_policy_count")
            != summary["independent_policy_count"]
            or space.get("selectable_candidate_id_count")
            != len(summary["candidate_ids"])):
        raise FrontierError(f"{label}: current graph does not match frozen suite")
    return value, summary


def frontier_record(
    label: str, current: dict[str, Any], expanded: dict[str, Any],
    change: dict[str, Any], *, dormant_schedule_candidate_id: str | None,
) -> dict[str, Any]:
    old = graph_summary(current)
    new = graph_summary(expanded)
    old_ids = set(old["candidate_ids"])
    new_ids = set(new["candidate_ids"])
    additions = sorted(new_ids - old_ids)
    expected = EXPECTED_COUNTS.get(label)
    if expected is None:
        raise FrontierError(f"unexpected conditional frontier label: {label}")
    if (old["independent_policy_count"], new["independent_policy_count"]) != expected:
        raise FrontierError(
            f"{label}: expected policy transition {expected[0]}->{expected[1]}"
        )
    if (not old_ids < new_ids or len(additions) != 1
            or change.get("candidate_id") != additions[0]
            or change.get("prior_candidate_ids_preserved") is not True
            or change.get("prior_selectable_candidate_count") != len(old_ids)
            or change.get("expanded_selectable_candidate_count") != len(new_ids)):
        raise FrontierError(f"{label}: expansion is not an exact one-candidate superset")
    return {
        "label": label,
        "current_graph_id": old["graph_id"],
        "conditional_graph_id": new["graph_id"],
        "current_independent_policy_count": old["independent_policy_count"],
        "conditional_independent_policy_count": new["independent_policy_count"],
        "per_entry_policy_delta": (
            new["independent_policy_count"] - old["independent_policy_count"]
        ),
        "current_selectable_candidate_count": len(old_ids),
        "conditional_selectable_candidate_count": len(new_ids),
        "new_graph_candidate_id": additions[0],
        "new_candidate_kind": change.get("candidate_kind"),
        "compiler_materializer_transform": change.get("materializer_transform"),
        "dormant_schedule_candidate_id": dormant_schedule_candidate_id,
        "prior_candidate_ids_preserved": True,
        "runtime_status": "unconfirmed",
        "model_visibility": "forbidden_until_positive_confirmation_and_refreeze",
        "performance_claim_supported": False,
    }


def verified_producer_candidate(
    report_path: Path,
) -> tuple[dict[str, Any], list[Path]]:
    report = read_json(report_path)
    candidate, _ = producer_proof.validate_dormant_candidate(report)
    internal = report.get("cases", {}).get("jacobi")
    if not isinstance(internal, dict):
        raise FrontierError("schedule coverage lacks the Jacobi compiler case")
    features = internal.get("features", {})
    if not isinstance(features.get("path"), str):
        raise FrontierError("Jacobi schedule coverage lacks compiler features")
    feature_path = Path(features["path"]).resolve()
    regenerated_internal, regenerated_model = producer_proof.coverage.load_case(
        "jacobi", feature_path,
    )
    model_cases = [
        item for item in report.get("model_graph", {}).get("cases", [])
        if isinstance(item, dict) and item.get("case_id") == internal.get("case_id")
    ]
    if (regenerated_internal != internal or len(model_cases) != 1
            or regenerated_model != model_cases[0]
            or regenerated_model.get("dormant_compiler_oracle") != candidate):
        raise FrontierError("Jacobi schedule coverage does not regenerate")
    paths = [feature_path]
    for item in internal.get("templates", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise FrontierError("Jacobi schedule coverage has invalid templates")
        paths.append(Path(item["path"]).resolve())
    return candidate, paths


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    suite_path = args.suite.resolve()
    prompt_dir = args.prompt_dir.resolve()
    suite = decision_suite.verified_suite(read_json(suite_path), prompt_dir)
    entries = {item["label"]: item for item in suite["entries"]}
    if any(label not in entries for label in EXPECTED_COUNTS):
        raise FrontierError("frozen suite lacks a conditional-frontier entry")

    producer_graph, producer_current = verify_current_graph(
        "jacobi", args.producer_graph.resolve(), entries["jacobi"],
    )
    producer_templates = [read_json(path.resolve()) for path in args.producer_template]
    _, producer_expanded, producer_change = producer_expansion.expand_graph(
        read_json(args.producer_dossier.resolve()),
        producer_templates,
        producer_graph,
    )
    producer_candidate, producer_proof_paths = verified_producer_candidate(
        args.producer_coverage_report.resolve()
    )
    if (producer_candidate.get("kind") != "producer_frontier_two_phase"
            or producer_change.get("candidate_kind")
            != "group_producer_frontier_two_phase"):
        raise FrontierError("producer schedule proof and graph transform diverged")
    producer_record = frontier_record(
        "jacobi", producer_graph, producer_expanded, producer_change,
        dormant_schedule_candidate_id=producer_candidate["candidate_id"],
    )

    guarded_graph, guarded_current = verify_current_graph(
        "mm_minimal", args.guarded_graph.resolve(), entries["mm_minimal"],
    )
    guarded_candidate, guarded_paths = guarded_proof.compiler_candidate(
        args.guarded_binary_dir.resolve()
    )
    if guarded_candidate.get("model_visible") is not False:
        raise FrontierError("guarded compiler candidate became model-visible")
    guarded_metadata = read_json(guarded_paths["baseline_kernel_metadata"])
    _, guarded_expanded, guarded_change = guarded_expansion.expand_graph(
        read_json(args.guarded_dossier.resolve()),
        read_json(args.guarded_template.resolve()),
        guarded_graph,
        guarded_candidate,
        guarded_metadata,
    )
    guarded_record = frontier_record(
        "mm_minimal", guarded_graph, guarded_expanded, guarded_change,
        dormant_schedule_candidate_id=guarded_candidate["candidate_id"],
    )

    reused_graph, reused_current = verify_current_graph(
        "loop_lto", args.reused_graph.resolve(), entries["loop_lto"],
    )
    reused_candidate, reused_paths = reused_proof.compiler_candidate(
        args.reused_binary_dir.resolve()
    )
    if reused_candidate.get("model_visible") is not False:
        raise FrontierError("reused-descriptor compiler candidate became model-visible")
    _, reused_expanded, reused_change = reused_expansion.expand_graph(
        read_json(args.reused_dossier.resolve()),
        read_json(args.reused_template.resolve()),
        reused_graph,
        reused_candidate,
        read_json(reused_paths["baseline_features"]),
        read_json(reused_paths["baseline_kernel_metadata"]),
    )
    reused_record = frontier_record(
        "loop_lto", reused_graph, reused_expanded, reused_change,
        dormant_schedule_candidate_id=reused_candidate["candidate_id"],
    )

    records = sorted(
        [producer_record, guarded_record, reused_record],
        key=lambda item: item["label"],
    )
    if [record["per_entry_policy_delta"] for record in records] != [1, 1, 1]:
        raise FrontierError("conditional frontier must add one policy per entry")
    if any(record["performance_claim_supported"] for record in records):
        raise FrontierError("an unconfirmed candidate cannot support performance")

    input_paths = {
        "auditor": Path(__file__).resolve(),
        "suite": suite_path,
        "producer_dossier": args.producer_dossier.resolve(),
        "producer_graph": args.producer_graph.resolve(),
        "producer_coverage_report": args.producer_coverage_report.resolve(),
        **{
            f"producer_template_{index}": path.resolve()
            for index, path in enumerate(args.producer_template, start=1)
        },
        **{
            f"producer_proof_artifact_{index}": path
            for index, path in enumerate(producer_proof_paths, start=1)
        },
        "guarded_dossier": args.guarded_dossier.resolve(),
        "guarded_template": args.guarded_template.resolve(),
        "guarded_graph": args.guarded_graph.resolve(),
        **{f"guarded_{role}": path for role, path in guarded_paths.items()},
        "reused_dossier": args.reused_dossier.resolve(),
        "reused_template": args.reused_template.resolve(),
        "reused_graph": args.reused_graph.resolve(),
        **{f"reused_{role}": path for role, path in reused_paths.items()},
    }
    payload = {
        "schema_version": REPORT_SCHEMA,
        "boundary": dict(BOUNDARY),
        "suite_id": suite["suite_id"],
        "composition": {
            "counts_reported_per_independent_entry": True,
            "global_joint_policy_count": None,
            "conditional_candidates_require_independent_runtime_gates": True,
        },
        "conditional_frontier": records,
        "summary": {
            "compiler_expressibility_supported": True,
            "conditional_entry_count": len(records),
            "current_policy_counts": {
                label: EXPECTED_COUNTS[label][0] for label in sorted(EXPECTED_COUNTS)
            },
            "conditional_policy_counts": {
                label: EXPECTED_COUNTS[label][1] for label in sorted(EXPECTED_COUNTS)
            },
            "runtime_confirmed_candidate_count": 0,
            "model_visible_conditional_candidate_count": 0,
            "provider_calls": 0,
            "llm_performance_claim_supported": False,
        },
        "evidence": {
            role: evidence(path) for role, path in sorted(input_paths.items())
        },
    }
    report = {"frontier_id": bridge._fingerprint(payload), **payload}
    # Keep the current summaries live so an accidental unused-input regression
    # cannot silently weaken the suite binding above.
    if any(summary["independent_policy_count"] <= 0 for summary in (
            producer_current, guarded_current, reused_current)):
        raise FrontierError("current suite contains an empty decision space")
    return report


def verify_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict) or report.get("schema_version") != REPORT_SCHEMA:
        raise FrontierError(f"expected report schema {REPORT_SCHEMA}")
    payload = dict(report)
    frontier_id = payload.pop("frontier_id", None)
    if frontier_id != bridge._fingerprint(payload):
        raise FrontierError("frontier_id does not match report content")
    if report.get("boundary") != BOUNDARY:
        raise FrontierError("frontier report crossed its offline boundary")
    records = report.get("conditional_frontier")
    if (not isinstance(records, list) or len(records) != len(EXPECTED_COUNTS)
            or {item.get("label") for item in records} != set(EXPECTED_COUNTS)):
        raise FrontierError("frontier report has the wrong conditional entries")
    for item in records:
        expected = EXPECTED_COUNTS[item["label"]]
        if (item.get("current_independent_policy_count"),
                item.get("conditional_independent_policy_count")) != expected:
            raise FrontierError(f"{item['label']}: report policy counts changed")
        if (item.get("per_entry_policy_delta") != 1
                or item.get("runtime_status") != "unconfirmed"
                or item.get("performance_claim_supported") is not False):
            raise FrontierError(f"{item['label']}: report overstates evidence")
    return report


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
    value.add_argument("--producer-dossier", type=Path, required=True)
    value.add_argument("--producer-template", type=Path, action="append", required=True)
    value.add_argument("--producer-graph", type=Path, required=True)
    value.add_argument("--producer-coverage-report", type=Path, required=True)
    value.add_argument("--guarded-dossier", type=Path, required=True)
    value.add_argument("--guarded-template", type=Path, required=True)
    value.add_argument("--guarded-graph", type=Path, required=True)
    value.add_argument("--guarded-binary-dir", type=Path, required=True)
    value.add_argument("--reused-dossier", type=Path, required=True)
    value.add_argument("--reused-template", type=Path, required=True)
    value.add_argument("--reused-graph", type=Path, required=True)
    value.add_argument("--reused-binary-dir", type=Path, required=True)
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
