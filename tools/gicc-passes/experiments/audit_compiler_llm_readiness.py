#!/usr/bin/env python3
"""Derive conservative LLM-evaluation readiness from compiler evidence.

This audit has no provider, scheduler, compiler, or source-edit path.  A
positive exploratory scout never authorizes a model call: it advances only to
an independently frozen runtime-confirmation stage.
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
PASS_PYTHON = HERE.parent / "python"
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(HERE / "collective"))
sys.path.insert(0, str(HERE / "producer_fission"))
sys.path.insert(0, str(HERE / "guarded_early_trigger"))
sys.path.insert(0, str(HERE / "reused_loop_descriptor"))

import analyze_compiler_collective_n8_confirmation as n8_confirmation  # noqa: E402
import analyze_guarded_early_trigger_confirmation as guarded_confirmation  # noqa: E402
import analyze_producer_fission_confirmation as producer_confirmation  # noqa: E402
import analyze_reused_loop_descriptor_confirmation as reused_confirmation  # noqa: E402
import gicc_comm_plan_bridge as structural  # noqa: E402
import gicc_compiler_decision_suite as decision_suite  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import prepare_confirmed_guarded_early_graph as guarded_expansion  # noqa: E402
import prepare_guarded_early_suite_refreeze as guarded_refreeze  # noqa: E402
import prepare_confirmed_producer_fission_graph as producer_expansion  # noqa: E402
import prepare_producer_fission_suite_refreeze as producer_refreeze  # noqa: E402


REPORT_SCHEMA = "gicc-compiler-llm-readiness-v1"
EXPECTED_LABELS = {
    "coalescing_placement",
    "collective_n8",
    "jacobi",
    "loop_lto",
    "minimod",
    "mixed_lto",
    "mm_minimal",
}
CAPACITY_ONLY_LABELS = ("minimod", "mixed_lto")


class ReadinessError(RuntimeError):
    """The suite or its runtime evidence cannot support a readiness claim."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReadinessError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise ReadinessError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def evidence(path: Path) -> dict[str, Any]:
    record = {"path": display_path(path), "present": path.is_file()}
    if path.is_file():
        record.update({
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    return record


def state_phase(path: Path) -> str:
    if not path.is_file():
        return "missing"
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) != 1:
        raise ReadinessError(f"invalid controller state {path}")
    fields = lines[0].split("\t")
    if len(fields) != 3 or not fields[1]:
        raise ReadinessError(f"invalid controller state {path}")
    return fields[1]


def _base_entry(entry: dict[str, Any], status: str,
                next_stage: str) -> dict[str, Any]:
    return {
        "suite_entry_id": entry["entry_id"],
        "compiler_graph_id": entry["graph_id"],
        "decision_family": entry["decision_family"],
        "status": status,
        "next_stage": next_stage,
        "provider_protocol_permitted": status == "provider_protocol_permitted",
        "provider_call_authorized": False,
        "llm_performance_measured": False,
    }


def classify_placement(entry: dict[str, Any], summary: Any,
                       graphs_equivalent: bool) -> dict[str, Any]:
    if (not isinstance(summary, dict)
            or summary.get("schema_version")
            != "gicc-communication-plan-placement-runtime-v1"
            or summary.get("model_invoked") is not False
            or summary.get("source_visible_to_model") is not False
            or not graphs_equivalent):
        raise ReadinessError("placement evidence violates its compiler-only contract")
    gate = summary.get("placement_llm_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
        raise ReadinessError("placement evidence lacks its preregistered gate")
    if gate["passed"]:
        result = _base_entry(
            entry, "provider_protocol_permitted",
            "freeze_exact_provider_request_and_request_authorization",
        )
    else:
        result = _base_entry(
            entry, "closed_negative", "do_not_run_model_for_this_graph",
        )
    result["runtime_gate_passed"] = gate["passed"]
    return result


def classify_collective(
    entry: dict[str, Any], phase: str, analysis: Any | None,
    confirmation_phase: str = "missing",
    confirmation_passed: bool | None = None,
) -> dict[str, Any]:
    if analysis is None:
        if confirmation_passed is not None:
            raise ReadinessError(
                "collective confirmation exists without a passed scout"
            )
        status = {
            "submitting": "awaiting_scout",
            "monitoring": "awaiting_scout",
            "missing": "awaiting_scout",
            "failed": "scout_failed",
        }.get(phase)
        if status is None:
            raise ReadinessError(
                f"collective state {phase!r} requires a matching analysis"
            )
        next_stage = (
            "wait_for_existing_n8_pdebug_scout"
            if status == "awaiting_scout" else "diagnose_without_model_call"
        )
        return _base_entry(entry, status, next_stage)
    if (not isinstance(analysis, dict)
            or analysis.get("schema_version")
            != "gicc-collective-hierpipe-n8-scout-v1"
            or analysis.get("graph_id") != entry["graph_id"]
            or analysis.get("model_invoked") is not False
            or analysis.get("application_source_modified") is not False):
        raise ReadinessError("collective scout violates its compiler-only contract")
    payload = dict(analysis)
    result_id = payload.pop("result_id", None)
    if result_id != bridge._fingerprint(payload):
        raise ReadinessError("collective scout result ID does not match content")
    gate = analysis.get("n8_capacity_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
        raise ReadinessError("collective scout lacks its preregistered gate")
    expected_phase = "promising" if gate["passed"] else "negative"
    if phase != expected_phase:
        raise ReadinessError(
            "collective controller state disagrees with scout analysis"
        )
    if gate["passed"]:
        if confirmation_passed is None:
            status = {
                "missing": "confirmation_required",
                "building": "awaiting_confirmation",
                "submitting": "awaiting_confirmation",
                "monitoring": "awaiting_confirmation",
                "analyzing": "awaiting_confirmation",
                "failed": "confirmation_failed",
            }.get(confirmation_phase)
            if status is None:
                raise ReadinessError(
                    "collective confirmation state requires an analysis"
                )
            next_stage = {
                "confirmation_required": (
                    "freeze_and_run_confirmatory_compiler_oracle"
                ),
                "awaiting_confirmation": (
                    "wait_for_existing_n8_pdebug_confirmation"
                ),
                "confirmation_failed": "diagnose_without_model_call",
            }[status]
            result = _base_entry(entry, status, next_stage)
        else:
            expected_confirmation_phase = (
                "confirmed" if confirmation_passed else "negative"
            )
            if confirmation_phase != expected_confirmation_phase:
                raise ReadinessError(
                    "collective confirmation state disagrees with analysis"
                )
            if confirmation_passed:
                result = _base_entry(
                    entry, "provider_protocol_permitted",
                    "freeze_exact_provider_request_and_request_authorization",
                )
            else:
                result = _base_entry(
                    entry, "closed_negative",
                    "do_not_run_model_for_this_graph",
                )
            result["runtime_confirmation_gate_passed"] = confirmation_passed
    else:
        if confirmation_passed is not None:
            raise ReadinessError(
                "collective confirmation exists after a negative scout"
            )
        result = _base_entry(
            entry, "closed_negative", "do_not_run_model_for_this_graph",
        )
    result["runtime_gate_passed"] = gate["passed"]
    return result


def verified_collective_confirmation(
    path: Path, expected_graph_id: str,
) -> bool:
    """Replay a completed N8 confirmation before granting request eligibility."""
    value = read_json(path)
    if (not isinstance(value, dict)
            or value.get("schema_version")
            != "gicc-collective-n8-confirmation-v1"):
        raise ReadinessError("wrong collective confirmation schema")
    payload = dict(value)
    result_id = payload.pop("result_id", None)
    if result_id != bridge._fingerprint(payload):
        raise ReadinessError("collective confirmation result ID changed")
    for key, expected in {
        "model_invoked": False,
        "application_source_modified": False,
        "provider_call_authorized": False,
    }.items():
        if value.get(key) is not expected:
            raise ReadinessError(
                f"collective confirmation boundary changed: {key}"
            )
    transition_path = Path(value.get("transition", ""))
    if (not transition_path.is_absolute() or not transition_path.is_file()
            or sha256_file(transition_path) != value.get("transition_sha256")):
        raise ReadinessError("collective confirmation transition changed")
    _, transition_files = n8_confirmation.validate_transition(transition_path)
    confirmed_graph = read_json(transition_files["compiler_graph"])
    if confirmed_graph.get("graph_id") != expected_graph_id:
        raise ReadinessError(
            "collective confirmation binds another compiler graph"
        )
    summaries = value.get("allocation_monitors")
    if not isinstance(summaries, list) or len(summaries) != 3:
        raise ReadinessError(
            "collective confirmation lacks three allocation monitors"
        )
    monitors = [Path(item.get("monitor", "")) for item in summaries]
    if any(not monitor.is_absolute() for monitor in monitors):
        raise ReadinessError(
            "collective confirmation monitor path is not absolute"
        )
    regenerated = n8_confirmation.analyze_monitors(
        transition_path, monitors,
    )
    if regenerated != value:
        raise ReadinessError(
            "collective confirmation does not replay from raw evidence"
        )
    gate = value.get("confirmation_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
        raise ReadinessError("collective confirmation lacks its gate")
    return gate["passed"]


def _hidden_candidate_result(
    entry: dict[str, Any], *, confirmation_passed: bool,
) -> dict[str, Any]:
    if confirmation_passed:
        result = _base_entry(
            entry, "graph_expansion_required",
            "regenerate_and_refreeze_graph_before_provider_request",
        )
    else:
        result = _base_entry(
            entry, "closed_negative", "keep_candidate_model_invisible",
        )
    result.update({
        "runtime_confirmation_gate_passed": confirmation_passed,
        "candidate_model_visible": False,
        "current_suite_graph_expanded": False,
    })
    return result


def _confirmation_pending_result(
    entry: dict[str, Any], *, phase: str, label: str,
) -> dict[str, Any]:
    status = {
        "missing": "confirmation_required",
        "waiting_scheduler_idle": "awaiting_confirmation",
        "submitting": "awaiting_confirmation",
        "monitoring": "awaiting_confirmation",
        "analyzing": "awaiting_confirmation",
        "failed": "confirmation_failed",
    }.get(phase)
    if status is None:
        raise ReadinessError(f"{label} confirmation state requires an analysis")
    next_stage = {
        "confirmation_required": "freeze_and_run_confirmatory_compiler_oracle",
        "awaiting_confirmation": f"wait_for_existing_{label}_confirmation",
        "confirmation_failed": "diagnose_without_model_call",
    }[status]
    result = _base_entry(entry, status, next_stage)
    result.update({
        "candidate_model_visible": False,
        "current_suite_graph_expanded": False,
    })
    return result


def classify_producer(
    entry: dict[str, Any], phase: str, analysis: Any | None,
    confirmation_phase: str = "missing",
    confirmation_passed: bool | None = None,
    graph_expansion_phase: str = "missing",
) -> dict[str, Any]:
    if analysis is None:
        if confirmation_passed is not None or graph_expansion_phase != "missing":
            raise ReadinessError(
                "producer-fission downstream evidence exists without a passed scout"
            )
        status = {
            "waiting_predecessor": "awaiting_predecessor",
            "submitting": "awaiting_scout",
            "monitoring": "awaiting_scout",
            "analyzing": "awaiting_scout",
            "missing": "awaiting_predecessor",
            "failed": "scout_failed",
        }.get(phase)
        if status is None:
            raise ReadinessError(
                f"producer-fission state {phase!r} requires an analysis"
            )
        next_stage = {
            "awaiting_predecessor": "wait_for_existing_n8_pdebug_scout",
            "awaiting_scout": "wait_for_existing_producer_fission_scout",
            "scout_failed": "diagnose_without_model_call",
        }[status]
        return _base_entry(entry, status, next_stage)
    if (not isinstance(analysis, dict)
            or analysis.get("schema_version")
            != "gicc-producer-fission-oracle-analysis-v1"
            or analysis.get("correctness_gate", {}).get("passed") is not True):
        raise ReadinessError("producer-fission scout violates its correctness gate")
    gate = analysis.get("oracle_headroom_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
        raise ReadinessError("producer-fission scout lacks its oracle gate")
    expected_phase = "promising" if gate["passed"] else "negative"
    if phase != expected_phase:
        raise ReadinessError(
            "producer-fission controller state disagrees with scout analysis"
        )
    if gate["passed"]:
        if confirmation_passed is None:
            if graph_expansion_phase != "missing":
                raise ReadinessError(
                    "producer-fission graph expanded without confirmation"
                )
            result = _confirmation_pending_result(
                entry, phase=confirmation_phase, label="producer_fission",
            )
        else:
            expected_confirmation_phase = (
                "confirmed" if confirmation_passed else "negative"
            )
            if confirmation_phase != expected_confirmation_phase:
                raise ReadinessError(
                    "producer-fission confirmation state disagrees with analysis"
                )
            if not confirmation_passed and graph_expansion_phase != "missing":
                raise ReadinessError(
                    "producer-fission graph expanded after negative confirmation"
                )
            if not confirmation_passed or graph_expansion_phase == "missing":
                result = _hidden_candidate_result(
                    entry, confirmation_passed=confirmation_passed,
                )
            elif graph_expansion_phase == "bundle_ready":
                result = _base_entry(
                    entry, "suite_refreeze_required",
                    "refreeze_and_audit_suite_with_expanded_graph",
                )
                result.update({
                    "runtime_confirmation_gate_passed": True,
                    "candidate_model_visible": False,
                    "expanded_graph_bundle_verified": True,
                    "current_suite_graph_expanded": False,
                })
            elif graph_expansion_phase == "suite_refrozen":
                result = _base_entry(
                    entry, "provider_protocol_permitted",
                    "freeze_exact_provider_request_and_request_authorization",
                )
                result.update({
                    "runtime_confirmation_gate_passed": True,
                    "candidate_model_visible": True,
                    "expanded_graph_bundle_verified": True,
                    "current_suite_graph_expanded": True,
                })
            else:
                raise ReadinessError(
                    f"unknown producer-fission graph phase "
                    f"{graph_expansion_phase!r}"
                )
    else:
        if confirmation_passed is not None:
            raise ReadinessError(
                "producer-fission confirmation exists after a negative scout"
            )
        result = _base_entry(
            entry, "closed_negative", "mask_producer_fission_from_model",
        )
    result["runtime_gate_passed"] = gate["passed"]
    return result


def _record_by_role(
    records: Any, role: str, *, label: str,
) -> dict[str, Any]:
    matches = [
        record for record in records
        if isinstance(record, dict) and record.get("role") == role
    ] if isinstance(records, list) else []
    if len(matches) != 1:
        raise ReadinessError(f"{label} lacks exact role {role}")
    return matches[0]


def _recorded_path(record: dict[str, Any], *, label: str) -> Path:
    value = record.get("path")
    if not isinstance(value, str):
        raise ReadinessError(f"{label} lacks a recorded path")
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    if (not path.is_file() or sha256_file(path) != record.get("sha256")
            or path.stat().st_size != record.get("bytes")):
        raise ReadinessError(f"{label} input changed")
    return path


def verified_refreeze_predecessor(
    manifest: dict[str, Any], *, label: str,
) -> tuple[Path, Path, dict[str, Any]]:
    """Verify and return the exact suite/prompt predecessor of a refreeze."""
    record = _record_by_role(
        manifest.get("inputs"), "current_suite", label=label,
    )
    suite_path = _recorded_path(record, label=f"{label} current suite")
    prompt_value = manifest.get("current_prompt_dir")
    if not isinstance(prompt_value, str):
        raise ReadinessError(f"{label} lacks its current prompt directory")
    prompt_dir = Path(prompt_value)
    prompt_dir = (
        prompt_dir.resolve() if prompt_dir.is_absolute()
        else (ROOT / prompt_dir).resolve()
    )
    suite = decision_suite.verified_suite(read_json(suite_path), prompt_dir)
    if suite.get("suite_id") != manifest.get("current_suite_id"):
        raise ReadinessError(f"{label} predecessor suite ID changed")
    return suite_path, prompt_dir, suite


def verified_producer_expansion(
    path: Path, confirmation_value: dict[str, Any],
) -> dict[str, Any]:
    """Replay and bind the expanded graph to this exact confirmation."""
    manifest = producer_expansion.verify_contained(path)
    if (manifest.get("status")
            != "expanded_graph_ready_for_suite_refreeze"
            or manifest.get("confirmation", {}).get("result_id")
            != confirmation_value.get("result_id")
            or manifest.get("confirmation", {}).get(
                "confirmation_gate_passed") is not True):
        raise ReadinessError(
            "producer-fission expansion binds another confirmation"
        )
    boundary = manifest.get("boundary", {})
    for key, expected in {
        "compiler_lto_decisions_only": True,
        "application_source_input": False,
        "application_source_modified": False,
        "model_invoked": False,
        "provider_call_authorized": False,
        "scheduler_job_submitted": False,
        "frozen_current_graph_modified": False,
        "current_decision_suite_modified": False,
    }.items():
        if boundary.get(key) is not expected:
            raise ReadinessError(
                f"producer-fission expansion boundary changed: {key}"
            )
    return manifest


def verified_producer_refreeze(
    path: Path, expansion_path: Path, suite_path: Path,
    suite: dict[str, Any], confirmation_value: dict[str, Any],
) -> dict[str, Any]:
    """Replay the suite transition and bind it to the audited suite input."""
    manifest = producer_refreeze.verify_contained(path)
    if (manifest.get("status")
            != "refrozen_suite_ready_for_readiness_audit"
            or manifest.get("refrozen_suite_id") != suite.get("suite_id")):
        raise ReadinessError("producer-fission refreeze binds another suite")
    boundary = manifest.get("boundary", {})
    for key, expected in {
        "compiler_lto_decisions_only": True,
        "application_source_input": False,
        "application_source_modified": False,
        "model_invoked": False,
        "provider_call_authorized": False,
        "scheduler_job_submitted": False,
        "current_suite_modified": False,
    }.items():
        if boundary.get(key) is not expected:
            raise ReadinessError(
                f"producer-fission refreeze boundary changed: {key}"
            )
    expansion_record = _record_by_role(
        manifest.get("inputs"), "producer_expansion_manifest",
        label="producer-fission refreeze",
    )
    recorded_expansion = Path(expansion_record.get("path", ""))
    recorded_expansion = (
        recorded_expansion.resolve() if recorded_expansion.is_absolute()
        else (ROOT / recorded_expansion).resolve()
    )
    if (recorded_expansion != expansion_path.resolve()
            or sha256_file(recorded_expansion)
            != expansion_record.get("sha256")):
        raise ReadinessError("producer-fission refreeze uses another expansion")
    expansion_manifest = verified_producer_expansion(
        expansion_path, confirmation_value,
    )
    suite_record = _record_by_role(
        manifest.get("outputs"), "refrozen_suite",
        label="producer-fission refreeze",
    )
    if (sha256_file(suite_path) != suite_record.get("sha256")
            or suite_path.stat().st_size != suite_record.get("bytes")):
        raise ReadinessError("readiness suite differs from refrozen suite")
    entries = {entry["label"]: entry for entry in suite["entries"]}
    jacobi = entries.get("jacobi", {})
    transition = expansion_manifest.get("graph_transition", {})
    delta = manifest.get("entry_transition", {})
    if (jacobi.get("graph_id") != transition.get("expanded_graph_id")
            or delta.get("new_graph_id") != jacobi.get("graph_id")
            or delta.get("new_candidate_id") != transition.get("candidate_id")
            or delta.get("all_other_entries_preserved") is not True):
        raise ReadinessError("refrozen Jacobi entry changed after expansion")
    return manifest


def verified_guarded_expansion(
    path: Path, confirmation_value: dict[str, Any],
) -> dict[str, Any]:
    """Replay and bind the guarded graph to its exact confirmation."""
    manifest = guarded_expansion.verify_contained(path)
    confirmation = manifest.get("confirmation", {})
    if (manifest.get("status")
            != "expanded_graph_ready_for_suite_refreeze"
            or confirmation.get("result_id")
            != confirmation_value.get("result_id")
            or confirmation.get("confirmation_gate_passed") is not True
            or confirmation.get("correctness_gate_passed") is not True
            or confirmation.get("runtime_guard_gate_passed") is not True):
        raise ReadinessError(
            "guarded-trigger expansion binds another confirmation"
        )
    boundary = manifest.get("boundary", {})
    for key, expected in {
        "compiler_lto_decisions_only": True,
        "application_source_hash_verified": True,
        "application_source_visible_to_model": False,
        "application_source_modified": False,
        "model_invoked": False,
        "provider_call_authorized": False,
        "scheduler_job_submitted": False,
        "frozen_current_graph_modified": False,
        "current_decision_suite_modified": False,
    }.items():
        if boundary.get(key) is not expected:
            raise ReadinessError(
                f"guarded-trigger expansion boundary changed: {key}"
            )
    return manifest


def verified_guarded_refreeze(
    path: Path, expansion_path: Path, suite_path: Path,
    suite: dict[str, Any], confirmation_value: dict[str, Any],
) -> dict[str, Any]:
    """Replay the guarded suite transition and bind its full lineage."""
    manifest = guarded_refreeze.verify_contained(path)
    if (manifest.get("status")
            != "refrozen_suite_ready_for_readiness_audit"
            or manifest.get("refrozen_suite_id") != suite.get("suite_id")):
        raise ReadinessError("guarded-trigger refreeze binds another suite")
    boundary = manifest.get("boundary", {})
    for key, expected in {
        "compiler_lto_decisions_only": True,
        "application_source_hash_verified": True,
        "application_source_visible_to_model": False,
        "application_source_modified": False,
        "model_invoked": False,
        "provider_call_authorized": False,
        "scheduler_job_submitted": False,
        "current_suite_modified": False,
    }.items():
        if boundary.get(key) is not expected:
            raise ReadinessError(
                f"guarded-trigger refreeze boundary changed: {key}"
            )
    expansion_record = _record_by_role(
        manifest.get("inputs"), "guarded_expansion_manifest",
        label="guarded-trigger refreeze",
    )
    recorded_expansion = _recorded_path(
        expansion_record, label="guarded-trigger refreeze expansion",
    )
    if recorded_expansion != expansion_path.resolve():
        raise ReadinessError("guarded-trigger refreeze uses another expansion")
    expansion_manifest = verified_guarded_expansion(
        expansion_path, confirmation_value,
    )
    suite_record = _record_by_role(
        manifest.get("outputs"), "refrozen_suite",
        label="guarded-trigger refreeze",
    )
    if (sha256_file(suite_path) != suite_record.get("sha256")
            or suite_path.stat().st_size != suite_record.get("bytes")):
        raise ReadinessError("readiness suite differs from guarded refreeze")
    _, _, predecessor = verified_refreeze_predecessor(
        manifest, label="guarded-trigger refreeze",
    )
    entries = {entry["label"]: entry for entry in suite["entries"]}
    old_entries = {
        entry["label"]: entry for entry in predecessor["entries"]
    }
    if (set(entries) != set(old_entries)
            or any(entries[label] != old_entries[label]
                   for label in set(entries) - {"mm_minimal"})):
        raise ReadinessError(
            "guarded-trigger refreeze changed an unrelated suite entry"
        )
    mm_entry = entries.get("mm_minimal", {})
    transition = expansion_manifest.get("graph_transition", {})
    delta = manifest.get("entry_transition", {})
    if (delta.get("label") != "mm_minimal"
            or mm_entry.get("graph_id") != transition.get("expanded_graph_id")
            or delta.get("new_graph_id") != mm_entry.get("graph_id")
            or delta.get("new_candidate_id") != transition.get("candidate_id")
            or delta.get("all_other_entries_preserved") is not True):
        raise ReadinessError(
            "refrozen mm_minimal entry changed after guarded expansion"
        )
    return manifest


def classify_guarded_early_trigger(
    entry: dict[str, Any], phase: str, analysis: Any | None,
    confirmation_phase: str = "missing",
    confirmation_passed: bool | None = None,
    graph_expansion_phase: str = "missing",
) -> dict[str, Any]:
    if analysis is None:
        if confirmation_passed is not None or graph_expansion_phase != "missing":
            raise ReadinessError(
                "guarded-trigger downstream evidence exists without a passed scout"
            )
        status = {
            "waiting_predecessor": "awaiting_predecessor",
            "submitting": "awaiting_scout",
            "monitoring": "awaiting_scout",
            "analyzing": "awaiting_scout",
            "missing": "runtime_labels_missing",
            "failed": "scout_failed",
        }.get(phase)
        if status is None:
            raise ReadinessError(
                f"guarded-early-trigger state {phase!r} requires an analysis"
            )
        next_stage = {
            "awaiting_predecessor": "wait_for_serial_compiler_headroom_campaign",
            "awaiting_scout": "wait_for_guarded_early_trigger_scout",
            "runtime_labels_missing": (
                "establish_preregistered_compiler_oracle_headroom_first"
            ),
            "scout_failed": "diagnose_without_model_call",
        }[status]
        return _base_entry(entry, status, next_stage)
    if (not isinstance(analysis, dict)
            or analysis.get("schema_version")
            != "gicc-guarded-early-trigger-analysis-v1"
            or analysis.get("correctness_gate", {}).get("passed") is not True):
        raise ReadinessError(
            "guarded-early-trigger scout violates its correctness gate"
        )
    gate = analysis.get("oracle_headroom_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
        raise ReadinessError("guarded-early-trigger scout lacks its oracle gate")
    expected_phase = "promising" if gate["passed"] else "negative"
    if phase != expected_phase:
        raise ReadinessError(
            "guarded-early-trigger controller state disagrees with scout analysis"
        )
    if gate["passed"]:
        if confirmation_passed is None:
            if graph_expansion_phase != "missing":
                raise ReadinessError(
                    "guarded-trigger graph expanded without confirmation"
                )
            result = _confirmation_pending_result(
                entry, phase=confirmation_phase, label="guarded_early_trigger",
            )
        else:
            expected_confirmation_phase = (
                "confirmed" if confirmation_passed else "negative"
            )
            if confirmation_phase != expected_confirmation_phase:
                raise ReadinessError(
                    "guarded-trigger confirmation state disagrees with analysis"
                )
            if not confirmation_passed and graph_expansion_phase != "missing":
                raise ReadinessError(
                    "guarded-trigger graph expanded after negative confirmation"
                )
            if not confirmation_passed or graph_expansion_phase == "missing":
                result = _hidden_candidate_result(
                    entry, confirmation_passed=confirmation_passed,
                )
            elif graph_expansion_phase == "bundle_ready":
                result = _base_entry(
                    entry, "suite_refreeze_required",
                    "refreeze_and_audit_suite_with_expanded_graph",
                )
                result.update({
                    "runtime_confirmation_gate_passed": True,
                    "candidate_model_visible": False,
                    "expanded_graph_bundle_verified": True,
                    "current_suite_graph_expanded": False,
                })
            elif graph_expansion_phase == "suite_refrozen":
                result = _base_entry(
                    entry, "provider_protocol_permitted",
                    "freeze_exact_provider_request_and_request_authorization",
                )
                result.update({
                    "runtime_confirmation_gate_passed": True,
                    "candidate_model_visible": True,
                    "expanded_graph_bundle_verified": True,
                    "current_suite_graph_expanded": True,
                })
            else:
                raise ReadinessError(
                    f"unknown guarded-trigger graph phase "
                    f"{graph_expansion_phase!r}"
                )
    else:
        if confirmation_passed is not None:
            raise ReadinessError(
                "guarded-trigger confirmation exists after a negative scout"
            )
        result = _base_entry(
            entry, "closed_negative",
            "mask_guarded_early_trigger_from_model",
        )
    result["runtime_gate_passed"] = gate["passed"]
    return result


def verified_hidden_candidate_confirmation(
    path: Path, *, schema: str, analyzer: Any, label: str,
    require_runtime_guard: bool = False,
    require_source_invisible: bool = False,
) -> bool:
    """Replay a model-invisible candidate's three confirmation allocations."""
    value = read_json(path)
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise ReadinessError(f"wrong {label} confirmation schema")
    payload = dict(value)
    result_id = payload.pop("result_id", None)
    if result_id != bridge._fingerprint(payload):
        raise ReadinessError(f"{label} confirmation result ID changed")
    for key, expected in {
        "model_invoked": False,
        "application_source_modified": False,
        "provider_call_authorized": False,
    }.items():
        if value.get(key) is not expected:
            raise ReadinessError(f"{label} confirmation boundary changed: {key}")
    if value.get("correctness_gate", {}).get("passed") is not True:
        raise ReadinessError(f"{label} confirmation correctness gate failed")
    if (require_runtime_guard
            and value.get("runtime_guard_gate", {}).get("passed") is not True):
        raise ReadinessError(f"{label} runtime guard gate failed")
    if (require_source_invisible
            and value.get("application_source_visible_to_model") is not False):
        raise ReadinessError(f"{label} source-visibility boundary changed")
    transition_path = Path(value.get("transition", ""))
    if (not transition_path.is_absolute() or not transition_path.is_file()
            or sha256_file(transition_path) != value.get("transition_sha256")):
        raise ReadinessError(f"{label} confirmation transition changed")
    summaries = value.get("allocation_monitors")
    if not isinstance(summaries, list) or len(summaries) != 3:
        raise ReadinessError(f"{label} confirmation lacks three monitors")
    monitors = [Path(item.get("monitor", "")) for item in summaries]
    if any(not monitor.is_absolute() for monitor in monitors):
        raise ReadinessError(f"{label} confirmation monitor is not absolute")
    regenerated = analyzer.analyze_monitors(transition_path, monitors)
    if regenerated != value:
        raise ReadinessError(
            f"{label} confirmation does not replay from raw evidence"
        )
    gate = value.get("confirmation_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("passed"), bool):
        raise ReadinessError(f"{label} confirmation lacks its gate")
    return gate["passed"]


def classify_reused_loop_descriptor(
    entry: dict[str, Any], phase: str, analysis: Any | None,
    confirmation_phase: str = "missing",
    confirmation_passed: bool | None = None,
) -> dict[str, Any]:
    """Classify the model-invisible loop graph-expansion oracle."""
    if analysis is None:
        if confirmation_passed is not None:
            raise ReadinessError(
                "reused-loop confirmation exists without a passed scout"
            )
        status = {
            "waiting_predecessor": "awaiting_predecessor",
            "waiting_scheduler_idle": "awaiting_predecessor",
            "submitting": "awaiting_scout",
            "monitoring": "awaiting_scout",
            "analyzing": "awaiting_scout",
            "missing": "runtime_labels_missing",
            "failed": "scout_failed",
        }.get(phase)
        if status is None:
            raise ReadinessError(
                f"reused-loop-descriptor state {phase!r} requires an analysis"
            )
        next_stage = {
            "awaiting_predecessor": "wait_for_serial_compiler_headroom_campaign",
            "awaiting_scout": "wait_for_reused_loop_descriptor_scout",
            "runtime_labels_missing": (
                "establish_preregistered_compiler_oracle_headroom_first"
            ),
            "scout_failed": "diagnose_without_model_call",
        }[status]
        result = _base_entry(entry, status, next_stage)
    else:
        if (not isinstance(analysis, dict)
                or analysis.get("schema_version")
                != "gicc-reused-loop-descriptor-analysis-v1"
                or analysis.get("correctness_gate", {}).get("passed") is not True):
            raise ReadinessError(
                "reused-loop-descriptor scout violates its correctness gate"
            )
        gate = analysis.get("oracle_headroom_gate")
        if (
            not isinstance(gate, dict)
            or not isinstance(gate.get("passed"), bool)
            or gate.get("paper_claim") is not False
            or gate.get("provider_protocol_permitted") is not False
        ):
            raise ReadinessError(
                "reused-loop-descriptor scout violates its oracle gate"
            )
        expected_phase = "promising" if gate["passed"] else "negative"
        if phase != expected_phase:
            raise ReadinessError(
                "reused-loop-descriptor controller state disagrees with analysis"
            )
        if gate["passed"]:
            if confirmation_passed is None:
                result = _confirmation_pending_result(
                    entry, phase=confirmation_phase,
                    label="reused_loop_descriptor",
                )
            else:
                expected_confirmation_phase = (
                    "confirmed" if confirmation_passed else "negative"
                )
                if confirmation_phase != expected_confirmation_phase:
                    raise ReadinessError(
                        "reused-loop confirmation state disagrees with analysis"
                    )
                result = _hidden_candidate_result(
                    entry, confirmation_passed=confirmation_passed,
                )
        else:
            if confirmation_passed is not None:
                raise ReadinessError(
                    "reused-loop confirmation exists after a negative scout"
                )
            result = _base_entry(
                entry, "closed_negative",
                "keep_reused_loop_descriptor_model_invisible",
            )
        result["runtime_gate_passed"] = gate["passed"]
    result.update({
        "candidate_kind": "trigger_reused_descriptor_loop",
        "candidate_model_visible": False,
        "current_suite_graph_expanded": False,
    })
    return result


def _normalized_structural_graph(value: Any) -> tuple[dict[str, Any], str]:
    graph = dict(structural.verified_graph(value))
    graph.pop("graph_id")
    graph.pop("compiler_dossier_id")
    return graph, bridge._fingerprint(graph)


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    suite_value = read_json(args.suite)
    suite = decision_suite.verified_suite(suite_value, args.prompt_dir)
    entries = {entry["label"]: entry for entry in suite["entries"]}
    if set(entries) != EXPECTED_LABELS:
        raise ReadinessError(
            f"expected suite labels {sorted(EXPECTED_LABELS)}, got {sorted(entries)}"
        )

    current_graph_value = read_json(args.placement_current_graph)
    historical_graph_value = read_json(args.placement_historical_graph)
    current_graph = structural.verified_graph(current_graph_value)
    historical_graph = structural.verified_graph(historical_graph_value)
    placement_entry = entries["coalescing_placement"]
    if (current_graph["graph_id"] != placement_entry["graph_id"]
            or sha256_file(args.placement_current_graph)
            != placement_entry["graph_file_sha256"]):
        raise ReadinessError("suite does not bind the current placement graph")
    normalized_current, normalized_id = _normalized_structural_graph(
        current_graph
    )
    normalized_historical, historical_normalized_id = (
        _normalized_structural_graph(historical_graph)
    )
    graphs_equivalent = normalized_current == normalized_historical
    if normalized_id != historical_normalized_id or not graphs_equivalent:
        raise ReadinessError("historical/current placement graphs are not equivalent")
    placement_summary = read_json(args.placement_summary)
    if placement_summary.get("graph_id") != historical_graph["graph_id"]:
        raise ReadinessError("placement summary does not bind the historical graph")

    collective_analysis = (
        read_json(args.collective_analysis)
        if args.collective_analysis.is_file() else None
    )
    collective_confirmation_analysis = (
        read_json(args.collective_confirmation_analysis)
        if args.collective_confirmation_analysis.is_file() else None
    )
    producer_analysis = (
        read_json(args.producer_analysis)
        if args.producer_analysis.is_file() else None
    )
    producer_confirmation_analysis = (
        read_json(args.producer_confirmation_analysis)
        if args.producer_confirmation_analysis.is_file() else None
    )
    guarded_analysis = (
        read_json(args.guarded_analysis)
        if args.guarded_analysis.is_file() else None
    )
    guarded_confirmation_analysis = (
        read_json(args.guarded_confirmation_analysis)
        if args.guarded_confirmation_analysis.is_file() else None
    )
    reused_analysis = (
        read_json(args.reused_analysis)
        if args.reused_analysis.is_file() else None
    )
    reused_confirmation_analysis = (
        read_json(args.reused_confirmation_analysis)
        if args.reused_confirmation_analysis.is_file() else None
    )
    collective_confirmation_passed = (
        verified_collective_confirmation(
            args.collective_confirmation_analysis,
            entries["collective_n8"]["graph_id"],
        )
        if collective_confirmation_analysis is not None else None
    )
    producer_confirmation_passed = (
        verified_hidden_candidate_confirmation(
            args.producer_confirmation_analysis,
            schema="gicc-producer-fission-confirmation-v1",
            analyzer=producer_confirmation, label="producer-fission",
        )
        if producer_confirmation_analysis is not None else None
    )
    guarded_confirmation_passed = (
        verified_hidden_candidate_confirmation(
            args.guarded_confirmation_analysis,
            schema="gicc-guarded-early-trigger-confirmation-v1",
            analyzer=guarded_confirmation, label="guarded-trigger",
            require_runtime_guard=True,
        )
        if guarded_confirmation_analysis is not None else None
    )
    reused_confirmation_passed = (
        verified_hidden_candidate_confirmation(
            args.reused_confirmation_analysis,
            schema="gicc-reused-loop-descriptor-confirmation-v1",
            analyzer=reused_confirmation, label="reused-loop-descriptor",
            require_source_invisible=True,
        )
        if reused_confirmation_analysis is not None else None
    )
    guarded_graph_phase = "missing"
    guarded_expansion_present = args.guarded_expansion_manifest.is_file()
    guarded_refreeze_present = args.guarded_refreeze_manifest.is_file()
    if guarded_refreeze_present and not guarded_expansion_present:
        raise ReadinessError(
            "guarded-trigger suite refreeze lacks its expansion manifest"
        )
    if ((guarded_expansion_present or guarded_refreeze_present)
            and guarded_confirmation_passed is not True):
        raise ReadinessError(
            "guarded-trigger graph evidence exists without passed confirmation"
        )

    # Suite refreezes are serialized by the campaign controller.  If guarded
    # trigger is the final transition, its exact predecessor is the suite that
    # producer fission must bind.  This accepts only an A -> B hash lineage; it
    # does not treat either final graph as proof of the missing transition.
    producer_suite_path = args.suite
    producer_suite = suite
    if guarded_refreeze_present:
        guarded_manifest = verified_guarded_refreeze(
            args.guarded_refreeze_manifest,
            args.guarded_expansion_manifest,
            args.suite, suite, guarded_confirmation_analysis,
        )
        producer_suite_path, _, producer_suite = (
            verified_refreeze_predecessor(
                guarded_manifest, label="guarded-trigger refreeze",
            )
        )
        guarded_graph_phase = "suite_refrozen"
    elif guarded_expansion_present:
        expansion_manifest = verified_guarded_expansion(
            args.guarded_expansion_manifest,
            guarded_confirmation_analysis,
        )
        if entries["mm_minimal"]["graph_id"] != expansion_manifest[
                "graph_transition"]["current_graph_id"]:
            raise ReadinessError(
                "suite changed before guarded-trigger refreeze was audited"
            )
        guarded_graph_phase = "bundle_ready"

    producer_entries = {
        entry["label"]: entry for entry in producer_suite["entries"]
    }
    if set(producer_entries) != EXPECTED_LABELS:
        raise ReadinessError("guarded refreeze predecessor labels changed")
    producer_graph_phase = "missing"
    producer_expansion_present = args.producer_expansion_manifest.is_file()
    producer_refreeze_present = args.producer_refreeze_manifest.is_file()
    if producer_refreeze_present and not producer_expansion_present:
        raise ReadinessError(
            "producer-fission suite refreeze lacks its expansion manifest"
        )
    if ((producer_expansion_present or producer_refreeze_present)
            and producer_confirmation_passed is not True):
        raise ReadinessError(
            "producer-fission graph evidence exists without passed confirmation"
        )
    if producer_refreeze_present:
        verified_producer_refreeze(
            args.producer_refreeze_manifest,
            args.producer_expansion_manifest,
            producer_suite_path, producer_suite,
            producer_confirmation_analysis,
        )
        producer_graph_phase = "suite_refrozen"
    elif producer_expansion_present:
        expansion_manifest = verified_producer_expansion(
            args.producer_expansion_manifest,
            producer_confirmation_analysis,
        )
        if producer_entries["jacobi"]["graph_id"] != expansion_manifest[
                "graph_transition"]["current_graph_id"]:
            raise ReadinessError(
                "suite changed before producer-fission refreeze was audited"
            )
        producer_graph_phase = "bundle_ready"
    records = {
        "coalescing_placement": classify_placement(
            placement_entry, placement_summary, graphs_equivalent,
        ),
        "collective_n8": classify_collective(
            entries["collective_n8"], state_phase(args.collective_state),
            collective_analysis,
            state_phase(args.collective_confirmation_state),
            collective_confirmation_passed,
        ),
        "jacobi": classify_producer(
            entries["jacobi"], state_phase(args.producer_state),
            producer_analysis,
            state_phase(args.producer_confirmation_state),
            producer_confirmation_passed,
            producer_graph_phase,
        ),
        "mm_minimal": classify_guarded_early_trigger(
            entries["mm_minimal"], state_phase(args.guarded_state),
            guarded_analysis,
            state_phase(args.guarded_confirmation_state),
            guarded_confirmation_passed,
            guarded_graph_phase,
        ),
        "loop_lto": classify_reused_loop_descriptor(
            entries["loop_lto"], state_phase(args.reused_state),
            reused_analysis,
            state_phase(args.reused_confirmation_state),
            reused_confirmation_passed,
        ),
    }
    for label in CAPACITY_ONLY_LABELS:
        records[label] = _base_entry(
            entries[label], "runtime_labels_missing",
            "establish_preregistered_compiler_oracle_headroom_first",
        )

    ordered = {label: records[label] for label in sorted(records)}
    permitted = [
        label for label, record in ordered.items()
        if record["provider_protocol_permitted"]
    ]
    payload = {
        "schema_version": REPORT_SCHEMA,
        "suite_id": suite["suite_id"],
        "suite_file_sha256": sha256_file(args.suite),
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_modified": False,
            "provider_invoked": False,
            "provider_call_authorized": False,
            "scout_pass_authorizes_provider": False,
        },
        "placement_graph_equivalence": {
            "historical_graph_id": historical_graph["graph_id"],
            "current_graph_id": current_graph["graph_id"],
            "normalized_graph_id": normalized_id,
            "equal_after_removing_only_graph_and_dossier_ids": True,
        },
        "entries": ordered,
        "summary": {
            "entry_count": len(ordered),
            "provider_protocol_permitted_entries": permitted,
            "provider_protocol_permitted_count": len(permitted),
            "provider_call_authorized_count": 0,
            "llm_performance_measured_count": 0,
            "paper_llm_performance_claim_ready": False,
        },
        "evidence": {
            "suite": evidence(args.suite),
            "placement_summary": evidence(args.placement_summary),
            "placement_historical_graph": evidence(
                args.placement_historical_graph
            ),
            "placement_current_graph": evidence(args.placement_current_graph),
            "collective_state": evidence(args.collective_state),
            "collective_analysis": evidence(args.collective_analysis),
            "collective_confirmation_state": evidence(
                args.collective_confirmation_state
            ),
            "collective_confirmation_analysis": evidence(
                args.collective_confirmation_analysis
            ),
            "collective_confirmation_analyzer": evidence(
                HERE / "collective/analyze_compiler_collective_n8_confirmation.py"
            ),
            "producer_state": evidence(args.producer_state),
            "producer_analysis": evidence(args.producer_analysis),
            "producer_confirmation_state": evidence(
                args.producer_confirmation_state
            ),
            "producer_confirmation_analysis": evidence(
                args.producer_confirmation_analysis
            ),
            "producer_confirmation_analyzer": evidence(
                HERE / "producer_fission/analyze_producer_fission_confirmation.py"
            ),
            "producer_expansion_manifest": evidence(
                args.producer_expansion_manifest
            ),
            "producer_expansion_preparer": evidence(
                HERE / "producer_fission/"
                "prepare_confirmed_producer_fission_graph.py"
            ),
            "producer_refreeze_manifest": evidence(
                args.producer_refreeze_manifest
            ),
            "producer_refreeze_preparer": evidence(
                HERE / "producer_fission/"
                "prepare_producer_fission_suite_refreeze.py"
            ),
            "guarded_state": evidence(args.guarded_state),
            "guarded_analysis": evidence(args.guarded_analysis),
            "guarded_confirmation_state": evidence(
                args.guarded_confirmation_state
            ),
            "guarded_confirmation_analysis": evidence(
                args.guarded_confirmation_analysis
            ),
            "guarded_confirmation_analyzer": evidence(
                HERE
                / "guarded_early_trigger/"
                "analyze_guarded_early_trigger_confirmation.py"
            ),
            "guarded_expansion_manifest": evidence(
                args.guarded_expansion_manifest
            ),
            "guarded_expansion_preparer": evidence(
                HERE
                / "guarded_early_trigger/"
                "prepare_confirmed_guarded_early_graph.py"
            ),
            "guarded_refreeze_manifest": evidence(
                args.guarded_refreeze_manifest
            ),
            "guarded_refreeze_preparer": evidence(
                HERE
                / "guarded_early_trigger/"
                "prepare_guarded_early_suite_refreeze.py"
            ),
            "reused_state": evidence(args.reused_state),
            "reused_analysis": evidence(args.reused_analysis),
            "reused_confirmation_state": evidence(
                args.reused_confirmation_state
            ),
            "reused_confirmation_analysis": evidence(
                args.reused_confirmation_analysis
            ),
            "reused_confirmation_analyzer": evidence(
                HERE
                / "reused_loop_descriptor/"
                "analyze_reused_loop_descriptor_confirmation.py"
            ),
            "reused_confirmation_runner": evidence(
                HERE
                / "reused_loop_descriptor/"
                "run_reused_loop_descriptor_confirmation.sh"
            ),
            "reused_confirmation_monitor": evidence(
                HERE
                / "reused_loop_descriptor/"
                "monitor_reused_loop_descriptor_confirmation.py"
            ),
            "reused_confirmation_controller": evidence(
                HERE
                / "reused_loop_descriptor/"
                "continue_reused_loop_descriptor_confirmation.sh"
            ),
            "reused_confirmation_successor": evidence(
                HERE
                / "reused_loop_descriptor/"
                "continue_reused_loop_descriptor_after_scout.sh"
            ),
            "reused_confirmation_protocol": evidence(
                HERE
                / "reused_loop_descriptor/"
                "REUSED_LOOP_DESCRIPTOR_CONFIRMATION_TRANSITION.md"
            ),
            "reused_confirmation_preparer": evidence(
                HERE
                / "reused_loop_descriptor/"
                "prepare_reused_loop_descriptor_confirmation.py"
            ),
        },
    }
    result = dict(payload)
    result["readiness_id"] = bridge._fingerprint(payload)
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


def add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--placement-summary", type=Path, required=True)
    parser.add_argument("--placement-historical-graph", type=Path, required=True)
    parser.add_argument("--placement-current-graph", type=Path, required=True)
    parser.add_argument("--collective-state", type=Path, required=True)
    parser.add_argument("--collective-analysis", type=Path, required=True)
    parser.add_argument(
        "--collective-confirmation-state", type=Path,
        default=(
            ROOT / "build_ofi/compiler_collective_n8_confirmation_20260904.state"
        ),
    )
    parser.add_argument(
        "--collective-confirmation-analysis", type=Path,
        default=(
            ROOT
            / "build_ofi/compiler_collective_n8_confirmation_20260904/analysis.json"
        ),
    )
    parser.add_argument("--producer-state", type=Path, required=True)
    parser.add_argument("--producer-analysis", type=Path, required=True)
    parser.add_argument(
        "--producer-confirmation-state", type=Path,
        default=(
            ROOT / "build_ofi/"
            "producer_fission_confirmation_7687377_20260904.state"
        ),
    )
    parser.add_argument(
        "--producer-confirmation-analysis", type=Path,
        default=(
            ROOT / "build_ofi/"
            "producer_fission_confirmation_7687377_20260904/analysis.json"
        ),
    )
    parser.add_argument(
        "--producer-expansion-manifest", type=Path,
        default=(
            ROOT / "build_ofi/producer_fission_graph_expansion_20260904/"
            "manifest.json"
        ),
    )
    parser.add_argument(
        "--producer-refreeze-manifest", type=Path,
        default=(
            ROOT / "build_ofi/producer_fission_suite_refreeze_20260904/"
            "manifest.json"
        ),
    )
    parser.add_argument("--guarded-state", type=Path, required=True)
    parser.add_argument("--guarded-analysis", type=Path, required=True)
    parser.add_argument(
        "--guarded-confirmation-state", type=Path,
        default=(
            ROOT / "build_ofi/"
            "guarded_early_trigger_confirmation_77897d9_20260904.state"
        ),
    )
    parser.add_argument(
        "--guarded-confirmation-analysis", type=Path,
        default=(
            ROOT / "build_ofi/"
            "guarded_early_trigger_confirmation_77897d9_20260904/analysis.json"
        ),
    )
    parser.add_argument(
        "--guarded-expansion-manifest", type=Path,
        default=(
            ROOT / "build_ofi/guarded_early_trigger_graph_expansion_20260904/"
            "manifest.json"
        ),
    )
    parser.add_argument(
        "--guarded-refreeze-manifest", type=Path,
        default=(
            ROOT / "build_ofi/guarded_early_trigger_suite_refreeze_20260904/"
            "manifest.json"
        ),
    )
    parser.add_argument("--reused-state", type=Path, required=True)
    parser.add_argument("--reused-analysis", type=Path, required=True)
    parser.add_argument(
        "--reused-confirmation-state", type=Path,
        default=(
            ROOT / "build_ofi/"
            "reused_loop_descriptor_confirmation_aff76f9_20260904.state"
        ),
    )
    parser.add_argument(
        "--reused-confirmation-analysis", type=Path,
        default=(
            ROOT / "build_ofi/"
            "reused_loop_descriptor_confirmation_aff76f9_20260904/"
            "analysis.json"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit")
    add_inputs(emit)
    emit.add_argument("--out", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    add_inputs(verify)
    verify.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build_report(args)
        if args.command == "emit":
            write_json_atomic(args.out, result)
            action = "wrote"
        else:
            if read_json(args.report) != result:
                raise ReadinessError("readiness report does not match current evidence")
            action = "verified"
        summary = result["summary"]
        print(
            f"compiler-llm-readiness: {action} {len(result['entries'])} entries; "
            f"provider_protocol_permitted="
            f"{summary['provider_protocol_permitted_count']}; "
            f"provider_calls_authorized=0; readiness_id="
            f"{result['readiness_id']}"
        )
        return 0
    except (
        ReadinessError, decision_suite.SuiteError,
        structural.PlanBridgeError, n8_confirmation.ConfirmError,
        n8_confirmation.monitor_base.MonitorError,
        producer_confirmation.ConfirmError,
        producer_confirmation.common.MonitorError,
        producer_expansion.ExpansionError,
        producer_expansion.groups.GroupPlanError,
        producer_expansion.bridge.BridgeError,
        producer_refreeze.RefreezeError,
        producer_refreeze.suites.SuiteError,
        producer_refreeze.suites.collective.CollectivePlanError,
        producer_refreeze.suites.structural.PlanBridgeError,
        guarded_expansion.ExpansionError,
        guarded_expansion.groups.GroupPlanError,
        guarded_expansion.bridge.BridgeError,
        guarded_refreeze.RefreezeError,
        guarded_confirmation.ConfirmError,
        guarded_confirmation.common.MonitorError,
        reused_confirmation.ConfirmError,
        reused_confirmation.common.MonitorError,
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"compiler-llm-readiness: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
