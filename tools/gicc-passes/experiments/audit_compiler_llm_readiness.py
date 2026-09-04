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

import analyze_compiler_collective_n8_confirmation as n8_confirmation  # noqa: E402
import gicc_comm_plan_bridge as structural  # noqa: E402
import gicc_compiler_decision_suite as decision_suite  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


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


def classify_producer(entry: dict[str, Any], phase: str,
                      analysis: Any | None) -> dict[str, Any]:
    if analysis is None:
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
        result = _base_entry(
            entry, "confirmation_required",
            "freeze_and_run_confirmatory_compiler_oracle",
        )
    else:
        result = _base_entry(
            entry, "closed_negative", "mask_producer_fission_from_model",
        )
    result["runtime_gate_passed"] = gate["passed"]
    return result


def classify_guarded_early_trigger(entry: dict[str, Any], phase: str,
                                   analysis: Any | None) -> dict[str, Any]:
    if analysis is None:
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
        result = _base_entry(
            entry, "confirmation_required",
            "freeze_and_run_confirmatory_compiler_oracle",
        )
    else:
        result = _base_entry(
            entry, "closed_negative",
            "mask_guarded_early_trigger_from_model",
        )
    result["runtime_gate_passed"] = gate["passed"]
    return result


def classify_reused_loop_descriptor(entry: dict[str, Any], phase: str,
                                    analysis: Any | None) -> dict[str, Any]:
    """Classify the model-invisible loop graph-expansion oracle."""
    if analysis is None:
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
            result = _base_entry(
                entry, "confirmation_required",
                "freeze_and_run_confirmatory_graph_expansion_oracle",
            )
        else:
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
    guarded_analysis = (
        read_json(args.guarded_analysis)
        if args.guarded_analysis.is_file() else None
    )
    reused_analysis = (
        read_json(args.reused_analysis)
        if args.reused_analysis.is_file() else None
    )
    collective_confirmation_passed = (
        verified_collective_confirmation(
            args.collective_confirmation_analysis,
            entries["collective_n8"]["graph_id"],
        )
        if collective_confirmation_analysis is not None else None
    )
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
        ),
        "mm_minimal": classify_guarded_early_trigger(
            entries["mm_minimal"], state_phase(args.guarded_state),
            guarded_analysis,
        ),
        "loop_lto": classify_reused_loop_descriptor(
            entries["loop_lto"], state_phase(args.reused_state),
            reused_analysis,
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
            "guarded_state": evidence(args.guarded_state),
            "guarded_analysis": evidence(args.guarded_analysis),
            "reused_state": evidence(args.reused_state),
            "reused_analysis": evidence(args.reused_analysis),
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
    parser.add_argument("--guarded-state", type=Path, required=True)
    parser.add_argument("--guarded-analysis", type=Path, required=True)
    parser.add_argument("--reused-state", type=Path, required=True)
    parser.add_argument("--reused-analysis", type=Path, required=True)


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
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"compiler-llm-readiness: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
