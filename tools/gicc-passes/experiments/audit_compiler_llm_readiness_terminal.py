#!/usr/bin/env python3
"""Finalize compiler-only readiness with replayed terminal-negative evidence.

This is a conservative adapter over ``audit_compiler_llm_readiness``.  It does
not invent completed scout analyses for interrupted experiments.  Instead, it
first runs the existing suite/readiness audit, then replaces only the two
fail-closed placeholder statuses whose separately audited terminal-negative
evidence proves that the positive gate cannot be used.  The v1 readiness
schema is retained because all existing fields and authority semantics remain
compatible; the added evidence and fields are covered by a new fingerprint.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "collective"))

import analyze_compiler_collective_n6_confirmation as n6_confirmation  # noqa: E402
import audit_compiler_llm_readiness as base  # noqa: E402
import audit_compiler_terminal_negatives as terminal  # noqa: E402
import prepare_collective_n6_suite_refreeze as n6_refreeze  # noqa: E402


PROGRAM_NAME = "compiler-llm-readiness-terminal"
TERMINAL_LABELS = {"jacobi", "loop_lto", "mm_minimal"}


def configure_collective(label: str) -> None:
    """Select the topology-matched verifier before constructing the report."""
    if label == "collective_n6":
        base.EXPECTED_LABELS = (
            base.BASE_EXPECTED_LABELS - {"collective_n8"} | {"collective_n6"}
        )
        base.COLLECTIVE_LABEL = "collective_n6"
        base.COLLECTIVE_TOPOLOGY_LABEL = "n6"
        base.COLLECTIVE_SCOUT_SCHEMA = "gicc-collective-hierpipe-n6-scout-v1"
        base.COLLECTIVE_CAPACITY_GATE_KEY = "n6_capacity_gate"
        base.COLLECTIVE_CONFIRMATION_SCHEMA = n6_confirmation.base.RESULT_SCHEMA
        base.COLLECTIVE_CONFIRMATION = n6_confirmation.base
        base.COLLECTIVE_CONFIRMATION_ANALYZER = Path(
            n6_confirmation.__file__
        ).resolve()
        base.COLLECTIVE_REFREEZE = n6_refreeze
        base.COLLECTIVE_REFREEZE_ERROR = n6_refreeze.RefreezeError
        base.COLLECTIVE_REFREEZER = Path(n6_refreeze.__file__).resolve()
        return
    if label != "collective_n8":
        raise base.ReadinessError(f"unknown collective label: {label}")
    base.EXPECTED_LABELS = set(base.BASE_EXPECTED_LABELS)
    base.COLLECTIVE_LABEL = "collective_n8"
    base.COLLECTIVE_TOPOLOGY_LABEL = "n8"
    base.COLLECTIVE_SCOUT_SCHEMA = "gicc-collective-hierpipe-n8-scout-v1"
    base.COLLECTIVE_CAPACITY_GATE_KEY = "n8_capacity_gate"
    base.COLLECTIVE_CONFIRMATION_SCHEMA = "gicc-collective-n8-confirmation-v1"
    base.COLLECTIVE_CONFIRMATION = base.n8_confirmation
    base.COLLECTIVE_CONFIRMATION_ANALYZER = (
        HERE / "collective/analyze_compiler_collective_n8_confirmation.py"
    )
    base.COLLECTIVE_REFREEZE = None
    base.COLLECTIVE_REFREEZE_ERROR = base.ReadinessError
    base.COLLECTIVE_REFREEZER = (
        HERE / "collective/prepare_collective_n6_suite_refreeze.py"
    )


def apply_terminal_negatives(
    readiness: dict[str, Any], terminal_report: dict[str, Any],
    terminal_path: Path,
) -> dict[str, Any]:
    """Apply only authority-reducing terminal dispositions."""
    result = copy.deepcopy(readiness)
    candidates = terminal_report.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != TERMINAL_LABELS:
        raise base.ReadinessError(
            "terminal-negative report does not bind the expected candidates"
        )
    entries = result.get("entries")
    if not isinstance(entries, dict) or not TERMINAL_LABELS <= set(entries):
        raise base.ReadinessError(
            "readiness report lacks terminal-negative suite entries"
        )
    expected_initial = {
        "jacobi": {"scout_failed"},
        "mm_minimal": {"scout_failed"},
        "loop_lto": {"closed_negative"},
    }
    next_stages = {
        "jacobi": "keep_producer_fission_model_invisible",
        "loop_lto": "keep_reused_loop_descriptor_model_invisible",
        "mm_minimal": "keep_guarded_early_trigger_model_invisible",
    }
    for label in sorted(TERMINAL_LABELS):
        candidate = candidates[label]
        record = entries[label]
        if (
            not isinstance(candidate, dict)
            or candidate.get("status") != "closed_negative"
            or candidate.get("model_visible") is not False
            or candidate.get("confirmation_eligible") is not False
            or candidate.get("paper_performance_claim") is not False
            or candidate.get("runtime_values_are_positive_performance_evidence")
            is not False
        ):
            raise base.ReadinessError(
                f"{label}: terminal-negative disposition is not fail-closed"
            )
        if record.get("status") not in expected_initial[label]:
            raise base.ReadinessError(
                f"{label}: terminal overlay would replace non-terminal status "
                f"{record.get('status')!r}"
            )
        if (
            record.get("provider_protocol_permitted") is not False
            or record.get("provider_call_authorized") is not False
            or record.get("llm_performance_measured") is not False
        ):
            raise base.ReadinessError(
                f"{label}: terminal overlay would reduce existing authority"
            )
        record.update({
            "status": "closed_negative",
            "next_stage": next_stages[label],
            "provider_protocol_permitted": False,
            "provider_call_authorized": False,
            "llm_performance_measured": False,
            "candidate_model_visible": False,
            "current_suite_graph_expanded": False,
            "terminal_negative_reason_code": candidate["reason_code"],
            "terminal_report_sha256": candidate["terminal_report_sha256"],
            "terminal_negatives_id": terminal_report["terminal_negatives_id"],
            "runtime_values_are_positive_performance_evidence": False,
        })
        if label == "jacobi":
            record["compiler_correctness_gate_passed"] = False
        elif label == "mm_minimal":
            record["positive_gate_mathematically_reachable"] = False

    permitted = sorted(
        label for label, record in entries.items()
        if record.get("provider_protocol_permitted") is True
    )
    if TERMINAL_LABELS & set(permitted):
        raise base.ReadinessError(
            "terminal-negative candidate became provider-protocol eligible"
        )
    summary = result.get("summary")
    if not isinstance(summary, dict):
        raise base.ReadinessError("readiness report lacks a summary")
    summary.update({
        "provider_protocol_permitted_entries": permitted,
        "provider_protocol_permitted_count": len(permitted),
        "provider_call_authorized_count": 0,
        "llm_performance_measured_count": 0,
        "paper_llm_performance_claim_ready": False,
        "terminal_negative_entries": sorted(TERMINAL_LABELS),
        "terminal_negative_count": len(TERMINAL_LABELS),
    })
    old_id = result.pop("readiness_id", None)
    if not isinstance(old_id, str):
        raise base.ReadinessError("base readiness report lacks an identity")
    result["pre_terminal_overlay_readiness_id"] = old_id
    result["terminal_negatives_id"] = terminal_report["terminal_negatives_id"]
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        raise base.ReadinessError("base readiness report lacks evidence")
    evidence["terminal_negatives"] = base.evidence(terminal_path)
    result["readiness_id"] = base.bridge._fingerprint(result)
    return result


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    configure_collective(args.collective_label)
    base_report = base.build_report(args)
    terminal_report = terminal.verify_contained(args.terminal_negatives)
    return apply_terminal_negatives(
        base_report, terminal_report, args.terminal_negatives
    )


def add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--collective-label",
        choices=("collective_n8", "collective_n6"),
        required=True,
    )
    parser.add_argument("--terminal-negatives", type=Path, required=True)
    base.add_inputs(parser)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("command", choices=("emit", "verify"))
    probe.add_argument(
        "--collective-label",
        choices=("collective_n8", "collective_n6"),
        required=True,
    )
    known, _ = probe.parse_known_args(arguments)
    configure_collective(known.collective_label)

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit")
    add_inputs(emit)
    emit.add_argument("--out", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    add_inputs(verify)
    verify.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(arguments)
    try:
        result = build_report(args)
        if args.command == "emit":
            base.write_json_atomic(args.out, result)
            action = "wrote"
        else:
            if base.read_json(args.report) != result:
                raise base.ReadinessError(
                    "terminal readiness report does not match current evidence"
                )
            action = "verified"
        summary = result["summary"]
        print(
            f"{PROGRAM_NAME}: {action} {len(result['entries'])} entries; "
            f"terminal_negative={summary['terminal_negative_count']}; "
            f"provider_protocol_permitted="
            f"{summary['provider_protocol_permitted_count']}; "
            f"provider_calls_authorized=0; readiness_id="
            f"{result['readiness_id']}"
        )
        return 0
    except Exception as exc:
        print(f"{PROGRAM_NAME}: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
