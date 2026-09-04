#!/usr/bin/env python3
"""Freeze the suite-level, compiler-only LLM capability evaluation contract.

This tool is deliberately incapable of invoking a provider, compiler, or
scheduler.  It binds the frozen decision suite to the independently derived
input-separation and runtime-readiness audits.  Entries without confirmed
compiler-oracle headroom remain in the report with zero permitted calls.
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
sys.path.insert(0, str(HERE.parent / "python"))
METRICS_IMPLEMENTATION = HERE.parent / "python" / "gicc_llm_capability_metrics.py"

import gicc_compiler_decision_suite as decision_suite  # noqa: E402


REPORT_SCHEMA = "gicc-compiler-llm-capability-protocol-v1"
READINESS_SCHEMA = "gicc-compiler-llm-readiness-v1"
SEPARATION_SCHEMA = "gicc-compiler-input-separation-v1"
VIEWS = ("relational", "descriptors", "opaque")
TRIALS_PER_VIEW = 20


class CapabilityProtocolError(RuntimeError):
    """The frozen inputs cannot support the claimed evaluation contract."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapabilityProtocolError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise CapabilityProtocolError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def evidence(path: Path) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def verified_fingerprint(value: Any, schema: str, id_key: str) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != schema:
        raise CapabilityProtocolError(f"expected {schema}")
    payload = dict(value)
    observed = payload.pop(id_key, None)
    if observed != fingerprint(payload):
        raise CapabilityProtocolError(f"{id_key} does not match report content")
    return value


def verify_separation(value: Any, suite: dict[str, Any],
                      suite_sha256: str) -> dict[str, Any]:
    report = verified_fingerprint(
        value, SEPARATION_SCHEMA, "audit_id",
    )
    if report.get("suite_id") != suite["suite_id"]:
        raise CapabilityProtocolError("input audit binds another suite")
    if report.get("evidence", {}).get("suite", {}).get("sha256") != suite_sha256:
        raise CapabilityProtocolError("input audit binds different suite bytes")
    if report.get("boundary") != {
        "application_source_visible": False,
        "application_source_modified": False,
        "model_invoked": False,
        "provider_invoked": False,
        "compiler_lto_decisions_only": True,
    }:
        raise CapabilityProtocolError("input audit violates compiler-only boundary")
    interpretation = report.get("controlled_interpretation", {})
    required = {
        "gbt_and_llm_inputs_identical": False,
        "llm_has_richer_relational_compiler_context": True,
        "same_selectable_ids_across_llm_information_ablation": True,
        "gbt_vs_llm_is_not_an_action-controlled_model_comparison": True,
        "valid_future_llm_ablation": list(VIEWS),
        "performance_superiority_claimed": False,
    }
    if any(interpretation.get(key) != wanted for key, wanted in required.items()):
        raise CapabilityProtocolError("input audit does not support controlled ablation")
    return report


def verify_readiness(value: Any, suite: dict[str, Any],
                     suite_sha256: str) -> dict[str, Any]:
    report = verified_fingerprint(value, READINESS_SCHEMA, "readiness_id")
    if (report.get("suite_id") != suite["suite_id"]
            or report.get("suite_file_sha256") != suite_sha256):
        raise CapabilityProtocolError("readiness report binds another suite")
    boundary = report.get("boundary", {})
    if boundary != {
        "compiler_lto_decisions_only": True,
        "application_source_modified": False,
        "provider_invoked": False,
        "provider_call_authorized": False,
        "scout_pass_authorizes_provider": False,
    }:
        raise CapabilityProtocolError("readiness violates compiler-only boundary")
    summary = report.get("summary", {})
    if (summary.get("provider_call_authorized_count") != 0
            or summary.get("llm_performance_measured_count") != 0
            or summary.get("paper_llm_performance_claim_ready") is not False):
        raise CapabilityProtocolError("readiness already claims unauthorized evidence")
    return report


def entry_contract(entry: dict[str, Any], readiness: dict[str, Any],
                   separation: dict[str, Any]) -> dict[str, Any]:
    label = entry["label"]
    ready = readiness.get("entries", {}).get(label)
    inputs = separation.get("entries", {}).get(label)
    if not isinstance(ready, dict) or not isinstance(inputs, dict):
        raise CapabilityProtocolError(f"{label}: missing audit entry")
    if (ready.get("suite_entry_id") != entry["entry_id"]
            or ready.get("compiler_graph_id") != entry["graph_id"]
            or ready.get("decision_family") != entry["decision_family"]):
        raise CapabilityProtocolError(f"{label}: readiness identity mismatch")
    if inputs.get("decision_family") != entry["decision_family"]:
        raise CapabilityProtocolError(f"{label}: input-audit family mismatch")
    if inputs.get("same_selectable_ids_across_llm_views") is not True:
        raise CapabilityProtocolError(f"{label}: LLM views have unequal authority")
    for view in VIEWS:
        audited = inputs.get("views", {}).get(view, {}).get("prompt", {})
        frozen = entry.get("views", {}).get(view, {})
        if (audited.get("sha256") != frozen.get("prompt_sha256")
                or audited.get("bytes") != frozen.get("prompt_bytes")):
            raise CapabilityProtocolError(f"{label}/{view}: prompt identity mismatch")

    permitted = ready.get("provider_protocol_permitted") is True
    if permitted and ready.get("status") != "provider_protocol_permitted":
        raise CapabilityProtocolError(f"{label}: inconsistent provider readiness")
    if ready.get("provider_call_authorized") is not False:
        raise CapabilityProtocolError(f"{label}: provider call already authorized")
    if ready.get("llm_performance_measured") is not False:
        raise CapabilityProtocolError(f"{label}: model performance already claimed")
    if permitted and ready.get("candidate_model_visible") is False:
        raise CapabilityProtocolError(f"{label}: model-invisible candidate is ineligible")

    return {
        "suite_entry_id": entry["entry_id"],
        "compiler_graph_id": entry["graph_id"],
        "decision_family": entry["decision_family"],
        "runtime_readiness_status": ready.get("status"),
        "next_runtime_stage": ready.get("next_stage"),
        "eligible_to_freeze_provider_request": permitted,
        "provider_call_authorized": False,
        "current_permitted_provider_calls": 0,
        "conditional_protocol": {
            "views": list(VIEWS),
            "independent_responses_per_view": TRIALS_PER_VIEW,
            "calls_if_entry_becomes_eligible": TRIALS_PER_VIEW * len(VIEWS),
            "same_selectable_ids_across_views": True,
            "prompt_sha256_by_view": {
                view: entry["views"][view]["prompt_sha256"] for view in VIEWS
            },
            "response_schema_sha256": entry["response_schema"]["file_sha256"],
        },
    }


def build_report(suite_path: Path, prompt_dir: Path, readiness_path: Path,
                 separation_path: Path) -> dict[str, Any]:
    suite = decision_suite.verified_suite(read_json(suite_path), prompt_dir)
    suite_sha256 = sha256_file(suite_path)
    readiness = verify_readiness(
        read_json(readiness_path), suite, suite_sha256,
    )
    separation = verify_separation(
        read_json(separation_path), suite, suite_sha256,
    )
    entries = {
        entry["label"]: entry_contract(entry, readiness, separation)
        for entry in suite["entries"]
    }
    entries = {label: entries[label] for label in sorted(entries)}
    eligible = [
        label for label, entry in entries.items()
        if entry["eligible_to_freeze_provider_request"]
    ]
    readiness_permitted = readiness["summary"].get(
        "provider_protocol_permitted_entries"
    )
    if eligible != sorted(readiness_permitted or []):
        raise CapabilityProtocolError("readiness summary disagrees with its entries")

    payload = {
        "schema_version": REPORT_SCHEMA,
        "status": (
            "eligible_entries_require_separate_content_addressed_authorization"
            if eligible else "blocked_no_runtime_eligible_entries"
        ),
        "suite_id": suite["suite_id"],
        "readiness_id": readiness["readiness_id"],
        "input_separation_audit_id": separation["audit_id"],
        "boundary": {
            "local_protocol_audit_only": True,
            "compiler_lto_decisions_only": True,
            "application_source_visible_to_model": False,
            "application_source_modified": False,
            "model_output_is_existing_graph_bound_ids_only": True,
            "compiler_revalidates_before_materialization": True,
            "evaluation_runtime_labels_visible_to_model": False,
            "provider_request_frozen": False,
            "provider_call_authorized": False,
            "provider_invoked": False,
            "scheduler_invoked": False,
        },
        "entries": entries,
        "trial_design": {
            "views": list(VIEWS),
            "independent_responses_per_view": TRIALS_PER_VIEW,
            "fresh_stateless_context_per_response": True,
            "same_model_decoding_and_response_schema_across_views": True,
            "view_order": (
                "deterministic rotating relational/descriptors/opaque order "
                "across response index"
            ),
            "all_raw_responses_archived_before_validation": True,
            "invalid_response_policy": "atomic compiler-anchor fallback",
            "runtime_labels_and_oracle_are_held_out_from_all_prompts": True,
        },
        "preregistered_scoring": {
            "intention_to_treat": [
                "invalid_output_rate",
                "oracle_normalized_geomean_regret_with_fallback",
                "deterministic_compiler_control_normalized_regret",
                "exact_oracle_policy_rate",
                "mean_decision_slot_accuracy",
            ],
            "stability": [
                "modal_accepted_policy_rate",
                "unique_accepted_policy_count",
                "accepted_policy_entropy",
            ],
            "representatives": {
                "primary": "modal_policy_per_view",
                "capability_upper_bound": "best_of_20_accepted_policies_per_view",
                "upper_bound_is_posthoc": True,
            },
            "semantic_context_effect": (
                "compare relational/descriptors/opaque under the identical "
                "compiler action set; never interpret scalar GBT versus LLM "
                "as an action-controlled model comparison"
            ),
            "offline_oracle_scoring_is_runtime_speedup_evidence": False,
        },
        "runtime_validation": {
            "required_for_every_reported_representative": True,
            "compile_through_strict_lto_bridge": True,
            "audit_materialized_ir": True,
            "matching_build_and_topology": True,
            "same_allocation_paired_blocks": True,
            "queue": "pdebug",
            "maximum_active_or_queued_campaign_jobs": 1,
            "deduplicate_identical_representatives_before_measurement": True,
        },
        "paper_claim_separation": {
            "method_claim": (
                "compiler-only graph decisions and equal-authority LLM "
                "ablations are implemented"
            ),
            "stable_policy_claim": "uses the preregistered modal representative",
            "capability_ceiling_claim": (
                "uses best-of-20 only and must be labeled post-hoc upper bound"
            ),
            "relational_context_claim": (
                "requires relational improvement over opaque/descriptors with "
                "identical selectable IDs"
            ),
            "performance_claim": (
                "requires paired runtime validation against semantic anchor "
                "and deterministic compiler control"
            ),
            "portfolio_generalization_claim": (
                "requires at least two independent eligible entries spanning "
                "at least two compiler decision families"
            ),
            "current_performance_claim_ready": False,
        },
        "summary": {
            "entry_count": len(entries),
            "runtime_eligible_entry_count": len(eligible),
            "runtime_eligible_entries": eligible,
            "provider_requests_frozen_count": 0,
            "provider_calls_authorized_count": 0,
            "provider_calls_made_count": 0,
            "currently_permitted_provider_calls": 0,
            "conditional_calls_per_eligible_entry": (
                TRIALS_PER_VIEW * len(VIEWS)
            ),
            "paper_llm_performance_claim_ready": False,
        },
        "implementation": {
            "capability_metrics": evidence(METRICS_IMPLEMENTATION),
        },
        "evidence": {
            "suite": evidence(suite_path),
            "readiness": evidence(readiness_path),
            "input_separation": evidence(separation_path),
        },
    }
    result = dict(payload)
    result["protocol_id"] = fingerprint(payload)
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
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--input-separation", type=Path, required=True)


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
        result = build_report(
            args.suite, args.prompt_dir, args.readiness, args.input_separation,
        )
        if args.command == "emit":
            write_json_atomic(args.out, result)
            action = "wrote"
        else:
            if read_json(args.report) != result:
                raise CapabilityProtocolError(
                    "capability protocol does not match current evidence"
                )
            action = "verified"
        print(
            f"compiler-llm-capability-protocol: {action}; "
            f"eligible={result['summary']['runtime_eligible_entry_count']}; "
            f"authorized_calls=0; protocol_id={result['protocol_id']}"
        )
        return 0
    except (CapabilityProtocolError, decision_suite.SuiteError, OSError,
            KeyError, TypeError, ValueError) as exc:
        print(f"compiler-llm-capability-protocol: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
