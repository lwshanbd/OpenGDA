#!/usr/bin/env python3
"""Derive the pre-inference paper claim matrix from final compiler evidence.

This audit distinguishes a compiler-method result, compiler-oracle headroom,
and actual LLM performance.  It runs only after the N6 runtime chain reaches a
terminal state and cannot invoke a compiler, scheduler, model, or provider.
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
sys.path.insert(0, str(HERE))

import audit_compiler_action_authority as authority_base  # noqa: E402
import audit_compiler_llm_capability_protocol as protocol_audit  # noqa: E402
import audit_compiler_terminal_negatives as terminal_audit  # noqa: E402
import audit_llm_sampling_null as sampling_audit  # noqa: E402
import gicc_compiler_decision_suite as decision_suite  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


REPORT_SCHEMA = "gicc-compiler-llm-mainline-claims-v1"
TERMINAL_N6_PHASES = {
    "confirmed", "negative", "skipped", "skipped_no_incremental_policy",
    "failed",
}
BOUNDARY = {
    "compiler_lto_decisions_only": True,
    "application_source_visible_to_model": False,
    "application_source_modified": False,
    "model_output_can_modify_application_source": False,
    "model_invoked": False,
    "provider_invoked": False,
    "provider_call_authorized": False,
    "provider_request_frozen": False,
    "compiler_invoked": False,
    "scheduler_invoked": False,
    "runtime_benchmark_invoked": False,
}


class MainlineClaimError(RuntimeError):
    """Final compiler evidence cannot support the requested claim matrix."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise MainlineClaimError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MainlineClaimError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise MainlineClaimError(f"cannot hash {path}: {exc}") from exc
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


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"missing mainline evidence: {resolved}")
    return {
        "path": display_path(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def verify_evidence(record: Any, *, label: str) -> Path:
    require(
        isinstance(record, dict)
        and set(record) == {"path", "sha256", "bytes"}
        and isinstance(record.get("path"), str)
        and isinstance(record.get("sha256"), str)
        and isinstance(record.get("bytes"), int)
        and not isinstance(record.get("bytes"), bool),
        f"malformed {label} evidence",
    )
    path = recorded_path(record["path"])
    require(
        path.is_file()
        and sha256_file(path) == record["sha256"]
        and path.stat().st_size == record["bytes"],
        f"{label} evidence changed",
    )
    return path


def state_record(path: Path) -> tuple[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MainlineClaimError(f"cannot read N6 chain state: {exc}") from exc
    require(len(lines) == 1, "N6 chain state is not one record")
    fields = lines[0].split("\t")
    require(len(fields) == 3 and fields[1] in TERMINAL_N6_PHASES,
            "N6 chain has not reached a terminal state")
    return fields[1], fields[2]


def _verified_fingerprint(
    value: Any, *, schema: str, id_key: str, label: str,
) -> dict[str, Any]:
    require(isinstance(value, dict) and value.get("schema_version") == schema,
            f"unexpected {label} schema")
    payload = dict(value)
    require(payload.pop(id_key, None) == bridge._fingerprint(payload),
            f"{label} identity does not match content")
    return value


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    suite_path = args.suite.resolve()
    prompt_dir = args.prompt_dir.resolve()
    suite = decision_suite.verified_suite(read_json(suite_path), prompt_dir)
    suite_sha256 = sha256_file(suite_path)
    entries = {entry["label"]: entry for entry in suite["entries"]}
    collective_labels = set(entries) & {"collective_n8", "collective_n6"}
    require(len(collective_labels) == 1,
            "final suite does not contain exactly one collective topology")
    collective_label = next(iter(collective_labels))

    terminal = terminal_audit.verify_contained(args.terminal_negatives)
    readiness = protocol_audit.verify_readiness(
        read_json(args.readiness), suite, suite_sha256,
    )
    require(
        readiness.get("terminal_negatives_id")
        == terminal["terminal_negatives_id"]
        and readiness.get("summary", {}).get("terminal_negative_count") == 3,
        "readiness does not bind all terminal-negative candidates",
    )

    authority = authority_base.verify_report(read_json(args.action_authority))
    authority_base.verify_recorded_evidence(authority.get("evidence"))
    require(authority.get("suite_id") == suite["suite_id"],
            "action authority binds another suite")
    authority_terminal = authority.get(
        "conditional_compiler_policy_authority", {}
    )
    require(
        authority.get("terminal_negatives_id")
        == terminal["terminal_negatives_id"]
        and set(authority_terminal.get("closed_terminal_entries", []))
        == set(terminal["candidates"])
        and authority_terminal.get("unresolved_conditional_entries") == []
        and authority_terminal.get("realized_current_entries") == []
        and authority_terminal.get("runtime_unconfirmed_transform_count") == 0
        and authority_terminal.get("model_visible_transform_count") == 0,
        "action authority did not close the terminal-negative frontier",
    )

    sampling = sampling_audit.verify_report(read_json(args.sampling_null))
    authority_from_sampling = verify_evidence(
        sampling.get("evidence", {}).get("action_authority"),
        label="sampling-null action authority",
    )
    require(
        authority_from_sampling == args.action_authority.resolve()
        and sampling.get("suite_id") == suite["suite_id"]
        and sampling.get("frontier_resolution", {}).get(
            "closed_terminal_entries"
        ) == sorted(terminal["candidates"])
        and sampling.get("frontier_resolution", {}).get(
            "unresolved_conditional_entries"
        ) == [],
        "sampling null did not remove the terminal-negative frontier",
    )

    expected_protocol = protocol_audit.build_report(
        suite_path, prompt_dir, args.readiness.resolve(),
        args.input_separation.resolve(), args.sampling_null.resolve(),
    )
    observed_protocol = read_json(args.capability_protocol)
    require(observed_protocol == expected_protocol,
            "capability protocol does not regenerate")
    protocol = _verified_fingerprint(
        observed_protocol,
        schema=protocol_audit.REPORT_SCHEMA,
        id_key="protocol_id",
        label="capability protocol",
    )
    n6_phase, n6_detail = state_record(args.n6_chain_state)
    eligible = protocol.get("summary", {}).get("runtime_eligible_entries")
    require(isinstance(eligible, list)
            and all(isinstance(label, str) for label in eligible),
            "capability protocol has invalid eligible entries")
    if collective_label == "collective_n6":
        require(n6_phase == "confirmed" and eligible == ["collective_n6"],
                "N6 suite lacks its sole confirmed eligible entry")
        status = "single_family_evaluation_requires_exact_authorization"
    else:
        require(n6_phase != "confirmed" and eligible == [],
                "N8 suite retained despite a confirmed or eligible N6 result")
        status = "method_and_terminal_negative_result_only"

    current_nonroute = authority.get("claim_separation", {}).get(
        "current_nonroute_compiler_candidate_or_option_ids"
    )
    require(isinstance(current_nonroute, int) and current_nonroute > 0,
            "action authority does not prove a wider compiler interface")
    claims = {
        "compiler_only_decision_method_implemented": True,
        "application_source_unchanged_boundary_enforced": True,
        "compiler_action_space_expressibility_supported": True,
        "richer_relational_input_than_frozen_scalar_gbt_supported": True,
        "equal_authority_llm_information_ablation_frozen": True,
        "terminal_negative_compiler_actions": 3,
        "confirmed_new_compiler_oracle_headroom_entry_count": (
            1 if collective_label == "collective_n6" else 0
        ),
        "runtime_eligible_llm_entry_count": len(eligible),
        "provider_request_frozen_count": 0,
        "provider_calls_authorized_count": 0,
        "provider_calls_made_count": 0,
        "new_llm_policies_measured_count": 0,
        "stable_llm_runtime_improvement_supported": False,
        "relational_context_runtime_effect_supported": False,
        "llm_over_gbt_superiority_supported": False,
        "llm_over_structured_ml_superiority_supported": False,
        "cross_family_generalization_supported": False,
        "paper_llm_performance_claim_ready": False,
    }
    payload = {
        "schema_version": REPORT_SCHEMA,
        "status": status,
        "suite_id": suite["suite_id"],
        "collective_label": collective_label,
        "n6_chain": {"phase": n6_phase, "detail": n6_detail},
        "boundary": dict(BOUNDARY),
        "claim_matrix": claims,
        "interpretation": {
            "compiler_interface_width_is_not_llm_quality": True,
            "terminal_negative_actions_are_not_evidence_of_weak_llm_reasoning": True,
            "single_family_evidence_cannot_support_portfolio_generalization": True,
            "next_stage": (
                "freeze one exact zero-authority request and ask for matching authorization"
                if eligible
                else "report method and negative compiler-oracle result; do not call a model"
            ),
        },
        "evidence": {
            "auditor": evidence(Path(__file__)),
            "suite": evidence(suite_path),
            "terminal_negatives": evidence(args.terminal_negatives),
            "readiness": evidence(args.readiness),
            "action_authority": evidence(args.action_authority),
            "input_separation": evidence(args.input_separation),
            "sampling_null": evidence(args.sampling_null),
            "capability_protocol": evidence(args.capability_protocol),
            "n6_chain_state": evidence(args.n6_chain_state),
        },
        "identities": {
            "terminal_negatives_id": terminal["terminal_negatives_id"],
            "readiness_id": readiness["readiness_id"],
            "authority_id": authority["authority_id"],
            "sampling_null_id": sampling["null_id"],
            "protocol_id": protocol["protocol_id"],
        },
        "current_nonroute_compiler_candidate_or_option_ids": current_nonroute,
        "runtime_eligible_entries": eligible,
    }
    return {**payload, "claim_audit_id": bridge._fingerprint(payload)}


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
    parser.add_argument("--terminal-negatives", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--action-authority", type=Path, required=True)
    parser.add_argument("--input-separation", type=Path, required=True)
    parser.add_argument("--sampling-null", type=Path, required=True)
    parser.add_argument("--capability-protocol", type=Path, required=True)
    parser.add_argument("--n6-chain-state", type=Path, required=True)


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
                raise MainlineClaimError(
                    "mainline claim report does not match current evidence"
                )
            action = "verified"
        claims = result["claim_matrix"]
        print(
            f"compiler-llm-mainline-claims: {action}; status={result['status']}; "
            f"eligible={claims['runtime_eligible_llm_entry_count']}; "
            f"terminal_negative={claims['terminal_negative_compiler_actions']}; "
            f"llm_performance_claim_ready=false; claim_audit_id="
            f"{result['claim_audit_id']}"
        )
        return 0
    except Exception as exc:
        print(f"compiler-llm-mainline-claims: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
