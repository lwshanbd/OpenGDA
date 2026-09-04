#!/usr/bin/env python3
"""Freeze a non-executable proposal for exact compiler LLM authorization."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PASS_PYTHON = HERE.parent / "python"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PASS_PYTHON))

import gicc_llm_bridge as bridge  # noqa: E402
import prepare_compiler_llm_capability_request as request_freezer  # noqa: E402
import run_compiler_llm_capability_trials as trials  # noqa: E402


PROPOSAL_SCHEMA = "gicc-compiler-llm-authorization-proposal-v1"


class AuthorizationProposalError(RuntimeError):
    """An exact, non-executable authorization proposal cannot be verified."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorizationProposalError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthorizationProposalError(f"cannot read JSON {path}: {exc}") from exc


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


def authorization(
    request: dict[str, Any], *, executable: str, cli_version: str,
    model: str, effort: str, maximum_attempts: int, timeout_seconds: int,
) -> dict[str, Any]:
    payload = {
        "schema_version": trials.AUTHORIZATION_SCHEMA,
        "granted": True,
        "request_id": request["request_id"],
        "provider_delivery": trials.delivery_contract(request),
        "data_boundary": {
            "model_tools": [],
            "source_visible": False,
            "source_locations_visible": False,
            "llvm_ir_visible": False,
            "evaluation_runtime_labels_visible": False,
            "evaluation_oracle_visible": False,
            "model_may_generate_code_or_ir": False,
        },
        "provider": {
            "kind": "anthropic-claude-cli",
            "executable": executable,
            "cli_version": cli_version,
            "requested_model": model,
            "effort": effort,
            "fresh_session_per_trial": True,
            "structured_output": True,
        },
        "transport_retry": {
            "max_attempts_per_trial": maximum_attempts,
            "timeout_seconds": timeout_seconds,
        },
    }
    result = {"authorization_id": bridge._fingerprint(payload), **payload}
    trials.verify_authorization(result, request)
    return result


def proposal_payload(
    request: dict[str, Any], proposed: dict[str, Any],
) -> dict[str, Any]:
    semantic_trials = proposed["provider_delivery"][
        "total_conditional_calls"
    ]
    attempts_per_trial = proposed["transport_retry"][
        "max_attempts_per_trial"
    ]
    return {
        "schema_version": PROPOSAL_SCHEMA,
        "status": "awaiting_explicit_exact_user_authorization",
        "request_id": request["request_id"],
        "proposed_authorization_id": proposed["authorization_id"],
        "proposed_authorization": proposed,
        "proposed_provider_budget": {
            "semantic_trials": semantic_trials,
            "max_transport_attempts_per_trial": attempts_per_trial,
            "maximum_provider_process_invocations": (
                semantic_trials * attempts_per_trial
            ),
        },
        "boundary": {
            "proposal_is_not_an_authorization_file": True,
            "runner_rejects_this_outer_schema": True,
            "authorization_granted_by_proposal": False,
            "currently_permitted_provider_calls": 0,
            "provider_cli_invoked": False,
            "provider_inference_invoked": False,
            "application_source_visible_or_modified": False,
            "model_output_authority": "existing graph-bound compiler IDs only",
        },
        "grant_contract": {
            "user_must_explicitly_repeat_proposal_id": True,
            "user_must_explicitly_repeat_authorization_id": True,
            "grant_materializes_exact_embedded_authorization_only": True,
            "request_or_provider_setting_change_requires_new_proposal": True,
        },
    }


def build_proposal(args: argparse.Namespace) -> dict[str, Any]:
    inputs = request_freezer.verified_inputs(
        args.suite.resolve(), args.prompt_dir.resolve(),
        args.readiness.resolve(), args.input_separation.resolve(),
        args.sampling_null.resolve(), args.capability_protocol.resolve(),
        args.label, args.graph.resolve(),
    )
    request = request_freezer.verify_bundle(inputs, args.request_dir.resolve())
    proposed = authorization(
        request, executable=args.provider_executable,
        cli_version=args.provider_cli_version, model=args.model,
        effort=args.effort, maximum_attempts=args.max_attempts_per_trial,
        timeout_seconds=args.timeout_seconds,
    )
    payload = proposal_payload(request, proposed)
    return {"proposal_id": bridge._fingerprint(payload), **payload}


def verify_proposal(value: Any) -> dict[str, Any]:
    require(
        isinstance(value, dict)
        and value.get("schema_version") == PROPOSAL_SCHEMA,
        f"expected {PROPOSAL_SCHEMA}",
    )
    payload = dict(value)
    observed = payload.pop("proposal_id", None)
    require(observed == bridge._fingerprint(payload),
            "authorization proposal ID does not match content")
    require(set(value) == {
        "proposal_id", "schema_version", "status", "request_id",
        "proposed_authorization_id", "proposed_authorization",
        "proposed_provider_budget", "boundary", "grant_contract",
    }, "authorization proposal fields do not match schema")
    require(
        value.get("status") == "awaiting_explicit_exact_user_authorization",
        "authorization proposal has invalid status",
    )
    boundary = value.get("boundary")
    require(
        boundary == {
            "proposal_is_not_an_authorization_file": True,
            "runner_rejects_this_outer_schema": True,
            "authorization_granted_by_proposal": False,
            "currently_permitted_provider_calls": 0,
            "provider_cli_invoked": False,
            "provider_inference_invoked": False,
            "application_source_visible_or_modified": False,
            "model_output_authority": (
                "existing graph-bound compiler IDs only"
            ),
        },
        "authorization proposal crossed its zero-call boundary",
    )
    require(value.get("grant_contract") == {
        "user_must_explicitly_repeat_proposal_id": True,
        "user_must_explicitly_repeat_authorization_id": True,
        "grant_materializes_exact_embedded_authorization_only": True,
        "request_or_provider_setting_change_requires_new_proposal": True,
    }, "authorization proposal has invalid grant contract")
    proposed = value.get("proposed_authorization")
    proposed_payload = trials.authorization_payload(proposed)
    require(
        value.get("proposed_authorization_id")
        == proposed.get("authorization_id"),
        "proposal and embedded authorization IDs differ",
    )
    require(
        value.get("request_id") == proposed_payload.get("request_id"),
        "proposal and embedded authorization request IDs differ",
    )
    semantic_trials = proposed_payload["provider_delivery"][
        "total_conditional_calls"
    ]
    attempts_per_trial = proposed_payload["transport_retry"][
        "max_attempts_per_trial"
    ]
    require(value.get("proposed_provider_budget") == {
        "semantic_trials": semantic_trials,
        "max_transport_attempts_per_trial": attempts_per_trial,
        "maximum_provider_process_invocations": (
            semantic_trials * attempts_per_trial
        ),
    }, "authorization proposal has invalid provider budget")
    return value


def explicitly_authorized_payload(
    proposal: dict[str, Any], *, authorized_proposal_id: str,
    authorized_authorization_id: str,
) -> dict[str, Any]:
    require(
        authorized_proposal_id == proposal["proposal_id"],
        "explicit user proposal ID does not match",
    )
    require(
        authorized_authorization_id
        == proposal["proposed_authorization_id"],
        "explicit user authorization ID does not match",
    )
    return proposal["proposed_authorization"]


def add_inputs(parser: argparse.ArgumentParser) -> None:
    request_freezer.add_inputs(parser)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--provider-executable", required=True)
    parser.add_argument("--provider-cli-version", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", required=True)
    parser.add_argument("--max-attempts-per-trial", type=int, default=3)
    parser.add_argument("--timeout-seconds", type=int, default=300)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    emit = children.add_parser("emit")
    add_inputs(emit)
    emit.add_argument("--out", type=Path, required=True)
    verify = children.add_parser("verify")
    add_inputs(verify)
    verify.add_argument("--proposal", type=Path, required=True)
    grant = children.add_parser("grant")
    add_inputs(grant)
    grant.add_argument("--proposal", type=Path, required=True)
    grant.add_argument("--authorized-proposal-id", required=True)
    grant.add_argument("--authorized-authorization-id", required=True)
    grant.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        proposal = verify_proposal(build_proposal(args))
        if args.command == "emit":
            require(not args.out.exists(), f"refusing to overwrite {args.out}")
            write_json_atomic(args.out.resolve(), proposal)
            action = "proposed_not_authorized"
        else:
            observed = verify_proposal(read_json(args.proposal.resolve()))
            require(observed == proposal,
                    "authorization proposal does not match current evidence")
            if args.command == "verify":
                action = "verified_not_authorized"
            else:
                granted = explicitly_authorized_payload(
                    proposal,
                    authorized_proposal_id=args.authorized_proposal_id,
                    authorized_authorization_id=(
                        args.authorized_authorization_id
                    ),
                )
                require(not args.out.exists(),
                        f"refusing to overwrite {args.out}")
                write_json_atomic(args.out.resolve(), granted)
                action = "exact_authorization_materialized"
        semantic_trials = proposal["proposed_provider_budget"][
            "semantic_trials"
        ]
        maximum_invocations = proposal["proposed_provider_budget"][
            "maximum_provider_process_invocations"
        ]
        print(json.dumps({
            "action": action,
            "proposal_id": proposal["proposal_id"],
            "proposed_authorization_id": proposal[
                "proposed_authorization_id"
            ],
            "currently_permitted_semantic_trials": (
                semantic_trials if args.command == "grant" else 0
            ),
            "maximum_permitted_provider_process_invocations": (
                maximum_invocations if args.command == "grant" else 0
            ),
        }, sort_keys=True))
        return 0
    except (
        AuthorizationProposalError, trials.CapabilityTrialError,
        request_freezer.CapabilityRequestError, OSError, KeyError,
        TypeError, ValueError,
    ) as exc:
        print(f"compiler-llm-authorization-proposal: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
