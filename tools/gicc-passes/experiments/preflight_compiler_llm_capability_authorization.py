#!/usr/bin/env python3
"""Validate one exact capability authorization without model inference."""

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
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PASS_PYTHON))

import gicc_llm_bridge as bridge  # noqa: E402
import prepare_compiler_llm_capability_request as request_freezer  # noqa: E402
import run_compiler_llm_capability_trials as trials  # noqa: E402


PREFLIGHT_SCHEMA = "gicc-compiler-llm-authorization-preflight-v1"


class AuthorizationPreflightError(RuntimeError):
    """The request is not ready for exactly authorized model trials."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuthorizationPreflightError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AuthorizationPreflightError(
            f"cannot read JSON {path}: {exc}"
        ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise AuthorizationPreflightError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"preflight evidence is absent: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def payload(
    request: dict[str, Any], authorization_value: dict[str, Any],
    authorization: dict[str, Any], observed_version: str,
    request_path: Path, authorization_path: Path,
) -> dict[str, Any]:
    provider = authorization["provider"]
    delivery = trials.delivery_contract(request)
    command = trials.provider_command(
        provider, "<system-prompt-not-recorded>", {"type": "object"}
    )
    require(command[command.index("--tools") + 1] == "",
            "provider command would expose tools")
    require("--safe-mode" in command and "--no-session-persistence" in command,
            "provider command would retain unsafe context")
    return {
        "schema_version": PREFLIGHT_SCHEMA,
        "status": "exact_authorization_verified_ready_for_serial_trials",
        "request_id": request["request_id"],
        "authorization_id": authorization_value["authorization_id"],
        "provider_delivery": delivery,
        "provider": {
            "kind": provider["kind"],
            "executable": provider["executable"],
            "requested_model": provider["requested_model"],
            "effort": provider["effort"],
            "authorized_cli_version": provider["cli_version"],
            "observed_cli_version": observed_version,
            "fresh_session_per_trial": provider["fresh_session_per_trial"],
            "structured_output": provider["structured_output"],
        },
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_visible": False,
            "source_locations_visible": False,
            "llvm_ir_visible": False,
            "evaluation_runtime_labels_visible": False,
            "evaluation_oracle_visible": False,
            "model_tools": [],
            "model_may_generate_code_or_ir": False,
            "provider_version_command_invoked": True,
            "provider_inference_invoked": False,
            "model_output_materialization_invoked": False,
            "scheduler_invoked": False,
        },
        "execution_contract": {
            "calls_strictly_sequential": True,
            "fresh_session_per_trial": True,
            "total_conditional_calls": delivery["total_conditional_calls"],
            "transport_retry": authorization["transport_retry"],
            "runner_performs_full_revalidation_before_first_call": True,
        },
        "evidence": {
            "request": evidence(request_path),
            "authorization": evidence(authorization_path),
            "preflight": evidence(Path(__file__)),
            "trial_runner": evidence(Path(trials.__file__)),
            "request_freezer": evidence(Path(request_freezer.__file__)),
        },
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    inputs = request_freezer.verified_inputs(
        args.suite.resolve(), args.prompt_dir.resolve(),
        args.readiness.resolve(), args.input_separation.resolve(),
        args.sampling_null.resolve(), args.capability_protocol.resolve(),
        args.label, args.graph.resolve(),
    )
    request_path = args.request_dir.resolve() / "request.json"
    request = request_freezer.verify_bundle(inputs, args.request_dir.resolve())
    authorization_path = args.authorization.resolve()
    authorization_value = read_json(authorization_path)
    authorization = trials.verify_authorization(authorization_value, request)
    observed_version = trials.cli_version(authorization["provider"]["executable"])
    require(observed_version == authorization["provider"]["cli_version"],
            "provider CLI version differs from exact authorization")
    body = payload(
        request, authorization_value, authorization, observed_version,
        request_path, authorization_path,
    )
    return {"preflight_id": bridge._fingerprint(body), **body}


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    for command in ("emit", "verify"):
        child = children.add_parser(command)
        request_freezer.add_inputs(child)
        child.add_argument("--request-dir", type=Path, required=True)
        child.add_argument("--authorization", type=Path, required=True)
        child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(args)
        if args.command == "emit":
            require(not args.out.exists(), f"refusing to overwrite {args.out}")
            write_json_atomic(args.out.resolve(), report)
            action = "wrote"
        else:
            require(read_json(args.out.resolve()) == report,
                    "authorization preflight does not match current evidence")
            action = "verified"
        print(json.dumps({
            "action": action,
            "preflight_id": report["preflight_id"],
            "request_id": report["request_id"],
            "authorization_id": report["authorization_id"],
            "provider_inference_invoked": False,
        }, sort_keys=True))
        return 0
    except (
        AuthorizationPreflightError, trials.CapabilityTrialError,
        request_freezer.CapabilityRequestError, OSError, KeyError,
        TypeError, ValueError,
    ) as exc:
        print(f"compiler-llm-authorization-preflight: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
