#!/usr/bin/env python3
"""Run authorized matched-calibration LTO LLM trials, strictly sequentially."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/gicc-passes/python"))
sys.path.insert(0, str(HERE))

import gicc_llm_bridge as bridge  # noqa: E402
import prepare_compiler_lto_llm_calibrated as prepare  # noqa: E402
import run_compiler_lto_llm_trials as zero_shot  # noqa: E402


AUTHORIZATION_SCHEMA = "gicc-compiler-lto-llm-calibrated-authorization-v1"
RUN_SCHEMA = "gicc-compiler-lto-llm-calibrated-trial-v1"
INDEX_SCHEMA = "gicc-compiler-lto-llm-calibrated-run-index-v1"


class TrialError(RuntimeError):
    """Authorization, provider transport, or archived evidence is invalid."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise TrialError(f"cannot read JSON {path}: {exc}") from exc


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def delivery_hash(request: dict[str, Any], role: str) -> str:
    records = request["provider_delivery"]["files"]
    matches = [record["sha256"] for record in records if record["role"] == role]
    if len(matches) != 1:
        raise TrialError(f"request lacks one delivery file: {role}")
    return matches[0]


def authorization_payload(value: Any) -> dict[str, Any]:
    if (not isinstance(value, dict)
            or value.get("schema_version") != AUTHORIZATION_SCHEMA):
        raise TrialError(f"expected {AUTHORIZATION_SCHEMA}")
    payload = dict(value)
    authorization_id = payload.pop("authorization_id", None)
    required_fields = {
        "schema_version", "granted", "request_id", "system_prompt_sha256",
        "prompt_sha256", "response_schema_sha256", "independent_responses",
        "data_boundary", "provider", "transport_retry",
    }
    if set(payload) != required_fields:
        raise TrialError("authorization fields do not match the frozen schema")
    if authorization_id != bridge._fingerprint(payload):
        raise TrialError("authorization_id does not match content")
    return payload


def verify_authorization(value: Any,
                         request: dict[str, Any]) -> dict[str, Any]:
    payload = authorization_payload(value)
    if payload.get("granted") is not True:
        raise TrialError("provider authorization is not explicitly granted")
    if (payload.get("request_id") != request["request_id"]
            or payload.get("system_prompt_sha256")
            != delivery_hash(request, "system_prompt")
            or payload.get("prompt_sha256")
            != delivery_hash(request, "user_prompt")
            or payload.get("response_schema_sha256")
            != delivery_hash(request, "response_schema")
            or payload.get("independent_responses")
            != request["provider_delivery"]["independent_responses"]):
        raise TrialError("authorization does not bind the exact request inputs")
    if payload.get("data_boundary") != {
        "source_visible": False,
        "llvm_ir_visible": False,
        "evaluation_runtime_labels_visible": False,
        "evaluation_oracle_visible": False,
        "calibration_labels_visible": True,
        "model_tools": [],
    }:
        raise TrialError("authorization does not preserve the data boundary")
    provider = payload.get("provider")
    required_provider = {
        "kind", "executable", "cli_version", "requested_model", "effort",
        "fresh_session_per_trial", "structured_output",
    }
    if (not isinstance(provider, dict)
            or set(provider) != required_provider
            or provider.get("kind") != "anthropic-claude-cli"
            or provider.get("fresh_session_per_trial") is not True
            or provider.get("structured_output") is not True
            or any(not isinstance(provider.get(key), str) or not provider[key]
                   for key in (
                       "executable", "cli_version", "requested_model", "effort"
                   ))):
        raise TrialError("authorization has invalid provider settings")
    retry = payload.get("transport_retry")
    if (not isinstance(retry, dict)
            or set(retry) != {"max_attempts_per_trial", "timeout_seconds"}
            or not isinstance(retry.get("max_attempts_per_trial"), int)
            or not 1 <= retry["max_attempts_per_trial"] <= 3
            or not isinstance(retry.get("timeout_seconds"), int)
            or not 30 <= retry["timeout_seconds"] <= 600):
        raise TrialError("authorization has invalid transport limits")
    return payload


def provider_command(provider: dict[str, Any], system_prompt: str,
                     response_schema: dict[str, Any]) -> list[str]:
    return [
        provider["executable"],
        "--print",
        "--safe-mode",
        "--disable-slash-commands",
        "--no-chrome",
        "--no-session-persistence",
        "--prompt-suggestions", "false",
        "--tools", "",
        "--model", provider["requested_model"],
        "--effort", provider["effort"],
        "--output-format", "json",
        "--json-schema", json.dumps(
            response_schema, sort_keys=True, separators=(",", ":"),
        ),
        "--system-prompt", system_prompt,
    ]


def cli_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "--version"], check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise TrialError(f"provider --version failed: {result.stderr.strip()}")
    return result.stdout.strip()


def run_one(*, trial: int, prompt: str, dossier: dict[str, Any],
            request: dict[str, Any], authorization: dict[str, Any],
            authorization_id: str, system_prompt: str,
            response_schema: dict[str, Any], output_dir: Path,
            observed_cli_version: str) -> dict[str, Any]:
    trial_dir = output_dir / f"trial{trial:02d}"
    if trial_dir.exists():
        raise TrialError(f"refusing partial or duplicate trial: {trial_dir}")
    trial_dir.mkdir(parents=True)
    provider = authorization["provider"]
    retry = authorization["transport_retry"]
    command = provider_command(provider, system_prompt, response_schema)
    attempts = []
    envelope: Any = None
    response: Any = None
    parse_error = None
    transport_succeeded = False
    for attempt in range(1, retry["max_attempts_per_trial"] + 1):
        started = utc_now()
        with tempfile.TemporaryDirectory(
                prefix="gicc-lto-calibrated-llm-") as temporary_cwd:
            try:
                completed = subprocess.run(
                    command, input=prompt, text=True, capture_output=True,
                    cwd=temporary_cwd, timeout=retry["timeout_seconds"],
                    check=False,
                )
                returncode = completed.returncode
                stdout = completed.stdout
                stderr = completed.stderr
                transport_error = None
            except subprocess.TimeoutExpired as exc:
                returncode = None
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                transport_error = (
                    f"timeout after {retry['timeout_seconds']} seconds"
                )
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        prefix = trial_dir / f"attempt{attempt}"
        write_text_atomic(prefix.with_suffix(".stdout.json"), stdout)
        write_text_atomic(prefix.with_suffix(".stderr.txt"), stderr)
        attempts.append({
            "attempt": attempt,
            "started_at": started,
            "ended_at": utc_now(),
            "returncode": returncode,
            "transport_error": transport_error,
            "stdout_sha256": sha256_bytes(stdout.encode()),
            "stderr_sha256": sha256_bytes(stderr.encode()),
        })
        if transport_error is not None or returncode != 0:
            continue
        transport_succeeded = True
        try:
            envelope = json.loads(stdout)
            response = zero_shot.extract_response(envelope)
        except (json.JSONDecodeError, zero_shot.TrialError) as exc:
            parse_error = str(exc)
        break

    hint, accepted, errors = bridge.decision_to_hint(dossier, response)
    if parse_error:
        errors = [parse_error, *errors]
    response_path = trial_dir / "response.json"
    hint_path = trial_dir / "hint.json"
    write_json_atomic(response_path, response)
    write_json_atomic(hint_path, hint)
    record = {
        "schema_version": RUN_SCHEMA,
        "trial": trial,
        "request_id": request["request_id"],
        "authorization_id": authorization_id,
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "system_prompt_sha256": sha256_bytes(system_prompt.encode()),
        "response_schema_sha256": delivery_hash(
            request, "response_schema",
        ),
        "response_schema_sha256_canonical": sha256_bytes(
            canonical(response_schema)
        ),
        "provider_cli_version": observed_cli_version,
        "requested_model": provider["requested_model"],
        "actual_models": zero_shot.actual_models(envelope),
        "attempts": attempts,
        "provider_call_succeeded": transport_succeeded,
        "response_parse_error": parse_error,
        "response_sha256": sha256_file(response_path),
        "response_sha256_canonical": sha256_bytes(canonical(response)),
        "hint_sha256": sha256_file(hint_path),
        "bridge_accepted": accepted,
        "bridge_errors": errors,
        "fallback_applied": not accepted,
    }
    write_json_atomic(trial_dir / "run.json", record)
    return record


def verify_archived_run(record: Any, output_dir: Path,
                        request: dict[str, Any],
                        authorization_id: str) -> int:
    if (not isinstance(record, dict)
            or record.get("schema_version") != RUN_SCHEMA
            or record.get("request_id") != request["request_id"]
            or record.get("authorization_id") != authorization_id
            or not isinstance(record.get("trial"), int)):
        raise TrialError("run index contains an invalid trial")
    trial = record["trial"]
    trial_dir = output_dir / f"trial{trial:02d}"
    if read_json(trial_dir / "run.json") != record:
        raise TrialError(f"trial{trial:02d}: run/index mismatch")
    response_path = trial_dir / "response.json"
    hint_path = trial_dir / "hint.json"
    if (sha256_file(response_path) != record.get("response_sha256")
            or sha256_file(hint_path) != record.get("hint_sha256")
            or sha256_bytes(canonical(read_json(response_path)))
            != record.get("response_sha256_canonical")):
        raise TrialError(f"trial{trial:02d}: response archive changed")
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise TrialError(f"trial{trial:02d}: no archived attempts")
    for wanted, attempt in enumerate(attempts, 1):
        prefix = trial_dir / f"attempt{wanted}"
        if (not isinstance(attempt, dict) or attempt.get("attempt") != wanted
                or sha256_file(prefix.with_suffix(".stdout.json"))
                != attempt.get("stdout_sha256")
                or sha256_file(prefix.with_suffix(".stderr.txt"))
                != attempt.get("stderr_sha256")):
            raise TrialError(f"trial{trial:02d}: attempt archive changed")
    return trial


def run_trials(*, request_path: Path, authorization_path: Path,
               calibration_dossier_path: Path,
               calibration_results_path: Path,
               evaluation_dossier_path: Path,
               output_dir: Path) -> dict[str, Any]:
    request = prepare.verify(
        request_path=request_path,
        calibration_dossier_path=calibration_dossier_path,
        calibration_results_path=calibration_results_path,
        evaluation_dossier_path=evaluation_dossier_path,
    )
    package = request_path.resolve().parent
    authorization_value = read_json(authorization_path)
    authorization = verify_authorization(authorization_value, request)
    authorization_id = authorization_value["authorization_id"]
    provider = authorization["provider"]
    observed_version = cli_version(provider["executable"])
    if observed_version != provider["cli_version"]:
        raise TrialError(
            f"provider CLI version changed: {observed_version!r} != "
            f"{provider['cli_version']!r}"
        )
    dossier = bridge._verified_dossier(read_json(evaluation_dossier_path))
    prompt = (package / "prompt.txt").read_text()
    system_prompt = (package / "system-prompt.txt").read_text()
    response_schema = read_json(package / "response-schema.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "run-index.json"
    if index_path.exists():
        index = read_json(index_path)
    else:
        index = {
            "schema_version": INDEX_SCHEMA,
            "status": "running",
            "request_id": request["request_id"],
            "authorization_id": authorization_id,
            "runs": [],
        }
    if (not isinstance(index, dict)
            or index.get("schema_version") != INDEX_SCHEMA
            or index.get("request_id") != request["request_id"]
            or index.get("authorization_id") != authorization_id
            or not isinstance(index.get("runs"), list)):
        raise TrialError("existing run index belongs to another protocol")
    completed = [
        verify_archived_run(row, output_dir, request, authorization_id)
        for row in index["runs"]
    ]
    if completed != list(range(1, len(completed) + 1)):
        raise TrialError("existing trials are not a valid sequential prefix")
    total = request["provider_delivery"]["independent_responses"]
    if index.get("status") == "complete":
        if completed != list(range(1, total + 1)):
            raise TrialError("complete run index is missing trials")
        return index
    if index.get("status") == "transport_failed":
        raise TrialError("provider transport exhausted this authorization")
    if index.get("status") != "running":
        raise TrialError("run index has an invalid state")
    for trial in range(len(completed) + 1, total + 1):
        print(f"MODEL_TRIAL_BEGIN trial={trial}", flush=True)
        record = run_one(
            trial=trial, prompt=prompt, dossier=dossier, request=request,
            authorization=authorization, authorization_id=authorization_id,
            system_prompt=system_prompt, response_schema=response_schema,
            output_dir=output_dir, observed_cli_version=observed_version,
        )
        index["runs"].append(record)
        if not record["provider_call_succeeded"]:
            index["status"] = "transport_failed"
            write_json_atomic(index_path, index)
            raise TrialError(f"provider transport exhausted at trial{trial:02d}")
        write_json_atomic(index_path, index)
        print(
            f"MODEL_TRIAL_END trial={trial} "
            f"accepted={record['bridge_accepted']}", flush=True,
        )
    index["status"] = "complete"
    index["completed_at"] = utc_now()
    write_json_atomic(index_path, index)
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--calibration-dossier", type=Path, required=True)
    parser.add_argument("--calibration-results", type=Path, required=True)
    parser.add_argument("--evaluation-dossier", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        index = run_trials(
            request_path=args.request.resolve(),
            authorization_path=args.authorization.resolve(),
            calibration_dossier_path=args.calibration_dossier.resolve(),
            calibration_results_path=args.calibration_results.resolve(),
            evaluation_dossier_path=args.evaluation_dossier.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        print(
            f"calibrated LTO model trials complete: {len(index['runs'])}"
        )
        return 0
    except (TrialError, prepare.PrepareError, bridge.BridgeError, OSError,
            KeyError, TypeError, ValueError) as exc:
        print(f"run-compiler-lto-llm-calibrated: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
