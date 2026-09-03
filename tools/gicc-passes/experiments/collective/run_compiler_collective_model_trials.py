#!/usr/bin/env python3
"""Run authorization-bound, source-free collective model trials sequentially.

The provider receives only one frozen Gate-E system prompt, one frozen
compiler-fact prompt, and the exact option-ID response schema.  No provider
process is started unless a content-addressed authorization matches all three
prompt views and the provider settings exactly.
"""

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
ROOT = HERE.parents[3]
PASS_PYTHON = ROOT / "tools/gicc-passes/python"
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(HERE))

import gicc_collective_plan_bridge as plans  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import prepare_compiler_collective_gate_e as gate_e  # noqa: E402


AUTHORIZATION_SCHEMA = "gicc-collective-provider-authorization-v1"
RUN_SCHEMA = "gicc-collective-model-trial-v1"
INDEX_SCHEMA = "gicc-collective-model-run-index-v1"


class TrialError(RuntimeError):
    """The authorization, provider transport, or trial archive is invalid."""


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


def authorization_payload(value: Any) -> dict[str, Any]:
    if (not isinstance(value, dict)
            or value.get("schema_version") != AUTHORIZATION_SCHEMA):
        raise TrialError(f"expected {AUTHORIZATION_SCHEMA}")
    payload = dict(value)
    authorization_id = payload.pop("authorization_id", None)
    if authorization_id != bridge._fingerprint(payload):
        raise TrialError("authorization_id does not match content")
    return payload


def verify_authorization(value: Any, request: dict[str, Any]) -> dict[str, Any]:
    payload = authorization_payload(value)
    if payload.get("granted") is not True:
        raise TrialError("provider authorization is not explicitly granted")
    delivery = request["provider_delivery"]
    wanted_views = {
        row["view"]: {
            "prompt_sha256": row["prompt_sha256"],
            "independent_responses": row["independent_responses"],
        }
        for row in delivery["views"]
    }
    if (payload.get("request_id") != request["request_id"]
            or payload.get("system_prompt_sha256")
            != delivery["system_prompt_sha256"]
            or payload.get("response_schema_sha256")
            != delivery["response_schema_sha256"]
            or payload.get("views") != wanted_views):
        raise TrialError("authorization does not bind the exact Gate-E inputs")
    boundary = payload.get("data_boundary")
    if boundary != {
        "model_tools": [],
        "source_visible": False,
        "llvm_ir_visible": False,
        "evaluation_runtime_labels_visible": False,
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


def cli_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "--version"], check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise TrialError(
            f"provider --version failed: {result.stderr.strip()}"
        )
    return result.stdout.strip()


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
            response_schema, sort_keys=True, separators=(",", ":")
        ),
        "--system-prompt", system_prompt,
    ]


def extract_response(envelope: Any) -> Any:
    if not isinstance(envelope, dict):
        raise TrialError("provider envelope is not an object")
    structured = envelope.get("structured_output")
    if isinstance(structured, dict):
        return structured
    result = envelope.get("result")
    if isinstance(result, dict):
        return result
    if isinstance(result, str):
        text = result.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3 and lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1]).strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise TrialError(f"provider result is not JSON: {exc}") from exc
    raise TrialError("provider envelope has no structured output")


def actual_models(envelope: Any) -> list[str]:
    if not isinstance(envelope, dict):
        return []
    models = set()
    for key in ("model", "model_name"):
        if isinstance(envelope.get(key), str):
            models.add(envelope[key])
    usage = envelope.get("modelUsage")
    if isinstance(usage, dict):
        models.update(key for key in usage if isinstance(key, str))
    return sorted(models)


def run_one(*, view: str, trial: int, prompt: str, graph: dict[str, Any],
            request: dict[str, Any], authorization: dict[str, Any],
            authorization_id: str, system_prompt: str,
            response_schema: dict[str, Any], output_dir: Path,
            observed_cli_version: str) -> dict[str, Any]:
    trial_dir = output_dir / view / f"trial{trial:02d}"
    if trial_dir.exists():
        raise TrialError(f"refusing partial or duplicate trial: {trial_dir}")
    trial_dir.mkdir(parents=True)
    provider = authorization["provider"]
    retry = authorization["transport_retry"]
    attempts = []
    envelope: Any = None
    response: Any = None
    parse_error = None
    transport_succeeded = False
    command = provider_command(provider, system_prompt, response_schema)
    for attempt in range(1, retry["max_attempts_per_trial"] + 1):
        started = utc_now()
        with tempfile.TemporaryDirectory(
                prefix="gicc-collective-model-trial-") as temporary_cwd:
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
            response = extract_response(envelope)
        except (json.JSONDecodeError, TrialError) as exc:
            parse_error = str(exc)
        break

    hint, accepted, errors = plans.decision_to_hint(graph, response)
    if parse_error:
        errors = [parse_error, *errors]
    response_path = trial_dir / "response.json"
    hint_path = trial_dir / "hint.json"
    write_json_atomic(response_path, response)
    write_json_atomic(hint_path, hint)
    record = {
        "schema_version": RUN_SCHEMA,
        "view": view,
        "trial": trial,
        "request_id": request["request_id"],
        "authorization_id": authorization_id,
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "system_prompt_sha256": sha256_bytes(system_prompt.encode()),
        "response_schema_sha256_canonical": sha256_bytes(
            canonical(response_schema)
        ),
        "provider_cli_version": observed_cli_version,
        "requested_model": provider["requested_model"],
        "actual_models": actual_models(envelope),
        "attempts": attempts,
        "provider_call_succeeded": transport_succeeded,
        "response_parse_error": parse_error,
        "response_sha256": sha256_file(response_path),
        "response_sha256_canonical": sha256_bytes(canonical(response)),
        "hint_sha256": sha256_file(hint_path),
        "bridge_accepted": accepted,
        "bridge_errors": errors,
        "fallback_applied": not accepted,
        "selected_option_ids": hint["llm_metadata"]["selected_option_ids"],
    }
    write_json_atomic(trial_dir / "run.json", record)
    return record


def expected_trials(request: dict[str, Any]) -> list[tuple[str, int]]:
    return [
        (row["view"], trial)
        for row in request["provider_delivery"]["views"]
        for trial in range(1, row["independent_responses"] + 1)
    ]


def verify_archived_run(record: Any, output_dir: Path,
                        request: dict[str, Any],
                        authorization_id: str) -> tuple[str, int]:
    if (not isinstance(record, dict)
            or record.get("schema_version") != RUN_SCHEMA
            or record.get("request_id") != request["request_id"]
            or record.get("authorization_id") != authorization_id
            or not isinstance(record.get("view"), str)
            or not isinstance(record.get("trial"), int)):
        raise TrialError("run index has an invalid archived trial")
    view = record["view"]
    trial = record["trial"]
    trial_dir = output_dir / view / f"trial{trial:02d}"
    if read_json(trial_dir / "run.json") != record:
        raise TrialError(f"archived run/index mismatch: {view}/trial{trial:02d}")
    response_path = trial_dir / "response.json"
    hint_path = trial_dir / "hint.json"
    if (sha256_file(response_path) != record.get("response_sha256")
            or sha256_file(hint_path) != record.get("hint_sha256")
            or sha256_bytes(canonical(read_json(response_path)))
            != record.get("response_sha256_canonical")):
        raise TrialError(f"archived response changed: {view}/trial{trial:02d}")
    attempts = record.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        raise TrialError(f"archived trial has no attempts: {view}/trial{trial:02d}")
    for wanted_attempt, attempt in enumerate(attempts, 1):
        if (not isinstance(attempt, dict)
                or attempt.get("attempt") != wanted_attempt):
            raise TrialError(f"invalid attempt archive: {view}/trial{trial:02d}")
        prefix = trial_dir / f"attempt{wanted_attempt}"
        if (sha256_file(prefix.with_suffix(".stdout.json"))
                != attempt.get("stdout_sha256")
                or sha256_file(prefix.with_suffix(".stderr.txt"))
                != attempt.get("stderr_sha256")):
            raise TrialError(f"attempt archive changed: {view}/trial{trial:02d}")
    return view, trial


def existing_index(path: Path, request: dict[str, Any],
                   authorization_id: str) -> dict[str, Any]:
    if not path.exists():
        return {
            "schema_version": INDEX_SCHEMA,
            "status": "running",
            "request_id": request["request_id"],
            "authorization_id": authorization_id,
            "runs": [],
        }
    value = read_json(path)
    if (not isinstance(value, dict)
            or value.get("schema_version") != INDEX_SCHEMA
            or value.get("request_id") != request["request_id"]
            or value.get("authorization_id") != authorization_id
            or not isinstance(value.get("runs"), list)):
        raise TrialError("existing run index belongs to another protocol")
    return value


def run_trials(request_path: Path, authorization_path: Path, bundle: Path,
               output_dir: Path, repo_root: Path) -> dict[str, Any]:
    request = gate_e.verify(request_path, bundle, repo_root)
    authorization_value = read_json(authorization_path)
    authorization = verify_authorization(authorization_value, request)
    authorization_id = authorization_value["authorization_id"]
    provider = authorization["provider"]
    version = cli_version(provider["executable"])
    if version != provider["cli_version"]:
        raise TrialError(
            f"provider CLI version changed: {version!r} != "
            f"{provider['cli_version']!r}"
        )
    graph = plans.verified_graph(read_json(bundle / "discovery/graph.json"))
    system_path = request_path.parent / "system-prompt.txt"
    schema_path = request_path.parent / "response-schema.json"
    system_prompt = system_path.read_text()
    response_schema = read_json(schema_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "run-index.json"
    index = existing_index(index_path, request, authorization_id)
    wanted = expected_trials(request)
    if index.get("status") == "transport_failed":
        raise TrialError(
            "a provider transport failure exhausted this authorization"
        )
    if index.get("status") not in {"running", "complete"}:
        raise TrialError("existing model run index has an invalid state")
    completed = [
        verify_archived_run(row, output_dir, request, authorization_id)
        for row in index["runs"]
    ]
    if completed != wanted[:len(completed)]:
        raise TrialError("existing model trials are not a valid prefix")
    if index.get("status") == "complete":
        if completed != wanted:
            raise TrialError("complete model run index is missing trials")
        return index
    for view, trial in wanted[len(completed):]:
        prompt_path = bundle / f"prompts/{view}.txt"
        prompt = prompt_path.read_text()
        print(f"MODEL_TRIAL_BEGIN view={view} trial={trial}", flush=True)
        record = run_one(
            view=view, trial=trial, prompt=prompt, graph=graph,
            request=request, authorization=authorization,
            authorization_id=authorization_id, system_prompt=system_prompt,
            response_schema=response_schema, output_dir=output_dir,
            observed_cli_version=version,
        )
        index["runs"].append(record)
        if not record["provider_call_succeeded"]:
            index["status"] = "transport_failed"
            write_json_atomic(index_path, index)
            raise TrialError(
                f"provider transport exhausted at {view}/trial{trial:02d}"
            )
        write_json_atomic(index_path, index)
        print(
            f"MODEL_TRIAL_END view={view} trial={trial} "
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
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        index = run_trials(
            args.request.resolve(), args.authorization.resolve(),
            args.bundle.resolve(), args.output_dir.resolve(),
            args.repo_root.resolve(),
        )
        print(f"MODEL_TRIALS_COMPLETE count={len(index['runs'])}")
        return 0
    except (TrialError, gate_e.GateEError, plans.CollectivePlanError,
            OSError, KeyError, TypeError, ValueError) as exc:
        print(f"compiler-collective-model-trials: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
