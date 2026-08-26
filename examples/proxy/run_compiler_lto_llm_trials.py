#!/usr/bin/env python3
"""Run replayable, source-free LLM trials for the frozen LTO dossier.

The provider process receives exactly two text inputs: the checked-in system
prompt and the frozen compiler-fact prompt.  It receives no tools, repository
context, source file, calibration labels, or evaluation results.  Every raw
provider envelope is retained before the response is checked by the same
``gicc_llm_bridge`` used by the LTO candidate build.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/gicc-passes/python"))
import gicc_llm_bridge as bridge  # noqa: E402


class TrialError(RuntimeError):
    pass


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


def write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(value)
    tmp.replace(path)


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def exact_response_schema(dossier: dict[str, Any]) -> dict[str, Any]:
    decision_properties: dict[str, Any] = {}
    decision_required: list[str] = []
    for site in dossier["sites"]:
        site_id = site["site_id"]
        decision_required.append(site_id)
        decision_properties[site_id] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "action": {"type": "string", "enum": site["legal_actions"]},
                "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "rationale": {"type": "string", "maxLength": 512},
            },
            "required": ["action", "confidence", "rationale"],
        }
    # Claude Code validates a practical JSON-Schema subset but rejects the
    # otherwise standard draft URI before making a provider request.  Omitting
    # the declaration does not relax any constraint below.
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema_version": {"const": bridge.DECISION_SCHEMA},
            "dossier_id": {"const": dossier["dossier_id"]},
            "decisions": {
                "type": "object",
                "additionalProperties": False,
                "properties": decision_properties,
                "required": decision_required,
            },
        },
        "required": ["schema_version", "dossier_id", "decisions"],
    }


def verify_inputs(protocol_path: Path, protocol: dict[str, Any]) -> tuple[
    dict[str, Any], str, str, dict[str, Any]
]:
    if protocol.get("schema_version") != "gicc-compiler-lto-llm-protocol-v1":
        raise TrialError("wrong protocol schema")
    inputs = protocol.get("frozen_inputs")
    if not isinstance(inputs, dict):
        raise TrialError("protocol lacks frozen_inputs")

    paths: dict[str, Path] = {}
    for name in ("dossier", "prompt", "system_prompt"):
        record = inputs.get(name)
        if not isinstance(record, dict):
            raise TrialError(f"protocol lacks frozen input {name}")
        path = ROOT / record.get("path", "")
        expected = record.get("sha256")
        actual = sha256_file(path)
        if actual != expected:
            raise TrialError(f"{name} hash changed: {actual} != {expected}")
        paths[name] = path

    dossier = bridge._verified_dossier(read_json(paths["dossier"]))
    if dossier["dossier_id"] != protocol.get("dossier_id"):
        raise TrialError("protocol/dossier ID mismatch")
    prompt = paths["prompt"].read_text()
    if prompt != bridge.render_prompt(dossier):
        raise TrialError("frozen prompt is not the canonical dossier rendering")
    system_prompt = paths["system_prompt"].read_text()
    schema = exact_response_schema(dossier)
    return dossier, prompt, system_prompt, schema


def cli_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "--version"], check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


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
    raise TrialError("provider envelope has no structured_output/result")


def actual_models(envelope: Any) -> list[str]:
    if not isinstance(envelope, dict):
        return []
    found: set[str] = set()
    for key in ("model", "model_name"):
        if isinstance(envelope.get(key), str):
            found.add(envelope[key])
    usage = envelope.get("modelUsage")
    if isinstance(usage, dict):
        found.update(key for key in usage if isinstance(key, str))
    return sorted(found)


def command_for(protocol: dict[str, Any], system_prompt: str,
                schema: dict[str, Any]) -> list[str]:
    provider = protocol["provider"]
    command = [
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
        "--json-schema", json.dumps(schema, sort_keys=True, separators=(",", ":")),
        "--system-prompt", system_prompt,
    ]
    return command


def run_one(index: int, protocol_path: Path, protocol: dict[str, Any],
            dossier: dict[str, Any], prompt: str, system_prompt: str,
            schema: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    trial_dir = output_dir / f"trial{index:02d}"
    if trial_dir.exists():
        raise TrialError(f"refusing to overwrite existing {trial_dir}")
    trial_dir.mkdir(parents=True)

    command = command_for(protocol, system_prompt, schema)
    retry = protocol["transport_retry"]
    max_attempts = int(retry["max_attempts_per_trial"])
    timeout = int(retry["timeout_seconds"])
    attempts: list[dict[str, Any]] = []
    envelope: Any = None
    response: Any = None
    parse_error: str | None = None

    for attempt in range(1, max_attempts + 1):
        started = utc_now()
        with tempfile.TemporaryDirectory(prefix="gicc-llm-trial-") as temp_cwd:
            try:
                completed = subprocess.run(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    cwd=temp_cwd,
                    timeout=timeout,
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
                transport_error = f"timeout after {timeout}s"
        ended = utc_now()
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        prefix = trial_dir / f"attempt{attempt}"
        write_text_atomic(prefix.with_suffix(".stdout.json"), stdout)
        write_text_atomic(prefix.with_suffix(".stderr.txt"), stderr)
        attempt_record = {
            "attempt": attempt,
            "ended_at": ended,
            "returncode": returncode,
            "started_at": started,
            "stderr_sha256": sha256_bytes(stderr.encode()),
            "stdout_sha256": sha256_bytes(stdout.encode()),
            "transport_error": transport_error,
        }
        attempts.append(attempt_record)
        if transport_error is not None or returncode != 0:
            continue
        try:
            envelope = json.loads(stdout)
            response = extract_response(envelope)
        except (json.JSONDecodeError, TrialError) as exc:
            # A successful provider call with malformed model output is a scored
            # semantic failure, not a transport condition and is never retried.
            parse_error = str(exc)
        break

    hint, accepted, bridge_errors = bridge.decision_to_hint(dossier, response)
    write_json_atomic(trial_dir / "response.json", response)
    write_json_atomic(trial_dir / "hint.json", hint)
    protocol_sha = sha256_file(protocol_path)
    run = {
        "schema_version": "gicc-compiler-lto-llm-trial-v1",
        "trial": index,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": protocol_sha,
        "dossier_id": dossier["dossier_id"],
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "system_prompt_sha256": sha256_bytes(system_prompt.encode()),
        "response_schema_sha256_canonical": sha256_bytes(canonical(schema)),
        "provider_cli_version": cli_version(protocol["provider"]["executable"]),
        "requested_model": protocol["provider"]["requested_model"],
        "actual_models": actual_models(envelope),
        "attempts": attempts,
        "provider_call_succeeded": envelope is not None,
        "response_parse_error": parse_error,
        "response_sha256_canonical": sha256_bytes(canonical(response)),
        "bridge_accepted": accepted,
        "bridge_errors": bridge_errors,
        "fallback_applied": not accepted,
    }
    write_json_atomic(trial_dir / "run.json", run)
    return run


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        protocol_path = args.protocol.resolve()
        protocol = read_json(protocol_path)
        dossier, prompt, system_prompt, schema = verify_inputs(protocol_path, protocol)
        version = cli_version(protocol["provider"]["executable"])
        if version != protocol["provider"]["cli_version"]:
            raise TrialError(
                f"provider CLI version changed: {version!r} != "
                f"{protocol['provider']['cli_version']!r}"
            )
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output_dir / "response-schema.json", schema)
        runs = []
        for index in range(1, int(protocol["trials"]) + 1):
            print(f"LLM_TRIAL_BEGIN {index}/{protocol['trials']}", flush=True)
            run = run_one(
                index, protocol_path, protocol, dossier, prompt, system_prompt,
                schema, output_dir,
            )
            runs.append(run)
            print(
                f"LLM_TRIAL_END {index}/{protocol['trials']} "
                f"provider={run['provider_call_succeeded']} "
                f"accepted={run['bridge_accepted']} "
                f"models={','.join(run['actual_models']) or 'unknown'}",
                flush=True,
            )
        write_json_atomic(output_dir / "run-index.json", {
            "schema_version": "gicc-compiler-lto-llm-run-index-v1",
            "protocol_id": protocol["protocol_id"],
            "protocol_sha256": sha256_file(protocol_path),
            "trials": runs,
        })
        return 0
    except (OSError, subprocess.SubprocessError, TrialError, bridge.BridgeError) as exc:
        print(f"compiler-lto-llm-trials: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
