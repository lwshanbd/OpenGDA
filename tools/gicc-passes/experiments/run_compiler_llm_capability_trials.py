#!/usr/bin/env python3
"""Run exactly authorized compiler-only capability trials sequentially.

The runner supports every graph family handled by the unified compiler policy
bridge.  It verifies the full runtime-eligibility chain and frozen request
before it inspects an authorization.  Provider transport is attempted only
after a content-addressed authorization matches the exact source-free delivery,
provider settings, retry limits, and rotating trial order.  Semantic failures
are archived once and receive the compiler anchor fallback; they are never
retried to search for a more favorable decision.
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
ROOT = HERE.parents[2]
PASS_PYTHON = HERE.parent / "python"
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(HERE))

import gicc_compiler_policy_bridge as policy_bridge  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import prepare_compiler_llm_capability_request as request_freezer  # noqa: E402


AUTHORIZATION_SCHEMA = "gicc-compiler-llm-capability-authorization-v1"
RUN_SCHEMA = "gicc-compiler-llm-capability-trial-v1"
INDEX_SCHEMA = "gicc-compiler-llm-capability-run-index-v1"


class CapabilityTrialError(RuntimeError):
    """The request, authorization, transport, or archive is invalid."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    try:
        return sha256_bytes(path.read_bytes())
    except OSError as exc:
        raise CapabilityTrialError(f"cannot hash {path}: {exc}") from exc


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapabilityTrialError(f"cannot read JSON {path}: {exc}") from exc


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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CapabilityTrialError(message)


def authorization_payload(value: Any) -> dict[str, Any]:
    if (not isinstance(value, dict)
            or value.get("schema_version") != AUTHORIZATION_SCHEMA):
        raise CapabilityTrialError(f"expected {AUTHORIZATION_SCHEMA}")
    payload = dict(value)
    authorization_id = payload.pop("authorization_id", None)
    required = {
        "schema_version", "granted", "request_id", "provider_delivery",
        "data_boundary", "provider", "transport_retry",
    }
    require(set(payload) == required, "authorization fields do not match schema")
    require(authorization_id == bridge._fingerprint(payload),
            "authorization ID does not match content")
    return payload


def delivery_contract(request: dict[str, Any]) -> dict[str, Any]:
    delivery = request["provider_delivery"]
    return {
        "system_prompt_sha256": delivery["system_prompt_sha256"],
        "response_schema_sha256": delivery["response_schema_sha256"],
        "views": {
            row["view"]: {
                "prompt_sha256": row["prompt_sha256"],
                "independent_responses": row["independent_responses"],
            }
            for row in delivery["views"]
        },
        "trial_order": delivery["trial_order"],
        "total_conditional_calls": delivery["total_conditional_calls"],
    }


def verify_authorization(value: Any,
                         request: dict[str, Any]) -> dict[str, Any]:
    payload = authorization_payload(value)
    require(payload.get("granted") is True,
            "provider authorization is not explicitly granted")
    require(payload.get("request_id") == request["request_id"],
            "authorization binds another request")
    require(payload.get("provider_delivery") == delivery_contract(request),
            "authorization does not bind the exact provider delivery")
    require(payload.get("data_boundary") == {
        "model_tools": [],
        "source_visible": False,
        "source_locations_visible": False,
        "llvm_ir_visible": False,
        "evaluation_runtime_labels_visible": False,
        "evaluation_oracle_visible": False,
        "model_may_generate_code_or_ir": False,
    }, "authorization violates the compiler-only data boundary")
    provider = payload.get("provider")
    require(isinstance(provider, dict) and set(provider) == {
        "kind", "executable", "cli_version", "requested_model", "effort",
        "fresh_session_per_trial", "structured_output",
    }, "authorization has invalid provider fields")
    require(provider.get("kind") == "anthropic-claude-cli",
            "authorization has unsupported provider kind")
    require(provider.get("fresh_session_per_trial") is True,
            "authorization permits shared provider context")
    require(provider.get("structured_output") is True,
            "authorization does not require structured output")
    for key in ("executable", "cli_version", "requested_model", "effort"):
        require(isinstance(provider.get(key), str) and bool(provider[key]),
                f"authorization has invalid provider setting: {key}")
    retry = payload.get("transport_retry")
    require(isinstance(retry, dict) and set(retry) == {
        "max_attempts_per_trial", "timeout_seconds",
    }, "authorization has invalid retry fields")
    maximum = retry.get("max_attempts_per_trial")
    timeout = retry.get("timeout_seconds")
    require(isinstance(maximum, int) and not isinstance(maximum, bool)
            and 1 <= maximum <= 3,
            "authorization has invalid transport attempt limit")
    require(isinstance(timeout, int) and not isinstance(timeout, bool)
            and 30 <= timeout <= 600,
            "authorization has invalid transport timeout")
    return payload


def cli_version(executable: str) -> str:
    result = subprocess.run(
        [executable, "--version"], check=False, capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise CapabilityTrialError(
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
        raise CapabilityTrialError("provider envelope is not an object")
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
            raise CapabilityTrialError(
                f"provider result is not JSON: {exc}"
            ) from exc
    raise CapabilityTrialError("provider envelope has no structured output")


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


def expected_trials(request: dict[str, Any]) -> list[tuple[str, int]]:
    delivery = request["provider_delivery"]
    rows = delivery["views"]
    views = [row["view"] for row in rows]
    require(views == list(request_freezer.VIEWS),
            "request has an unexpected view order")
    require(delivery.get("trial_order") == {
        "kind": "response_index_major_rotating_views",
        "base_view_order": views,
        "rotation_offset_for_trial": "(trial - 1) modulo view count",
    }, "request lacks the frozen rotating view order")
    counts = {row["independent_responses"] for row in rows}
    require(counts == {request_freezer.TRIALS_PER_VIEW},
            "request has an unexpected response count")
    result = []
    for trial in range(1, request_freezer.TRIALS_PER_VIEW + 1):
        offset = (trial - 1) % len(views)
        rotated = views[offset:] + views[:offset]
        result.extend((view, trial) for view in rotated)
    require(len(result) == delivery["total_conditional_calls"],
            "request total call count is inconsistent")
    return result


def run_one(
    *, view: str, trial: int, prompt: str, graph: dict[str, Any],
    request: dict[str, Any], authorization: dict[str, Any],
    authorization_id: str, system_prompt: str,
    response_schema: dict[str, Any], output_dir: Path,
    observed_cli_version: str,
) -> dict[str, Any]:
    trial_dir = output_dir / view / f"trial{trial:02d}"
    if trial_dir.exists():
        raise CapabilityTrialError(
            f"refusing partial or duplicate trial: {view}/trial{trial:02d}"
        )
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
                prefix="gicc-compiler-llm-capability-") as temporary_cwd:
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
        except (json.JSONDecodeError, CapabilityTrialError) as exc:
            parse_error = str(exc)
        break

    bridged = policy_bridge.decision_to_policy(graph, response)
    response_path = trial_dir / "response.json"
    hint_path = trial_dir / "private/compiler-hint.json"
    write_json_atomic(response_path, response)
    write_json_atomic(hint_path, bridged["compiler_hint"])
    record = {
        "schema_version": RUN_SCHEMA,
        "view": view,
        "trial": trial,
        "request_id": request["request_id"],
        "authorization_id": authorization_id,
        "compiler_graph_id": bridged["graph_id"],
        "decision_family": bridged["decision_family"],
        "prompt_sha256": sha256_bytes(prompt.encode()),
        "system_prompt_sha256": sha256_bytes(system_prompt.encode()),
        "response_schema_sha256": sha256_bytes(
            (json.dumps(response_schema, indent=2, sort_keys=True) + "\n").encode()
        ),
        "provider_cli_version": observed_cli_version,
        "requested_model": provider["requested_model"],
        "actual_models": actual_models(envelope),
        "attempts": attempts,
        "provider_call_succeeded": transport_succeeded,
        "response_parse_error": parse_error,
        "semantic_retry_count": 0,
        "response_sha256": sha256_file(response_path),
        "response_sha256_canonical": sha256_bytes(canonical(response)),
        "compiler_hint_sha256": sha256_file(hint_path),
        "compiler_hint_id": bridged["compiler_hint_id"],
        "compiler_hint_is_private_and_never_provider_input": True,
        "bridge_accepted": bridged["bridge_accepted"],
        "family_bridge_errors": bridged["bridge_errors"],
        "fallback_applied": bridged["fallback_applied"],
        "selected_ids_by_slot": bridged["selected_ids_by_slot"],
        "policy_id": bridged["policy_id"],
    }
    write_json_atomic(trial_dir / "run.json", record)
    return record


def verify_archived_run(
    record: Any, output_dir: Path, request: dict[str, Any],
    authorization: dict[str, Any], authorization_id: str,
    graph: dict[str, Any],
) -> tuple[str, int]:
    require(
        isinstance(record, dict) and record.get("schema_version") == RUN_SCHEMA
        and record.get("request_id") == request["request_id"]
        and record.get("authorization_id") == authorization_id
        and isinstance(record.get("view"), str)
        and isinstance(record.get("trial"), int),
        "run index has an invalid archived trial",
    )
    view = record["view"]
    trial = record["trial"]
    matching_views = [
        row for row in request["provider_delivery"]["views"]
        if row["view"] == view
    ]
    require(len(matching_views) == 1,
            f"archived trial has an unknown view: {view}")
    require(1 <= trial <= matching_views[0]["independent_responses"],
            f"archived trial index is out of range: {view}/trial{trial:02d}")
    require(
        record.get("prompt_sha256") == matching_views[0]["prompt_sha256"]
        and record.get("system_prompt_sha256")
        == request["provider_delivery"]["system_prompt_sha256"]
        and record.get("response_schema_sha256")
        == request["provider_delivery"]["response_schema_sha256"],
        f"archived provider input identity changed: {view}/trial{trial:02d}",
    )
    require(
        record.get("provider_cli_version")
        == authorization["provider"]["cli_version"]
        and record.get("requested_model")
        == authorization["provider"]["requested_model"],
        f"archived provider identity changed: {view}/trial{trial:02d}",
    )
    require(isinstance(record.get("actual_models"), list)
            and all(isinstance(model, str) and model
                    for model in record["actual_models"]),
            f"archived actual model identities are invalid: {view}/trial{trial:02d}")
    trial_dir = output_dir / view / f"trial{trial:02d}"
    require(read_json(trial_dir / "run.json") == record,
            f"archived run/index mismatch: {view}/trial{trial:02d}")
    response_path = trial_dir / "response.json"
    hint_path = trial_dir / "private/compiler-hint.json"
    response = read_json(response_path)
    require(
        sha256_file(response_path) == record.get("response_sha256")
        and sha256_bytes(canonical(response))
        == record.get("response_sha256_canonical")
        and sha256_file(hint_path) == record.get("compiler_hint_sha256"),
        f"archived response or hint changed: {view}/trial{trial:02d}",
    )
    bridged = policy_bridge.decision_to_policy(graph, response)
    require(read_json(hint_path) == bridged["compiler_hint"],
            f"compiler hint does not regenerate: {view}/trial{trial:02d}")
    for key in (
        "compiler_graph_id", "decision_family", "compiler_hint_id",
        "bridge_accepted", "family_bridge_errors", "fallback_applied",
        "selected_ids_by_slot", "policy_id",
    ):
        expected_key = {
            "compiler_graph_id": "graph_id",
            "family_bridge_errors": "bridge_errors",
        }.get(key, key)
        require(record.get(key) == bridged[expected_key],
                f"archived bridge field changed: {view}/trial{trial:02d}/{key}")
    require(record.get("semantic_retry_count") == 0,
            f"semantic retry recorded: {view}/trial{trial:02d}")
    require(isinstance(record.get("provider_call_succeeded"), bool),
            f"archived transport result is invalid: {view}/trial{trial:02d}")
    if record.get("response_parse_error") is not None:
        require(record.get("fallback_applied") is True,
                f"parse failure did not fall back: {view}/trial{trial:02d}")
    attempts = record.get("attempts")
    require(isinstance(attempts, list) and attempts,
            f"archived trial has no attempts: {view}/trial{trial:02d}")
    require(len(attempts) <= authorization["transport_retry"][
        "max_attempts_per_trial"
    ], f"too many transport attempts: {view}/trial{trial:02d}")
    for wanted, attempt in enumerate(attempts, 1):
        require(isinstance(attempt, dict) and attempt.get("attempt") == wanted,
                f"invalid attempt archive: {view}/trial{trial:02d}")
        prefix = trial_dir / f"attempt{wanted}"
        require(
            sha256_file(prefix.with_suffix(".stdout.json"))
            == attempt.get("stdout_sha256")
            and sha256_file(prefix.with_suffix(".stderr.txt"))
            == attempt.get("stderr_sha256"),
            f"attempt archive changed: {view}/trial{trial:02d}",
        )
        if wanted < len(attempts):
            require(
                attempt.get("transport_error") is not None
                or attempt.get("returncode") != 0,
                f"successful transport was retried: {view}/trial{trial:02d}",
            )
    final = attempts[-1]
    require(
        record["provider_call_succeeded"]
        == (final.get("transport_error") is None and final.get("returncode") == 0),
        f"archived transport status is inconsistent: {view}/trial{trial:02d}",
    )
    final_stdout = (
        trial_dir / f"attempt{len(attempts)}.stdout.json"
    ).read_text(encoding="utf-8")
    if record["provider_call_succeeded"]:
        envelope: Any = None
        extracted: Any = None
        observed_parse_error = None
        try:
            envelope = json.loads(final_stdout)
            extracted = extract_response(envelope)
        except (json.JSONDecodeError, CapabilityTrialError) as exc:
            observed_parse_error = str(exc)
        require(record.get("response_parse_error") == observed_parse_error,
                f"parse outcome changed: {view}/trial{trial:02d}")
        require(record.get("actual_models") == actual_models(envelope),
                f"actual model identities changed: {view}/trial{trial:02d}")
        require(extracted == response,
                f"extracted response changed: {view}/trial{trial:02d}")
    else:
        require(response is None and record.get("response_parse_error") is None
                and record.get("actual_models") == [],
                f"failed transport contains semantic output: {view}/trial{trial:02d}")
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
    require(
        isinstance(value, dict) and value.get("schema_version") == INDEX_SCHEMA
        and value.get("request_id") == request["request_id"]
        and value.get("authorization_id") == authorization_id
        and isinstance(value.get("runs"), list),
        "existing run index belongs to another protocol",
    )
    return value


def archive_authorization(value: Any, output_dir: Path) -> None:
    path = output_dir / "authorization.json"
    if path.exists():
        require(read_json(path) == value,
                "archived authorization differs from requested run")
    else:
        write_json_atomic(path, value)


def run_trials(
    *, suite_path: Path, prompt_dir: Path, readiness_path: Path,
    separation_path: Path, protocol_path: Path, label: str,
    graph_path: Path, request_dir: Path, authorization_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    inputs = request_freezer.verified_inputs(
        suite_path, prompt_dir, readiness_path, separation_path,
        protocol_path, label, graph_path,
    )
    request = request_freezer.verify_bundle(inputs, request_dir)
    authorization_value = read_json(authorization_path)
    authorization = verify_authorization(authorization_value, request)
    authorization_id = authorization_value["authorization_id"]
    provider = authorization["provider"]
    version = cli_version(provider["executable"])
    require(version == provider["cli_version"],
            "provider CLI version differs from authorization")

    graph = policy_bridge.verified_graph(
        read_json(request_dir / "private/compiler-graph.json")
    )
    system_prompt = (request_dir / "system-prompt.txt").read_text(encoding="utf-8")
    response_schema = read_json(request_dir / "response-schema.json")
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_authorization(authorization_value, output_dir)
    index_path = output_dir / "run-index.json"
    index = existing_index(index_path, request, authorization_id)
    wanted = expected_trials(request)
    require(index.get("status") != "transport_failed",
            "a provider transport failure exhausted this authorization")
    require(index.get("status") in {"running", "complete"},
            "existing run index has an invalid state")
    completed = [
        verify_archived_run(
            row, output_dir, request, authorization, authorization_id, graph,
        )
        for row in index["runs"]
    ]
    require(completed == wanted[:len(completed)],
            "existing trials are not a valid rotating-order prefix")
    if index["status"] == "complete":
        require(completed == wanted, "complete run index is missing trials")
        return index

    for view, trial in wanted[len(completed):]:
        prompt = (request_dir / f"prompts/{view}.txt").read_text(encoding="utf-8")
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
            raise CapabilityTrialError(
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
    request_freezer.add_inputs(parser)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        index = run_trials(
            suite_path=args.suite.resolve(),
            prompt_dir=args.prompt_dir.resolve(),
            readiness_path=args.readiness.resolve(),
            separation_path=args.input_separation.resolve(),
            protocol_path=args.capability_protocol.resolve(),
            label=args.label, graph_path=args.graph.resolve(),
            request_dir=args.request_dir.resolve(),
            authorization_path=args.authorization.resolve(),
            output_dir=args.output_dir.resolve(),
        )
        print(f"MODEL_TRIALS_COMPLETE count={len(index['runs'])}")
        return 0
    except (
        CapabilityTrialError, request_freezer.CapabilityRequestError,
        policy_bridge.CompilerPolicyBridgeError, OSError, KeyError,
        TypeError, ValueError,
    ) as exc:
        print(f"compiler-llm-capability-trials: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
