#!/usr/bin/env python3
"""Freeze one eligibility-gated, compiler-only suite provider request.

Preparation is local and deliberately has no provider, compiler, scheduler, or
application-source path.  It refuses to create a request unless the current
machine-audited capability protocol marks the exact suite entry runtime-ready.
The resulting bundle exposes only source-free prompt views, a response schema,
and a system prompt to a future provider runner; the compiler graph remains a
private downstream bridge input.
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

import audit_compiler_llm_capability_protocol as capability  # noqa: E402
import gicc_compiler_decision_suite as decision_suite  # noqa: E402
import gicc_compiler_policy_bridge as policy_bridge  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


REQUEST_SCHEMA = "gicc-compiler-llm-capability-request-v1"
VIEWS = ("relational", "descriptors", "opaque")
TRIALS_PER_VIEW = 20
SYSTEM_PROMPT = (
    "Act only as a compiler policy selector. Use only the source-free compiler "
    "facts in the user prompt and return exactly one JSON object matching the "
    "provided schema. Select only existing graph-bound compiler IDs. Do not "
    "use tools, source code, LLVM IR, external context, runtime measurements, "
    "or oracle labels. Do not generate code or compiler transformations.\n"
)


class CapabilityRequestError(RuntimeError):
    """The entry is ineligible or the request bundle is not exact."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapabilityRequestError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise CapabilityRequestError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


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


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CapabilityRequestError(message)


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


def eligible_entry(protocol: Any, suite: dict[str, Any],
                   label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if (not isinstance(protocol, dict)
            or protocol.get("schema_version") != capability.REPORT_SCHEMA):
        raise CapabilityRequestError("wrong capability protocol schema")
    payload = dict(protocol)
    protocol_id = payload.pop("protocol_id", None)
    require(protocol_id == capability.fingerprint(payload),
            "capability protocol ID does not match content")
    require(protocol.get("suite_id") == suite["suite_id"],
            "capability protocol binds another suite")
    boundary = protocol.get("boundary", {})
    for key, wanted in {
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
    }.items():
        require(boundary.get(key) is wanted,
                f"capability boundary mismatch: {key}")
    require(protocol.get("summary", {}).get("provider_calls_made_count") == 0,
            "capability protocol already records provider calls")
    require(protocol.get("summary", {}).get("provider_calls_authorized_count") == 0,
            "capability protocol already records authorization")

    suite_entries = {
        entry["label"]: entry for entry in suite["entries"]
    }
    require(label in suite_entries, f"suite entry is absent: {label}")
    protocol_entry = protocol.get("entries", {}).get(label)
    require(isinstance(protocol_entry, dict),
            f"capability entry is absent: {label}")
    suite_entry = suite_entries[label]
    require(
        protocol_entry.get("suite_entry_id") == suite_entry["entry_id"]
        and protocol_entry.get("compiler_graph_id") == suite_entry["graph_id"]
        and protocol_entry.get("decision_family") == suite_entry["decision_family"],
        f"{label}: capability/suite identity mismatch",
    )
    require(protocol_entry.get("eligible_to_freeze_provider_request") is True,
            f"{label}: runtime evidence does not permit request freezing")
    require(protocol_entry.get("runtime_readiness_status")
            == "provider_protocol_permitted",
            f"{label}: runtime readiness status is not permitted")
    require(protocol_entry.get("provider_call_authorized") is False,
            f"{label}: protocol unexpectedly records authorization")
    require(protocol_entry.get("current_permitted_provider_calls") == 0,
            f"{label}: provider calls are permitted before authorization")
    conditional = protocol_entry.get("conditional_protocol", {})
    require(conditional.get("views") == list(VIEWS),
            f"{label}: prompt views differ from the capability contract")
    require(conditional.get("independent_responses_per_view") == TRIALS_PER_VIEW,
            f"{label}: response count differs from the capability contract")
    require(conditional.get("same_selectable_ids_across_views") is True,
            f"{label}: prompt views have unequal action authority")
    return suite_entry, protocol_entry


def verified_inputs(
    suite_path: Path,
    prompt_dir: Path,
    readiness_path: Path,
    separation_path: Path,
    protocol_path: Path,
    label: str,
    graph_path: Path,
) -> dict[str, Any]:
    suite = decision_suite.verified_suite(read_json(suite_path), prompt_dir)
    expected_protocol = capability.build_report(
        suite_path, prompt_dir, readiness_path, separation_path,
    )
    protocol = read_json(protocol_path)
    require(protocol == expected_protocol,
            "capability protocol does not regenerate from readiness evidence")
    suite_entry, protocol_entry = eligible_entry(protocol, suite, label)
    graph = policy_bridge.verified_graph(read_json(graph_path))
    require(graph.get("graph_id") == suite_entry["graph_id"],
            f"{label}: graph ID differs from suite")
    require(graph.get("schema_version") == suite_entry["graph_schema"],
            f"{label}: graph schema differs from suite")
    require(sha256_file(graph_path) == suite_entry["graph_file_sha256"],
            f"{label}: graph bytes differ from suite")

    schema_path = prompt_dir / label / "response-schema.json"
    schema = read_json(schema_path)
    require(schema == policy_bridge.decision_response_schema(graph),
            f"{label}: response schema is not canonical for graph")
    schema_record = suite_entry["response_schema"]
    require(
        sha256_file(schema_path) == schema_record["file_sha256"]
        and schema_path.stat().st_size == schema_record["bytes"]
        and bridge._fingerprint(schema) == schema_record["response_schema_id"],
        f"{label}: response schema identity mismatch",
    )
    prompts = {view: prompt_dir / label / f"{view}.txt" for view in VIEWS}
    for view, path in prompts.items():
        record = suite_entry["views"][view]
        require(
            sha256_file(path) == record["prompt_sha256"]
            and path.stat().st_size == record["prompt_bytes"],
            f"{label}/{view}: prompt identity mismatch",
        )
    return {
        "suite_path": suite_path,
        "suite": suite,
        "readiness_path": readiness_path,
        "separation_path": separation_path,
        "protocol_path": protocol_path,
        "protocol": protocol,
        "suite_entry": suite_entry,
        "protocol_entry": protocol_entry,
        "graph_path": graph_path,
        "graph": graph,
        "schema_path": schema_path,
        "schema": schema,
        "prompts": prompts,
    }


def bundle_record(bundle: Path, path: Path, role: str,
                  provider_visible: bool) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(bundle.resolve()).as_posix()
    except ValueError as exc:
        raise CapabilityRequestError(f"bundle file escapes output: {path}") from exc
    return {
        "role": role,
        "path": relative,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "provider_visible": provider_visible,
    }


def build_request(inputs: dict[str, Any], bundle: Path) -> dict[str, Any]:
    entry = inputs["suite_entry"]
    protocol_entry = inputs["protocol_entry"]
    system_path = bundle / "system-prompt.txt"
    schema_path = bundle / "response-schema.json"
    graph_path = bundle / "private/compiler-graph.json"
    prompt_paths = {view: bundle / f"prompts/{view}.txt" for view in VIEWS}
    files = [
        bundle_record(bundle, system_path, "system_prompt", True),
        bundle_record(bundle, schema_path, "response_schema", True),
        bundle_record(bundle, graph_path, "compiler_graph", False),
        *[
            bundle_record(bundle, prompt_paths[view], f"prompt_{view}", True)
            for view in VIEWS
        ],
    ]
    views = [{
        "view": view,
        "prompt_path": f"prompts/{view}.txt",
        "prompt_sha256": sha256_file(prompt_paths[view]),
        "prompt_bytes": prompt_paths[view].stat().st_size,
        "independent_responses": TRIALS_PER_VIEW,
    } for view in VIEWS]
    payload = {
        "schema_version": REQUEST_SCHEMA,
        "status": "awaiting_explicit_content_addressed_authorization",
        "capability_protocol_id": inputs["protocol"]["protocol_id"],
        "suite_id": inputs["suite"]["suite_id"],
        "suite_entry_id": entry["entry_id"],
        "label": entry["label"],
        "decision_family": entry["decision_family"],
        "compiler_graph_id": entry["graph_id"],
        "runtime_readiness_status": protocol_entry["runtime_readiness_status"],
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_visible": False,
            "source_locations_visible": False,
            "llvm_ir_visible": False,
            "evaluation_runtime_labels_visible": False,
            "evaluation_oracle_visible": False,
            "model_tools": [],
            "model_may_generate_code_or_ir": False,
            "model_output_is_existing_graph_bound_ids_only": True,
            "compiler_revalidates_every_response": True,
            "compiler_graph_is_private_downstream_input": True,
        },
        "provider_delivery": {
            "system_prompt_path": "system-prompt.txt",
            "system_prompt_sha256": sha256_file(system_path),
            "response_schema_path": "response-schema.json",
            "response_schema_sha256": sha256_file(schema_path),
            "views": views,
            "one_prompt_per_fresh_session": True,
            "calls_are_strictly_sequential": True,
            "trial_order": {
                "kind": "response_index_major_rotating_views",
                "base_view_order": list(VIEWS),
                "rotation_offset_for_trial": "(trial - 1) modulo view count",
            },
            "total_conditional_calls": TRIALS_PER_VIEW * len(VIEWS),
        },
        "authorization": {
            "required": True,
            "granted": False,
            "currently_permitted_provider_calls": 0,
            "must_bind_exactly": [
                "request_id", "system_prompt_sha256",
                "response_schema_sha256", "all view prompt SHA-256 values",
                "provider kind/executable/version/model/effort",
                "fresh-session and structured-output settings",
                "transport-only retry limits", "trial order",
            ],
        },
        "preregistered_scoring": inputs["protocol"]["preregistered_scoring"],
        "runtime_validation": inputs["protocol"]["runtime_validation"],
        "implementation": {
            "request_preparer": evidence(Path(__file__)),
            "model_trial_runner": evidence(
                HERE / "run_compiler_llm_capability_trials.py"
            ),
            "capability_analyzer": evidence(
                HERE / "analyze_compiler_llm_capability_trials.py"
            ),
            "unified_compiler_policy_bridge": evidence(
                PASS_PYTHON / "gicc_compiler_policy_bridge.py"
            ),
        },
        "evidence": {
            "capability_protocol": evidence(inputs["protocol_path"]),
            "suite": evidence(inputs["suite_path"]),
            "readiness": evidence(inputs["readiness_path"]),
            "input_separation": evidence(inputs["separation_path"]),
            "source_graph": evidence(inputs["graph_path"]),
            "bundle_files": files,
        },
    }
    result = dict(payload)
    result["request_id"] = bridge._fingerprint(payload)
    return result


def materialize_bundle(inputs: dict[str, Any], output_dir: Path) -> None:
    write_text_atomic(output_dir / "system-prompt.txt", SYSTEM_PROMPT)
    write_text_atomic(
        output_dir / "response-schema.json",
        inputs["schema_path"].read_text(encoding="utf-8"),
    )
    write_text_atomic(
        output_dir / "private/compiler-graph.json",
        inputs["graph_path"].read_text(encoding="utf-8"),
    )
    for view in VIEWS:
        write_text_atomic(
            output_dir / f"prompts/{view}.txt",
            inputs["prompts"][view].read_text(encoding="utf-8"),
        )


def prepare(inputs: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise CapabilityRequestError(
            f"refusing to overwrite request bundle: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    try:
        materialize_bundle(inputs, output_dir)
        request = build_request(inputs, output_dir)
        write_json_atomic(output_dir / "request.json", request)
        return request
    except BaseException:
        # Retain a partial directory for forensic inspection; it is never a
        # valid request because request.json is written last and prepare will
        # refuse to resume or overwrite it.
        raise


def verify_bundle(inputs: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    request_path = output_dir / "request.json"
    request = read_json(request_path)
    require(request.get("schema_version") == REQUEST_SCHEMA,
            "wrong request schema")
    payload = dict(request)
    request_id = payload.pop("request_id", None)
    require(request_id == bridge._fingerprint(payload),
            "request ID does not match content")
    require((output_dir / "system-prompt.txt").read_text(encoding="utf-8")
            == SYSTEM_PROMPT, "system prompt changed")
    require((output_dir / "response-schema.json").read_bytes()
            == inputs["schema_path"].read_bytes(), "response schema changed")
    require((output_dir / "private/compiler-graph.json").read_bytes()
            == inputs["graph_path"].read_bytes(), "private graph changed")
    for view in VIEWS:
        require((output_dir / f"prompts/{view}.txt").read_bytes()
                == inputs["prompts"][view].read_bytes(),
                f"{view} prompt changed")
    require(request == build_request(inputs, output_dir),
            "request does not regenerate from current evidence")
    return request


def add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--input-separation", type=Path, required=True)
    parser.add_argument("--capability-protocol", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--graph", type=Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    add_inputs(freeze)
    freeze.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    add_inputs(verify)
    verify.add_argument("--request-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        inputs = verified_inputs(
            args.suite.resolve(), args.prompt_dir.resolve(),
            args.readiness.resolve(), args.input_separation.resolve(),
            args.capability_protocol.resolve(), args.label,
            args.graph.resolve(),
        )
        if args.command == "freeze":
            request = prepare(inputs, args.output_dir.resolve())
            action = "frozen but NOT authorized"
        else:
            request = verify_bundle(inputs, args.request_dir.resolve())
            action = "verified and still NOT authorized"
        print(
            f"compiler-llm-capability-request: {action}; "
            f"label={request['label']}; calls_permitted=0; "
            f"request_id={request['request_id']}"
        )
        return 0
    except (
        CapabilityRequestError, capability.CapabilityProtocolError,
        decision_suite.SuiteError, policy_bridge.CompilerPolicyBridgeError,
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"compiler-llm-capability-request: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
