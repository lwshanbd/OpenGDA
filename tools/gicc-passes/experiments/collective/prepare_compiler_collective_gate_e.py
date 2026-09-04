#!/usr/bin/env python3
"""Freeze an authorization-gated, compiler-only Gate-E model protocol.

This tool never invokes a provider or submits a scheduler job.  Preparation is
allowed only after the paired Gate-D analysis can be regenerated from its raw
monitors and confirms non-zero compiler-policy headroom.
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
ROOT = HERE.parents[3]
PASS_PYTHON = ROOT / "tools/gicc-passes/python"
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(HERE))

import compiler_collective_eval as controls  # noqa: E402
import gicc_collective_plan_bridge as plans  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


REQUEST_SCHEMA = "gicc-collective-gate-e-request-v1"
TRIALS_PER_VIEW = 20
SYSTEM_PROMPT = (
    "Act only as a compiler policy selector. Use only the source-free compiler "
    "facts in the user prompt and return exactly one JSON object matching its "
    "schema. Do not use tools, source code, external context, or runtime labels.\n"
)


class GateEError(RuntimeError):
    """Gate E is incomplete, inconsistent, or not safe to authorize."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GateEError(f"cannot read JSON {path}: {exc}") from exc


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


def exact_response_schema(graph_value: Any) -> dict[str, Any]:
    graph = plans.verified_graph(graph_value)
    if len(graph["opportunities"]) != 1:
        raise GateEError("Gate E v1 requires exactly one opportunity")
    return plans.decision_response_schema(graph)


def validate_gate_d_result(value: Any, graph: dict[str, Any],
                           manifest: dict[str, Any]) -> None:
    if (not isinstance(value, dict)
            or value.get("schema_version")
            != "gicc-collective-confirmatory-analysis-v1"
            or value.get("graph_id") != graph["graph_id"]
            or value.get("manifest_id") != manifest["manifest_id"]):
        raise GateEError("Gate E requires the matching Gate-D analysis")
    primary = value.get("primary_headroom_confirmation")
    if (not isinstance(primary, dict)
            or primary.get("positive_point_estimate") is not True
            or primary.get("confidence_interval_excludes_one") is not True
            or not isinstance(primary.get("point_estimate"), (int, float))
            or primary["point_estimate"] <= 1.0):
        raise GateEError(
            "Gate D did not confirm compiler-policy headroom; model calls stop"
        )
    monitors = value.get("replicate_monitors")
    if (not isinstance(monitors, list) or len(monitors) != 3
            or {row.get("replicate") for row in monitors} != {1, 2, 3}):
        raise GateEError("Gate-D analysis lacks three paired replicate monitors")


def validated_bundle(bundle: Path, repo_root: Path) -> dict[str, Any]:
    bundle = bundle.resolve()
    freeze_path = bundle / "FROZEN_V3_MANIFEST.json"
    graph_path = bundle / "discovery/graph.json"
    manifest_path = bundle / "controls/manifest.json"
    screen_path = bundle / "gate-b/analysis.json"
    gate_d_path = bundle / "gate-d/analysis.json"
    freeze = read_json(freeze_path)
    graph = plans.verified_graph(read_json(graph_path))
    manifest = read_json(manifest_path)
    controls.verify_offline_freeze(
        freeze, manifest_path=freeze_path, repo_root=repo_root.resolve(),
    )
    controls.verify_manifest(graph, manifest, manifest_path.parent)
    screen = read_json(screen_path)
    gate_d = read_json(gate_d_path)
    validate_gate_d_result(gate_d, graph, manifest)

    monitor_paths = [
        Path(row["monitor"]) for row in sorted(
            gate_d["replicate_monitors"], key=lambda row: row["replicate"]
        )
    ]
    regenerated = controls.confirmatory_analysis(
        graph, manifest, screen, monitor_paths,
    )
    regenerated["screen_analysis_sha256"] = sha256_file(screen_path)
    if regenerated != gate_d:
        raise GateEError("Gate-D analysis does not regenerate from raw monitors")

    prompts = {}
    for view in plans.MODEL_VIEW_KINDS:
        path = bundle / f"prompts/{view}.txt"
        try:
            text = path.read_text()
        except OSError as exc:
            raise GateEError(f"cannot read {view} prompt: {exc}") from exc
        if text != plans.render_prompt(graph, view):
            raise GateEError(f"{view} prompt is not the canonical graph view")
        prompts[view] = path
    return {
        "bundle": bundle,
        "freeze_path": freeze_path,
        "freeze": freeze,
        "graph_path": graph_path,
        "graph": graph,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "gate_d_path": gate_d_path,
        "gate_d": gate_d,
        "prompts": prompts,
    }


def bundle_record(path: Path, bundle: Path, role: str) -> dict[str, Any]:
    try:
        relative = path.resolve().relative_to(bundle.resolve())
    except ValueError as exc:
        raise GateEError(f"Gate-E bundle file escapes bundle: {path}") from exc
    return {
        "role": role,
        "path": relative.as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_request(inputs: dict[str, Any], response_schema_path: Path,
                  system_prompt_path: Path) -> dict[str, Any]:
    bundle = inputs["bundle"]
    graph = inputs["graph"]
    gate_d = inputs["gate_d"]
    schema_sha = sha256_file(response_schema_path)
    system_sha = sha256_file(system_prompt_path)
    views = [{
        "view": view,
        "prompt_path": inputs["prompts"][view].relative_to(bundle).as_posix(),
        "prompt_sha256": sha256_file(inputs["prompts"][view]),
        "prompt_bytes": inputs["prompts"][view].stat().st_size,
        "response_schema_sha256": schema_sha,
        "independent_responses": TRIALS_PER_VIEW,
    } for view in plans.MODEL_VIEW_KINDS]
    files = [
        bundle_record(inputs["freeze_path"], bundle, "offline_freeze"),
        bundle_record(inputs["graph_path"], bundle, "compiler_graph"),
        bundle_record(inputs["manifest_path"], bundle, "control_manifest"),
        bundle_record(inputs["gate_d_path"], bundle, "gate_d_analysis"),
        bundle_record(system_prompt_path, bundle, "system_prompt"),
        bundle_record(response_schema_path, bundle, "response_schema"),
        *[
            bundle_record(inputs["prompts"][view], bundle, f"prompt_{view}")
            for view in plans.MODEL_VIEW_KINDS
        ],
    ]
    payload = {
        "schema_version": REQUEST_SCHEMA,
        "status": "awaiting_explicit_provider_authorization",
        "compiler_only": True,
        "model_output_scope": "existing compiler-generated option IDs only",
        "graph_id": graph["graph_id"],
        "control_manifest_id": inputs["manifest"]["manifest_id"],
        "offline_freeze_manifest_id": inputs["freeze"]["manifest_id"],
        "gate_d_analysis_sha256": sha256_file(inputs["gate_d_path"]),
        "confirmed_compiler_bin_headroom": gate_d[
            "primary_headroom_confirmation"
        ],
        "data_boundary": {
            "source_visible": False,
            "source_locations_visible": False,
            "llvm_ir_visible": False,
            "evaluation_runtime_labels_visible": False,
            "model_tools": [],
            "model_may_generate_code": False,
            "model_may_generate_ir": False,
            "compiler_revalidates_every_response": True,
        },
        "provider_delivery": {
            "system_prompt_path": system_prompt_path.relative_to(bundle).as_posix(),
            "system_prompt_sha256": system_sha,
            "one_user_prompt_per_call": True,
            "response_schema_sha256": schema_sha,
            "views": views,
            "calls_are_sequential": True,
            "total_provider_calls": TRIALS_PER_VIEW * len(views),
        },
        "authorization": {
            "required": True,
            "granted": False,
            "must_bind_exactly": [
                "request_id", "system_prompt_sha256", "prompt_sha256",
                "provider.kind", "provider.executable",
                "provider.cli_version", "provider.requested_model",
                "provider.effort", "provider.fresh_session_per_trial",
                "provider.structured_output", "transport_retry",
                "independent_responses",
            ],
            "note": (
                "Earlier Minimod/Jacobi authorization does not cover these "
                "collective prompt hashes."
            ),
        },
        "preregistered_analysis": {
            "per_view_metrics": [
                "invalid_output_rate", "exact_oracle_policy_rate",
                "mean_bin_choice_accuracy", "modal_policy_rate",
                "distance_to_confirmatory_compiler_bin_oracle",
            ],
            "runtime_representatives_per_view": [
                "modal_accepted_policy",
                "best_of_20_control_screen_policy_upper_bound",
            ],
            "representatives_are_deduplicated_across_views": True,
            "runtime_design": (
                "three same-allocation paired pdebug blocks; baseline, frozen "
                "compiler controls, and every representative run sequentially"
            ),
            "runtime_jobs_active_or_queued_at_once": 1,
            "best_of_20_is_labeled_posthoc_upper_bound": True,
        },
        "comparators": {
            "semantic_compiler_baseline": "eligible",
            "best_uniform_compiler_target": "eligible",
            "confirmatory_compiler_bin_oracle": "capacity_bound_only",
            "deterministic_source_free_heuristic": "required_before_runtime",
            "gbt": (
                "ineligible until a disjoint collective-algorithm training "
                "dataset exists; old path-selection labels are not reusable"
            ),
        },
        "implementation": {
            "gate_e_preparer": {
                "path": (HERE / "prepare_compiler_collective_gate_e.py")
                .relative_to(ROOT).as_posix(),
                "sha256": sha256_file(
                    HERE / "prepare_compiler_collective_gate_e.py"
                ),
            },
            "model_trial_runner": {
                "path": (HERE / "run_compiler_collective_model_trials.py")
                .relative_to(ROOT).as_posix(),
                "sha256": sha256_file(
                    HERE / "run_compiler_collective_model_trials.py"
                ),
            },
        },
        "files": files,
    }
    request = dict(payload)
    request["request_id"] = bridge._fingerprint(payload)
    return request


def prepare(bundle: Path, output_dir: Path, repo_root: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise GateEError(f"refusing to overwrite Gate-E output: {output_dir}")
    inputs = validated_bundle(bundle, repo_root)
    output_dir.mkdir(parents=True)
    system_path = output_dir / "system-prompt.txt"
    schema_path = output_dir / "response-schema.json"
    write_text_atomic(system_path, SYSTEM_PROMPT)
    write_json_atomic(schema_path, exact_response_schema(inputs["graph"]))
    request = build_request(inputs, schema_path, system_path)
    write_json_atomic(output_dir / "request.json", request)
    return request


def verify(request_path: Path, bundle: Path, repo_root: Path) -> dict[str, Any]:
    request = read_json(request_path)
    if not isinstance(request, dict) or request.get("schema_version") != REQUEST_SCHEMA:
        raise GateEError(f"expected {REQUEST_SCHEMA}")
    payload = dict(request)
    request_id = payload.pop("request_id", None)
    if request_id != bridge._fingerprint(payload):
        raise GateEError("Gate-E request_id does not match content")
    inputs = validated_bundle(bundle, repo_root)
    output_dir = request_path.resolve().parent
    schema_path = output_dir / "response-schema.json"
    system_path = output_dir / "system-prompt.txt"
    if system_path.read_text() != SYSTEM_PROMPT:
        raise GateEError("Gate-E system prompt changed")
    if read_json(schema_path) != exact_response_schema(inputs["graph"]):
        raise GateEError("Gate-E response schema changed")
    expected = build_request(inputs, schema_path, system_path)
    if request != expected:
        raise GateEError("Gate-E request does not match regenerated protocol")
    return request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--bundle", type=Path, required=True)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    prepare_parser.add_argument("--repo-root", type=Path, default=ROOT)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--request", type=Path, required=True)
    verify_parser.add_argument("--bundle", type=Path, required=True)
    verify_parser.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            request = prepare(args.bundle, args.output_dir, args.repo_root)
            print(
                "Gate E frozen but NOT AUTHORIZED: "
                f"request_id={request['request_id']} "
                f"provider_calls={request['provider_delivery']['total_provider_calls']}"
            )
        else:
            request = verify(args.request, args.bundle, args.repo_root)
            print(
                "Gate E request verified and still NOT AUTHORIZED: "
                f"request_id={request['request_id']}"
            )
        return 0
    except (GateEError, controls.EvalError, plans.CollectivePlanError,
            OSError, KeyError, TypeError, ValueError) as exc:
        print(f"compiler-collective-gate-e: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
