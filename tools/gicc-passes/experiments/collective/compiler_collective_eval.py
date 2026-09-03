#!/usr/bin/env python3
"""Generate, verify, and analyze compiler-only collective controls.

Uniform compiler catalog arms establish the measurable performance envelope
without invoking a model.  Their per-size results also define a compiler-stage
piecewise-policy oracle over the exact thresholds exposed to the LLM.  This
tool never reads or rewrites the fixed benchmark source; its hash is supplied
and frozen in the control manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PASS_PYTHON = ROOT / "tools" / "gicc-passes" / "python"
sys.path.insert(0, str(PASS_PYTHON))

import gicc_collective_plan_bridge as plans  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


MANIFEST_SCHEMA = "gicc-collective-uniform-controls-v1"
BUILD_PROVENANCE_SCHEMA = "gicc-collective-build-provenance-v1"
OFFLINE_FREEZE_SCHEMA = "gicc-collective-offline-freeze-v2"
RESULT_RE = re.compile(r"([a-z_]+)=([^ ]+)")
HDIR_DEVICE_KERNEL = (
    "_ZN9gicc_coll27hier_direct_rs_cross_kernelEPN4gicc9DeviceCtxE"
    "iiiiiiiiPfS3_PVj"
)
GATE_A_SIZES = [1024, 4096]
GATE_B_SIZES = [
    1024, 4096, 8192, 65536, 262144,
    1048576, 4194304, 8388608, 16777216,
]
GATE_SPECS = {
    "a": {
        "nodes": 2,
        "ranks": 16,
        "ppn": 8,
        "runs": 1,
        "warmup": 0,
        "sizes": GATE_A_SIZES,
        "catalog_complete": False,
    },
    "b": {
        "nodes": 2,
        "ranks": 16,
        "ppn": 8,
        "runs": 3,
        "warmup": 1,
        "sizes": GATE_B_SIZES,
        "catalog_complete": True,
    },
}


class EvalError(ValueError):
    """A compiler control or runtime result is incomplete or inconsistent."""


def _parse_assignments(
    values: list[str], *, kind: str, allow_unset: bool = False,
) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for value in values:
        if "=" in value:
            name, payload = value.split("=", 1)
        elif allow_unset:
            name, payload = value, None
        else:
            raise EvalError(f"{kind} must use NAME=VALUE: {value}")
        if not name or name in result:
            raise EvalError(f"invalid or duplicate {kind} name: {name!r}")
        if payload == "" and not allow_unset:
            raise EvalError(f"{kind} has an empty value: {name}")
        result[name] = payload
    return result


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _file_record(
    path: Path, *, role: str, repo_root: Path, build_root: Path,
) -> dict[str, Any]:
    absolute = _lexical_absolute(path)
    if not absolute.is_file():
        raise EvalError(f"missing provenance file for {role}: {absolute}")
    try:
        relative = absolute.relative_to(build_root)
        scope = "build"
        locator = relative.as_posix()
    except ValueError:
        try:
            relative = absolute.relative_to(repo_root)
            scope = "repository"
            locator = relative.as_posix()
        except ValueError:
            scope = "external"
            locator = os.fspath(absolute)
    resolved = absolute.resolve()
    record = {
        "role": role,
        "locator": {"scope": scope, "path": locator},
        "size_bytes": absolute.stat().st_size,
        "sha256": _sha256(absolute),
    }
    if resolved != absolute:
        record["resolved_path"] = os.fspath(resolved)
    return record


def _record_path(
    record: dict[str, Any], *, repo_root: Path, build_root: Path,
) -> Path:
    locator = record.get("locator")
    if not isinstance(locator, dict):
        raise EvalError("provenance record lacks a locator")
    scope = locator.get("scope")
    path = locator.get("path")
    if not isinstance(path, str) or not path:
        raise EvalError("provenance locator has no path")
    if scope == "build":
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise EvalError("build provenance locator escapes its root")
        result = build_root / path
    elif scope == "repository":
        if Path(path).is_absolute() or ".." in Path(path).parts:
            raise EvalError("repository provenance locator escapes its root")
        result = repo_root / path
    elif scope == "external":
        result = Path(path)
        if not result.is_absolute():
            raise EvalError("external provenance path must be absolute")
    else:
        raise EvalError(f"unknown provenance locator scope: {scope}")
    return _lexical_absolute(result)


def _dependency_paths(path: Path, *, compile_root: Path) -> list[Path]:
    try:
        text = path.read_text()
    except OSError as exc:
        raise EvalError(f"cannot read dependency file {path}: {exc}") from exc
    logical = text.replace("\\\n", " ")
    if ":" not in logical:
        raise EvalError(f"invalid compiler dependency file: {path}")
    payload = logical.split(":", 1)[1]
    try:
        tokens = shlex.split(payload, posix=True)
    except ValueError as exc:
        raise EvalError(f"cannot parse dependency file {path}: {exc}") from exc
    if not tokens:
        raise EvalError(f"compiler dependency file is empty: {path}")
    result = []
    for token in tokens:
        dependency = Path(token)
        if not dependency.is_absolute():
            dependency = compile_root / dependency
        result.append(_lexical_absolute(dependency))
    return result


def record_build_command(path: Path, argv: list[str], *, reset: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if reset:
        path.write_text("")
        if argv:
            raise EvalError("record-command --reset does not accept a command")
        return
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        raise EvalError("record-command requires a command after --")
    with path.open("a") as stream:
        stream.write(json.dumps({"argv": argv}, sort_keys=True) + "\n")


def _read_command_log(path: Path) -> list[list[str]]:
    commands = []
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise EvalError(f"cannot read build command log {path}: {exc}") from exc
    for number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvalError(
                f"invalid command-log JSON at {path}:{number}: {exc}"
            ) from exc
        argv = value.get("argv") if isinstance(value, dict) else None
        if (not isinstance(argv, list) or not argv
                or any(not isinstance(item, str) for item in argv)):
            raise EvalError(f"invalid argv at {path}:{number}")
        commands.append(argv)
    if not commands:
        raise EvalError(f"build command log has no commands: {path}")
    return commands


def generate_build_provenance(
    *, mode: str, repo_root: Path, build_root: Path,
    inputs: dict[str, Path], artifacts: dict[str, Path],
    dependency_files: list[Path], environment: dict[str, str | None],
    commands_path: Path,
) -> dict[str, Any]:
    if mode not in {"discover", "lower"}:
        raise EvalError(f"unknown collective build mode: {mode}")
    repo_root = _lexical_absolute(repo_root)
    build_root = _lexical_absolute(build_root)
    required_inputs = {
        "benchmark_source", "catalog_source", "build_script", "evaluator",
        "compiler", "pass_plugin",
    }
    required_artifacts = {
        "eval_object", "inventory", "host_ir", "device_ir",
        "device_ir_audit", "compile_log", "host_ir_log", "device_ir_log",
        "command_log",
    }
    if mode == "lower":
        required_inputs.add("collective_hint")
        required_artifacts.update({
            "binary", "link_log", "runtime_helpers_object",
            "proxy_thread_object", "proxy_libfabric_object",
        })
    missing_inputs = required_inputs - set(inputs)
    missing_artifacts = required_artifacts - set(artifacts)
    if missing_inputs:
        raise EvalError(
            f"build provenance lacks inputs: {sorted(missing_inputs)}"
        )
    if missing_artifacts:
        raise EvalError(
            f"build provenance lacks artifacts: {sorted(missing_artifacts)}"
        )
    if environment.get("GICC_COLLECTIVE_ONLY") != "1":
        raise EvalError("collective build must record GICC_COLLECTIVE_ONLY=1")
    if environment.get("GICC_HINT_IN") is not None:
        raise EvalError("collective-only build must record GICC_HINT_IN unset")
    if mode == "lower" and not environment.get("GICC_COLLECTIVE_HINT_IN"):
        raise EvalError("lower build lacks GICC_COLLECTIVE_HINT_IN")
    if mode == "discover" and environment.get("GICC_COLLECTIVE_HINT_IN") is not None:
        raise EvalError("discover build must not use GICC_COLLECTIVE_HINT_IN")

    input_records = [
        _file_record(path, role=role, repo_root=repo_root,
                     build_root=build_root)
        for role, path in sorted(inputs.items())
    ]
    artifact_records = [
        _file_record(path, role=role, repo_root=repo_root,
                     build_root=build_root)
        for role, path in sorted(artifacts.items())
    ]
    dependencies: dict[str, Path] = {}
    depfile_records = []
    for index, depfile in enumerate(dependency_files):
        depfile_records.append(_file_record(
            depfile, role=f"compiler_dependency_file_{index}",
            repo_root=repo_root, build_root=build_root,
        ))
        for dependency in _dependency_paths(depfile, compile_root=repo_root):
            dependencies[os.fspath(dependency)] = dependency
    dependency_records = [
        _file_record(path, role="translation_unit_dependency",
                     repo_root=repo_root, build_root=build_root)
        for path in sorted(dependencies.values(), key=os.fspath)
    ]
    dependency_locators = {
        (item["locator"]["scope"], item["locator"]["path"])
        for item in dependency_records
    }
    required_dependencies = {
        ("repository", "examples/proxy/coll_common.hpp"),
        ("repository", "tools/gicc-passes/experiments/collective/"
                       "compiler_collective_catalog.hpp"),
        ("repository", "tools/gicc-passes/experiments/collective/"
                       "compiler_collective_eval.cpp"),
    }
    missing_dependencies = required_dependencies - dependency_locators
    if missing_dependencies:
        raise EvalError(
            "compiler dependency closure lacks required sources: "
            f"{sorted(missing_dependencies)}"
        )
    commands = _read_command_log(commands_path)
    compiler = inputs["compiler"]
    try:
        completed = subprocess.run(
            [os.fspath(compiler), "--version"], check=True,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvalError(f"cannot identify compiler {compiler}: {exc}") from exc
    payload = {
        "schema_version": BUILD_PROVENANCE_SCHEMA,
        "build_mode": mode,
        "compiler_only": True,
        "collective_only": True,
        "model_invoked": False,
        "application_source_modified": False,
        "environment": dict(sorted(environment.items())),
        "compiler_version": completed.stdout.strip(),
        "inputs": input_records,
        "dependency_closure": {
            "compile_root": ".",
            "dependency_files": depfile_records,
            "file_count": len(dependency_records),
            "files": dependency_records,
        },
        "commands": {
            "count": len(commands),
            "log_sha256": _sha256(commands_path),
            "argv": commands,
        },
        "artifacts": artifact_records,
    }
    manifest = dict(payload)
    manifest["manifest_id"] = bridge._fingerprint(payload)
    return manifest


def verify_build_provenance(
    value: Any, *, manifest_path: Path, repo_root: Path,
) -> None:
    if (not isinstance(value, dict)
            or value.get("schema_version") != BUILD_PROVENANCE_SCHEMA):
        raise EvalError(f"expected provenance schema {BUILD_PROVENANCE_SCHEMA}")
    payload = dict(value)
    manifest_id = payload.pop("manifest_id", None)
    if manifest_id != bridge._fingerprint(payload):
        raise EvalError("build provenance manifest_id does not match content")
    if (value.get("compiler_only") is not True
            or value.get("collective_only") is not True
            or value.get("model_invoked") is not False
            or value.get("application_source_modified") is not False):
        raise EvalError("build provenance violates the compiler-only boundary")
    mode = value.get("build_mode")
    if mode not in {"discover", "lower"}:
        raise EvalError("build provenance has an invalid mode")
    environment = value.get("environment")
    if (not isinstance(environment, dict)
            or environment.get("GICC_COLLECTIVE_ONLY") != "1"
            or environment.get("GICC_HINT_IN") is not None):
        raise EvalError("build provenance has an invalid lowering scope")
    if mode == "lower" and not environment.get("GICC_COLLECTIVE_HINT_IN"):
        raise EvalError("lower provenance lacks a collective hint")
    if mode == "discover" and environment.get("GICC_COLLECTIVE_HINT_IN") is not None:
        raise EvalError("discover provenance unexpectedly records a hint")

    repo_root = _lexical_absolute(repo_root)
    build_root = _lexical_absolute(manifest_path).parent
    all_records = []
    roles: dict[str, dict[str, Any]] = {}
    for section in ("inputs", "artifacts"):
        records = value.get(section)
        if not isinstance(records, list):
            raise EvalError(f"build provenance lacks {section}")
        for record in records:
            if not isinstance(record, dict) or not isinstance(record.get("role"), str):
                raise EvalError(f"invalid file record in {section}")
            if record["role"] in roles:
                raise EvalError(f"duplicate provenance role: {record['role']}")
            roles[record["role"]] = record
            all_records.append(record)
    closure = value.get("dependency_closure")
    if not isinstance(closure, dict):
        raise EvalError("build provenance lacks dependency closure")
    dependency_records = closure.get("files")
    depfile_records = closure.get("dependency_files")
    if not isinstance(dependency_records, list) or not isinstance(depfile_records, list):
        raise EvalError("invalid provenance dependency closure")
    if closure.get("file_count") != len(dependency_records):
        raise EvalError("dependency-closure file count does not match")
    all_records.extend(dependency_records)
    all_records.extend(depfile_records)
    for record in all_records:
        if not isinstance(record, dict):
            raise EvalError("invalid provenance file record")
        path = _record_path(record, repo_root=repo_root, build_root=build_root)
        if not path.is_file():
            raise EvalError(f"provenance file is missing: {path}")
        expected_hash = record.get("sha256")
        if not isinstance(expected_hash, str):
            raise EvalError(f"provenance record lacks hash: {path}")
        _check_sha(expected_hash, name=f"sha256 for {path}")
        if _sha256(path) != expected_hash:
            raise EvalError(f"provenance hash mismatch: {path}")
        if path.stat().st_size != record.get("size_bytes"):
            raise EvalError(f"provenance size mismatch: {path}")
        resolved = record.get("resolved_path")
        if resolved is not None and os.fspath(path.resolve()) != resolved:
            raise EvalError(f"provenance symlink target mismatch: {path}")

    declared_dependency_locators = {
        (record["locator"].get("scope"), record["locator"].get("path"))
        for record in dependency_records
    }
    actual_dependency_locators = set()
    for record in depfile_records:
        depfile = _record_path(
            record, repo_root=repo_root, build_root=build_root
        )
        for dependency in _dependency_paths(depfile, compile_root=repo_root):
            actual = _file_record(
                dependency, role="translation_unit_dependency",
                repo_root=repo_root, build_root=build_root,
            )
            actual_dependency_locators.add((
                actual["locator"]["scope"], actual["locator"]["path"]
            ))
    if declared_dependency_locators != actual_dependency_locators:
        raise EvalError(
            "stored dependency closure does not match compiler depfiles"
        )
    required_dependencies = {
        ("repository", "examples/proxy/coll_common.hpp"),
        ("repository", "tools/gicc-passes/experiments/collective/"
                       "compiler_collective_catalog.hpp"),
        ("repository", "tools/gicc-passes/experiments/collective/"
                       "compiler_collective_eval.cpp"),
    }
    if not required_dependencies.issubset(declared_dependency_locators):
        raise EvalError("provenance lacks the required catalog dependency chain")

    required_roles = {
        "benchmark_source", "catalog_source", "build_script", "evaluator",
        "compiler", "pass_plugin", "eval_object", "inventory", "host_ir",
        "device_ir", "device_ir_audit", "command_log",
    }
    if mode == "lower":
        required_roles.update({
            "collective_hint", "binary", "runtime_helpers_object",
            "proxy_thread_object", "proxy_libfabric_object",
        })
    missing_roles = required_roles - set(roles)
    if missing_roles:
        raise EvalError(f"build provenance lacks roles: {sorted(missing_roles)}")
    if mode == "lower":
        for role in (
            "runtime_helpers_object", "proxy_thread_object",
            "proxy_libfabric_object",
        ):
            if roles[role]["locator"].get("scope") != "build":
                raise EvalError(f"{role} is not a same-build artifact")
    command_record = roles["command_log"]
    command_path = _record_path(
        command_record, repo_root=repo_root, build_root=build_root
    )
    commands = value.get("commands")
    parsed_commands = _read_command_log(command_path)
    if (not isinstance(commands, dict)
            or commands.get("count") != len(parsed_commands)
            or commands.get("argv") != parsed_commands
            or commands.get("log_sha256") != _sha256(command_path)):
        raise EvalError("recorded build commands do not match the command log")


def _provenance_roles(value: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = [*value.get("inputs", []), *value.get("artifacts", [])]
    if any(not isinstance(record, dict) for record in records):
        raise EvalError("invalid nested build-provenance records")
    result = {record.get("role"): record for record in records}
    if (None in result or len(result) != len(records)
            or any(not isinstance(role, str) for role in result)):
        raise EvalError("duplicate or invalid nested provenance role")
    return result


def _nested_role_path(
    value: dict[str, Any], manifest_path: Path, repo_root: Path, role: str,
) -> Path:
    roles = _provenance_roles(value)
    if role not in roles:
        raise EvalError(f"nested build provenance lacks {role}")
    return _record_path(
        roles[role], repo_root=repo_root,
        build_root=_lexical_absolute(manifest_path).parent,
    )


def _validate_gate_a_monitor(value: Any) -> None:
    if (not isinstance(value, dict)
            or value.get("schema_version") != "gicc-collective-job-monitor-v1"
            or value.get("state") != "passed"):
        raise EvalError("v3 freeze requires a passed Gate-A monitor")
    expected = value.get("expected")
    benchmark = value.get("benchmark")
    scheduler = value.get("scheduler")
    jobspec = value.get("jobspec")
    if (not isinstance(expected, dict) or not isinstance(benchmark, dict)
            or not isinstance(scheduler, dict) or not isinstance(jobspec, dict)):
        raise EvalError("Gate-A monitor is incomplete")
    contract = {
        "nodes": 2, "ranks": 16, "ppn": 8, "runs": 1, "warmup": 0,
    }
    if any(expected.get(name) != expected_value
           for name, expected_value in contract.items()):
        raise EvalError("Gate-A monitor has the wrong runtime contract")
    if expected.get("sizes") != GATE_A_SIZES:
        raise EvalError("Gate-A monitor has the wrong message sizes")
    if benchmark.get("config") != {
        "label": expected.get("label"), **contract,
    }:
        raise EvalError("Gate-A benchmark config does not match its contract")
    if (set(benchmark.get("results", {}))
            != {str(size) for size in GATE_A_SIZES}
            or benchmark.get("total_errors") != 0):
        raise EvalError("Gate-A monitor lacks complete correct results")
    if (scheduler.get("exit_code") != 0
            or scheduler.get("exceptions") != []
            or jobspec.get("queue") != "pdebug"):
        raise EvalError("Gate-A monitor is not a clean pdebug completion")


def _validate_gate_a_logs(value: dict[str, Any], stdout: Path, stderr: Path) -> None:
    record = _parse_log_record(stdout)
    expected = value["expected"]
    contract = {
        "label": expected["label"],
        "nodes": expected["nodes"],
        "ranks": expected["ranks"],
        "ppn": expected["ppn"],
        "runs": expected["runs"],
        "warmup": expected["warmup"],
    }
    if any(record[name] != expected_value
           for name, expected_value in contract.items()):
        raise EvalError("Gate-A stdout does not match the frozen monitor")
    observed_rows = {
        str(size): latency for size, latency in sorted(record["rows"].items())
    }
    if observed_rows != value["benchmark"]["results"]:
        raise EvalError("Gate-A stdout timings do not match the monitor")
    try:
        stdout_bytes = stdout.stat().st_size
        stderr_bytes = stderr.stat().st_size
    except OSError as exc:
        raise EvalError(f"cannot stat Gate-A logs: {exc}") from exc
    if (stdout_bytes != value.get("stdout_bytes")
            or stderr_bytes != value.get("stderr_bytes")):
        raise EvalError("Gate-A raw-log sizes do not match the monitor")


def _compare_generated_hint(stored_value: Any, generated_value: Any) -> None:
    if not isinstance(stored_value, dict) or not isinstance(generated_value, dict):
        raise EvalError("compiler hint is not an object")
    stored = json.loads(json.dumps(stored_value))
    generated = json.loads(json.dumps(generated_value))
    for value in (stored, generated):
        metadata = value.get("llm_metadata")
        if not isinstance(metadata, dict):
            raise EvalError("compiler hint lacks llm_metadata")
        metadata.pop("producer", None)
        metadata.pop("model_invoked", None)
    if stored != generated:
        raise EvalError("stored compiler hint does not match its decision")


def generate_offline_freeze(
    *, bundle_root: Path, repo_root: Path, platform_path: Path,
    calibration_paths: list[Path], gate_a_monitor_path: Path,
    gate_a_stdout_path: Path, gate_a_stderr_path: Path,
    preparation_script: Path,
) -> dict[str, Any]:
    """Validate and content-address a complete v3 offline bundle."""
    bundle_root = _lexical_absolute(bundle_root)
    repo_root = _lexical_absolute(repo_root)
    graph_path = bundle_root / "discovery/graph.json"
    controls_path = bundle_root / "controls/manifest.json"
    capacity_path = bundle_root / "capacity-audit.json"
    discovery_provenance_path = (
        bundle_root / "discovery/build/build-provenance.json"
    )
    canary_decision_path = bundle_root / "canary/decision.json"
    canary_hint_path = bundle_root / "canary/hint.json"
    canary_provenance_path = bundle_root / "canary/build/build-provenance.json"

    gate_a = _read_json(gate_a_monitor_path)
    _validate_gate_a_monitor(gate_a)
    _validate_gate_a_logs(gate_a, gate_a_stdout_path, gate_a_stderr_path)
    profile = plans._verified_profile(_read_json(platform_path))
    plans._verify_calibration_artifacts(profile, calibration_paths)

    discovery_provenance = _read_json(discovery_provenance_path)
    verify_build_provenance(
        discovery_provenance, manifest_path=discovery_provenance_path,
        repo_root=repo_root,
    )
    if discovery_provenance.get("build_mode") != "discover":
        raise EvalError("discovery provenance is not a discover build")
    inventory_path = _nested_role_path(
        discovery_provenance, discovery_provenance_path, repo_root, "inventory"
    )
    expected_graph = plans.make_graph(_read_json(inventory_path), profile)
    graph = plans.verified_graph(_read_json(graph_path))
    if graph != expected_graph:
        raise EvalError("graph does not match frozen inventory and platform")

    prompt_paths = {
        view: bundle_root / f"prompts/{view}.txt"
        for view in plans.MODEL_VIEW_KINDS
    }
    for view, path in prompt_paths.items():
        try:
            prompt = path.read_text()
        except OSError as exc:
            raise EvalError(f"cannot read {view} prompt: {exc}") from exc
        if prompt != plans.render_prompt(graph, view):
            raise EvalError(f"{view} prompt does not match the frozen graph")

    controls_manifest = _read_json(controls_path)
    verify_manifest(graph, controls_manifest, controls_path.parent)
    benchmark_path = _nested_role_path(
        discovery_provenance, discovery_provenance_path, repo_root,
        "benchmark_source",
    )
    catalog_path = _nested_role_path(
        discovery_provenance, discovery_provenance_path, repo_root,
        "catalog_source",
    )
    if (controls_manifest.get("source_sha256") != _sha256(benchmark_path)
            or controls_manifest.get("catalog_sha256") != _sha256(catalog_path)):
        raise EvalError("control manifest source hashes do not match discovery")

    capacity = _read_json(capacity_path)
    if capacity != audit_capacity(graph):
        raise EvalError("capacity audit does not match the frozen graph")

    freeze_files: dict[str, Path] = {
        "preparation_script": preparation_script,
        "platform_profile": platform_path,
        "gate_a_monitor": gate_a_monitor_path,
        "gate_a_stdout": gate_a_stdout_path,
        "gate_a_stderr": gate_a_stderr_path,
        "discovery_build_provenance": discovery_provenance_path,
        "graph": graph_path,
        "controls_manifest": controls_path,
        "capacity_audit": capacity_path,
        "canary_decision": canary_decision_path,
        "canary_hint": canary_hint_path,
        "canary_build_provenance": canary_provenance_path,
    }
    for index, path in enumerate(calibration_paths):
        freeze_files[f"calibration_artifact_{index}"] = path
    for view, path in prompt_paths.items():
        freeze_files[f"prompt_{view}"] = path

    provenances = [discovery_provenance]
    arms_summary = []
    for arm in controls_manifest["arms"]:
        name = arm["name"]
        response_path = controls_path.parent / arm["response"]
        hint_path = controls_path.parent / arm["hint"]
        provenance_path = bundle_root / f"binaries/{name}/build-provenance.json"
        provenance = _read_json(provenance_path)
        verify_build_provenance(
            provenance, manifest_path=provenance_path, repo_root=repo_root
        )
        if provenance.get("build_mode") != "lower":
            raise EvalError(f"{name}: build provenance is not lower mode")
        roles = _provenance_roles(provenance)
        if (roles["benchmark_source"].get("sha256")
                != controls_manifest["source_sha256"]
                or roles["catalog_source"].get("sha256")
                != controls_manifest["catalog_sha256"]
                or roles["collective_hint"].get("sha256")
                != arm["hint_sha256"]):
            raise EvalError(f"{name}: build inputs do not match controls")
        hint = _read_json(hint_path)
        host_ir = _nested_role_path(
            provenance, provenance_path, repo_root, "host_ir"
        )
        device_ir = _nested_role_path(
            provenance, provenance_path, repo_root, "device_ir"
        )
        verify_plan_ir(graph, hint, host_ir)
        verify_device_ir(device_ir)
        freeze_files[f"control_response_{name}"] = response_path
        freeze_files[f"control_hint_{name}"] = hint_path
        freeze_files[f"control_build_provenance_{name}"] = provenance_path
        arms_summary.append({
            "name": name,
            "algorithm": arm["algorithm"],
            "hint_sha256": arm["hint_sha256"],
            "build_provenance_manifest_id": provenance["manifest_id"],
            "binary_sha256": roles["binary"]["sha256"],
            "host_ir_sha256": roles["host_ir"]["sha256"],
            "device_ir_sha256": roles["device_ir"]["sha256"],
        })
        provenances.append(provenance)
    verify_ir(controls_manifest, controls_path.parent, bundle_root / "binaries")

    canary_decision_value = _read_json(canary_decision_path)
    canary_hint_value = _read_json(canary_hint_path)
    generated_hint, accepted, errors = plans.decision_to_hint(
        graph, canary_decision_value
    )
    if not accepted:
        raise EvalError(f"mixed-policy canary decision rejected: {errors}")
    _compare_generated_hint(canary_hint_value, generated_hint)
    if canary_hint_value.get("llm_metadata", {}).get("model_invoked") is not False:
        raise EvalError("mixed-policy canary must record model_invoked=false")
    canary_provenance = _read_json(canary_provenance_path)
    verify_build_provenance(
        canary_provenance, manifest_path=canary_provenance_path,
        repo_root=repo_root,
    )
    canary_roles = _provenance_roles(canary_provenance)
    if (canary_provenance.get("build_mode") != "lower"
            or canary_roles["collective_hint"].get("sha256")
            != _sha256(canary_hint_path)
            or canary_roles["benchmark_source"].get("sha256")
            != controls_manifest["source_sha256"]
            or canary_roles["catalog_source"].get("sha256")
            != controls_manifest["catalog_sha256"]):
        raise EvalError("mixed-policy canary build inputs do not match controls")
    verify_plan_ir(
        graph, canary_hint_value,
        _nested_role_path(
            canary_provenance, canary_provenance_path, repo_root, "host_ir"
        ),
    )
    verify_device_ir(_nested_role_path(
        canary_provenance, canary_provenance_path, repo_root, "device_ir"
    ))
    provenances.append(canary_provenance)

    repository_dependencies: dict[str, str] = {}
    for provenance in provenances:
        for record in provenance["dependency_closure"]["files"]:
            locator = record["locator"]
            if locator["scope"] != "repository":
                continue
            path = locator["path"]
            previous = repository_dependencies.setdefault(path, record["sha256"])
            if previous != record["sha256"]:
                raise EvalError(f"inconsistent dependency hash across builds: {path}")
    dependency_set = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(repository_dependencies.items())
    ]
    file_records = [
        _file_record(path, role=role, repo_root=repo_root,
                     build_root=bundle_root)
        for role, path in sorted(freeze_files.items())
    ]
    opportunity = graph["opportunities"][0]
    canary_selection = next(iter(canary_hint_value["selections"].values()))
    payload = {
        "schema_version": OFFLINE_FREEZE_SCHEMA,
        "bundle_version": 3,
        "status": "offline_frozen_pending_gate_b",
        "compiler_only": True,
        "model_invoked": False,
        "performance_evidence": False,
        "application_source_modified": False,
        "gate_a": {
            "job_id": gate_a["job_id"],
            "monitor_sha256": _sha256(gate_a_monitor_path),
            "stdout_sha256": _sha256(gate_a_stdout_path),
            "stderr_sha256": _sha256(gate_a_stderr_path),
            "role": "diagnostic compiler-pipeline safety only",
        },
        "graph": {
            "graph_id": graph["graph_id"],
            "platform_id": graph["compiler_inputs"]["platform_id"],
            "joint_action_space_size": opportunity["joint_action_space_size"],
        },
        "source_dependency_closure": {
            "file_count": len(dependency_set),
            "set_id": bridge._fingerprint(dependency_set),
            "files": dependency_set,
        },
        "prompts": {
            view: {"sha256": _sha256(path), "bytes": path.stat().st_size}
            for view, path in sorted(prompt_paths.items())
        },
        "uniform_controls": {
            "manifest_id": controls_manifest["manifest_id"],
            "arm_count": len(arms_summary),
            "arms": arms_summary,
        },
        "mixed_policy_canary": {
            "candidate_id": canary_selection["candidate_id"],
            "kind": canary_selection["kind"],
            "build_provenance_manifest_id": canary_provenance["manifest_id"],
            "binary_sha256": canary_roles["binary"]["sha256"],
        },
        "capacity_audit": {
            "accepted_action_count": capacity["accepted_action_count"],
            "unique_composite_candidate_id_count": capacity[
                "unique_composite_candidate_id_count"
            ],
            "candidate_id_set_sha256": capacity["candidate_id_set_sha256"],
        },
        "files": file_records,
    }
    result = dict(payload)
    result["manifest_id"] = bridge._fingerprint(payload)
    return result


def verify_offline_freeze(
    value: Any, *, manifest_path: Path, repo_root: Path,
) -> None:
    if (not isinstance(value, dict)
            or value.get("schema_version") != OFFLINE_FREEZE_SCHEMA):
        raise EvalError(f"expected offline freeze schema {OFFLINE_FREEZE_SCHEMA}")
    payload = dict(value)
    manifest_id = payload.pop("manifest_id", None)
    if manifest_id != bridge._fingerprint(payload):
        raise EvalError("offline freeze manifest_id does not match content")
    if (value.get("bundle_version") != 3
            or value.get("compiler_only") is not True
            or value.get("model_invoked") is not False
            or value.get("performance_evidence") is not False
            or value.get("application_source_modified") is not False):
        raise EvalError("offline freeze violates the v3 compiler-only boundary")
    bundle_root = _lexical_absolute(manifest_path).parent
    repo_root = _lexical_absolute(repo_root)
    records = value.get("files")
    if not isinstance(records, list):
        raise EvalError("offline freeze lacks file records")
    by_role = {}
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("role"), str):
            raise EvalError("offline freeze has an invalid file record")
        if record["role"] in by_role:
            raise EvalError(f"duplicate offline-freeze role: {record['role']}")
        path = _record_path(record, repo_root=repo_root, build_root=bundle_root)
        if not path.is_file() or _sha256(path) != record.get("sha256"):
            raise EvalError(f"offline-freeze file mismatch: {path}")
        by_role[record["role"]] = path
    required = {
        "preparation_script", "platform_profile", "gate_a_monitor",
        "gate_a_stdout", "gate_a_stderr",
        "discovery_build_provenance", "graph", "controls_manifest",
        "capacity_audit", "canary_decision", "canary_hint",
        "canary_build_provenance",
        *{f"prompt_{view}" for view in plans.MODEL_VIEW_KINDS},
    }
    if not required.issubset(by_role):
        raise EvalError("offline freeze is missing required files")
    calibration_paths = [
        path for role, path in sorted(by_role.items())
        if role.startswith("calibration_artifact_")
    ]
    regenerated = generate_offline_freeze(
        bundle_root=bundle_root,
        repo_root=repo_root,
        platform_path=by_role["platform_profile"],
        calibration_paths=calibration_paths,
        gate_a_monitor_path=by_role["gate_a_monitor"],
        gate_a_stdout_path=by_role["gate_a_stdout"],
        gate_a_stderr_path=by_role["gate_a_stderr"],
        preparation_script=by_role["preparation_script"],
    )
    if regenerated != value:
        raise EvalError("offline freeze does not match regenerated v3 bundle")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot read JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _check_sha(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef"
                               for character in value):
        raise EvalError(f"{name} must be 64 lowercase hex characters")


def _manifest_arms(value: Any) -> list[dict[str, Any]]:
    if (not isinstance(value, dict)
            or value.get("schema_version") != MANIFEST_SCHEMA):
        raise EvalError(f"expected manifest schema {MANIFEST_SCHEMA}")
    payload = dict(value)
    manifest_id = payload.pop("manifest_id", None)
    if manifest_id != bridge._fingerprint(payload):
        raise EvalError("manifest_id does not match manifest content")
    if value.get("model_invoked") is not False:
        raise EvalError("compiler controls must record model_invoked=false")
    arms = value.get("arms")
    if (not isinstance(arms, list) or not arms
            or any(not isinstance(arm, dict) for arm in arms)):
        raise EvalError("manifest has no valid compiler controls")
    return arms


def _algorithm_options(opportunity: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    slots = opportunity["decision_slots"]
    result: dict[str, list[dict[str, Any]]] = {}
    for slot in slots:
        algorithms = {item["algorithm"]: item for item in slot["options"]}
        if len(algorithms) != len(slot["options"]):
            raise EvalError("algorithm names must be unique within a decision slot")
        if not result:
            result = {algorithm: [option]
                      for algorithm, option in algorithms.items()}
        elif set(algorithms) != set(result):
            raise EvalError("every message bin must expose the same catalog")
        else:
            for algorithm, option in algorithms.items():
                result[algorithm].append(option)
    return result


def generate_controls(
    graph_value: Any, outdir: Path, source_sha256: str,
    catalog_sha256: str,
) -> dict[str, Any]:
    graph = plans.verified_graph(graph_value)
    if len(graph["opportunities"]) != 1:
        raise EvalError("capacity v1 requires one semantic collective opportunity")
    _check_sha(source_sha256, name="source_sha256")
    _check_sha(catalog_sha256, name="catalog_sha256")
    opportunity = graph["opportunities"][0]
    choices = _algorithm_options(opportunity)
    outdir.mkdir(parents=True, exist_ok=True)
    arms = []
    for algorithm in sorted(choices):
        options = choices[algorithm]
        response = {
            "schema_version": plans.DECISION_SCHEMA,
            "graph_id": graph["graph_id"],
            "selections": {
                opportunity["opportunity_id"]: {
                    "slot_candidate_ids": {
                        slot["slot_id"]: option["option_id"]
                        for slot, option in zip(
                            opportunity["decision_slots"], options, strict=True
                        )
                    },
                    "confidence": 1.0,
                    "rationale": "compiler-generated uniform catalog control",
                }
            },
        }
        hint, accepted, errors = plans.decision_to_hint(graph, response)
        if not accepted:
            raise EvalError(f"generated {algorithm} control rejected: {errors}")
        selection = hint["selections"][opportunity["opportunity_id"]]
        if selection.get("kind") != "uniform":
            raise EvalError(f"{algorithm}: uniform control did not collapse")
        hint["llm_metadata"].update({
            "producer": "compiler-generated uniform collective control",
            "model_invoked": False,
        })
        safe_name = re.sub(r"[^a-z0-9_]+", "_", algorithm.lower()).strip("_")
        response_path = outdir / f"{safe_name}-response.json"
        hint_path = outdir / f"{safe_name}-hint.json"
        bridge._write_json_atomic(response_path, response)
        bridge._write_json_atomic(hint_path, hint)
        arms.append({
            "name": safe_name,
            "algorithm": algorithm,
            "candidate_id": selection["candidate_id"],
            "target_id": selection["target_id"],
            "response": response_path.name,
            "response_sha256": _sha256(response_path),
            "hint": hint_path.name,
            "hint_sha256": _sha256(hint_path),
        })
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "graph_id": graph["graph_id"],
        "source_sha256": source_sha256,
        "catalog_sha256": catalog_sha256,
        "model_invoked": False,
        "model_output_scope": "compiler-generated option IDs only",
        "opportunity_id": opportunity["opportunity_id"],
        "decision_slot_count": len(opportunity["decision_slots"]),
        "joint_action_space_size": opportunity["joint_action_space_size"],
        "arms": arms,
    }
    manifest = dict(payload)
    manifest["manifest_id"] = bridge._fingerprint(payload)
    bridge._write_json_atomic(outdir / "manifest.json", manifest)
    return manifest


def verify_manifest(graph_value: Any, manifest_value: Any, root: Path) -> None:
    graph = plans.verified_graph(graph_value)
    if not isinstance(manifest_value, dict) or manifest_value.get(
            "schema_version") != MANIFEST_SCHEMA:
        raise EvalError(f"expected manifest schema {MANIFEST_SCHEMA}")
    payload = dict(manifest_value)
    manifest_id = payload.pop("manifest_id", None)
    if manifest_id != bridge._fingerprint(payload):
        raise EvalError("manifest_id does not match manifest content")
    if manifest_value.get("graph_id") != graph["graph_id"]:
        raise EvalError("manifest graph_id does not match")
    if manifest_value.get("model_invoked") is not False:
        raise EvalError("uniform controls must record model_invoked=false")
    opportunity = graph["opportunities"][0]
    expected_algorithms = set(_algorithm_options(opportunity))
    arms = manifest_value.get("arms")
    if not isinstance(arms, list) or len(arms) != len(expected_algorithms):
        raise EvalError("manifest does not cover the full uniform catalog")
    seen = set()
    for arm in arms:
        if not isinstance(arm, dict) or arm.get("algorithm") in seen:
            raise EvalError("invalid or duplicate uniform arm")
        algorithm = arm["algorithm"]
        seen.add(algorithm)
        response_path = root / arm["response"]
        hint_path = root / arm["hint"]
        if _sha256(response_path) != arm["response_sha256"]:
            raise EvalError(f"{response_path}: response hash mismatch")
        if _sha256(hint_path) != arm["hint_sha256"]:
            raise EvalError(f"{hint_path}: hint hash mismatch")
        hint, accepted, errors = plans.decision_to_hint(
            graph, _read_json(response_path)
        )
        if not accepted:
            raise EvalError(f"{response_path}: rejected: {errors}")
        stored = _read_json(hint_path)
        stored["llm_metadata"].pop("producer", None)
        stored["llm_metadata"].pop("model_invoked", None)
        if stored != hint:
            raise EvalError(f"{hint_path}: hint does not match response")
        selection = hint["selections"][opportunity["opportunity_id"]]
        if (selection.get("kind") != "uniform"
                or selection.get("candidate_id") != arm["candidate_id"]
                or selection.get("target_id") != arm["target_id"]):
            raise EvalError(f"{arm['name']}: materializer summary mismatch")
    if seen != expected_algorithms:
        raise EvalError("uniform arms do not match graph algorithms")


def verify_ir(manifest_value: Any, root: Path, ir_dir: Path) -> None:
    arms = manifest_value.get("arms", [])
    for arm in arms:
        path = ir_dir / arm["name"] / "materialized.ll"
        try:
            text = path.read_text()
        except OSError as exc:
            raise EvalError(f"cannot read materialized IR {path}: {exc}") from exc
        candidate_marker = f'!{{!"{arm["candidate_id"]}"}}'
        target_marker = f'!{{!"{arm["target_id"]}"}}'
        if candidate_marker not in text or target_marker not in text:
            raise EvalError(f"{path}: missing compiler plan metadata")
        if text.count("!gicc.collective.candidate_id") != 1:
            raise EvalError(f"{path}: expected one materialized collective call")


def verify_plan_ir(graph_value: Any, hint_value: Any, path: Path) -> None:
    graph = plans.verified_graph(graph_value)
    if (not isinstance(hint_value, dict)
            or hint_value.get("schema_version") != plans.HINT_SCHEMA
            or hint_value.get("llm_metadata", {}).get("compiler_only_output")
            is not True):
        raise EvalError("expected a compiler-only collective hint")
    selections = hint_value.get("selections")
    if (not isinstance(selections, dict) or len(selections) != 1
            or len(graph["opportunities"]) != 1):
        raise EvalError("plan-IR audit v1 requires one collective selection")
    opportunity = graph["opportunities"][0]
    opportunity_id = opportunity["opportunity_id"]
    if (set(selections) != {opportunity_id}
            or hint_value["llm_metadata"].get("graph_id") != graph["graph_id"]):
        raise EvalError("collective hint does not match compiler graph")
    selection = selections[opportunity_id]
    try:
        text = path.read_text()
    except OSError as exc:
        raise EvalError(f"cannot read materialized IR {path}: {exc}") from exc
    candidate_id = selection.get("candidate_id")
    if not isinstance(candidate_id, str):
        raise EvalError("collective hint lacks candidate_id")
    candidate_marker = f'!{{!"{candidate_id}"}}'
    if candidate_marker not in text:
        raise EvalError(f"{path}: missing compiler plan candidate metadata")
    if selection.get("kind") == "uniform":
        target_ids = [selection.get("target_id")]
    elif selection.get("kind") == "size_policy":
        rules = selection.get("rules")
        if not isinstance(rules, list) or not rules:
            raise EvalError("size policy has no rules")
        target_ids = [rule.get("target_id") for rule in rules]
    else:
        raise EvalError("unknown collective materializer kind")
    if any(not isinstance(target_id, str) for target_id in target_ids):
        raise EvalError("collective plan lacks target IDs")
    for target_id in target_ids:
        if f'!{{!"{target_id}"}}' not in text:
            raise EvalError(f"{path}: missing target metadata {target_id}")
    expected_calls = len(target_ids)
    if text.count("!gicc.collective.candidate_id") != expected_calls:
        raise EvalError(
            f"{path}: expected {expected_calls} materialized collective calls"
        )
    if text.count("!gicc.collective.target_id") != expected_calls:
        raise EvalError(f"{path}: target metadata count does not match plan")
    functions = [
        match.group(0)
        for match in re.finditer(
            r"^define\b.*?^}\s*$", text, flags=re.MULTILINE | re.DOTALL
        )
        if "!gicc.collective.candidate_id" in match.group(0)
    ]
    if len(functions) != 1:
        raise EvalError(f"{path}: expected one compiler-policy function body")
    if selection.get("kind") == "size_policy":
        expected_bounds = [
            slot["message_bytes"]["max"]
            for slot in opportunity["decision_slots"]
        ]
        if [rule.get("max_bytes") for rule in selection["rules"]] != expected_bounds:
            raise EvalError("size-policy bounds do not match compiler graph")
        element_bytes = opportunity["compiler_facts"]["call"]["element_bytes"]
        expected_cutoffs = [
            maximum // element_bytes + 1
            for maximum in expected_bounds if maximum is not None
        ]
        actual_cutoffs = [
            int(value) for value in re.findall(
                r"icmp ult i32 [^,\n]+, ([0-9]+)", functions[0]
            )
        ]
        if actual_cutoffs != expected_cutoffs:
            raise EvalError(
                f"{path}: policy cutoffs {actual_cutoffs}, expected {expected_cutoffs}"
            )


def verify_device_ir(path: Path) -> int:
    """Prove that collective-only compilation retained proxy-ring commands."""
    try:
        text = path.read_text()
    except OSError as exc:
        raise EvalError(f"cannot read materialized device IR {path}: {exc}") from exc
    match = re.search(
        rf"^define\b[^\n]*@{re.escape(HDIR_DEVICE_KERNEL)}\(.*?^\}}\s*$",
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise EvalError(f"{path}: missing hierarchical-direct device kernel")
    # Each of the two puts and two quiets reserves one ProxyRing slot through
    # a cmpxchg loop after inlining. The accidentally full lowering pipeline
    # erased all four operations, leaving zero cmpxchg instructions and a
    # receiver that waited forever for a flag no peer could send.
    reservations = len(re.findall(r"\bcmpxchg\b", match.group(0)))
    if reservations < 4:
        raise EvalError(
            f"{path}: hierarchical-direct kernel retains only "
            f"{reservations}/4 proxy-ring reservations"
        )
    return reservations


def _int_field(fields: dict[str, str], name: str, path: Path) -> int:
    try:
        return int(fields[name])
    except (KeyError, ValueError) as exc:
        raise EvalError(f"{path}: missing or invalid integer field {name}") from exc


def _float_field(fields: dict[str, str], name: str, path: Path) -> float:
    try:
        value = float(fields[name])
    except (KeyError, ValueError) as exc:
        raise EvalError(f"{path}: missing or invalid numeric field {name}") from exc
    if value <= 0 or not math.isfinite(value):
        raise EvalError(f"{path}: {name} must be positive and finite")
    return value


def _parse_log_record(path: Path) -> dict[str, Any]:
    config = None
    done = None
    rows: dict[int, float] = {}
    result_fields: dict[int, dict[str, str]] = {}
    for line in path.read_text().splitlines():
        if line.startswith("COLLECTIVE_CONFIG "):
            if config is not None:
                raise EvalError(f"{path}: duplicate COLLECTIVE_CONFIG")
            config = dict(RESULT_RE.findall(line))
        elif line.startswith("RESULT "):
            fields = dict(RESULT_RE.findall(line))
            size = _int_field(fields, "bytes", path)
            errors = _int_field(fields, "errors", path)
            if errors != 0:
                raise EvalError(f"{path}: correctness errors at {size} B")
            if size in rows:
                raise EvalError(f"{path}: duplicate result at {size} B")
            rows[size] = _float_field(fields, "median_us", path)
            result_fields[size] = fields
        elif line.startswith("COLLECTIVE_DONE "):
            if done is not None:
                raise EvalError(f"{path}: duplicate COLLECTIVE_DONE")
            done = dict(RESULT_RE.findall(line))
    if config is None or done is None or not rows:
        raise EvalError(f"{path}: incomplete or failing collective log")
    label = config.get("plan")
    if not label:
        raise EvalError(f"{path}: missing plan label")
    ranks = _int_field(config, "ranks", path)
    ppn = _int_field(config, "ppn", path)
    runs = _int_field(config, "runs", path)
    warmup = _int_field(config, "warmup", path)
    if ranks <= 0 or ppn <= 0 or ranks % ppn != 0 or runs <= 0 or warmup < 0:
        raise EvalError(f"{path}: invalid runtime configuration")
    for size, fields in result_fields.items():
        if fields.get("plan") != label:
            raise EvalError(f"{path}: RESULT plan does not match config")
        if (_int_field(fields, "ranks", path) != ranks
                or _int_field(fields, "ppn", path) != ppn
                or _int_field(fields, "nodes", path) != ranks // ppn):
            raise EvalError(f"{path}: RESULT topology does not match config")
    if done.get("plan") != label or _int_field(done, "total_errors", path) != 0:
        raise EvalError(f"{path}: incomplete or failing collective log")
    return {
        "label": label,
        "nodes": ranks // ppn,
        "ranks": ranks,
        "ppn": ppn,
        "runs": runs,
        "warmup": warmup,
        "rows": rows,
    }


def parse_log(path: Path) -> tuple[str, dict[int, float]]:
    record = _parse_log_record(path)
    return record["label"], record["rows"]


def qualify_logs(
    manifest_value: Any, logs: list[Path], gate: str,
) -> dict[str, Any]:
    if gate not in GATE_SPECS:
        raise EvalError(f"unknown qualification gate {gate}")
    arms = _manifest_arms(manifest_value)
    by_name = {arm.get("name"): arm for arm in arms}
    if len(by_name) != len(arms) or any(not isinstance(name, str) for name in by_name):
        raise EvalError("manifest has invalid or duplicate arm names")
    baseline_names = [
        name for name, arm in by_name.items()
        if arm.get("algorithm") == "baseline_auto"
    ]
    if len(baseline_names) != 1:
        raise EvalError("manifest must have exactly one baseline_auto arm")
    spec = GATE_SPECS[gate]
    expected_labels = set(by_name) if spec["catalog_complete"] else set(baseline_names)
    if len(logs) != len(expected_labels):
        raise EvalError(
            f"Gate {gate.upper()} requires {len(expected_labels)} log(s), got {len(logs)}"
        )
    observed = {}
    for path in logs:
        record = _parse_log_record(path)
        label = record["label"]
        if label not in expected_labels or label in observed:
            raise EvalError(f"{path}: unexpected or duplicate plan label {label}")
        for field in ("nodes", "ranks", "ppn", "runs", "warmup"):
            if record[field] != spec[field]:
                raise EvalError(
                    f"{path}: {field}={record[field]}, expected {spec[field]}"
                )
        if sorted(record["rows"]) != spec["sizes"]:
            raise EvalError(
                f"{path}: sizes={sorted(record['rows'])}, expected {spec['sizes']}"
            )
        observed[label] = {
            "log": str(path),
            "log_sha256": _sha256(path),
            "result_rows": len(record["rows"]),
        }
    if set(observed) != expected_labels:
        raise EvalError(f"Gate {gate.upper()}: catalog coverage is incomplete")
    return {
        "schema_version": "gicc-collective-qualification-v1",
        "gate": gate.upper(),
        "passed": True,
        "manifest_id": manifest_value.get("manifest_id"),
        "graph_id": manifest_value.get("graph_id"),
        "source_sha256": manifest_value.get("source_sha256"),
        "catalog_sha256": manifest_value.get("catalog_sha256"),
        "runtime": {
            key: spec[key]
            for key in ("nodes", "ranks", "ppn", "runs", "warmup")
        },
        "sizes": spec["sizes"],
        "logs": observed,
    }


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value)
                         for value in values):
        raise EvalError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def analyze_logs(
    graph_value: Any, manifest_value: Any, logs: list[Path]
) -> dict[str, Any]:
    graph = plans.verified_graph(graph_value)
    arms = _manifest_arms(manifest_value)
    if manifest_value.get("graph_id") != graph["graph_id"]:
        raise EvalError("control manifest does not match the compiler graph")
    try:
        algorithms = {arm["name"]: arm["algorithm"] for arm in arms}
    except KeyError as exc:
        raise EvalError("control manifest arm lacks name or algorithm") from exc
    if len(algorithms) != len(arms):
        raise EvalError("control manifest has duplicate arm names")
    samples: dict[str, dict[int, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for path in logs:
        label, rows = parse_log(path)
        if label not in algorithms:
            raise EvalError(f"{path}: unknown plan label {label}")
        for size, latency in rows.items():
            samples[algorithms[label]][size].append(latency)
    expected_algorithms = set(algorithms.values())
    if set(samples) != expected_algorithms:
        raise EvalError(
            f"logs cover {sorted(samples)}, expected {sorted(expected_algorithms)}"
        )
    sizes = sorted(next(iter(samples.values())))
    for algorithm, by_size in samples.items():
        if sorted(by_size) != sizes:
            raise EvalError(f"{algorithm}: size coverage differs")
    replicate_counts = {
        len(samples[algorithm][size])
        for algorithm in samples for size in sizes
    }
    if len(replicate_counts) != 1:
        raise EvalError("compiler controls have unequal replicate coverage")
    medians = {
        algorithm: {
            size: statistics.median(by_size[size]) for size in sizes
        }
        for algorithm, by_size in samples.items()
    }
    baseline = "baseline_auto"
    if baseline not in medians:
        raise EvalError("baseline_auto arm is missing")
    per_size = {}
    for size in sizes:
        winner = min(
            medians, key=lambda algorithm: (medians[algorithm][size], algorithm)
        )
        per_size[str(size)] = {
            "winner": winner,
            "winner_us": medians[winner][size],
            "baseline_us": medians[baseline][size],
            "baseline_over_oracle": medians[baseline][size] / medians[winner][size],
            "algorithm_median_us": {
                algorithm: medians[algorithm][size]
                for algorithm in sorted(medians)
            },
        }
    aggregate = {
        algorithm: _geomean([medians[algorithm][size] for size in sizes])
        for algorithm in medians
    }
    best_uniform = min(aggregate, key=lambda algorithm: (aggregate[algorithm], algorithm))
    oracle_geomean = _geomean(
        [per_size[str(size)]["winner_us"] for size in sizes]
    )
    opportunity = graph["opportunities"][0]
    bin_choices = []
    for slot in opportunity["decision_slots"]:
        lower = slot["message_bytes"]["min"]
        upper = slot["message_bytes"]["max"]
        in_bin = [size for size in sizes
                  if (lower is None or size >= lower)
                  and (upper is None or size <= upper)]
        if not in_bin:
            raise EvalError(f"no measured size in compiler slot {slot['slot_id']}")
        score = {
            algorithm: _geomean([medians[algorithm][size] for size in in_bin])
            for algorithm in medians
        }
        winner = min(score, key=lambda algorithm: (score[algorithm], algorithm))
        option = next(item for item in slot["options"]
                      if item["algorithm"] == winner)
        bin_choices.append({
            "slot_id": slot["slot_id"],
            "algorithm": winner,
            "option_id": option["option_id"],
            "sizes": in_bin,
            "geomean_us": score[winner],
        })
    bin_by_slot = {item["slot_id"]: item["algorithm"] for item in bin_choices}
    compiler_bin_latencies = []
    for size in sizes:
        slot = next(
            item for item in opportunity["decision_slots"]
            if (item["message_bytes"]["min"] is None
                or size >= item["message_bytes"]["min"])
            and (item["message_bytes"]["max"] is None
                 or size <= item["message_bytes"]["max"])
        )
        compiler_bin_latencies.append(medians[bin_by_slot[slot["slot_id"]]][size])
    compiler_bin_geomean = _geomean(compiler_bin_latencies)
    pointwise_ratio = aggregate[baseline] / oracle_geomean
    maximum_size_headroom = max(
        item["baseline_over_oracle"] for item in per_size.values()
    )
    distinct_winners = sorted({item["winner"] for item in per_size.values()})
    bin_differs_from_baseline = any(
        item["algorithm"] != baseline for item in bin_choices
    )
    gate_c_criteria = {
        "at_least_two_distinct_size_winners": {
            "observed": len(distinct_winners),
            "threshold": 2,
            "passed": len(distinct_winners) >= 2,
        },
        "baseline_over_pointwise_oracle_geomean": {
            "observed": pointwise_ratio,
            "threshold": 1.05,
            "passed": pointwise_ratio >= 1.05,
        },
        "maximum_single_size_headroom": {
            "observed": maximum_size_headroom,
            "threshold": 1.10,
            "passed": maximum_size_headroom >= 1.10,
        },
        "compiler_bin_policy_differs_from_baseline": {
            "observed": bin_differs_from_baseline,
            "threshold": True,
            "passed": bin_differs_from_baseline,
        },
    }
    return {
        "schema_version": "gicc-collective-control-analysis-v1",
        "graph_id": graph["graph_id"],
        "replicates_per_algorithm_size": {
            algorithm: {str(size): len(samples[algorithm][size]) for size in sizes}
            for algorithm in sorted(samples)
        },
        "per_size": per_size,
        "aggregate": {
            "algorithm_geomean_us": aggregate,
            "best_uniform_algorithm": best_uniform,
            "best_uniform_geomean_us": aggregate[best_uniform],
            "baseline_geomean_us": aggregate[baseline],
            "per_size_oracle_geomean_us": oracle_geomean,
            "baseline_over_per_size_oracle": pointwise_ratio,
            "best_uniform_over_per_size_oracle": aggregate[best_uniform] / oracle_geomean,
            "compiler_bin_oracle_geomean_us": compiler_bin_geomean,
            "baseline_over_compiler_bin_oracle": (
                aggregate[baseline] / compiler_bin_geomean
            ),
            "best_uniform_over_compiler_bin_oracle": (
                aggregate[best_uniform] / compiler_bin_geomean
            ),
            "maximum_single_size_headroom": maximum_size_headroom,
            "distinct_per_size_winners": distinct_winners,
        },
        "compiler_bin_oracle": bin_choices,
        "gate_c": {
            "passed": all(item["passed"] for item in gate_c_criteria.values()),
            "criteria": gate_c_criteria,
        },
        "scope": (
            "Compiler-generated controls only; no LLM result. The oracle is "
            "restricted to the exact source-free catalog and compiler-owned "
            "message thresholds."
        ),
    }


def oracle_decision(graph_value: Any, analysis_value: Any) -> tuple[dict, dict]:
    graph = plans.verified_graph(graph_value)
    opportunity = graph["opportunities"][0]
    choices = analysis_value.get("compiler_bin_oracle")
    if not isinstance(choices, list):
        raise EvalError("analysis has no compiler-bin oracle")
    by_slot = {item["slot_id"]: item["option_id"] for item in choices}
    expected_slots = {item["slot_id"] for item in opportunity["decision_slots"]}
    if set(by_slot) != expected_slots:
        raise EvalError("oracle slots do not match graph")
    decision = {
        "schema_version": plans.DECISION_SCHEMA,
        "graph_id": graph["graph_id"],
        "selections": {
            opportunity["opportunity_id"]: {
                "slot_candidate_ids": by_slot,
                "confidence": 1.0,
                "rationale": "compiler-control measurement oracle; no model invoked",
            }
        },
    }
    hint, accepted, errors = plans.decision_to_hint(graph, decision)
    if not accepted:
        raise EvalError(f"compiler-bin oracle rejected: {errors}")
    hint["llm_metadata"].update({
        "producer": "measured compiler-bin oracle",
        "model_invoked": False,
    })
    return decision, hint


def canary_decision(graph_value: Any) -> tuple[dict, dict]:
    graph = plans.verified_graph(graph_value)
    if len(graph["opportunities"]) != 1:
        raise EvalError("mixed-policy canary v1 requires one opportunity")
    opportunity = graph["opportunities"][0]
    selected = {}
    for index, slot in enumerate(opportunity["decision_slots"]):
        candidates = sorted(
            (option for option in slot["options"] if option["role"] != "anchor"),
            key=lambda option: (option["algorithm"], option["option_id"]),
        )
        if not candidates:
            raise EvalError("mixed-policy canary has no non-anchor candidate")
        selected[slot["slot_id"]] = candidates[index % len(candidates)][
            "option_id"
        ]
    decision = {
        "schema_version": plans.DECISION_SCHEMA,
        "graph_id": graph["graph_id"],
        "selections": {
            opportunity["opportunity_id"]: {
                "slot_candidate_ids": selected,
                "confidence": 1.0,
                "rationale": (
                    "compiler-generated mixed-policy materialization canary"
                ),
            }
        },
    }
    hint, accepted, errors = plans.decision_to_hint(graph, decision)
    if not accepted:
        raise EvalError(f"mixed-policy canary rejected: {errors}")
    selection = hint["selections"][opportunity["opportunity_id"]]
    if selection.get("kind") != "size_policy":
        raise EvalError("mixed-policy canary unexpectedly collapsed to uniform")
    hint["llm_metadata"].update({
        "producer": "compiler-generated mixed-policy canary",
        "model_invoked": False,
    })
    return decision, hint


def audit_capacity(graph_value: Any) -> dict[str, Any]:
    graph = plans.verified_graph(graph_value)
    if len(graph["opportunities"]) != 1:
        raise EvalError("capacity audit v1 requires one collective opportunity")
    opportunity = graph["opportunities"][0]
    slots = opportunity["decision_slots"]
    candidate_ids = set()
    kind_counts: Counter[str] = Counter()
    enumerated = 0
    for combination in itertools.product(
            *(slot["options"] for slot in slots)):
        response = {
            "schema_version": plans.DECISION_SCHEMA,
            "graph_id": graph["graph_id"],
            "selections": {
                opportunity["opportunity_id"]: {
                    "slot_candidate_ids": {
                        slot["slot_id"]: option["option_id"]
                        for slot, option in zip(slots, combination, strict=True)
                    },
                    "confidence": 1.0,
                    "rationale": "compiler action-space audit",
                }
            },
        }
        hint, accepted, errors = plans.decision_to_hint(graph, response)
        if not accepted:
            raise EvalError(
                f"declared compiler action rejected during audit: {errors}"
            )
        selection = hint["selections"][opportunity["opportunity_id"]]
        candidate_id = selection["candidate_id"]
        if candidate_id in candidate_ids:
            raise EvalError("composite compiler candidate-ID collision")
        candidate_ids.add(candidate_id)
        kind_counts[selection["kind"]] += 1
        enumerated += 1
    declared = opportunity["joint_action_space_size"]
    if enumerated != declared or len(candidate_ids) != declared:
        raise EvalError("enumerated compiler action space does not match graph")
    digest = hashlib.sha256(
        ("\n".join(sorted(candidate_ids)) + "\n").encode()
    ).hexdigest()
    return {
        "schema_version": "gicc-collective-capacity-audit-v1",
        "graph_id": graph["graph_id"],
        "declared_joint_action_space_size": declared,
        "enumerated_action_count": enumerated,
        "accepted_action_count": enumerated,
        "unique_composite_candidate_id_count": len(candidate_ids),
        "materializer_kind_counts": dict(sorted(kind_counts.items())),
        "candidate_id_set_sha256": digest,
        "model_invoked": False,
        "output_scope": "compiler-generated option IDs only",
    }


def score_decisions(
    graph_value: Any,
    analysis_value: Any,
    prompt: Path,
    prompt_view: str,
    responses: list[Path],
) -> dict[str, Any]:
    graph = plans.verified_graph(graph_value)
    if len(graph["opportunities"]) != 1:
        raise EvalError("decision scoring v1 requires one collective opportunity")
    if prompt_view not in plans.MODEL_VIEW_KINDS:
        raise EvalError(f"unknown prompt view {prompt_view}")
    try:
        prompt_text = prompt.read_text()
    except OSError as exc:
        raise EvalError(f"cannot read prompt {prompt}: {exc}") from exc
    expected_prompt = plans.render_prompt(graph, prompt_view)
    if prompt_text != expected_prompt:
        raise EvalError("prompt does not match the graph and declared view")
    if (not isinstance(analysis_value, dict)
            or analysis_value.get("schema_version")
            != "gicc-collective-control-analysis-v1"
            or analysis_value.get("graph_id") != graph["graph_id"]):
        raise EvalError("control analysis does not match the compiler graph")
    if not responses:
        raise EvalError("decision scoring requires at least one response")

    opportunity = graph["opportunities"][0]
    opportunity_id = opportunity["opportunity_id"]
    slots = {slot["slot_id"]: slot
             for slot in opportunity["decision_slots"]}
    oracle_rows = analysis_value.get("compiler_bin_oracle")
    if not isinstance(oracle_rows, list):
        raise EvalError("control analysis has no compiler-bin oracle")
    try:
        oracle_options = {
            row["slot_id"]: row["option_id"] for row in oracle_rows
        }
    except (KeyError, TypeError) as exc:
        raise EvalError("invalid compiler-bin oracle") from exc
    if set(oracle_options) != set(slots):
        raise EvalError("compiler-bin oracle slots do not match graph")
    per_size = analysis_value.get("per_size")
    aggregate = analysis_value.get("aggregate")
    if not isinstance(per_size, dict) or not isinstance(aggregate, dict):
        raise EvalError("control analysis lacks timing aggregates")
    try:
        baseline_geomean = float(aggregate["baseline_geomean_us"])
        bin_oracle_geomean = float(
            aggregate["compiler_bin_oracle_geomean_us"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvalError("control analysis lacks compiler-bin timing") from exc
    _geomean([baseline_geomean, bin_oracle_geomean])

    scored = []
    policy_counts: Counter[str] = Counter()
    for response_path in responses:
        parse_error = None
        try:
            response = json.loads(response_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            response = None
            parse_error = f"cannot parse response: {exc}"
        hint, accepted, errors = plans.decision_to_hint(graph, response)
        if parse_error is not None:
            errors = [parse_error, *errors]
        selected = hint["llm_metadata"]["selected_option_ids"][opportunity_id]
        selected_algorithms = {}
        for slot_id, option_id in selected.items():
            option = next(
                (item for item in slots[slot_id]["options"]
                 if item["option_id"] == option_id), None
            )
            if option is None:
                raise EvalError("validated hint contains an unknown option")
            selected_algorithms[slot_id] = option["algorithm"]
        selected_latencies = []
        for size_text, row in sorted(
                per_size.items(), key=lambda item: int(item[0])):
            size = int(size_text)
            slot = next(
                (item for item in opportunity["decision_slots"]
                 if (item["message_bytes"]["min"] is None
                     or size >= item["message_bytes"]["min"])
                 and (item["message_bytes"]["max"] is None
                      or size <= item["message_bytes"]["max"])),
                None,
            )
            if slot is None:
                raise EvalError(f"measured size {size} has no compiler slot")
            algorithm = selected_algorithms[slot["slot_id"]]
            try:
                selected_latencies.append(
                    float(row["algorithm_median_us"][algorithm])
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise EvalError(
                    f"control analysis lacks {algorithm} timing at {size} B"
                ) from exc
        screen_geomean = _geomean(selected_latencies)
        policy_key = bridge._fingerprint({
            "opportunity_id": opportunity_id,
            "slot_option_ids": selected,
        })
        policy_counts[policy_key] += 1
        exact_choices = sum(
            selected[slot_id] == oracle_options[slot_id] for slot_id in slots
        )
        materializer = hint["selections"][opportunity_id]
        scored.append({
            "response": str(response_path),
            "response_sha256": _sha256(response_path),
            "accepted": accepted,
            "errors": errors,
            "policy_key": policy_key,
            "candidate_id": materializer["candidate_id"],
            "materializer_kind": materializer["kind"],
            "selected_option_ids": selected,
            "selected_algorithms": selected_algorithms,
            "exact_compiler_bin_oracle_policy": exact_choices == len(slots),
            "bin_choice_accuracy": exact_choices / len(slots),
            "control_screen_geomean_us": screen_geomean,
            "baseline_speedup_screen": baseline_geomean / screen_geomean,
            "distance_to_compiler_bin_oracle": (
                screen_geomean / bin_oracle_geomean
            ),
        })

    modal_policy, modal_count = min(
        policy_counts.items(), key=lambda item: (-item[1], item[0])
    )
    return {
        "schema_version": "gicc-collective-decision-score-v1",
        "graph_id": graph["graph_id"],
        "prompt_view": prompt_view,
        "prompt_sha256": _sha256(prompt),
        "response_count": len(scored),
        "aggregate": {
            "accepted_count": sum(item["accepted"] for item in scored),
            "invalid_output_rate": (
                sum(not item["accepted"] for item in scored) / len(scored)
            ),
            "exact_oracle_policy_rate": (
                sum(item["exact_compiler_bin_oracle_policy"] for item in scored)
                / len(scored)
            ),
            "mean_bin_choice_accuracy": statistics.mean(
                item["bin_choice_accuracy"] for item in scored
            ),
            "median_baseline_speedup_screen": statistics.median(
                item["baseline_speedup_screen"] for item in scored
            ),
            "median_distance_to_compiler_bin_oracle": statistics.median(
                item["distance_to_compiler_bin_oracle"] for item in scored
            ),
            "unique_materialized_policy_count": len(policy_counts),
            "modal_policy_key": modal_policy,
            "modal_policy_rate": modal_count / len(scored),
        },
        "responses": scored,
        "scope": (
            "Counterfactual screen from frozen uniform compiler controls. "
            "It is not a runtime measurement of the materialized size policy; "
            "paper performance requires pdebug confirmation."
        ),
        "runtime_confirmation_required": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    controls = sub.add_parser("controls")
    controls.add_argument("--graph", type=Path, required=True)
    controls.add_argument("--out", type=Path, required=True)
    controls.add_argument("--source-sha256", required=True)
    controls.add_argument("--catalog-sha256", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--graph", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    verify_ir_parser = sub.add_parser("verify-ir")
    verify_ir_parser.add_argument("--manifest", type=Path, required=True)
    verify_ir_parser.add_argument("--ir", type=Path, required=True)
    verify_plan = sub.add_parser("verify-plan-ir")
    verify_plan.add_argument("--graph", type=Path, required=True)
    verify_plan.add_argument("--hint", type=Path, required=True)
    verify_plan.add_argument("--ir", type=Path, required=True)
    verify_device = sub.add_parser("verify-device-ir")
    verify_device.add_argument("--ir", type=Path, required=True)
    qualify = sub.add_parser("qualify")
    qualify.add_argument("--manifest", type=Path, required=True)
    qualify.add_argument("--gate", choices=sorted(GATE_SPECS), required=True)
    qualify.add_argument("--out", type=Path, required=True)
    qualify.add_argument("logs", type=Path, nargs="+")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--graph", type=Path, required=True)
    analyze.add_argument("--manifest", type=Path, required=True)
    analyze.add_argument("--out", type=Path, required=True)
    analyze.add_argument("logs", type=Path, nargs="+")
    oracle = sub.add_parser("oracle")
    oracle.add_argument("--graph", type=Path, required=True)
    oracle.add_argument("--analysis", type=Path, required=True)
    oracle.add_argument("--decision", type=Path, required=True)
    oracle.add_argument("--hint", type=Path, required=True)
    canary = sub.add_parser("canary")
    canary.add_argument("--graph", type=Path, required=True)
    canary.add_argument("--decision", type=Path, required=True)
    canary.add_argument("--hint", type=Path, required=True)
    capacity = sub.add_parser("capacity-audit")
    capacity.add_argument("--graph", type=Path, required=True)
    capacity.add_argument("--out", type=Path, required=True)
    score = sub.add_parser("score")
    score.add_argument("--graph", type=Path, required=True)
    score.add_argument("--analysis", type=Path, required=True)
    score.add_argument("--prompt", type=Path, required=True)
    score.add_argument("--prompt-view", choices=plans.MODEL_VIEW_KINDS,
                       required=True)
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("responses", type=Path, nargs="+")
    record_command = sub.add_parser("record-command")
    record_command.add_argument("--out", type=Path, required=True)
    record_command.add_argument("--reset", action="store_true")
    record_command.add_argument("argv", nargs=argparse.REMAINDER)
    provenance = sub.add_parser("build-provenance")
    provenance.add_argument("--mode", choices=("discover", "lower"),
                            required=True)
    provenance.add_argument("--repo-root", type=Path, required=True)
    provenance.add_argument("--build-root", type=Path, required=True)
    provenance.add_argument("--commands", type=Path, required=True)
    provenance.add_argument("--out", type=Path, required=True)
    provenance.add_argument("--input", action="append", default=[])
    provenance.add_argument("--artifact", action="append", default=[])
    provenance.add_argument("--dependency-file", type=Path, action="append",
                            default=[])
    provenance.add_argument("--environment", action="append", default=[])
    verify_provenance = sub.add_parser("verify-build-provenance")
    verify_provenance.add_argument("--manifest", type=Path, required=True)
    verify_provenance.add_argument("--repo-root", type=Path, required=True)
    freeze = sub.add_parser("freeze-offline")
    freeze.add_argument("--bundle-root", type=Path, required=True)
    freeze.add_argument("--repo-root", type=Path, required=True)
    freeze.add_argument("--platform", type=Path, required=True)
    freeze.add_argument("--calibration-artifact", type=Path, action="append",
                        default=[])
    freeze.add_argument("--gate-a-monitor", type=Path, required=True)
    freeze.add_argument("--gate-a-stdout", type=Path, required=True)
    freeze.add_argument("--gate-a-stderr", type=Path, required=True)
    freeze.add_argument("--preparation-script", type=Path, required=True)
    freeze.add_argument("--out", type=Path, required=True)
    verify_freeze = sub.add_parser("verify-offline-freeze")
    verify_freeze.add_argument("--manifest", type=Path, required=True)
    verify_freeze.add_argument("--repo-root", type=Path, required=True)
    verify_gate_a = sub.add_parser("verify-gate-a-monitor")
    verify_gate_a.add_argument("--monitor", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "controls":
            manifest = generate_controls(
                _read_json(args.graph), args.out,
                args.source_sha256, args.catalog_sha256,
            )
            print(
                f"compiler-collective-eval: generated {len(manifest['arms'])} "
                f"uniform controls; manifest_id={manifest['manifest_id']}"
            )
        elif args.command == "verify":
            verify_manifest(
                _read_json(args.graph), _read_json(args.manifest),
                args.manifest.parent,
            )
            print("compiler-collective-eval: verified uniform control manifest")
        elif args.command == "verify-ir":
            verify_ir(_read_json(args.manifest), args.manifest.parent, args.ir)
            print("compiler-collective-eval: verified compiler plan metadata in IR")
        elif args.command == "verify-plan-ir":
            verify_plan_ir(_read_json(args.graph), _read_json(args.hint), args.ir)
            print("compiler-collective-eval: verified one materialized plan in IR")
        elif args.command == "verify-device-ir":
            reservations = verify_device_ir(args.ir)
            print(
                "compiler-collective-eval: verified device proxy-ring "
                f"operations in IR ({reservations} reservations)"
            )
        elif args.command == "qualify":
            summary = qualify_logs(
                _read_json(args.manifest), args.logs, args.gate
            )
            bridge._write_json_atomic(args.out, summary)
            print(
                f"compiler-collective-eval: Gate {summary['gate']} passed; "
                f"verified {len(summary['logs'])} immutable log(s)"
            )
        elif args.command == "analyze":
            summary = analyze_logs(
                _read_json(args.graph), _read_json(args.manifest), args.logs
            )
            bridge._write_json_atomic(args.out, summary)
            print(json.dumps({
                "aggregate": summary["aggregate"],
                "gate_c": summary["gate_c"],
            }, indent=2, sort_keys=True))
        elif args.command == "oracle":
            decision, hint = oracle_decision(
                _read_json(args.graph), _read_json(args.analysis)
            )
            bridge._write_json_atomic(args.decision, decision)
            bridge._write_json_atomic(args.hint, hint)
            print(
                "compiler-collective-eval: wrote measured compiler-bin oracle; "
                "model_invoked=false"
            )
        elif args.command == "canary":
            decision, hint = canary_decision(_read_json(args.graph))
            bridge._write_json_atomic(args.decision, decision)
            bridge._write_json_atomic(args.hint, hint)
            print(
                "compiler-collective-eval: wrote mixed-policy canary; "
                "model_invoked=false"
            )
        elif args.command == "capacity-audit":
            summary = audit_capacity(_read_json(args.graph))
            bridge._write_json_atomic(args.out, summary)
            print(json.dumps(summary, indent=2, sort_keys=True))
        elif args.command == "score":
            summary = score_decisions(
                _read_json(args.graph), _read_json(args.analysis),
                args.prompt, args.prompt_view, args.responses,
            )
            bridge._write_json_atomic(args.out, summary)
            print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
        elif args.command == "record-command":
            record_build_command(args.out, args.argv, reset=args.reset)
        elif args.command == "build-provenance":
            input_values = _parse_assignments(args.input, kind="input")
            artifact_values = _parse_assignments(
                args.artifact, kind="artifact"
            )
            environment = _parse_assignments(
                args.environment, kind="environment", allow_unset=True
            )
            summary = generate_build_provenance(
                mode=args.mode,
                repo_root=args.repo_root,
                build_root=args.build_root,
                inputs={name: Path(path) for name, path in input_values.items()
                        if path is not None},
                artifacts={
                    name: Path(path) for name, path in artifact_values.items()
                    if path is not None
                },
                dependency_files=args.dependency_file,
                environment=environment,
                commands_path=args.commands,
            )
            bridge._write_json_atomic(args.out, summary)
            print(
                "compiler-collective-eval: wrote same-build provenance; "
                f"manifest_id={summary['manifest_id']}"
            )
        elif args.command == "verify-build-provenance":
            verify_build_provenance(
                _read_json(args.manifest), manifest_path=args.manifest,
                repo_root=args.repo_root,
            )
            print(
                "compiler-collective-eval: verified build inputs, dependency "
                "closure, commands, and artifacts"
            )
        elif args.command == "freeze-offline":
            summary = generate_offline_freeze(
                bundle_root=args.bundle_root,
                repo_root=args.repo_root,
                platform_path=args.platform,
                calibration_paths=args.calibration_artifact,
                gate_a_monitor_path=args.gate_a_monitor,
                gate_a_stdout_path=args.gate_a_stdout,
                gate_a_stderr_path=args.gate_a_stderr,
                preparation_script=args.preparation_script,
            )
            bridge._write_json_atomic(args.out, summary)
            print(
                "compiler-collective-eval: froze verified v3 offline bundle; "
                f"manifest_id={summary['manifest_id']}"
            )
        elif args.command == "verify-offline-freeze":
            verify_offline_freeze(
                _read_json(args.manifest), manifest_path=args.manifest,
                repo_root=args.repo_root,
            )
            print(
                "compiler-collective-eval: verified complete v3 offline bundle"
            )
        else:
            _validate_gate_a_monitor(_read_json(args.monitor))
            print(
                "compiler-collective-eval: verified passed pdebug Gate-A monitor"
            )
        return 0
    except (EvalError, plans.CollectivePlanError, OSError, ValueError) as exc:
        print(f"compiler-collective-eval: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
