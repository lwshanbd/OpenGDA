#!/usr/bin/env python3
"""Build a held-out collective policy screen after an LLM archive is complete.

The adapter never invokes a provider, compiler, scheduler, or application
source path.  It revalidates the complete compiler-only response archive, a
passed eight-node confirmation, and three same-allocation rotated blocks of
all uniform compiler-catalog arms.  Only then does it compose per-message-size
costs for every observed graph-bound policy.

The composed costs are an offline screen.  They select representatives for a
later paired runtime campaign; they are not runtime evidence for any composed
LLM size policy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import tempfile
import os
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
PASS_ROOT = HERE.parents[1]
PASS_PYTHON = PASS_ROOT / "python"
ROOT = HERE.parents[4]
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(HERE))

import analyze_compiler_collective_n8_confirmation as n8_confirmation  # noqa: E402
import analyze_compiler_llm_capability_trials as capability_analysis  # noqa: E402
import compiler_collective_eval as collective_eval  # noqa: E402
import gicc_collective_structural_heuristic as structural_heuristic  # noqa: E402
import gicc_compiler_policy_bridge as policy_bridge  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import monitor_compiler_collective_job as monitor_base  # noqa: E402
import run_compiler_llm_capability_trials as trial_runner  # noqa: E402


CONTROL_BLOCKS = 3
CONTROL_RUNS = 7
CONTROL_WARMUP = 2
CONTROL_NODES = 8
CONTROL_RANKS = 64
CONTROL_PPN = 8
CONTROL_DURATION_SECONDS = 3300.0
CONTROL_SCHEMA = "gicc-collective-llm-uniform-control-screen-v1"
RUNNER_NAME = "run_compiler_collective_llm_controls.sh"
CONTROLLER_NAME = "continue_compiler_collective_llm_controls.sh"


class CollectivePolicyScreenError(RuntimeError):
    """The archive or hidden collective controls cannot support a screen."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CollectivePolicyScreenError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CollectivePolicyScreenError(
            f"cannot read JSON {path}: {exc}"
        ) from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise CollectivePolicyScreenError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise CollectivePolicyScreenError(
            f"screen evidence escapes repository: {resolved}"
        ) from exc


def evidence(path: Path, repo_root: Path) -> dict[str, Any]:
    require(path.is_file(), f"screen evidence is absent: {path}")
    return {
        "path": display_path(path, repo_root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _monitor_dependency_paths(path: Path) -> set[Path]:
    """Return the immutable raw-log and artifact closure named by a monitor."""
    value = read_json(path)
    result = {path.resolve()}
    for record in value.get("artifacts", []):
        if isinstance(record, dict) and isinstance(record.get("path"), str):
            result.add(Path(record["path"]).resolve())
    for role in ("driver_stdout", "driver_stderr"):
        record = value.get(role)
        if isinstance(record, dict) and isinstance(record.get("path"), str):
            result.add(Path(record["path"]).resolve())
    for benchmark in value.get("benchmarks", {}).values():
        if not isinstance(benchmark, dict):
            continue
        for role in ("stdout", "stderr"):
            record = benchmark.get(role)
            if isinstance(record, dict) and isinstance(record.get("path"), str):
                result.add(Path(record["path"]).resolve())
    return result


def runtime_dependency_paths(
    confirmation_path: Path, control_monitor_paths: list[Path],
) -> list[Path]:
    """Bind every raw file used to verify or compute the held-out costs."""
    confirmation = read_json(confirmation_path)
    result = {
        confirmation_path.resolve(),
        (PASS_PYTHON / "gicc_collective_structural_heuristic.py").resolve(),
    }
    transition = confirmation.get("transition")
    if isinstance(transition, str) and transition:
        result.add(Path(transition).resolve())
    for summary in confirmation.get("allocation_monitors", []):
        if isinstance(summary, dict) and isinstance(summary.get("monitor"), str):
            result.update(_monitor_dependency_paths(Path(summary["monitor"])))
    for path in control_monitor_paths:
        result.update(_monitor_dependency_paths(path))
    require(all(path.is_file() for path in result),
            "held-out runtime dependency closure contains an absent file")
    return sorted(result, key=str)


def _request_payload(value: Any) -> dict[str, Any]:
    require(
        isinstance(value, dict)
        and value.get("schema_version") == trial_runner.request_freezer.REQUEST_SCHEMA,
        "wrong compiler LLM request schema",
    )
    payload = dict(value)
    request_id = payload.pop("request_id", None)
    require(request_id == bridge._fingerprint(payload),
            "request ID does not match content")
    return payload


def verify_request_archive(
    graph: dict[str, Any], request_dir: Path, archive_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Verify the immutable request snapshot and every archived response."""
    request_path = request_dir / "request.json"
    request = read_json(request_path)
    _request_payload(request)
    require(request.get("compiler_graph_id") == graph["graph_id"],
            "request binds another compiler graph")
    require(request.get("decision_family")
            == "collective_algorithm_and_size_policy",
            "request is not a collective policy request")
    boundary = request.get("boundary", {})
    for key, expected in {
        "compiler_lto_decisions_only": True,
        "application_source_visible": False,
        "llvm_ir_visible": False,
        "evaluation_runtime_labels_visible": False,
        "evaluation_oracle_visible": False,
        "model_may_generate_code_or_ir": False,
        "model_output_is_existing_graph_bound_ids_only": True,
        "compiler_revalidates_every_response": True,
    }.items():
        require(boundary.get(key) is expected,
                f"request boundary changed: {key}")
    private_graph = request_dir / "private/compiler-graph.json"
    require(
        private_graph.is_file()
        and read_json(private_graph) == graph,
        "request private graph differs from the screened graph",
    )
    graph_records = [
        item for item in request.get("evidence", {}).get("bundle_files", [])
        if isinstance(item, dict) and item.get("role") == "compiler_graph"
    ]
    require(len(graph_records) == 1, "request lacks one private graph record")
    record = graph_records[0]
    require(
        record.get("provider_visible") is False
        and record.get("sha256") == sha256_file(private_graph)
        and record.get("bytes") == private_graph.stat().st_size,
        "request private graph provenance changed",
    )

    authorization_path = archive_dir / "authorization.json"
    authorization = read_json(authorization_path)
    index, _ = trial_runner.verify_complete_archive(
        request=request,
        authorization_value=authorization,
        graph=graph,
        output_dir=archive_dir,
    )
    index_path = archive_dir / "run-index.json"
    require(index.get("schema_version") == trial_runner.INDEX_SCHEMA,
            "wrong archive index schema")
    require(index.get("request_id") == request["request_id"],
            "archive binds another request")
    require(len(index.get("runs", [])) == 60,
            "collective capability archive must contain exactly 60 trials")
    capability_analysis.observed_policies(index, graph)
    return request, index, index_path


def verify_confirmation(
    path: Path, graph: dict[str, Any],
) -> dict[str, Any]:
    """Replay the exact three-allocation N8 confirmation from raw monitors."""
    value = read_json(path)
    require(
        isinstance(value, dict)
        and value.get("schema_version") == "gicc-collective-n8-confirmation-v1",
        "collective policy screen requires an N8 confirmation",
    )
    payload = dict(value)
    result_id = payload.pop("result_id", None)
    require(result_id == bridge._fingerprint(payload),
            "N8 confirmation ID does not match content")
    require(value.get("confirmation_gate", {}).get("passed") is True,
            "N8 confirmation gate did not pass")
    for key, expected in {
        "model_invoked": False,
        "application_source_modified": False,
        "provider_call_authorized": False,
    }.items():
        require(value.get(key) is expected,
                f"N8 confirmation boundary changed: {key}")
    transition_path = Path(value.get("transition", ""))
    require(
        transition_path.is_absolute() and transition_path.is_file()
        and sha256_file(transition_path) == value.get("transition_sha256"),
        "N8 confirmation transition changed",
    )
    transition, transition_files = n8_confirmation.validate_transition(
        transition_path
    )
    confirmed_graph = policy_bridge.verified_graph(
        read_json(transition_files["compiler_graph"])
    )
    require(confirmed_graph["graph_id"] == graph["graph_id"],
            "N8 confirmation binds another compiler graph")
    monitors = value.get("allocation_monitors")
    require(isinstance(monitors, list) and len(monitors) == 3,
            "N8 confirmation lacks three allocation monitors")
    monitor_paths = [Path(item.get("monitor", "")) for item in monitors]
    require(all(item.is_absolute() for item in monitor_paths),
            "N8 confirmation monitor path is not absolute")
    regenerated = n8_confirmation.analyze_monitors(
        transition_path, monitor_paths,
    )
    require(regenerated == value,
            "N8 confirmation does not replay from current raw evidence")
    require(transition.get("transition_id") == value.get("transition_id"),
            "N8 confirmation transition identity changed")
    return value


def collective_catalog(
    graph_value: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[str]]:
    graph = policy_bridge.verified_graph(graph_value)
    require(graph.get("schema_version")
            == collective_eval.plans.GRAPH_SCHEMA,
            "policy screen requires a collective graph")
    opportunities = graph.get("opportunities", [])
    require(len(opportunities) == 1,
            "collective policy screen requires exactly one opportunity")
    opportunity = opportunities[0]
    slots = opportunity.get("decision_slots", [])
    require(len(slots) == 4,
            "collective policy screen requires four message-size slots")
    algorithms: list[str] | None = None
    for slot in slots:
        by_algorithm = {
            option.get("algorithm"): option for option in slot.get("options", [])
        }
        require(
            None not in by_algorithm
            and len(by_algorithm) == len(slot.get("options", [])),
            f"slot {slot.get('slot_id')} repeats or omits an algorithm",
        )
        current = sorted(by_algorithm)
        if algorithms is None:
            algorithms = current
        require(current == algorithms,
                "collective slots expose different algorithm catalogs")
    require(algorithms is not None and "baseline_auto" in algorithms,
            "collective catalog lacks baseline_auto")
    return opportunity, slots, algorithms


def _expected_resources() -> list[dict[str, Any]]:
    return [{
        "type": "node", "count": CONTROL_NODES,
        "with": [{
            "type": "slot", "count": CONTROL_PPN, "label": "task",
            "with": [
                {"type": "core", "count": 8},
                {"type": "gpu", "count": 1},
            ],
        }],
    }]


def _artifact_paths(value: dict[str, Any]) -> set[Path]:
    records = value.get("artifacts")
    require(isinstance(records, list), "control monitor lacks frozen artifacts")
    paths: set[Path] = set()
    for record in records:
        require(
            isinstance(record, dict)
            and isinstance(record.get("path"), str)
            and isinstance(record.get("sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) is not None,
            "control monitor has an invalid artifact record",
        )
        path = Path(record["path"]).resolve()
        require(path not in paths, "control monitor repeats an artifact")
        require(path.is_file() and sha256_file(path) == record["sha256"],
                f"control artifact changed: {path}")
        paths.add(path)
    return paths


def _required_artifacts(bundle: Path, algorithms: list[str]) -> set[Path]:
    result = {
        (bundle / "FROZEN_V3_MANIFEST.json").resolve(),
        (bundle / "discovery/graph.json").resolve(),
        (bundle / "controls/manifest.json").resolve(),
        (HERE / RUNNER_NAME).resolve(),
        (HERE / CONTROLLER_NAME).resolve(),
        (HERE / "monitor_compiler_collective_replicate.py").resolve(),
        Path(__file__).resolve(),
    }
    for algorithm in algorithms:
        result.update({
            (bundle / f"binaries/{algorithm}/compiler_collective_eval").resolve(),
            (bundle / f"binaries/{algorithm}/build-provenance.json").resolve(),
            (bundle / f"controls/{algorithm}-hint.json").resolve(),
        })
    return result


def _arm_orders(algorithms: list[str]) -> dict[int, list[str]]:
    stride = math.ceil(len(algorithms) / CONTROL_BLOCKS)
    return {
        replicate: (
            algorithms[offset:] + algorithms[:offset]
        )
        for replicate in range(1, CONTROL_BLOCKS + 1)
        for offset in [((replicate - 1) * stride) % len(algorithms)]
    }


def _driver_lines(replicate: int, order: list[str]) -> list[str]:
    lines = [
        f"COLLECTIVE_LLM_CONTROL_CONFIG replicate={replicate} "
        f"arms={' '.join(order)}",
    ]
    for arm in order:
        lines.extend([
            f"COLLECTIVE_LLM_CONTROL_ARM_START replicate={replicate} arm={arm}",
            f"COLLECTIVE_LLM_CONTROL_ARM_DONE replicate={replicate} arm={arm}",
        ])
    lines.append(f"COLLECTIVE_LLM_CONTROL_DONE replicate={replicate}")
    return lines


def control_rows_from_monitors(
    graph: dict[str, Any], graph_path: Path, bundle: Path,
    monitor_paths: list[Path], repo_root: Path,
) -> tuple[dict[int, dict[str, dict[str, float]]], list[dict[str, Any]]]:
    """Reparse all raw logs under the frozen N8 full-catalog contract."""
    _, _, algorithms = collective_catalog(graph)
    require(len(monitor_paths) == CONTROL_BLOCKS,
            "full-catalog screen requires exactly three control monitors")
    freeze_path = bundle / "FROZEN_V3_MANIFEST.json"
    manifest_path = bundle / "controls/manifest.json"
    collective_eval.verify_offline_freeze(
        read_json(freeze_path), manifest_path=freeze_path,
        repo_root=repo_root,
    )
    manifest = read_json(manifest_path)
    collective_eval.verify_manifest(graph, manifest, manifest_path.parent)
    arms = {
        arm.get("name"): arm.get("algorithm") for arm in manifest.get("arms", [])
    }
    require(arms == {algorithm: algorithm for algorithm in algorithms},
            "uniform control manifest does not cover the graph catalog exactly")
    require(read_json(graph_path) == graph,
            "screen graph path changed during control validation")

    expected = {
        "nodes": CONTROL_NODES,
        "ranks": CONTROL_RANKS,
        "ppn": CONTROL_PPN,
        "runs": CONTROL_RUNS,
        "warmup": CONTROL_WARMUP,
        "sizes": list(collective_eval.GATE_B_SIZES),
    }
    required_artifacts = _required_artifacts(bundle, algorithms)
    orders = _arm_orders(algorithms)
    rows: dict[int, dict[str, dict[str, float]]] = {}
    summaries = []
    job_ids = set()
    node_sets = set()
    artifact_sets = set()
    for path in monitor_paths:
        value = read_json(path)
        require(
            isinstance(value, dict)
            and value.get("schema_version")
            == "gicc-collective-replicate-job-monitor-v1"
            and value.get("state") == "passed",
            f"full-catalog control monitor did not pass: {path}",
        )
        replicate = value.get("replicate")
        require(replicate in orders and replicate not in rows,
                "full-catalog controls repeat or misnumber a block")
        jobspec = value.get("jobspec", {})
        scheduler = value.get("scheduler", {})
        require(
            value.get("expected") == expected
            and scheduler.get("exit_code") == 0
            and scheduler.get("exception_types") == []
            and jobspec.get("queue") == "pdebug"
            and jobspec.get("duration_seconds") == CONTROL_DURATION_SECONDS
            and jobspec.get("resources") == _expected_resources(),
            f"control block {replicate} changed the N8 pdebug contract",
        )
        nodelist = value.get("resource_set", {}).get("nodelist")
        require(isinstance(nodelist, list) and nodelist,
                f"control block {replicate} lacks an exact node set")
        artifacts = _artifact_paths(value)
        require(artifacts == required_artifacts,
                f"control block {replicate} artifact set changed")
        runner = (HERE / RUNNER_NAME).resolve()
        require(jobspec.get("embedded_script_sha256") == sha256_file(runner),
                "full-catalog job did not execute the frozen runner")
        command = jobspec.get("command")
        require(isinstance(command, list),
                "full-catalog control job lacks a command")
        try:
            script_index = command.index("{{tmpdir}}/script")
        except ValueError:
            raise CollectivePolicyScreenError(
                "full-catalog control job lacks its batch script"
            ) from None
        driver_record = value.get("driver_stdout", {})
        driver_stderr = value.get("driver_stderr", {})
        driver = Path(driver_record.get("path", ""))
        driver_err = Path(driver_stderr.get("path", ""))
        output_root = driver.parent.parent.resolve()
        require(
            command[script_index + 1:]
            == [str(bundle.resolve()), str(output_root)],
            "full-catalog batch arguments changed",
        )
        require(
            driver.is_file()
            and sha256_file(driver) == driver_record.get("sha256")
            and driver.stat().st_size == driver_record.get("bytes")
            and driver_err.is_file()
            and sha256_file(driver_err) == driver_stderr.get("sha256")
            and driver_err.stat().st_size == driver_stderr.get("bytes") == 0,
            f"control block {replicate} driver logs changed",
        )
        require(
            driver.read_text(encoding="utf-8").splitlines()
            == _driver_lines(replicate, orders[replicate]),
            f"control block {replicate} did not finish its frozen arm order",
        )

        benchmarks = value.get("benchmarks")
        require(isinstance(benchmarks, dict)
                and set(benchmarks) == set(algorithms),
                f"control block {replicate} lacks the full catalog")
        block_rows = {}
        for algorithm in algorithms:
            record = benchmarks[algorithm]
            stdout_record = record.get("stdout", {})
            stderr_record = record.get("stderr", {})
            stdout = Path(stdout_record.get("path", ""))
            stderr = Path(stderr_record.get("path", ""))
            require(
                stdout.is_file()
                and sha256_file(stdout) == stdout_record.get("sha256")
                and stdout.stat().st_size == stdout_record.get("bytes")
                and stderr.is_file()
                and sha256_file(stderr) == stderr_record.get("sha256")
                and stderr.stat().st_size == stderr_record.get("bytes"),
                f"control block {replicate}/{algorithm} logs changed",
            )
            reparsed = monitor_base.validate_output(
                stdout, algorithm, list(collective_eval.GATE_B_SIZES),
                CONTROL_NODES, CONTROL_RANKS, CONTROL_PPN,
                CONTROL_RUNS, CONTROL_WARMUP,
            )
            require(reparsed == record.get("benchmark")
                    and reparsed.get("total_errors") == 0,
                    f"control block {replicate}/{algorithm} is not correct")
            block_rows[algorithm] = {
                key: float(number)
                for key, number in reparsed["results"].items()
            }
        rows[replicate] = block_rows
        artifact_set_id = bridge._fingerprint([
            {"path": str(item), "sha256": sha256_file(item)}
            for item in sorted(artifacts, key=str)
        ])
        summaries.append({
            "replicate": replicate,
            "job_id": value.get("job_id"),
            "nodelist": nodelist,
            "arm_order": orders[replicate],
            "monitor": str(path.resolve()),
            "monitor_sha256": sha256_file(path),
            "artifact_set_id": artifact_set_id,
        })
        job_ids.add(value.get("job_id"))
        node_sets.add(json.dumps(nodelist, sort_keys=True))
        artifact_sets.add(artifact_set_id)
    require(set(rows) == {1, 2, 3},
            "full-catalog control blocks are incomplete")
    require(len(job_ids) == len(node_sets) == len(artifact_sets) == 1,
            "full-catalog blocks did not share one allocation and artifact set")
    return rows, sorted(summaries, key=lambda item: item["replicate"])


def analyze_control_rows(
    graph_value: Any,
    rows: dict[int, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    """Pool three blocks and derive exact per-bin compiler controls."""
    graph = policy_bridge.verified_graph(graph_value)
    opportunity, slots, algorithms = collective_catalog(graph)
    sizes = list(collective_eval.GATE_B_SIZES)
    wanted_sizes = {str(size) for size in sizes}
    require(set(rows) == {1, 2, 3},
            "control rows require blocks 1, 2, and 3")
    for replicate, by_algorithm in rows.items():
        require(set(by_algorithm) == set(algorithms),
                f"control block {replicate} has incomplete algorithms")
        for algorithm, by_size in by_algorithm.items():
            require(set(by_size) == wanted_sizes,
                    f"control block {replicate}/{algorithm} has incomplete sizes")
            require(all(
                isinstance(number, (int, float))
                and not isinstance(number, bool)
                and math.isfinite(number) and number > 0
                for number in by_size.values()
            ), f"control block {replicate}/{algorithm} has invalid latency")

    pooled = {
        algorithm: {
            str(size): statistics.median([
                float(rows[replicate][algorithm][str(size)])
                for replicate in (1, 2, 3)
            ])
            for size in sizes
        }
        for algorithm in algorithms
    }
    size_to_slot: dict[int, dict[str, Any]] = {}
    sizes_by_slot: dict[str, list[int]] = {}
    for slot in slots:
        lower = slot["message_bytes"]["min"]
        upper = slot["message_bytes"]["max"]
        matching = [
            size for size in sizes
            if (lower is None or size >= lower)
            and (upper is None or size <= upper)
        ]
        require(matching, f"slot {slot['slot_id']} has no measured size")
        sizes_by_slot[slot["slot_id"]] = matching
        for size in matching:
            require(size not in size_to_slot,
                    f"message size {size} belongs to multiple slots")
            size_to_slot[size] = slot
    require(set(size_to_slot) == set(sizes),
            "message-size slots do not cover the runtime control sweep")

    raw_weights = opportunity.get("compiler_facts", {}).get(
        "message_distribution", {}
    ).get("bins_weight")
    if raw_weights is None:
        raw_weights = graph.get("platform_profile", {}).get(
            "message_distribution", {}
        ).get("bins_weight")
    require(
        isinstance(raw_weights, list) and len(raw_weights) == len(slots)
        and all(isinstance(weight, (int, float))
                and not isinstance(weight, bool)
                and math.isfinite(weight) and weight > 0
                for weight in raw_weights),
        "collective graph lacks positive weights for every message bin",
    )
    total_weight = sum(float(weight) for weight in raw_weights)
    unit_weights = {}
    for slot, raw_weight in zip(slots, raw_weights, strict=True):
        in_bin = sizes_by_slot[slot["slot_id"]]
        for size in in_bin:
            unit_weights[f"message_bytes:{size}"] = (
                float(raw_weight) / total_weight / len(in_bin)
            )

    opportunity_id = opportunity["opportunity_id"]
    oracle_selection = {}
    oracle_bins = []
    for slot in slots:
        in_bin = sizes_by_slot[slot["slot_id"]]
        scores = {
            algorithm: math.exp(sum(
                math.log(pooled[algorithm][str(size)]) for size in in_bin
            ) / len(in_bin))
            for algorithm in algorithms
        }
        winner = min(scores, key=lambda name: (scores[name], name))
        options = {
            option["algorithm"]: option["option_id"]
            for option in slot["options"]
        }
        key = f"{opportunity_id}/{slot['slot_id']}"
        oracle_selection[key] = options[winner]
        oracle_bins.append({
            "slot_id": slot["slot_id"],
            "algorithm": winner,
            "option_id": options[winner],
            "sizes": in_bin,
            "geomean_us": scores[winner],
        })
    oracle = policy_bridge.verified_policy(graph, oracle_selection)
    return {
        "schema_version": CONTROL_SCHEMA,
        "graph_id": graph["graph_id"],
        "replicates": 3,
        "sizes_bytes": sizes,
        "pooled_algorithm_median_us": pooled,
        "sizes_by_slot": sizes_by_slot,
        "unit_weights": unit_weights,
        "oracle_bins": oracle_bins,
        "oracle_selected_ids_by_slot": oracle["selected_ids_by_slot"],
    }


def policy_cost(
    graph_value: Any, selected_ids_by_slot: Any,
    controls: dict[str, Any],
) -> dict[str, float]:
    graph = policy_bridge.verified_graph(graph_value)
    opportunity, slots, _ = collective_catalog(graph)
    policy = policy_bridge.verified_policy(graph, selected_ids_by_slot)
    pooled = controls["pooled_algorithm_median_us"]
    result = {}
    for slot in slots:
        key = f"{opportunity['opportunity_id']}/{slot['slot_id']}"
        option_id = policy["selected_ids_by_slot"][key]
        option = next(
            item for item in slot["options"]
            if item["option_id"] == option_id
        )
        algorithm = option["algorithm"]
        for size in controls["sizes_by_slot"][slot["slot_id"]]:
            result[f"message_bytes:{size}"] = float(
                pooled[algorithm][str(size)]
            )
    require(set(result) == set(controls["unit_weights"]),
            "composed policy cost has incomplete message-size units")
    return result


def build_screen(
    *, graph: dict[str, Any], request: dict[str, Any],
    index: dict[str, Any], index_path: Path,
    controls: dict[str, Any], control_summaries: list[dict[str, Any]],
    confirmation_path: Path, monitor_paths: list[Path], repo_root: Path,
) -> dict[str, Any]:
    policies = capability_analysis.observed_policies(index, graph)
    anchor = policy_bridge.decision_to_policy(graph, None)
    deterministic_decision = structural_heuristic.make_decision(graph)
    deterministic = policy_bridge.decision_to_policy(
        graph, deterministic_decision
    )
    require(deterministic["bridge_accepted"] is True,
            "compiler rejected its deterministic collective control")
    oracle = policy_bridge.verified_policy(
        graph, controls["oracle_selected_ids_by_slot"]
    )
    screen_controls = {
        "oracle": {
            "selected_ids_by_slot": oracle["selected_ids_by_slot"],
            "cost_by_unit": policy_cost(
                graph, oracle["selected_ids_by_slot"], controls
            ),
        },
        "anchor": {
            "selected_ids_by_slot": anchor["selected_ids_by_slot"],
            "cost_by_unit": policy_cost(
                graph, anchor["selected_ids_by_slot"], controls
            ),
        },
        "deterministic": {
            "selected_ids_by_slot": deterministic["selected_ids_by_slot"],
            "cost_by_unit": policy_cost(
                graph, deterministic["selected_ids_by_slot"], controls
            ),
        },
    }
    policy_costs = {
        policy_id: policy_cost(graph, selected, controls)
        for policy_id, selected in sorted(policies.items())
    }
    runtime_evidence_paths = runtime_dependency_paths(
        confirmation_path, monitor_paths
    )
    payload = {
        "schema_version": capability_analysis.SCREEN_SCHEMA,
        "graph_id": graph["graph_id"],
        "request_id": request["request_id"],
        "run_index_sha256": sha256_file(index_path),
        "boundary": dict(capability_analysis.SCREEN_BOUNDARY),
        "controls": screen_controls,
        "unit_weights": controls["unit_weights"],
        "policy_cost_by_id": policy_costs,
        "collective_control": {
            "schema_version": controls["schema_version"],
            "replicates": controls["replicates"],
            "sizes_bytes": controls["sizes_bytes"],
            "pooled_algorithm_median_us": controls[
                "pooled_algorithm_median_us"
            ],
            "oracle_bins": controls["oracle_bins"],
            "block_monitors": control_summaries,
            "runtime_evidence_file_count": len(runtime_evidence_paths),
            "composition": (
                "Each option selects the held-out pooled uniform-arm median "
                "for each measured size in its frozen compiler message bin."
            ),
            "runtime_interpretation": (
                "Uniform-arm labels rank graph-bound policies offline; each "
                "chosen composed policy still requires paired materialized "
                "runtime confirmation."
            ),
        },
        "deterministic_control": {
            "heuristic_version": structural_heuristic.HEURISTIC_VERSION,
            "implementation": evidence(
                PASS_PYTHON / "gicc_collective_structural_heuristic.py",
                repo_root,
            ),
        },
        "evidence": {
            "family_adapter": evidence(Path(__file__), repo_root),
            "runtime_controls": [
                evidence(path, repo_root) for path in runtime_evidence_paths
            ],
        },
    }
    result = dict(payload)
    result["screen_id"] = bridge._fingerprint(payload)
    return result


def preflight(
    *, graph_path: Path, request_dir: Path, archive_dir: Path,
    confirmation_path: Path, bundle: Path, repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], Path]:
    graph = policy_bridge.verified_graph(read_json(graph_path))
    require(graph_path.resolve() == (bundle / "discovery/graph.json").resolve(),
            "screen graph must be the frozen bundle graph")
    collective_catalog(graph)
    freeze_path = bundle / "FROZEN_V3_MANIFEST.json"
    manifest_path = bundle / "controls/manifest.json"
    collective_eval.verify_offline_freeze(
        read_json(freeze_path), manifest_path=freeze_path,
        repo_root=repo_root,
    )
    collective_eval.verify_manifest(
        graph, read_json(manifest_path), manifest_path.parent,
    )
    request, index, index_path = verify_request_archive(
        graph, request_dir, archive_dir,
    )
    confirmation = verify_confirmation(confirmation_path, graph)
    return graph, request, index, index_path


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


def add_base_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--confirmation", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    check = subparsers.add_parser("preflight")
    add_base_inputs(check)
    emit = subparsers.add_parser("emit")
    add_base_inputs(emit)
    emit.add_argument("--monitor", type=Path, action="append", required=True)
    emit.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        graph, request, index, index_path = preflight(
            graph_path=args.graph.resolve(),
            request_dir=args.request_dir.resolve(),
            archive_dir=args.archive_dir.resolve(),
            confirmation_path=args.confirmation.resolve(),
            bundle=args.bundle.resolve(), repo_root=args.repo_root.resolve(),
        )
        if args.command == "preflight":
            print(
                "collective-llm-policy-screen: ready for one hidden N8 "
                "full-catalog control allocation; provider_calls=0 "
                "scheduler_jobs=0"
            )
            return 0
        monitor_paths = [path.resolve() for path in args.monitor]
        rows, summaries = control_rows_from_monitors(
            graph, args.graph.resolve(), args.bundle.resolve(),
            monitor_paths, args.repo_root.resolve(),
        )
        controls = analyze_control_rows(graph, rows)
        screen = build_screen(
            graph=graph, request=request, index=index, index_path=index_path,
            controls=controls, control_summaries=summaries,
            confirmation_path=args.confirmation.resolve(),
            monitor_paths=monitor_paths, repo_root=args.repo_root.resolve(),
        )
        if args.out.exists():
            raise CollectivePolicyScreenError(
                f"refusing to overwrite policy screen: {args.out}"
            )
        write_json_atomic(args.out.resolve(), screen)
        print(
            "collective-llm-policy-screen: wrote held-out offline screen; "
            f"policies={len(screen['policy_cost_by_id'])}; "
            f"screen_id={screen['screen_id']}"
        )
        return 0
    except (
        CollectivePolicyScreenError,
        capability_analysis.CapabilityAnalysisError,
        trial_runner.CapabilityTrialError,
        n8_confirmation.ConfirmError,
        collective_eval.EvalError,
        structural_heuristic.HeuristicError,
        policy_bridge.CompilerPolicyBridgeError,
        monitor_base.MonitorError,
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"collective-llm-policy-screen: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
