#!/usr/bin/env python3
"""Audit three independent N8 compiler-policy confirmation allocations."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import monitor_compiler_collective_job as monitor_base
import prepare_compiler_collective_n8_confirmation as transition_base


ARMS = (
    "derived_bin_policy",
    "scout_best_uniform",
    "frozen_structural_heuristic",
)
COMPARATORS = ARMS[1:]
ORDERS = {
    1: list(ARMS),
    2: [ARMS[1], ARMS[2], ARMS[0]],
    3: [ARMS[2], ARMS[0], ARMS[1]],
}
SIZES = tuple(transition_base.SIZES)
GENERATED_ROLES = {
    "derived_decision",
    "derived_hint",
    "uniform_decision",
    "uniform_hint",
    "heuristic_hint",
}
ConfirmError = transition_base.TransitionError
sha256 = transition_base.sha256_file
fingerprint = transition_base.bridge._fingerprint
TOPOLOGY_LABEL = "N8"
NODES = 8
RANKS = 64
PPN = 8
RUNS = 7
WARMUP = 2
BATCH_DURATION_SECONDS = 1200.0
RUNNER_FILENAME = "run_compiler_collective_n8_confirmation.sh"
CONTROLLER_FILENAME = "continue_compiler_collective_n8_confirmation.sh"
ANALYZER_PATH = Path(__file__).resolve()
SUPPORT_ANALYZER_FILES: tuple[Path, ...] = ()
LOG_PREFIX = "COLLECTIVE_N8_CONFIRM"
RESULT_SCHEMA = "gicc-collective-n8-confirmation-v1"
RESULT_SCOPE = (
    "Three sequential independent N8 pdebug allocations with Latin-"
    "rotated compiler/LTO policies; no model or source modification."
)
PROGRAM_NAME = "compiler-collective-n8-confirm"


def read_json(path: Path) -> Any:
    return transition_base.read_json(path)


def geomean(values: list[float]) -> float:
    return transition_base.geomean(values)


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_allocation_bootstrap(ratios: list[float]) -> dict[str, Any]:
    if len(ratios) != 3 or any(
            value <= 0 or not math.isfinite(value) for value in ratios):
        raise ConfirmError(
            "confirmation requires three finite positive allocation ratios"
        )
    estimates = [
        geomean([ratios[index] for index in sample])
        for sample in itertools.product(range(3), repeat=3)
    ]
    return {
        "estimate": geomean(ratios),
        "allocation_ratios": ratios,
        "allocation_wins": sum(value > 1.0 for value in ratios),
        "method": (
            "exact 3-out-of-3 paired cluster bootstrap over independent "
            "pdebug allocations"
        ),
        "bootstrap_samples": len(estimates),
        "lower_2_5_percent": percentile(estimates, 0.025),
        "upper_97_5_percent": percentile(estimates, 0.975),
    }


def analyze_rows(
    rows: dict[int, dict[str, dict[str, float]]],
) -> dict[str, Any]:
    if set(rows) != {1, 2, 3}:
        raise ConfirmError("confirmation requires allocations 1, 2, and 3")
    wanted_sizes = {str(size) for size in SIZES}
    for replicate, arms in rows.items():
        if set(arms) != set(ARMS):
            raise ConfirmError(
                f"allocation {replicate} has incomplete arm coverage"
            )
        for arm, values in arms.items():
            if set(values) != wanted_sizes:
                raise ConfirmError(
                    f"allocation {replicate}/{arm} has incomplete size coverage"
                )
            if any(value <= 0 or not math.isfinite(value)
                   for value in values.values()):
                raise ConfirmError("confirmation latency is not finite positive")

    comparisons = {}
    criteria = {}
    for comparator in COMPARATORS:
        allocation_ratios = []
        per_size_ratios = {str(size): [] for size in SIZES}
        for replicate in (1, 2, 3):
            derived_values = [
                rows[replicate][ARMS[0]][str(size)] for size in SIZES
            ]
            comparator_values = [
                rows[replicate][comparator][str(size)] for size in SIZES
            ]
            allocation_ratios.append(
                geomean(comparator_values) / geomean(derived_values)
            )
            for size in SIZES:
                key = str(size)
                per_size_ratios[key].append(
                    rows[replicate][comparator][key]
                    / rows[replicate][ARMS[0]][key]
                )
        bootstrap = paired_allocation_bootstrap(allocation_ratios)
        comparisons[comparator] = {
            "derived_policy_speedup": bootstrap,
            "per_size_allocation_speedups": per_size_ratios,
        }
        criteria[f"derived_over_{comparator}_point_estimate"] = {
            "observed": bootstrap["estimate"],
            "threshold": 1.03,
            "comparison": "greater_than_or_equal",
            "passed": bootstrap["estimate"] >= 1.03,
        }
        criteria[f"derived_over_{comparator}_lower_95"] = {
            "observed": bootstrap["lower_2_5_percent"],
            "threshold": 1.0,
            "comparison": "strictly_greater",
            "passed": bootstrap["lower_2_5_percent"] > 1.0,
        }
    criteria["all_runtime_correctness_checks"] = {
        "observed": True,
        "threshold": True,
        "passed": True,
    }
    return {
        "comparisons": comparisons,
        "confirmation_gate": {
            "passed": all(item["passed"] for item in criteria.values()),
            "criteria": criteria,
        },
    }


def _transition_file_paths(
    transition_path: Path, value: dict[str, Any],
) -> dict[str, Path]:
    records = value.get("files")
    if not isinstance(records, list):
        raise ConfirmError("transition lacks its content-addressed file set")
    result = {}
    for record in records:
        if (not isinstance(record, dict)
                or not isinstance(record.get("role"), str)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)):
            raise ConfirmError("transition has an invalid file record")
        role = record["role"]
        if role in result:
            raise ConfirmError(f"transition repeats file role {role}")
        raw = Path(record["path"])
        if raw.is_absolute():
            path = raw.resolve()
        elif role in GENERATED_ROLES:
            path = (transition_path.parent / raw).resolve()
        else:
            path = (transition_base.ROOT / raw).resolve()
        if (not path.is_file() or sha256(path) != record["sha256"]
                or path.stat().st_size != record["bytes"]):
            raise ConfirmError(f"transition file changed: {path}")
        result[role] = path
    expected_roles = {
        "compiler_graph", transition_base.SCOUT_FILE_ROLE,
        "frozen_structural_heuristic", "preregistered_transition_protocol",
        "transition_preparer", *transition_base.TRANSITION_SUPPORT_ROLES,
        *GENERATED_ROLES,
    }
    if set(result) != expected_roles:
        raise ConfirmError("transition file roles changed")
    return result


def validate_transition(
    transition_path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    value = read_json(transition_path)
    if (not isinstance(value, dict)
            or value.get("schema_version")
            != transition_base.TRANSITION_SCHEMA
            or value.get("status") != "confirmation_plan_ready"):
        raise ConfirmError(
            f"confirmation requires a ready {TOPOLOGY_LABEL} transition"
        )
    payload = dict(value)
    transition_id = payload.pop("transition_id", None)
    if transition_id != fingerprint(payload):
        raise ConfirmError("transition ID does not match content")
    boundary = value.get("boundary")
    if boundary != {
        "compiler_lto_decisions_only": True,
        "application_source_modified": False,
        "model_invoked": False,
        "provider_call_authorized": False,
        "scheduler_job_submitted": False,
    }:
        raise ConfirmError("transition compiler-only boundary changed")
    contract = value.get("confirmation_contract")
    required_contract = {
        "enabled": True,
        "queue": "pdebug",
        "nodes": NODES,
        "ranks": RANKS,
        "ranks_per_node": PPN,
        "cpu_cores_per_rank": 8,
        "gpus_per_rank": 1,
        "independent_allocations": 3,
        "maximum_active_or_queued_jobs": 1,
        "sizes_bytes": list(SIZES),
        "warmup_calls": 2,
        "timed_calls": 7,
        "arms": list(ARMS),
        "arm_order": {str(key): order for key, order in ORDERS.items()},
        "co_primary_comparators": list(COMPARATORS),
        "scout_or_confirmation_labels_visible_to_model": False,
    }
    if not isinstance(contract, dict) or any(
            contract.get(key) != expected
            for key, expected in required_contract.items()):
        raise ConfirmError(f"{TOPOLOGY_LABEL} confirmation contract changed")
    paths = _transition_file_paths(transition_path, value)
    regenerated = transition_base.verify(
        paths["compiler_graph"], paths[transition_base.SCOUT_FILE_ROLE],
        paths["frozen_structural_heuristic"], transition_path.parent,
    )
    if regenerated != value:
        raise ConfirmError(f"{TOPOLOGY_LABEL} transition does not regenerate")
    return value, paths


def _artifact_records(value: dict[str, Any]) -> dict[Path, str]:
    artifacts = value.get("artifacts")
    if not isinstance(artifacts, list):
        raise ConfirmError("confirmation monitor lacks frozen artifacts")
    records = {}
    for record in artifacts:
        if (not isinstance(record, dict)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])):
            raise ConfirmError("confirmation monitor has an invalid artifact")
        path = Path(record["path"]).resolve()
        if path in records or not path.is_file() or sha256(path) != record["sha256"]:
            raise ConfirmError(f"confirmation artifact changed: {path}")
        records[path] = record["sha256"]
    return records


def _validate_monitor(
    path: Path, transition_path: Path, transition_files: dict[str, Path],
) -> tuple[int, dict[str, dict[str, float]], dict[str, Any]]:
    value = read_json(path)
    if (not isinstance(value, dict)
            or value.get("schema_version")
            != "gicc-collective-replicate-job-monitor-v1"
            or value.get("state") != "passed"):
        raise ConfirmError(f"confirmation monitor did not pass: {path}")
    replicate = value.get("replicate")
    if replicate not in ORDERS:
        raise ConfirmError("confirmation allocation number is invalid")
    expected = {
        "nodes": NODES, "ranks": RANKS, "ppn": PPN,
        "runs": RUNS, "warmup": WARMUP,
        "sizes": list(SIZES),
    }
    resources = [{
        "type": "node", "count": NODES,
        "with": [{
            "type": "slot", "count": PPN, "label": "task",
            "with": [
                {"type": "core", "count": 8},
                {"type": "gpu", "count": 1},
            ],
        }],
    }]
    jobspec = value.get("jobspec", {})
    scheduler = value.get("scheduler", {})
    if (value.get("expected") != expected
            or scheduler.get("exit_code") != 0
            or scheduler.get("exception_types") != []
            or jobspec.get("queue") != "pdebug"
            or jobspec.get("duration_seconds") != BATCH_DURATION_SECONDS
            or jobspec.get("resources") != resources):
        raise ConfirmError(
            f"allocation {replicate} runtime contract changed"
        )
    nodelist = value.get("resource_set", {}).get("nodelist")
    if not isinstance(nodelist, list) or len(nodelist) != NODES:
        raise ConfirmError(
            f"allocation {replicate} lacks {NODES} exact nodes"
        )

    records = _artifact_records(value)
    here = ANALYZER_PATH.parent
    output_root = path.resolve().parent.parent
    binary_root = (output_root / "binaries").resolve()
    required = {
        transition_path.resolve(), *transition_files.values(),
        (here / "build_compiler_collective_eval.sh").resolve(),
        (here / "compiler_collective_eval.py").resolve(),
        (here / RUNNER_FILENAME).resolve(),
        (here / CONTROLLER_FILENAME).resolve(),
        (here / "monitor_compiler_collective_replicate.py").resolve(),
        ANALYZER_PATH,
    }
    required.update(path.resolve() for path in SUPPORT_ANALYZER_FILES)
    freezes = [item for item in records if item.name == "FROZEN_V3_MANIFEST.json"]
    if len(freezes) != 1:
        raise ConfirmError(
            f"confirmation lacks one frozen {TOPOLOGY_LABEL} bundle manifest"
        )
    required.add(freezes[0])
    for arm in ARMS:
        required.update({
            (binary_root / arm / "compiler_collective_eval").resolve(),
            (binary_root / arm / "build-provenance.json").resolve(),
        })
    if set(records) != required:
        raise ConfirmError("confirmation artifact set changed")

    runner = (here / RUNNER_FILENAME).resolve()
    if jobspec.get("embedded_script_sha256") != records[runner]:
        raise ConfirmError(
            f"Flux did not execute the frozen {TOPOLOGY_LABEL} runner"
        )
    command = jobspec.get("command")
    try:
        script_index = command.index("{{tmpdir}}/script")
    except (AttributeError, ValueError):
        raise ConfirmError("confirmation lacks its Flux batch command") from None
    arguments = command[script_index + 1:]
    if (len(arguments) != 3
            or Path(arguments[0]).resolve() != binary_root
            or Path(arguments[1]).resolve() != output_root
            or arguments[2] != str(replicate)):
        raise ConfirmError("confirmation batch arguments changed")

    driver_record = value.get("driver_stdout")
    driver_stderr = value.get("driver_stderr")
    if not isinstance(driver_record, dict) or not isinstance(driver_stderr, dict):
        raise ConfirmError(f"allocation {replicate} lacks a driver log")
    driver = Path(driver_record.get("path", ""))
    driver_err = Path(driver_stderr.get("path", ""))
    if (not driver.is_file() or sha256(driver) != driver_record.get("sha256")
            or driver.stat().st_size != driver_record.get("bytes")
            or not driver_err.is_file()
            or sha256(driver_err) != driver_stderr.get("sha256")
            or driver_err.stat().st_size != driver_stderr.get("bytes")
            or driver_err.stat().st_size != 0):
        raise ConfirmError(f"allocation {replicate} driver log changed")
    expected_lines = [
        f"{LOG_PREFIX}_CONFIG replicate={replicate} "
        f"arms={' '.join(ORDERS[replicate])}",
    ]
    for arm in ORDERS[replicate]:
        expected_lines.extend([
            f"{LOG_PREFIX}_ARM_START replicate={replicate} arm={arm}",
            f"{LOG_PREFIX}_ARM_DONE replicate={replicate} arm={arm}",
        ])
    expected_lines.append(
        f"{LOG_PREFIX}_DONE replicate={replicate}"
    )
    if driver.read_text(encoding="utf-8").splitlines() != expected_lines:
        raise ConfirmError(f"allocation {replicate} arm order changed")

    benchmarks = value.get("benchmarks")
    if not isinstance(benchmarks, dict) or set(benchmarks) != set(ARMS):
        raise ConfirmError(f"allocation {replicate} lacks complete arms")
    rows = {}
    for arm in ARMS:
        record = benchmarks[arm]
        if not isinstance(record, dict):
            raise ConfirmError(f"allocation {replicate}/{arm} is invalid")
        benchmark = record.get("benchmark")
        stdout = record.get("stdout")
        stderr = record.get("stderr")
        if (not isinstance(benchmark, dict) or not isinstance(stdout, dict)
                or not isinstance(stderr, dict)):
            raise ConfirmError(f"allocation {replicate}/{arm} is incomplete")
        log = Path(stdout.get("path", ""))
        err = Path(stderr.get("path", ""))
        if (not log.is_file() or sha256(log) != stdout.get("sha256")
                or log.stat().st_size != stdout.get("bytes")
                or not err.is_file() or sha256(err) != stderr.get("sha256")
                or err.stat().st_size != stderr.get("bytes")):
            raise ConfirmError(f"allocation {replicate}/{arm} log changed")
        reparsed = monitor_base.validate_output(
            log, arm, list(SIZES), NODES, RANKS, PPN, RUNS, WARMUP,
        )
        if reparsed != benchmark or benchmark.get("total_errors") != 0:
            raise ConfirmError(
                f"allocation {replicate}/{arm} monitor/raw log mismatch"
            )
        rows[arm] = {
            key: float(number)
            for key, number in benchmark["results"].items()
        }
    summary = {
        "allocation": replicate,
        "job_id": value.get("job_id"),
        "nodelist": nodelist,
        "arm_order": ORDERS[replicate],
        "monitor": str(path.resolve()),
        "monitor_sha256": sha256(path),
        "artifact_set_id": fingerprint([
            {"path": str(item), "sha256": records[item]}
            for item in sorted(records, key=str)
        ]),
    }
    return replicate, rows, summary


def analyze_monitors(
    transition_path: Path, monitor_paths: list[Path],
) -> dict[str, Any]:
    transition, transition_files = validate_transition(transition_path)
    if len(monitor_paths) != 3:
        raise ConfirmError("confirmation requires exactly three monitors")
    rows = {}
    summaries = []
    for path in monitor_paths:
        replicate, allocation, summary = _validate_monitor(
            path.resolve(), transition_path.resolve(), transition_files,
        )
        if replicate in rows:
            raise ConfirmError("confirmation repeats an allocation number")
        rows[replicate] = allocation
        summaries.append(summary)
    if set(rows) != {1, 2, 3}:
        raise ConfirmError("confirmation allocations are incomplete")
    if len({item["job_id"] for item in summaries}) != 3:
        raise ConfirmError("confirmation did not use three independent jobs")
    if len({item["artifact_set_id"] for item in summaries}) != 1:
        raise ConfirmError("confirmation allocations used different artifacts")
    analysis = analyze_rows(rows)
    payload = {
        "schema_version": RESULT_SCHEMA,
        "scope": RESULT_SCOPE,
        "model_invoked": False,
        "application_source_modified": False,
        "provider_call_authorized": False,
        "transition_id": transition["transition_id"],
        "transition": str(transition_path.resolve()),
        "transition_sha256": sha256(transition_path),
        "allocation_monitors": sorted(
            summaries, key=lambda item: item["allocation"]
        ),
        **analysis,
    }
    return {**payload, "result_id": fingerprint(payload)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transition", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = analyze_monitors(
            args.transition.resolve(),
            [path.resolve() for path in args.monitor],
        )
        if args.out.exists():
            raise ConfirmError(f"refusing to overwrite {args.out}")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(result["confirmation_gate"], sort_keys=True))
        return 0
    except (ConfirmError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"{PROGRAM_NAME}: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
