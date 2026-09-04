#!/usr/bin/env python3
"""Derive one preregistered N8 confirmation plan from a passed scout.

This tool never invokes a provider, compiler, or scheduler.  It maps the
scout's pooled compiler-control measurements to existing graph-bound option
IDs using a rule committed before the pending scout produced any results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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

import analyze_compiler_collective_hierpipe_n8_scout as scout  # noqa: E402
import gicc_collective_plan_bridge as plans  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


TRANSITION_SCHEMA = "gicc-collective-n8-confirmation-transition-v1"
GRAPH_ID = scout.GRAPH_ID
BUNDLE_ID = scout.BUNDLE_ID
ARMS = scout.ARMS
SIZES = scout.SIZES
SLOT_SIZES = {
    "message-bin-0": (1024, 4096),
    "message-bin-1": (8192, 65536, 262144),
    "message-bin-2": (1048576, 4194304, 8388608),
    "message-bin-3": (16777216,),
}
SLOT_INTERVALS = {
    "message-bin-0": {"min": None, "max": 4096},
    "message-bin-1": {"min": 4097, "max": 262144},
    "message-bin-2": {"min": 262145, "max": 8388608},
    "message-bin-3": {"min": 8388609, "max": None},
}
SYSTEM_PROTOCOL = HERE / "HIERPIPE_N8_CONFIRMATION_TRANSITION.md"
SCOUT_SCHEMA = "gicc-collective-hierpipe-n8-scout-v1"
CAPACITY_GATE_KEY = "n8_capacity_gate"
TOPOLOGY_LABEL = "N8"
NODES = 8
RANKS = 64
RANKS_PER_NODE = 8
SCOUT_FILE_ROLE = "passed_n8_scout"
DECISION_ORIGIN = "preregistered_n8_scout_to_confirmation_selector"
DERIVED_RATIONALE = (
    "preregistered per-bin geometric-mean selector from the N8 scout"
)
UNIFORM_RATIONALE = (
    "regenerated best uniform hierarchy-pipeline arm from the N8 scout"
)
PREPARER_PATH = Path(__file__).resolve()
SUPPORT_PREPARER_FILES: tuple[tuple[Path, str], ...] = ()
TRANSITION_SUPPORT_ROLES: set[str] = set()
PROGRAM_NAME = "compiler-collective-n8-confirmation"


class TransitionError(RuntimeError):
    """The scout cannot be mapped to the frozen confirmation contract."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise TransitionError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TransitionError(f"cannot read JSON {path}: {exc}") from exc


def geomean(values: list[float]) -> float:
    if not values or any(
            value <= 0 or not math.isfinite(value) for value in values):
        raise TransitionError("scout costs must be finite and positive")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def validate_scout(value: Any) -> dict[str, Any]:
    if (not isinstance(value, dict)
            or value.get("schema_version")
            != SCOUT_SCHEMA
            or value.get("graph_id") != GRAPH_ID
            or value.get("bundle_id") != BUNDLE_ID
            or value.get("model_invoked") is not False
            or value.get("application_source_modified") is not False
            or value.get(CAPACITY_GATE_KEY, {}).get("passed") is not True):
        raise TransitionError(
            f"{TOPOLOGY_LABEL} confirmation requires the passed "
            "compiler-only scout"
        )
    payload = dict(value)
    result_id = payload.pop("result_id", None)
    if result_id != bridge._fingerprint(payload):
        raise TransitionError("N8 scout result ID does not match content")
    rows = value.get("per_size_pooled_median")
    if not isinstance(rows, dict) or set(rows) != {str(size) for size in SIZES}:
        raise TransitionError("N8 scout has incomplete size coverage")
    all_values = {arm: [] for arm in ARMS}
    for size in SIZES:
        row = rows[str(size)]
        costs = row.get("algorithm_median_us") if isinstance(row, dict) else None
        if not isinstance(costs, dict) or set(costs) != set(ARMS):
            raise TransitionError("N8 scout has incomplete arm coverage")
        normalized = {}
        for arm in ARMS:
            cost = costs[arm]
            if (isinstance(cost, bool) or not isinstance(cost, (int, float))
                    or float(cost) <= 0 or not math.isfinite(float(cost))):
                raise TransitionError("N8 scout cost is not finite positive")
            normalized[arm] = float(cost)
            all_values[arm].append(float(cost))
        winner = min(ARMS, key=lambda arm: (normalized[arm], ARMS.index(arm)))
        if row.get("winner") != winner:
            raise TransitionError("N8 scout pooled winner does not match costs")
    scores = {arm: geomean(values) for arm, values in all_values.items()}
    best_uniform = min(
        ARMS, key=lambda arm: (scores[arm], ARMS.index(arm))
    )
    aggregate = value.get("aggregate")
    if (not isinstance(aggregate, dict)
            or aggregate.get("best_uniform_algorithm") != best_uniform):
        raise TransitionError("N8 scout best-uniform arm does not regenerate")
    return value


def validate_scout_provenance(path: Path, value: dict[str, Any]) -> None:
    summaries = value.get("block_monitors")
    if not isinstance(summaries, list) or len(summaries) != 3:
        raise TransitionError("N8 scout lacks three raw monitor bindings")
    monitor_paths = []
    for summary in summaries:
        if (not isinstance(summary, dict)
                or not isinstance(summary.get("monitor"), str)
                or not isinstance(summary.get("monitor_sha256"), str)):
            raise TransitionError("N8 scout has an invalid monitor binding")
        monitor_path = Path(summary["monitor"]).resolve()
        if (not monitor_path.is_file()
                or sha256_file(monitor_path) != summary["monitor_sha256"]):
            raise TransitionError(f"N8 scout monitor changed: {monitor_path}")
        monitor_paths.append(monitor_path)
    regenerated = scout.analyze_monitors(monitor_paths)
    if regenerated != value:
        raise TransitionError(
            f"N8 scout does not regenerate from raw monitors: {path}"
        )


def derive_bin_algorithms(value: Any) -> tuple[
    dict[str, str], dict[str, dict[str, Any]], str
]:
    scout_value = validate_scout(value)
    rows = scout_value["per_size_pooled_median"]
    selected = {}
    bins = {}
    all_values = {arm: [] for arm in ARMS}
    for slot_id, sizes in SLOT_SIZES.items():
        scores = {}
        for arm in ARMS:
            values = [
                float(rows[str(size)]["algorithm_median_us"][arm])
                for size in sizes
            ]
            all_values[arm].extend(values)
            scores[arm] = geomean(values)
        winner = min(
            ARMS, key=lambda arm: (scores[arm], ARMS.index(arm))
        )
        selected[slot_id] = winner
        bins[slot_id] = {
            "message_bytes": SLOT_INTERVALS[slot_id],
            "scout_sizes": list(sizes),
            "algorithm_geomean_us": scores,
            "selected_algorithm": winner,
            "tie_break_priority": list(ARMS),
        }
    uniform_scores = {
        arm: geomean(values) for arm, values in all_values.items()
    }
    best_uniform = min(
        ARMS, key=lambda arm: (uniform_scores[arm], ARMS.index(arm))
    )
    return selected, bins, best_uniform


def validate_graph(value: Any) -> dict[str, Any]:
    graph = plans.verified_graph(value)
    if graph["graph_id"] != GRAPH_ID or len(graph["opportunities"]) != 1:
        raise TransitionError("N8 graph identity or opportunity count changed")
    opportunity = graph["opportunities"][0]
    slots = {slot["slot_id"]: slot for slot in opportunity["decision_slots"]}
    if set(slots) != set(SLOT_SIZES):
        raise TransitionError("N8 compiler bins changed")
    for slot_id, slot in slots.items():
        if slot.get("message_bytes") != SLOT_INTERVALS[slot_id]:
            raise TransitionError("N8 compiler bin interval changed")
        by_algorithm = {option.get("algorithm"): option for option in slot["options"]}
        if not set(ARMS).issubset(by_algorithm):
            raise TransitionError("N8 graph lacks a hierarchy-pipeline arm")
    return graph


def make_decision(graph_value: Any, algorithms: dict[str, str],
                  rationale: str) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = validate_graph(graph_value)
    opportunity = graph["opportunities"][0]
    slots = {slot["slot_id"]: slot for slot in opportunity["decision_slots"]}
    if set(algorithms) != set(slots) or any(
            algorithm not in ARMS for algorithm in algorithms.values()):
        raise TransitionError("confirmation policy does not cover exact N8 bins")
    option_ids = {}
    for slot_id, algorithm in algorithms.items():
        matches = [
            option for option in slots[slot_id]["options"]
            if option.get("algorithm") == algorithm
        ]
        if len(matches) != 1:
            raise TransitionError("N8 algorithm does not map to one graph option")
        option_ids[slot_id] = matches[0]["option_id"]
    decision = {
        "schema_version": plans.DECISION_SCHEMA,
        "graph_id": graph["graph_id"],
        "selections": {
            opportunity["opportunity_id"]: {
                "slot_candidate_ids": option_ids,
                "confidence": 1.0,
                "rationale": rationale,
            },
        },
    }
    hint, accepted, errors = plans.decision_to_hint(graph, decision)
    if not accepted or errors:
        raise TransitionError(
            "compiler bridge rejected derived confirmation policy: "
            + "; ".join(errors)
        )
    hint["llm_metadata"].update({
        "model_invoked": False,
        "decision_origin": DECISION_ORIGIN,
    })
    return decision, hint


def selected_option_ids(decision: dict[str, Any]) -> dict[str, str]:
    selections = decision["selections"]
    return next(iter(selections.values()))["slot_candidate_ids"]


def has_incremental_policy(
    derived: dict[str, str], uniform: dict[str, str],
    heuristic: dict[str, str],
) -> bool:
    return derived != uniform and derived != heuristic


def validate_heuristic(graph: dict[str, Any], value: Any) -> dict[str, Any]:
    hint, accepted, errors = plans.decision_to_hint(graph, value)
    if not accepted or errors:
        raise TransitionError(
            "frozen structural heuristic is invalid: " + "; ".join(errors)
        )
    return value


def materialize_heuristic_hint(
    graph: dict[str, Any], value: dict[str, Any],
) -> dict[str, Any]:
    hint, accepted, errors = plans.decision_to_hint(graph, value)
    if not accepted or errors:
        raise TransitionError(
            "frozen structural heuristic is invalid: " + "; ".join(errors)
        )
    hint["llm_metadata"].update({
        "model_invoked": False,
        "decision_origin": "frozen_structural_heuristic",
    })
    return hint


def file_record(path: Path, role: str, base: Path | None = None) -> dict[str, Any]:
    resolved = path.resolve()
    if base is not None:
        try:
            displayed = resolved.relative_to(base.resolve()).as_posix()
        except ValueError as exc:
            raise TransitionError(f"generated file escapes output: {path}") from exc
    else:
        try:
            displayed = resolved.relative_to(ROOT).as_posix()
        except ValueError:
            displayed = str(resolved)
    return {
        "role": role,
        "path": displayed,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


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


def build_outputs(graph_path: Path, scout_path: Path,
                  heuristic_path: Path) -> dict[str, Any]:
    graph = validate_graph(read_json(graph_path))
    scout_value = validate_scout(read_json(scout_path))
    validate_scout_provenance(scout_path, scout_value)
    heuristic = validate_heuristic(graph, read_json(heuristic_path))
    heuristic_hint = materialize_heuristic_hint(graph, heuristic)
    selected, bin_records, best_uniform = derive_bin_algorithms(scout_value)
    derived_decision, derived_hint = make_decision(
        graph, selected, DERIVED_RATIONALE,
    )
    uniform_decision, uniform_hint = make_decision(
        graph, {slot_id: best_uniform for slot_id in SLOT_SIZES},
        UNIFORM_RATIONALE,
    )
    heuristic_ids = selected_option_ids(heuristic)
    derived_ids = selected_option_ids(derived_decision)
    uniform_ids = selected_option_ids(uniform_decision)
    incremental_policy = has_incremental_policy(
        derived_ids, uniform_ids, heuristic_ids,
    )
    return {
        "graph": graph,
        "scout": scout_value,
        "heuristic": heuristic,
        "heuristic_hint": heuristic_hint,
        "selected": selected,
        "bin_records": bin_records,
        "best_uniform": best_uniform,
        "derived_decision": derived_decision,
        "derived_hint": derived_hint,
        "uniform_decision": uniform_decision,
        "uniform_hint": uniform_hint,
        "derived_ids": derived_ids,
        "uniform_ids": uniform_ids,
        "heuristic_ids": heuristic_ids,
        "incremental_policy": incremental_policy,
    }


def transition_payload(inputs: dict[str, Any], graph_path: Path,
                       scout_path: Path, heuristic_path: Path,
                       output_dir: Path) -> dict[str, Any]:
    generated = {
        "derived_decision": output_dir / "derived-bin-policy-decision.json",
        "derived_hint": output_dir / "derived-bin-policy-hint.json",
        "uniform_decision": output_dir / "best-uniform-decision.json",
        "uniform_hint": output_dir / "best-uniform-hint.json",
        "heuristic_hint": output_dir / "frozen-structural-heuristic-hint.json",
    }
    payload = {
        "schema_version": TRANSITION_SCHEMA,
        "status": (
            "confirmation_plan_ready" if inputs["incremental_policy"]
            else "closed_no_incremental_policy"
        ),
        "graph_id": inputs["graph"]["graph_id"],
        "scout_result_id": inputs["scout"]["result_id"],
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_modified": False,
            "model_invoked": False,
            "provider_call_authorized": False,
            "scheduler_job_submitted": False,
        },
        "selector": {
            "rule": (
                "within each frozen compiler size bin, choose the arm with "
                "minimum geometric mean of pooled scout median latency"
            ),
            "tie_break_priority": list(ARMS),
            "bins": inputs["bin_records"],
            "selected_algorithms": inputs["selected"],
            "selected_option_ids": inputs["derived_ids"],
        },
        "comparators": {
            "scout_best_uniform_algorithm": inputs["best_uniform"],
            "scout_best_uniform_option_ids": inputs["uniform_ids"],
            "frozen_structural_heuristic_option_ids": inputs["heuristic_ids"],
            "derived_equals_best_uniform": (
                inputs["derived_ids"] == inputs["uniform_ids"]
            ),
            "derived_equals_frozen_structural_heuristic": (
                inputs["derived_ids"] == inputs["heuristic_ids"]
            ),
        },
        "confirmation_contract": {
            "enabled": inputs["incremental_policy"],
            "queue": "pdebug",
            "nodes": NODES,
            "ranks": RANKS,
            "ranks_per_node": RANKS_PER_NODE,
            "cpu_cores_per_rank": 8,
            "gpus_per_rank": 1,
            "independent_allocations": 3,
            "maximum_active_or_queued_jobs": 1,
            "sizes_bytes": list(SIZES),
            "warmup_calls": 2,
            "timed_calls": 7,
            "arms": [
                "derived_bin_policy", "scout_best_uniform",
                "frozen_structural_heuristic",
            ],
            "arm_order": {
                "1": [
                    "derived_bin_policy", "scout_best_uniform",
                    "frozen_structural_heuristic",
                ],
                "2": [
                    "scout_best_uniform", "frozen_structural_heuristic",
                    "derived_bin_policy",
                ],
                "3": [
                    "frozen_structural_heuristic", "derived_bin_policy",
                    "scout_best_uniform",
                ],
            },
            "co_primary_comparators": [
                "scout_best_uniform", "frozen_structural_heuristic",
            ],
            "co_primary_metrics": (
                "paired geometric-mean speedup of derived_bin_policy over "
                "each preregistered comparator across all frozen sizes"
            ),
            "pass_rule": (
                "for both co-primary comparisons, point estimate >=1.03 and "
                "exact paired-bootstrap 95% lower bound >1.0; all correctness "
                "checks must also pass"
            ),
            "scout_or_confirmation_labels_visible_to_model": False,
        },
        "files": [
            file_record(graph_path, "compiler_graph"),
            file_record(scout_path, SCOUT_FILE_ROLE),
            file_record(heuristic_path, "frozen_structural_heuristic"),
            file_record(SYSTEM_PROTOCOL, "preregistered_transition_protocol"),
            file_record(PREPARER_PATH, "transition_preparer"),
            *[
                file_record(path, role)
                for path, role in SUPPORT_PREPARER_FILES
            ],
            *[
                file_record(path, role, output_dir)
                for role, path in generated.items()
            ],
        ],
    }
    return payload


def prepare(graph_path: Path, scout_path: Path, heuristic_path: Path,
            output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise TransitionError(f"refusing to overwrite output {output_dir}")
    inputs = build_outputs(graph_path, scout_path, heuristic_path)
    output_dir.mkdir(parents=True)
    write_json_atomic(
        output_dir / "derived-bin-policy-decision.json",
        inputs["derived_decision"],
    )
    write_json_atomic(
        output_dir / "derived-bin-policy-hint.json", inputs["derived_hint"]
    )
    write_json_atomic(
        output_dir / "best-uniform-decision.json", inputs["uniform_decision"]
    )
    write_json_atomic(
        output_dir / "best-uniform-hint.json", inputs["uniform_hint"]
    )
    write_json_atomic(
        output_dir / "frozen-structural-heuristic-hint.json",
        inputs["heuristic_hint"],
    )
    payload = transition_payload(
        inputs, graph_path, scout_path, heuristic_path, output_dir,
    )
    transition = {**payload, "transition_id": bridge._fingerprint(payload)}
    write_json_atomic(output_dir / "transition.json", transition)
    return transition


def verify(graph_path: Path, scout_path: Path, heuristic_path: Path,
           output_dir: Path) -> dict[str, Any]:
    inputs = build_outputs(graph_path, scout_path, heuristic_path)
    expected_values = {
        "derived-bin-policy-decision.json": inputs["derived_decision"],
        "derived-bin-policy-hint.json": inputs["derived_hint"],
        "best-uniform-decision.json": inputs["uniform_decision"],
        "best-uniform-hint.json": inputs["uniform_hint"],
        "frozen-structural-heuristic-hint.json": inputs["heuristic_hint"],
    }
    for name, expected in expected_values.items():
        if read_json(output_dir / name) != expected:
            raise TransitionError(f"generated confirmation artifact changed: {name}")
    payload = transition_payload(
        inputs, graph_path, scout_path, heuristic_path, output_dir,
    )
    expected_transition = {
        **payload, "transition_id": bridge._fingerprint(payload),
    }
    if read_json(output_dir / "transition.json") != expected_transition:
        raise TransitionError("confirmation transition does not regenerate")
    return expected_transition


def add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--scout", type=Path, required=True)
    parser.add_argument("--heuristic-decision", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    add_inputs(prepare_parser)
    verify_parser = subparsers.add_parser("verify")
    add_inputs(verify_parser)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            transition = prepare(
                args.graph, args.scout, args.heuristic_decision,
                args.output_dir,
            )
            action = "prepared"
        else:
            transition = verify(
                args.graph, args.scout, args.heuristic_decision,
                args.output_dir,
            )
            action = "verified"
        print(
            f"{PROGRAM_NAME}: {action}; "
            f"model_invoked=false; scheduler_job_submitted=false; "
            f"transition_id={transition['transition_id']}"
        )
        return 0
    except (TransitionError, plans.CollectivePlanError, OSError, KeyError,
            TypeError, ValueError) as exc:
        print(f"{PROGRAM_NAME}: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
