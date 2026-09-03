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
import re
import statistics
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
        else:
            summary = score_decisions(
                _read_json(args.graph), _read_json(args.analysis),
                args.prompt, args.prompt_view, args.responses,
            )
            bridge._write_json_atomic(args.out, summary)
            print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
        return 0
    except (EvalError, plans.CollectivePlanError, OSError, ValueError) as exc:
        print(f"compiler-collective-eval: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
