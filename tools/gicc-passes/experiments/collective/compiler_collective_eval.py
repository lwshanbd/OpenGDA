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
import json
import math
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
PASS_PYTHON = ROOT / "tools" / "gicc-passes" / "python"
sys.path.insert(0, str(PASS_PYTHON))

import gicc_collective_plan_bridge as plans  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


MANIFEST_SCHEMA = "gicc-collective-uniform-controls-v1"
RESULT_RE = re.compile(r"([a-z_]+)=([^ ]+)")


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


def parse_log(path: Path) -> tuple[str, dict[int, float]]:
    label = None
    rows: dict[int, float] = {}
    done_errors = None
    for line in path.read_text().splitlines():
        if line.startswith("COLLECTIVE_CONFIG "):
            fields = dict(RESULT_RE.findall(line))
            label = fields.get("plan")
        elif line.startswith("RESULT "):
            fields = dict(RESULT_RE.findall(line))
            if int(fields["errors"]) != 0:
                raise EvalError(f"{path}: correctness errors at {fields['bytes']} B")
            size = int(fields["bytes"])
            if size in rows:
                raise EvalError(f"{path}: duplicate result at {size} B")
            rows[size] = float(fields["median_us"])
        elif line.startswith("COLLECTIVE_DONE "):
            fields = dict(RESULT_RE.findall(line))
            done_errors = int(fields["total_errors"])
    if not label or not rows or done_errors != 0:
        raise EvalError(f"{path}: incomplete or failing collective log")
    return label, rows


def _geomean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value)
                         for value in values):
        raise EvalError("geometric mean requires positive finite values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def analyze_logs(
    graph_value: Any, manifest_value: Any, logs: list[Path]
) -> dict[str, Any]:
    graph = plans.verified_graph(graph_value)
    algorithms = {arm["name"]: arm["algorithm"]
                  for arm in manifest_value["arms"]}
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
        winner = min(medians, key=lambda algorithm: medians[algorithm][size])
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
    best_uniform = min(aggregate, key=aggregate.get)
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
        winner = min(score, key=score.get)
        option = next(item for item in slot["options"]
                      if item["algorithm"] == winner)
        bin_choices.append({
            "slot_id": slot["slot_id"],
            "algorithm": winner,
            "option_id": option["option_id"],
            "sizes": in_bin,
            "geomean_us": score[winner],
        })
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
            "baseline_over_per_size_oracle": aggregate[baseline] / oracle_geomean,
            "best_uniform_over_per_size_oracle": aggregate[best_uniform] / oracle_geomean,
            "distinct_per_size_winners": sorted({
                item["winner"] for item in per_size.values()
            }),
        },
        "compiler_bin_oracle": bin_choices,
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
        elif args.command == "analyze":
            summary = analyze_logs(
                _read_json(args.graph), _read_json(args.manifest), args.logs
            )
            bridge._write_json_atomic(args.out, summary)
            print(json.dumps(summary["aggregate"], indent=2, sort_keys=True))
        else:
            decision, hint = oracle_decision(
                _read_json(args.graph), _read_json(args.analysis)
            )
            bridge._write_json_atomic(args.decision, decision)
            bridge._write_json_atomic(args.hint, hint)
            print(
                "compiler-collective-eval: wrote measured compiler-bin oracle; "
                "model_invoked=false"
            )
        return 0
    except (EvalError, plans.CollectivePlanError, OSError, ValueError) as exc:
        print(f"compiler-collective-eval: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
