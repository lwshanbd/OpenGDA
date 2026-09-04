#!/usr/bin/env python3
"""Recover route-only Minimod evidence without upgrading it to a new claim.

The historical experiment fixed either the serial or overlap schedule while
LTO materialized one of three uniform communication routes.  This analyzer
reparses every raw log, maps those routes to current graph candidate IDs, and
quantifies the route-only oracle.  Schema drift and unmeasured mixed routes
remain explicit, so this report cannot unlock a provider experiment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PASS_PYTHON = ROOT / "tools/gicc-passes/python"
MINIMOD_TOOLS = ROOT / "examples/minimod_paper"
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(MINIMOD_TOOLS))

import analyze as minimod  # noqa: E402
import gicc_comm_group_plan_bridge as group_plans  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


SCHEMA = "gicc-minimod-route-only-retrospective-v1"
ROUTES = ("default", "trigger", "proxy")
SCHEDULES = ("serial", "overlap")
ROUTE_DISPATCH = {
    "default": "IPC_OR_DWQ",
    "trigger": "DWQ_TRIGGER",
    "proxy": "CPU_PROXY_ENQUEUE",
}
MEASUREMENT_FIELDS = (
    "kind", "arm", "nodes", "rpn", "ranks", "rep", "grid", "steps",
    "kernel_s", "comm_s", "comp_s", "checksum_set", "staged", "pushed",
    "path",
)


class RetrospectiveError(RuntimeError):
    """Historical evidence is incomplete or incompatible."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def geomean(values: list[float]) -> float:
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise RetrospectiveError("runtime ratios must be finite and positive")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        displayed = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        displayed = str(resolved)
    return {
        "path": displayed,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _same_sample(row: dict[str, str], sample: minimod.Sample) -> bool:
    strings = ("kind", "arm", "checksum_set")
    integers = ("nodes", "rpn", "ranks", "rep", "grid", "steps", "staged", "pushed")
    floats = ("kernel_s", "comm_s", "comp_s")
    if any(row[name] != str(getattr(sample, name)) for name in strings):
        return False
    row_path = Path(row["path"])
    row_path = row_path.resolve() if row_path.is_absolute() else (ROOT / row_path).resolve()
    if row_path != Path(sample.path).resolve():
        return False
    if any(int(row[name]) != getattr(sample, name) for name in integers):
        return False
    return all(math.isclose(
        float(row[name]), getattr(sample, name), rel_tol=1e-12, abs_tol=1e-12,
    ) for name in floats)


def load_dataset(path: Path, label: str) -> list[minimod.Sample]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != MEASUREMENT_FIELDS:
                raise RetrospectiveError(f"measurement schema changed: {path}")
            rows = list(reader)
    except OSError as exc:
        raise RetrospectiveError(f"cannot read {path}: {exc}") from exc
    samples = []
    for row in rows:
        raw = Path(row["path"])
        log = raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()
        try:
            sample = minimod.parse_log(log)
        except (OSError, ValueError) as exc:
            raise RetrospectiveError(f"raw Minimod log rejected: {exc}") from exc
        if not _same_sample(row, sample):
            raise RetrospectiveError(
                f"derived measurement disagrees with raw log: {log}"
            )
        samples.append(sample)
    standard = [sample for sample in samples if sample.kind == "standard"]
    try:
        minimod.validate_full(standard, "standard", standard_nodes=(1, 2, 4))
        minimod.validate_checksums(standard)
    except ValueError as exc:
        raise RetrospectiveError(f"dataset {label} is incomplete: {exc}") from exc
    grids = {sample.grid for sample in standard}
    if grids != {int(label)}:
        raise RetrospectiveError(
            f"dataset label/grid mismatch: {label} versus {sorted(grids)}"
        )
    return standard


def feature_signature(value: Any) -> tuple[int, list[dict[str, Any]]]:
    if not isinstance(value, list) or not value:
        raise RetrospectiveError("Minimod feature file is empty")
    versions = {row.get("schema_version") for row in value if isinstance(row, dict)}
    if len(versions) != 1 or not all(isinstance(row, dict) for row in value):
        raise RetrospectiveError("Minimod feature schemas are inconsistent")
    signature = [
        {
            "site_id": row.get("site_id"),
            "op_kind": row.get("op_kind"),
            "legal_paths": row.get("legal_paths"),
        }
        for row in value
    ]
    return next(iter(versions)), signature


def validate_feature_compatibility(
    legacy_path: Path, current_path: Path,
) -> dict[str, Any]:
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    current = json.loads(current_path.read_text(encoding="utf-8"))
    legacy_version, legacy_signature = feature_signature(legacy)
    current_version, current_signature = feature_signature(current)
    if legacy_signature != current_signature:
        raise RetrospectiveError(
            "legacy/current operation sites or legal routes differ"
        )
    if legacy_version == current_version:
        raise RetrospectiveError("expected explicit feature-schema drift")
    return {
        "operation_signature_equal": True,
        "legacy_schema_version": legacy_version,
        "current_schema_version": current_version,
        "feature_files_byte_identical": False,
        "compatibility": "semantic_uniform_route_subset_only",
    }


def uniform_candidates(graph_value: Any) -> tuple[dict[str, str], list[str]]:
    graph = group_plans.verified_graph(graph_value)
    if len(graph["opportunities"]) != 1:
        raise RetrospectiveError("Minimod graph must have one route opportunity")
    opportunity = graph["opportunities"][0]
    result = {}
    measured_ids = set()
    for candidate in opportunity["candidates"]:
        requests = list(candidate["materializer"]["sites"].values())
        dispatches = {request["dispatch"] for request in requests}
        if len(dispatches) != 1:
            continue
        dispatch = next(iter(dispatches))
        for route, wanted in ROUTE_DISPATCH.items():
            if dispatch == wanted:
                if route in result:
                    raise RetrospectiveError(f"duplicate uniform route {route}")
                result[route] = candidate["candidate_id"]
                measured_ids.add(candidate["candidate_id"])
    if set(result) != set(ROUTES):
        raise RetrospectiveError("current graph lacks three uniform routes")
    unmeasured = [
        candidate["candidate_id"] for candidate in opportunity["candidates"]
        if candidate["candidate_id"] not in measured_ids
    ]
    return result, unmeasured


def validate_binaries(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split(maxsplit=1)
        if len(fields) != 2:
            raise RetrospectiveError("invalid historical binary hash record")
        digest, raw = fields
        binary = Path(raw).resolve()
        name = binary.name.removeprefix("minimod_")
        if name in ROUTES:
            if name in rows or not binary.is_file() or sha256_file(binary) != digest:
                raise RetrospectiveError(f"historical binary changed: {binary}")
            rows[name] = {
                "path": str(binary.relative_to(ROOT)),
                "sha256": digest,
            }
    if set(rows) != set(ROUTES):
        raise RetrospectiveError("historical route binaries are incomplete")
    return rows


def paired_bootstrap(
    ratios: list[float], seed: int = 0, draws: int = 10000,
) -> dict[str, Any]:
    estimate = geomean(ratios)
    rng = random.Random(seed)
    samples = sorted(
        geomean([ratios[rng.randrange(len(ratios))] for _ in ratios])
        for _ in range(draws)
    )
    return {
        "estimate": estimate,
        "paired_cell_allocation_ratios": ratios,
        "bootstrap_draws": draws,
        "seed": seed,
        "lower_2_5_percent": samples[int(0.025 * draws)],
        "upper_97_5_percent": samples[int(0.975 * draws) - 1],
        "interpretation": (
            "descriptive paired bootstrap conditional on retrospectively "
            "selected routes; not a preregistered confidence interval"
        ),
    }


def analyze_schedule(
    samples: list[minimod.Sample], schedule: str,
    candidate_ids: dict[str, str],
) -> dict[str, Any]:
    selected = [sample for sample in samples if sample.arm.endswith(f"_{schedule}")]
    cells = sorted({(sample.nodes, sample.rpn) for sample in selected} - {(1, 1)})
    by_key = {
        ((sample.nodes, sample.rpn), sample.rep, sample.arm.split("_", 1)[0]):
        sample.kernel_s
        for sample in selected
        if (sample.nodes, sample.rpn) in cells
    }
    expected = {
        (cell, replicate, route)
        for cell in cells for replicate in range(1, 6) for route in ROUTES
    }
    if set(by_key) != expected:
        raise RetrospectiveError("fixed-schedule route matrix is incomplete")
    medians = {
        cell: {
            route: statistics.median(
                by_key[(cell, replicate, route)] for replicate in range(1, 6)
            )
            for route in ROUTES
        }
        for cell in cells
    }
    winners = {
        cell: min(ROUTES, key=lambda route: (medians[cell][route], ROUTES.index(route)))
        for cell in cells
    }
    route_geomeans = {
        route: geomean([medians[cell][route] for cell in cells])
        for route in ROUTES
    }
    best_fixed = min(
        ROUTES, key=lambda route: (route_geomeans[route], ROUTES.index(route))
    )
    oracle_geomean = geomean([medians[cell][winners[cell]] for cell in cells])
    paired_ratios = [
        by_key[(cell, replicate, best_fixed)]
        / by_key[(cell, replicate, winners[cell])]
        for cell in cells for replicate in range(1, 6)
    ]
    per_cell = {}
    for cell in cells:
        label = f"n{cell[0]}-rpn{cell[1]}"
        winner = winners[cell]
        per_cell[label] = {
            "median_seconds": medians[cell],
            "winner": winner,
            "winner_candidate_id": candidate_ids[winner],
            "best_fixed_over_oracle": medians[cell][best_fixed] / medians[cell][winner],
        }
    return {
        "fixed_schedule": schedule,
        "cells": per_cell,
        "distinct_winners": sorted(set(winners.values())),
        "route_geometric_mean_seconds": route_geomeans,
        "best_fixed_route": best_fixed,
        "best_fixed_candidate_id": candidate_ids[best_fixed],
        "pointwise_oracle_geometric_mean_seconds": oracle_geomean,
        "best_fixed_over_pointwise_oracle": route_geomeans[best_fixed] / oracle_geomean,
        "maximum_cell_headroom": max(
            item["best_fixed_over_oracle"] for item in per_cell.values()
        ),
        "conditional_paired_bootstrap": paired_bootstrap(paired_ratios),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    datasets = {}
    for item in args.dataset:
        if "=" not in item:
            raise RetrospectiveError("dataset must be GRID=MEASUREMENTS.csv")
        label, raw = item.split("=", 1)
        if label in datasets or not label.isdigit():
            raise RetrospectiveError(f"duplicate or invalid dataset label: {label}")
        path = Path(raw).resolve()
        samples = load_dataset(path, label)
        datasets[label] = {
            "path": path,
            "samples": samples,
        }
    if len(datasets) < 2:
        raise RetrospectiveError("at least two independent grids are required")
    graph_value = json.loads(args.graph.read_text(encoding="utf-8"))
    graph = group_plans.verified_graph(graph_value)
    candidate_ids, unmeasured = uniform_candidates(graph)
    compatibility = validate_feature_compatibility(
        args.legacy_features, args.current_features,
    )
    binaries = validate_binaries(args.binary_sha256)

    analyses = {
        label: {
            schedule: analyze_schedule(
                row["samples"], schedule, candidate_ids,
            )
            for schedule in SCHEDULES
        }
        for label, row in sorted(datasets.items())
    }
    winner_consistency = {}
    labels = sorted(analyses)
    for schedule in SCHEDULES:
        common_cells = set.intersection(*(
            set(analyses[label][schedule]["cells"]) for label in labels
        ))
        matches = {
            cell: len({
                analyses[label][schedule]["cells"][cell]["winner"]
                for label in labels
            }) == 1
            for cell in sorted(common_cells)
        }
        winner_consistency[schedule] = {
            "matching_cells": sum(matches.values()),
            "total_cells": len(matches),
            "per_cell": matches,
        }

    payload = {
        "schema_version": SCHEMA,
        "scope": (
            "Retrospective extraction of compiler/LTO uniform-route effects "
            "under fixed hand-written schedules; no LLM or source decision."
        ),
        "boundary": {
            "model_invoked": False,
            "provider_call_authorized": False,
            "llm_output_modified_application_source": False,
            "measurement_source_was_instrumented": True,
            "compiler_lto_route_is_only_scored_decision_axis": True,
        },
        "graph_id": graph["graph_id"],
        "uniform_route_candidate_ids": candidate_ids,
        "measured_candidate_count": len(candidate_ids),
        "current_graph_candidate_count": len(graph["opportunities"][0]["candidates"]),
        "unmeasured_mixed_candidate_ids": unmeasured,
        "feature_compatibility": compatibility,
        "datasets": analyses,
        "cross_grid_winner_consistency": winner_consistency,
        "historical_binaries": binaries,
        "claim_gate": {
            "retrospective_motivating_evidence": True,
            "current_graph_runtime_labels_complete": False,
            "confirmatory_compiler_oracle": False,
            "llm_performance_measured": False,
            "provider_protocol_permitted": False,
            "reasons": [
                "candidate choices and analysis were not preregistered for this question",
                "legacy feature schema differs from the current compiler graph schema",
                "only three uniform routes are measured; six mixed routes are unmeasured",
                "the fixed serial/overlap schedule is hand-written rather than LTO-selected",
            ],
        },
        "evidence": {
            "measurements": {
                label: evidence(row["path"]) for label, row in sorted(datasets.items())
            },
            "legacy_features": evidence(args.legacy_features),
            "current_features": evidence(args.current_features),
            "current_graph": evidence(args.graph),
            "binary_sha256": evidence(args.binary_sha256),
        },
    }
    return {**payload, "result_id": bridge._fingerprint(payload)}


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


def add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset", action="append", required=True)
    parser.add_argument("--legacy-features", type=Path, required=True)
    parser.add_argument("--current-features", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--binary-sha256", type=Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit")
    add_inputs(emit)
    emit.add_argument("--out", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    add_inputs(verify)
    verify.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build_report(args)
        if args.command == "emit":
            if args.out.exists():
                raise RetrospectiveError(f"refusing to overwrite {args.out}")
            write_json_atomic(args.out, result)
            action = "wrote"
        else:
            if json.loads(args.report.read_text(encoding="utf-8")) != result:
                raise RetrospectiveError("retrospective report does not regenerate")
            action = "verified"
        print(
            f"minimod-route-only-retrospective: {action}; "
            f"current_graph_runtime_labels_complete=false; "
            f"provider_protocol_permitted=false; result_id={result['result_id']}"
        )
        return 0
    except (RetrospectiveError, group_plans.GroupPlanError, OSError,
            json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        print(f"minimod-route-only-retrospective: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
