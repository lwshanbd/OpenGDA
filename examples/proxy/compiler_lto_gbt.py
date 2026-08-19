#!/usr/bin/env python3
"""Emit a source-free GBT response for the frozen compiler LTO dossier.

The model is trained only on the hash-bound historical ctx_bench grid named
by the deployment profile.  It never reads application source, frozen runtime
logs, or oracle labels.  Only axes with compatible units are transferred:
message bytes, logical operation count, trigger batch size, proxy producer
count, and proxy worker count.  The historical D axis is deliberately held at
zero because it is measured in microseconds while LTO reports instruction/
FLOP distance; inventing a conversion would leak an unmeasured cost model.

Two policies are useful:

  history       train on every historical message size
  size-holdout  exclude every message size present in the frozen dossier

Both emit ordinary gicc-llm-decision-v1 JSON.  The same strict bridge and LTO
lowering used for an LLM candidate consume the response.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = ROOT / "tools" / "gicc-passes" / "python"
sys.path.insert(0, str(BRIDGE_DIR))

import gicc_llm_bridge as bridge  # noqa: E402


MODEL_SCHEMA = "gicc-compiler-lto-gbt-v1"
MODEL_FEATURES = (
    "log2(size_bytes)",
    "log2(logical_ops)",
    "log2(total_bytes)",
    "is_proxy",
    "log2(trigger_batch) or -1",
    "log2(proxy_producers) or -1",
    "log2(proxy_workers) or -1",
)


class ModelError(ValueError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def require_positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{label} must be a positive integer, got {value!r}")
    return value


def model_features(size_bytes: int, logical_ops: int, action: str,
                   trigger_batch: int, proxy_producers: int,
                   proxy_workers: int) -> list[float]:
    def optional_log2(value: int) -> float:
        return math.log2(value) if value > 0 else -1.0

    return [
        math.log2(size_bytes),
        math.log2(logical_ops),
        math.log2(size_bytes * logical_ops),
        1.0 if action == "proxy" else 0.0,
        optional_log2(trigger_batch),
        optional_log2(proxy_producers),
        optional_log2(proxy_workers),
    ]


def load_training_rows(path: Path, excluded_sizes: set[int]) \
        -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    spin: dict[int, float] = {}
    raw: list[dict[str, str]] = []
    try:
        with path.open(newline="") as source:
            for row in csv.DictReader(source):
                # The concatenated evidence contains repeated CSV headers.
                if not row or row.get("path") == "path":
                    continue
                if row.get("config") == "spin-only":
                    distance = int(row["D"])
                    median = float(row["median_us"])
                    spin[distance] = min(spin.get(distance, median), median)
                    continue
                raw.append(row)
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ModelError(f"cannot parse historical grid {path}: {exc}") from exc

    x: list[list[float]] = []
    y: list[float] = []
    cells: set[tuple[int, int, str]] = set()
    sizes: set[int] = set()
    for row in raw:
        distance = int(row["D"])
        if distance != 0:
            continue
        size_bytes = int(row["bytes"])
        if size_bytes in excluded_sizes:
            continue
        logical_ops = int(row["K"])
        action = row["path"]
        if action not in {"proxy", "trigger"}:
            raise ModelError(f"unknown historical action {action!r}")
        trigger_batch = int(row["B"]) if action == "trigger" else 0
        proxy_producers = int(row["P"]) if action == "proxy" else 0
        proxy_workers = int(row["L"]) if action == "proxy" else 0
        exposed_us = max(float(row["median_us"]) - spin.get(0, 0.0), 0.01)
        x.append(model_features(
            size_bytes, logical_ops, action, trigger_batch,
            proxy_producers, proxy_workers,
        ))
        y.append(math.log(exposed_us))
        cells.add((size_bytes, logical_ops, row["config"]))
        sizes.add(size_bytes)

    if not x:
        raise ModelError("training set is empty after leakage exclusions")
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float), {
        "rows": len(x),
        "unique_cells": len(cells),
        "message_sizes": sorted(sizes),
        "distance_us": 0,
        "target": "log(max(median_us - min_spin0_us, 0.01))",
        "min_spin0_us": spin.get(0, 0.0),
    }


def group_sites(dossier: dict[str, Any]) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for site in dossier["sites"]:
        kernel = site.get("kernel")
        if not isinstance(kernel, str) or not kernel:
            raise ModelError(f"site {site.get('site_id')!r} lacks a kernel")
        grouped[kernel].append(site)
    return [sorted(sites, key=lambda site: site["site_id"])
            for _, sites in sorted(grouped.items())]


def describe_group(sites: list[dict[str, Any]], proxy_workers: int) \
        -> dict[str, Any]:
    kernel = sites[0]["kernel"]
    sizes = {site.get("size_bytes") for site in sites}
    batches = {site.get("batch_size") for site in sites}
    grids = {site.get("grid_blocks") for site in sites}
    loops = {site.get("in_loop") for site in sites}
    if len(sizes) != 1 or len(batches) != 1 or len(grids) != 1:
        raise ModelError(f"{kernel}: inconsistent group facts")
    size_bytes = require_positive_int(next(iter(sizes)), f"{kernel}.size_bytes")
    logical_ops = require_positive_int(next(iter(batches)), f"{kernel}.batch_size")
    grid_blocks = require_positive_int(next(iter(grids)), f"{kernel}.grid_blocks")
    if len(loops) != 1:
        raise ModelError(f"{kernel}: inconsistent loop facts")
    in_loop = next(iter(loops)) is True

    # A modeled loop is one compiler operation and therefore has one proxy
    # producer. Independent static sites can occupy separate blocks/rings.
    independent_sites = 1 if in_loop else len(sites)
    proxy_producers = min(independent_sites, grid_blocks, proxy_workers)
    legal = set.intersection(*(set(site["legal_actions"]) for site in sites))
    return {
        "kernel": kernel,
        "site_ids": [site["site_id"] for site in sites],
        "size_bytes": size_bytes,
        "logical_ops": logical_ops,
        "grid_blocks": grid_blocks,
        "in_loop": in_loop,
        "proxy_producers": proxy_producers,
        "proxy_workers": proxy_workers,
        "legal_actions": sorted(legal),
        "compiler_facts_not_mapped_to_historical_grid": {
            "descriptor_reusable": sorted({site.get("descriptor_reusable")
                                             for site in sites}),
            "coalescable": sorted({site.get("coalescable") for site in sites}),
            "flops_to_first_use": sorted({site.get("flops_to_first_use")
                                           for site in sites}),
        },
    }


def make_response(dossier: dict[str, Any], model: GradientBoostingRegressor,
                  proxy_workers: int, policy: str, dataset_hash: str,
                  excluded_sizes: set[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    decisions: dict[str, Any] = {}
    predictions: list[dict[str, Any]] = []
    for sites in group_sites(dossier):
        group = describe_group(sites, proxy_workers)
        legal_physical = [action for action in ("proxy", "trigger")
                          if action in group["legal_actions"]]
        if not legal_physical:
            raise ModelError(f"{group['kernel']}: no modeled physical action")

        predicted: dict[str, float] = {}
        for action in legal_physical:
            candidate = model_features(
                group["size_bytes"], group["logical_ops"], action,
                group["logical_ops"] if action == "trigger" else 0,
                group["proxy_producers"] if action == "proxy" else 0,
                group["proxy_workers"] if action == "proxy" else 0,
            )
            predicted[action] = float(math.exp(model.predict(
                np.asarray([candidate], dtype=float))[0]))

        winner = min(predicted, key=predicted.get)
        if len(predicted) == 1:
            confidence = 1.0
        else:
            ordered = sorted(predicted.values())
            margin = abs(math.log(ordered[1] / ordered[0]))
            confidence = 1.0 / (1.0 + math.exp(-margin))
        rationale = (
            f"GBT {policy}; historical-grid prediction "
            + ", ".join(f"{name}={value:.3f}us"
                        for name, value in sorted(predicted.items()))
        )
        for site in sites:
            decisions[site["site_id"]] = {
                "action": winner,
                "confidence": round(confidence, 6),
                "rationale": rationale,
            }
        predictions.append({
            **group,
            "predicted_exposed_us": predicted,
            "chosen_action": winner,
            "confidence": confidence,
        })

    response = {
        "schema_version": bridge.DECISION_SCHEMA,
        "dossier_id": dossier["dossier_id"],
        "producer": {
            "kind": "gradient_boosted_regression",
            "schema_version": MODEL_SCHEMA,
            "policy": policy,
            "random_state": 0,
            "n_estimators": 200,
            "max_depth": 3,
            "historical_dataset_sha256": dataset_hash,
            "excluded_message_sizes": sorted(excluded_sizes),
        },
        "decisions": decisions,
    }
    hint, accepted, errors = bridge.decision_to_hint(dossier, response)
    if not accepted or errors:
        raise ModelError(f"strict bridge rejected generated response: {errors}")
    report = {
        "schema_version": MODEL_SCHEMA,
        "policy": policy,
        "dossier_id": dossier["dossier_id"],
        "historical_dataset_sha256": dataset_hash,
        "excluded_message_sizes": sorted(excluded_sizes),
        "model_features": list(MODEL_FEATURES),
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "distance_policy": (
            "train only historical D=0; do not convert compiler FLOPs to us"
        ),
        "oracle_or_runtime_results_read": False,
        "source_read": False,
        "response_sha256_canonical": hashlib.sha256(
            json.dumps(response, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")
        ).hexdigest(),
        "predictions": predictions,
        "accepted_hint_sha256": hashlib.sha256(
            json.dumps(hint, sort_keys=True, separators=(",", ":"))
            .encode("utf-8")
        ).hexdigest(),
    }
    return response, report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dossier", required=True, type=Path)
    ap.add_argument("--grid", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--policy", choices=("history", "size-holdout"),
                    default="history")
    args = ap.parse_args()

    try:
        dossier = bridge._verified_dossier(read_json(args.dossier))
        profile = dossier.get("platform_profile", {})
        provenance = profile.get("provenance", {})
        expected_hash = provenance.get("dataset_sha256")
        actual_hash = sha256_file(args.grid)
        if not isinstance(expected_hash, str) or actual_hash != expected_hash:
            raise ModelError(
                f"historical dataset hash mismatch: {actual_hash} != {expected_hash}"
            )
        deployment = profile.get("deployment_constraints", {})
        proxy_workers = require_positive_int(
            deployment.get("proxy_worker_lanes"), "proxy_worker_lanes"
        )
        frozen_sizes = {
            require_positive_int(site.get("size_bytes"), "site.size_bytes")
            for site in dossier["sites"]
        }
        excluded_sizes = frozen_sizes if args.policy == "size-holdout" else set()
        x, y, training = load_training_rows(args.grid, excluded_sizes)
        model = GradientBoostingRegressor(
            random_state=0, n_estimators=200, max_depth=3,
        ).fit(x, y)
        response, report = make_response(
            dossier, model, proxy_workers, args.policy,
            actual_hash, excluded_sizes,
        )
        report["training"] = training
        write_json(args.output, response)
        write_json(args.report, report)
        print(json.dumps({
            "policy": args.policy,
            "training_rows": training["rows"],
            "excluded_message_sizes": sorted(excluded_sizes),
            "choices": {
                row["kernel"]: row["chosen_action"]
                for row in report["predictions"]
            },
            "output": str(args.output),
            "report": str(args.report),
        }, indent=2, sort_keys=True))
        return 0
    except (ModelError, bridge.BridgeError, OSError, ValueError) as exc:
        print(f"compiler-lto-gbt: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
