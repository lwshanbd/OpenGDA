#!/usr/bin/env python3
"""Train on compiler-path calibration labels and decide the frozen LTO test.

The model reads two compiler dossiers and the calibration result summary.  It
does not accept source, historical runtime-grid data, frozen test logs, or
frozen oracle labels.  Exact frozen message sizes are rejected if they appear
in the calibration dossier.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import GradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gicc-passes" / "python"))

import gicc_llm_bridge as bridge  # noqa: E402


MODEL_SCHEMA = "gicc-compiler-path-calibrated-gbt-v2"
MODEL_FEATURES = (
    "log2(size_bytes)",
    "log2(batch_size)",
    "log2(size_bytes * batch_size)",
    "log2(grid_blocks)",
    "log2(site_count)",
    "in_loop",
    "descriptor_reusable",
    "coalescable",
    "log2(1 + flops_to_first_use)",
    "distance_exact",
    "is_proxy",
)


class ModelError(ValueError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelError(f"cannot read JSON {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ModelError(f"{label} must be a positive integer")
    return value


def group_dossier(dossier: dict[str, Any]) \
        -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for site in dossier["sites"]:
        output.setdefault(site["kernel"], []).append(site)
    for sites in output.values():
        sites.sort(key=lambda site: site["site_id"])
    return output


def group_facts(sites: list[dict[str, Any]]) -> dict[str, Any]:
    kernel = sites[0]["kernel"]
    fields = (
        "size_bytes", "batch_size", "grid_blocks", "in_loop",
        "descriptor_reusable", "coalescable", "flops_to_first_use",
        "distance_exact",
    )
    facts: dict[str, Any] = {"kernel": kernel, "site_count": len(sites)}
    for field in fields:
        values = {site.get(field) for site in sites}
        if len(values) != 1:
            raise ModelError(f"{kernel}: group has inconsistent {field}")
        facts[field] = next(iter(values))
    positive_int(facts["size_bytes"], f"{kernel}.size_bytes")
    positive_int(facts["batch_size"], f"{kernel}.batch_size")
    positive_int(facts["grid_blocks"], f"{kernel}.grid_blocks")
    if not isinstance(facts["flops_to_first_use"], int) or facts["flops_to_first_use"] < 0:
        raise ModelError(f"{kernel}: invalid flops_to_first_use")
    facts["legal_actions"] = sorted(set.intersection(*(
        set(site["legal_actions"]) for site in sites
    )))
    facts["site_ids"] = [site["site_id"] for site in sites]
    return facts


def vector(facts: dict[str, Any], action: str) -> list[float]:
    size = facts["size_bytes"]
    batch = facts["batch_size"]
    return [
        math.log2(size),
        math.log2(batch),
        math.log2(size * batch),
        math.log2(facts["grid_blocks"]),
        math.log2(facts["site_count"]),
        float(facts["in_loop"] is True),
        float(facts["descriptor_reusable"] is True),
        float(facts["coalescable"] is True),
        math.log2(1 + facts["flops_to_first_use"]),
        float(facts["distance_exact"] is True),
        float(action == "proxy"),
    ]


def new_model() -> GradientBoostingRegressor:
    return GradientBoostingRegressor(
        random_state=0,
        n_estimators=200,
        max_depth=2,
        learning_rate=0.04,
        loss="huber",
    )


def training_rows(calibration_dossier: dict[str, Any], results: dict[str, Any]) \
        -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, list[str]]:
    if results.get("schema_version") != "gicc-compiler-lto-calibration-results-v1":
        raise ModelError("wrong calibration result schema")
    if results.get("dossier_id") != calibration_dossier["dossier_id"]:
        raise ModelError("calibration result/dossier ID mismatch")
    if results.get("frozen_evaluation_results_read") is not False:
        raise ModelError("calibration provenance does not exclude frozen results")
    aggregate = results.get("aggregate")
    if not isinstance(aggregate, dict):
        raise ModelError("calibration result lacks aggregate labels")

    grouped = group_dossier(calibration_dossier)
    rows: list[dict[str, Any]] = []
    x: list[list[float]] = []
    y: list[float] = []
    seen_kernels: set[str] = set()
    excluded_unstable: list[str] = []
    for scenario, label in sorted(aggregate.items()):
        if not isinstance(label, dict):
            raise ModelError(f"{scenario}: malformed calibration label")
        kernel = label.get("kernel")
        if kernel not in grouped or kernel in seen_kernels:
            raise ModelError(f"{scenario}: invalid/duplicate kernel {kernel!r}")
        seen_kernels.add(kernel)
        if label.get("stable_winner") is not True:
            excluded_unstable.append(scenario)
            continue
        facts = group_facts(grouped[kernel])
        medians = label.get("geomean_median_us")
        if not isinstance(medians, dict) or set(medians) != {"proxy", "trigger"}:
            raise ModelError(f"{scenario}: labels must cover proxy and trigger")
        for action in ("proxy", "trigger"):
            cost = medians[action]
            if not isinstance(cost, (int, float)) or cost <= 0:
                raise ModelError(f"{scenario}: invalid {action} cost")
            row = {"scenario": scenario, "kernel": kernel, "action": action,
                   "cost_us": float(cost), "facts": facts}
            rows.append(row)
            x.append(vector(facts, action))
            y.append(math.log(float(cost)))
    if seen_kernels != set(grouped):
        raise ModelError("calibration labels do not cover every dossier kernel")
    if not rows:
        raise ModelError("no stable calibration scenarios remain")
    return (
        rows,
        np.asarray(x, dtype=float),
        np.asarray(y, dtype=float),
        excluded_unstable,
    )


def leave_one_scenario_out(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scenarios = sorted({row["scenario"] for row in rows})
    predictions: dict[str, Any] = {}
    regrets: list[float] = []
    matches = 0
    for heldout in scenarios:
        train = [row for row in rows if row["scenario"] != heldout]
        test = [row for row in rows if row["scenario"] == heldout]
        model = new_model().fit(
            np.asarray([vector(row["facts"], row["action"]) for row in train]),
            np.asarray([math.log(row["cost_us"]) for row in train]),
        )
        predicted = {
            row["action"]: float(math.exp(model.predict(np.asarray([
                vector(row["facts"], row["action"])
            ]))[0]))
            for row in test
        }
        measured = {row["action"]: row["cost_us"] for row in test}
        chosen = min(predicted, key=predicted.get)
        oracle = min(measured, key=measured.get)
        regret = measured[chosen] / measured[oracle]
        regrets.append(regret)
        matches += int(chosen == oracle)
        predictions[heldout] = {
            "predicted_us": predicted,
            "measured_us": measured,
            "chosen_action": chosen,
            "oracle_action": oracle,
            "regret": regret,
        }
    return {
        "protocol": "leave one complete scenario (both action labels) out",
        "scenario_count": len(scenarios),
        "action_matches": matches,
        "geomean_regret": math.exp(sum(math.log(x) for x in regrets) / len(regrets)),
        "predictions": predictions,
    }


def make_eval_response(eval_dossier: dict[str, Any], model: Any,
                       provenance: dict[str, Any]) \
        -> tuple[dict[str, Any], list[dict[str, Any]]]:
    decisions: dict[str, Any] = {}
    predictions: list[dict[str, Any]] = []
    for kernel, sites in sorted(group_dossier(eval_dossier).items()):
        facts = group_facts(sites)
        legal = [action for action in ("proxy", "trigger")
                 if action in facts["legal_actions"]]
        if not legal:
            raise ModelError(f"{kernel}: no calibrated physical action")
        predicted = {
            action: float(math.exp(model.predict(np.asarray([
                vector(facts, action)
            ]))[0]))
            for action in legal
        }
        chosen = min(predicted, key=predicted.get)
        confidence = 1.0 if len(predicted) == 1 else 1.0 / (
            1.0 + math.exp(-abs(math.log(max(predicted.values()) /
                                       min(predicted.values()))))
        )
        rationale = (
            "compiler-path calibrated GBT; "
            + ", ".join(f"{key}={value:.3f}us"
                        for key, value in sorted(predicted.items()))
        )
        for site in sites:
            decisions[site["site_id"]] = {
                "action": chosen,
                "confidence": round(confidence, 6),
                "rationale": rationale,
            }
        predictions.append({
            "kernel": kernel,
            "facts": facts,
            "predicted_us": predicted,
            "chosen_action": chosen,
            "confidence": confidence,
        })
    response = {
        "schema_version": bridge.DECISION_SCHEMA,
        "dossier_id": eval_dossier["dossier_id"],
        "producer": provenance,
        "decisions": decisions,
    }
    _, accepted, errors = bridge.decision_to_hint(eval_dossier, response)
    if not accepted or errors:
        raise ModelError(f"strict bridge rejected calibrated response: {errors}")
    return response, predictions


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--calibration-dossier", required=True, type=Path)
    ap.add_argument("--calibration-results", required=True, type=Path)
    ap.add_argument("--evaluation-dossier", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    args = ap.parse_args()
    try:
        cal = bridge._verified_dossier(read_json(args.calibration_dossier))
        results = read_json(args.calibration_results)
        evaluation = bridge._verified_dossier(read_json(args.evaluation_dossier))
        cal_sizes = {site["size_bytes"] for site in cal["sites"]}
        eval_sizes = {site["size_bytes"] for site in evaluation["sites"]}
        overlap = cal_sizes & eval_sizes
        if overlap:
            raise ModelError(f"calibration/evaluation sizes overlap: {sorted(overlap)}")

        rows, x, y, excluded_unstable = training_rows(cal, results)
        validation = leave_one_scenario_out(rows)
        model = new_model().fit(x, y)
        provenance = {
            "kind": "compiler_path_calibrated_gradient_boosting",
            "schema_version": MODEL_SCHEMA,
            "random_state": 0,
            "n_estimators": 200,
            "max_depth": 2,
            "learning_rate": 0.04,
            "loss": "huber",
            "calibration_dossier_id": cal["dossier_id"],
            "calibration_dossier_sha256": sha256_file(args.calibration_dossier),
            "calibration_results_sha256": sha256_file(args.calibration_results),
            "source_read": False,
            "historical_runtime_grid_read": False,
            "frozen_evaluation_results_read": False,
            "unstable_calibration_scenarios_excluded": excluded_unstable,
        }
        response, predictions = make_eval_response(evaluation, model, provenance)
        report = {
            "schema_version": MODEL_SCHEMA,
            "model_features": list(MODEL_FEATURES),
            "numpy_version": np.__version__,
            "scikit_learn_version": sklearn.__version__,
            "training_scenarios": len(rows) // 2,
            "training_rows": len(rows),
            "unstable_calibration_scenarios_excluded": excluded_unstable,
            "calibration_message_sizes": sorted(cal_sizes),
            "evaluation_message_sizes": sorted(eval_sizes),
            "message_size_overlap": [],
            "provenance": provenance,
            "calibration_validation": validation,
            "evaluation_predictions": predictions,
            "response_sha256_canonical": hashlib.sha256(
                json.dumps(response, sort_keys=True, separators=(",", ":"))
                .encode("utf-8")
            ).hexdigest(),
        }
        write_json(args.output, response)
        write_json(args.report, report)
        print(json.dumps({
            "training_scenarios": len(rows) // 2,
            "excluded_unstable": excluded_unstable,
            "loso_matches": validation["action_matches"],
            "loso_regret": validation["geomean_regret"],
            "evaluation_choices": {
                row["kernel"]: row["chosen_action"] for row in predictions
            },
            "output": str(args.output),
        }, indent=2, sort_keys=True))
        return 0
    except (ModelError, bridge.BridgeError, OSError, ValueError) as exc:
        print(f"compiler-lto-calibrated-gbt: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
