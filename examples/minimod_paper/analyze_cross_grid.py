#!/usr/bin/env python3
"""Evaluate whether a Minimod policy learned at one grid transfers to another."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor

from analyze import STANDARD_ARMS, cell_feature, geometric_mean


def parse_analysis(value: str) -> tuple[int, Path]:
    try:
        grid_text, path_text = value.split("=", 1)
        grid = int(grid_text)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("use GRID=ANALYSIS_DIR") from exc
    if grid < 1 or not path_text:
        raise argparse.ArgumentTypeError("use a positive GRID and directory")
    return grid, Path(path_text)


def load_analysis(grid: int, directory: Path):
    medians = {}
    with (directory / "cell-medians.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            cell = (int(row["nodes"]), int(row["rpn"]))
            medians[(cell, row["arm"])] = float(row["kernel_median_s"])

    samples = {}
    with (directory / "measurements.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            if row["kind"] != "standard":
                continue
            if int(row["grid"]) != grid:
                raise ValueError(
                    f"{directory}: measurement grid {row['grid']} != {grid}"
                )
            key = (
                (int(row["nodes"]), int(row["rpn"])),
                int(row["rep"]), row["arm"],
            )
            if key in samples:
                raise ValueError(f"{directory}: duplicate measurement {key}")
            samples[key] = float(row["kernel_s"])
    return medians, samples


def paired_speedup(samples, candidate, reference, nboot=10000):
    ratios = []
    for cell in sorted(candidate):
        for rep in range(1, 6):
            ratios.append(
                samples[(cell, rep, reference[cell])] /
                samples[(cell, rep, candidate[cell])]
            )
    logs = np.log(np.asarray(ratios))
    rng = np.random.default_rng(0)
    draws = np.exp(
        rng.choice(logs, size=(nboot, len(logs)), replace=True).mean(axis=1)
    )
    return {
        "gmean": float(np.exp(logs.mean())),
        "bootstrap95": [
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)),
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--analysis", action="append", required=True, type=parse_analysis,
        help="validated within-grid analysis as GRID=ANALYSIS_DIR",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    paths = dict(args.analysis)
    if len(paths) < 2 or len(paths) != len(args.analysis):
        raise SystemExit("provide at least two distinct grid analyses")

    medians, samples = {}, {}
    for grid, directory in sorted(paths.items()):
        medians[grid], samples[grid] = load_analysis(grid, directory)

    all_cells = [
        {cell for cell, _ in values if cell[0] * cell[1] > 1}
        for values in medians.values()
    ]
    if any(cells != all_cells[0] for cells in all_cells[1:]):
        raise ValueError("grid analyses do not contain the same policy cells")
    cells = sorted(all_cells[0])
    for grid in sorted(paths):
        for cell in cells:
            for arm in STANDARD_ARMS:
                if (cell, arm) not in medians[grid]:
                    raise ValueError(f"grid {grid}: missing {(cell, arm)}")
                for rep in range(1, 6):
                    if (cell, rep, arm) not in samples[grid]:
                        raise ValueError(
                            f"grid {grid}: missing {(cell, rep, arm)}"
                        )

    report = {"schema_version": 1, "grids": sorted(paths), "transfers": {}}
    for heldout in sorted(paths):
        train_grids = [grid for grid in sorted(paths) if grid != heldout]
        x, y = [], []
        for grid in train_grids:
            for cell in cells:
                for arm in STANDARD_ARMS:
                    x.append(cell_feature(cell, arm, grid))
                    y.append(math.log(medians[grid][(cell, arm)]))
        model = GradientBoostingRegressor(
            random_state=0, n_estimators=200, max_depth=2,
            learning_rate=0.04, loss="huber",
        ).fit(np.asarray(x), np.asarray(y))

        predictions, picks, oracle = {}, {}, {}
        for cell in cells:
            pred = {
                arm: float(math.exp(model.predict(np.asarray([
                    cell_feature(cell, arm, heldout)
                ]))[0]))
                for arm in STANDARD_ARMS
            }
            predictions[cell] = pred
            picks[cell] = min(pred, key=pred.get)
            oracle[cell] = min(
                STANDARD_ARMS,
                key=lambda arm: medians[heldout][(cell, arm)],
            )

        default = {cell: "default_serial" for cell in cells}
        hand = {
            cell: "trigger_overlap" if cell[1] == 1 else "default_overlap"
            for cell in cells
        }
        best_global_arm = min(
            STANDARD_ARMS,
            key=lambda arm: geometric_mean(
                medians[heldout][(cell, arm)] /
                medians[heldout][(cell, oracle[cell])]
                for cell in cells
            ),
        )
        best_global = {cell: best_global_arm for cell in cells}
        regret = geometric_mean(
            medians[heldout][(cell, picks[cell])] /
            medians[heldout][(cell, oracle[cell])]
            for cell in cells
        )
        key = f"train-{'-'.join(map(str, train_grids))}-test-{heldout}"
        report["transfers"][key] = {
            "train_grids": train_grids,
            "test_grid": heldout,
            "oracle_regret_gmean": regret,
            "oracle_matches": sum(picks[cell] == oracle[cell] for cell in cells),
            "policy_cells": len(cells),
            "best_global_action": best_global_arm,
            "picks": {f"n{c[0]}-rpn{c[1]}": picks[c] for c in cells},
            "oracle_picks": {f"n{c[0]}-rpn{c[1]}": oracle[c] for c in cells},
            "speedup_over_default": paired_speedup(
                samples[heldout], picks, default
            ),
            "speedup_over_best_global": paired_speedup(
                samples[heldout], picks, best_global
            ),
            "speedup_over_hand_rule": paired_speedup(
                samples[heldout], picks, hand
            ),
            "predicted_seconds": {
                f"n{c[0]}-rpn{c[1]}": predictions[c] for c in cells
            },
        }

    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "cross-grid.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
