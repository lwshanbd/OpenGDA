#!/usr/bin/env python3
"""Validate Minimod logs and score the paper policies.

The parser is intentionally strict: one value per rank, exact rank sets,
exact compiler-route counters, and identical full-domain checksums across all
arms and allocation-level replicates of a cell.  The primary application time
is the maximum rank time from each allocation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import geometric_mean, median

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


STANDARD_ARMS = (
    "default_serial",
    "default_overlap",
    "trigger_serial",
    "trigger_overlap",
    "proxy_serial",
    "proxy_overlap",
)
O5_ARMS = ("mirrored_serial", "unmirrored_serial")
FIXED_ARMS = (
    "trigger_serial",
    "trigger_overlap",
    "proxy_serial",
    "proxy_overlap",
)

RUN_RE = re.compile(
    r"^RUN kind=(\S+) arm=(\S+) nodes=(\d+) rpn=(\d+) ranks=(\d+) "
    r"rep=(\d+) grid=(\d+) steps=(\d+) binary=(\S+)$", re.M)
CHECK_RE = re.compile(r"^CHECKSUM rank (\d+) = ([0-9a-f]+)$", re.M)
ROUTE_RE = re.compile(
    r"^GICC_ROUTE rank (\d+) staged=(\d+) pushed=(\d+)$", re.M)
TIME_RE = re.compile(
    r"^I am rank (\d+) Time (comm|comp|kernel): ([0-9.eE+-]+) s$", re.M)


@dataclass(frozen=True)
class Sample:
    kind: str
    arm: str
    nodes: int
    rpn: int
    ranks: int
    rep: int
    grid: int
    steps: int
    kernel_s: float
    comm_s: float
    comp_s: float
    checksum_set: str
    staged: int
    pushed: int
    path: str


def exact_rank_map(matches, ranks: int, label: str, path: Path):
    out = {}
    for rank, *vals in matches:
        rank = int(rank)
        if rank in out:
            raise ValueError(f"{path}: duplicate {label} for rank {rank}")
        out[rank] = vals[0] if len(vals) == 1 else tuple(vals)
    if set(out) != set(range(ranks)):
        raise ValueError(
            f"{path}: {label} ranks {sorted(out)} != 0..{ranks - 1}")
    return out


def active_neighbors(rank: int, ranks: int) -> int:
    return int(rank > 0) + int(rank + 1 < ranks)


def cross_neighbors(rank: int, ranks: int, rpn: int) -> int:
    node = rank // rpn
    left = int(rank > 0 and (rank - 1) // rpn != node)
    right = int(rank + 1 < ranks and (rank + 1) // rpn != node)
    return left + right


def validate_route(sample_meta, routes, path: Path) -> None:
    arm = sample_meta["arm"]
    ranks = sample_meta["ranks"]
    rpn = sample_meta["rpn"]
    steps = sample_meta["steps"]
    for rank, (staged_s, pushed_s) in routes.items():
        staged, pushed = int(staged_s), int(pushed_s)
        active = active_neighbors(rank, ranks) * steps
        cross = cross_neighbors(rank, ranks, rpn) * steps
        if arm.startswith("proxy_") or arm == "unmirrored_serial":
            expected = (0, active)
        elif arm.startswith("trigger_") or arm == "mirrored_serial":
            expected = (active, 0)
        elif arm.startswith("default_"):
            expected = (cross, 0)
        else:
            raise ValueError(f"{path}: unknown arm {arm}")
        if (staged, pushed) != expected:
            raise ValueError(
                f"{path}: rank {rank} route {(staged, pushed)} != {expected}")


def parse_log(path: Path) -> Sample:
    text = path.read_text(errors="replace")
    run = RUN_RE.findall(text)
    if len(run) != 1:
        raise ValueError(f"{path}: expected one RUN line, got {len(run)}")
    kind, arm, nodes, rpn, ranks, rep, grid, steps, _binary = run[0]
    meta = {
        "kind": kind,
        "arm": arm,
        "nodes": int(nodes),
        "rpn": int(rpn),
        "ranks": int(ranks),
        "rep": int(rep),
        "grid": int(grid),
        "steps": int(steps),
    }
    if meta["ranks"] != meta["nodes"] * meta["rpn"]:
        raise ValueError(f"{path}: ranks != nodes*rpn")

    checks = exact_rank_map(CHECK_RE.findall(text), meta["ranks"], "checksum", path)
    routes = exact_rank_map(ROUTE_RE.findall(text), meta["ranks"], "route", path)
    validate_route(meta, routes, path)

    times = defaultdict(dict)
    for rank_s, field, value_s in TIME_RE.findall(text):
        rank = int(rank_s)
        if field in times[rank]:
            raise ValueError(f"{path}: duplicate Time {field} rank {rank}")
        times[rank][field] = float(value_s)
    if set(times) != set(range(meta["ranks"])):
        raise ValueError(f"{path}: incomplete time rank set")
    for rank, row in times.items():
        if set(row) != {"comm", "comp", "kernel"}:
            raise ValueError(f"{path}: incomplete times for rank {rank}: {row}")

    checksum_set = ";".join(f"{r}:{checks[r]}" for r in sorted(checks))
    return Sample(
        **meta,
        kernel_s=max(row["kernel"] for row in times.values()),
        comm_s=max(row["comm"] for row in times.values()),
        comp_s=max(row["comp"] for row in times.values()),
        checksum_set=checksum_set,
        staged=sum(int(v[0]) for v in routes.values()),
        pushed=sum(int(v[1]) for v in routes.values()),
        path=str(path),
    )


def load_dirs(dirs: list[Path], allow_incomplete: bool) -> tuple[list[Sample], list[str]]:
    samples, errors = [], []
    seen = set()
    for directory in dirs:
        for path in sorted((directory / "raw").glob("*.log")):
            try:
                sample = parse_log(path)
            except ValueError as exc:
                if not allow_incomplete:
                    raise
                errors.append(str(exc))
                continue
            key = (sample.kind, sample.nodes, sample.rpn, sample.rep, sample.arm)
            if key in seen:
                raise ValueError(f"duplicate sample key {key}")
            seen.add(key)
            samples.append(sample)
    return samples, errors


def validate_checksums(samples: list[Sample]) -> None:
    by_cell = defaultdict(set)
    for s in samples:
        by_cell[(s.kind, s.nodes, s.rpn, s.grid, s.steps)].add(s.checksum_set)
    bad = {k: v for k, v in by_cell.items() if len(v) != 1}
    if bad:
        raise ValueError(f"checksum mismatch by cell: {bad}")


def validate_full(samples: list[Sample], kind: str,
                  standard_nodes=(1, 2, 4, 8)) -> None:
    arms = STANDARD_ARMS if kind == "standard" else O5_ARMS
    expected_cells = (
        [(n, rpn) for n in standard_nodes for rpn in (1, 8)]
        if kind == "standard" else [(2, 1)]
    )
    reps = range(1, 6)
    found = {(s.nodes, s.rpn, s.rep, s.arm) for s in samples if s.kind == kind}
    missing = [
        (n, rpn, rep, arm)
        for n, rpn in expected_cells
        for rep in reps
        for arm in arms
        if (n, rpn, rep, arm) not in found
    ]
    if missing:
        raise ValueError(f"{kind}: missing {len(missing)} samples; first={missing[:8]}")


def quantiles(values):
    a = np.asarray(list(values), dtype=float)
    return float(np.median(a)), float(np.percentile(a, 25)), float(np.percentile(a, 75))


def cell_feature(cell, arm: str, grid: int):
    nodes, rpn = cell
    ranks = nodes * rpn
    local_x = grid / ranks
    local_edges = nodes * max(rpn - 1, 0)
    total_edges = max(ranks - 1, 1)
    local_frac = local_edges / total_edges if ranks > 1 else 0.0
    dispatch = arm.split("_", 1)[0]
    overlap = float(arm.endswith("_overlap"))
    return [
        math.log2(nodes),
        math.log2(rpn),
        math.log2(ranks),
        math.log2(max(local_x, 1.0)),
        local_frac,
        1.0 - local_frac if ranks > 1 else 0.0,
        float(ranks > 1),
        overlap,
        float(dispatch == "default"),
        float(dispatch == "trigger"),
        float(dispatch == "proxy"),
    ]


def gmean_ratio(picks, medians, oracle):
    return geometric_mean(medians[cell][arm] / oracle[cell] for cell, arm in picks.items())


def bootstrap_ratio(samples, picks, baseline="default_serial", nboot=10000):
    by_key = {(s.nodes, s.rpn, s.rep, s.arm): s.kernel_s for s in samples}
    ratios = []
    for cell, arm in picks.items():
        for rep in range(1, 6):
            ratios.append(by_key[(cell[0], cell[1], rep, baseline)] /
                          by_key[(cell[0], cell[1], rep, arm)])
    arr = np.log(np.asarray(ratios))
    rng = np.random.default_rng(0)
    draws = np.exp(rng.choice(arr, size=(nboot, len(arr)), replace=True).mean(axis=1))
    return float(np.exp(arr.mean())), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def bootstrap_policy_speedup(samples, candidate, reference, nboot=10000):
    """Paired speedup of candidate over reference across cells and reps."""
    by_key = {(s.nodes, s.rpn, s.rep, s.arm): s.kernel_s for s in samples}
    ratios = []
    for cell, candidate_arm in candidate.items():
        reference_arm = reference[cell]
        for rep in range(1, 6):
            ratios.append(
                by_key[(cell[0], cell[1], rep, reference_arm)] /
                by_key[(cell[0], cell[1], rep, candidate_arm)]
            )
    arr = np.log(np.asarray(ratios))
    rng = np.random.default_rng(0)
    draws = np.exp(rng.choice(arr, size=(nboot, len(arr)), replace=True).mean(axis=1))
    return float(np.exp(arr.mean())), float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def analyze_standard(samples: list[Sample], out: Path):
    std = [s for s in samples if s.kind == "standard"]
    grouped = defaultdict(list)
    for s in std:
        grouped[(s.nodes, s.rpn, s.arm)].append(s)
    cells = sorted({(s.nodes, s.rpn) for s in std})
    medians = {cell: {} for cell in cells}
    rows = []
    for cell in cells:
        for arm in STANDARD_ARMS:
            arm_samples = grouped[(cell[0], cell[1], arm)]
            if len(arm_samples) != 5:
                raise ValueError(
                    f"cell {cell} arm {arm}: expected 5 reps, "
                    f"got {len(arm_samples)}")
            kernel = quantiles(s.kernel_s for s in arm_samples)
            comm = quantiles(s.comm_s for s in arm_samples)
            comp = quantiles(s.comp_s for s in arm_samples)
            medians[cell][arm] = kernel[0]
            rows.append((cell[0], cell[1], arm,
                         *kernel, *comm, *comp))
    # n1-rpn1 is retained in the scaling table as a no-communication control,
    # but it cannot inform a communication policy.  Excluding it from policy
    # selection/scoring prevents random compute jitter from choosing a path.
    policy_cells = [cell for cell in cells if cell[0] * cell[1] > 1]
    oracle = {cell: min(medians[cell].values()) for cell in policy_cells}

    default = {cell: "default_serial" for cell in policy_cells}
    fixed_scores = {
        arm: geometric_mean(medians[cell][arm] / oracle[cell] for cell in policy_cells)
        for arm in FIXED_ARMS
    }
    best_fixed_arm = min(fixed_scores, key=fixed_scores.get)
    best_fixed = {cell: best_fixed_arm for cell in policy_cells}
    best_all_arm = min(
        STANDARD_ARMS,
        key=lambda arm: geometric_mean(
            medians[c][arm] / oracle[c] for c in policy_cells),
    )
    best_all = {cell: best_all_arm for cell in policy_cells}

    # Preregistered before the full run: no communication at one rank;
    # otherwise overlap, with fixed DWQ for pure cross-node and the hybrid
    # runtime locality branch when a rank may have an on-node neighbor.
    hand = {}
    for cell in policy_cells:
        nodes, rpn = cell
        ranks = nodes * rpn
        hand[cell] = (
            "default_serial" if ranks == 1 else
            "trigger_overlap" if rpn == 1 else
            "default_overlap"
        )

    # Leave one node count out.  Each prediction is scored only on a scale
    # whose six action costs were absent from training.
    gbt = {}
    gbt_pred = {}
    for heldout_nodes in sorted({c[0] for c in policy_cells}):
        train_cells = [c for c in policy_cells if c[0] != heldout_nodes]
        test_cells = [c for c in policy_cells if c[0] == heldout_nodes]
        x, y = [], []
        for cell in train_cells:
            for arm in STANDARD_ARMS:
                x.append(cell_feature(cell, arm, std[0].grid))
                y.append(math.log(medians[cell][arm]))
        model = GradientBoostingRegressor(
            random_state=0, n_estimators=200, max_depth=2,
            learning_rate=0.04, loss="huber",
        ).fit(np.asarray(x), np.asarray(y))
        for cell in test_cells:
            pred = {
                arm: float(math.exp(model.predict(
                    np.asarray([cell_feature(cell, arm, std[0].grid)]))[0]))
                for arm in STANDARD_ARMS
            }
            gbt[cell] = min(pred, key=pred.get)
            gbt_pred[f"n{cell[0]}-rpn{cell[1]}"] = pred

    oracle_picks = {
        cell: min(medians[cell], key=medians[cell].get)
        for cell in policy_cells
    }
    policies = {
        "default": default,
        "best_fixed": best_fixed,
        "best_global_all_actions": best_all,
        "hand_rule": hand,
        "gbt_lono": gbt,
        "oracle": oracle_picks,
    }
    summary = {}
    for name, picks in policies.items():
        speed, lo, hi = bootstrap_ratio(std, picks)
        summary[name] = {
            "oracle_regret_gmean": gmean_ratio(picks, medians, oracle),
            "speedup_over_default_gmean": speed,
            "speedup_bootstrap95": [lo, hi],
            "picks": {f"n{c[0]}-rpn{c[1]}": a for c, a in picks.items()},
        }
    summary["best_fixed"]["single_action"] = best_fixed_arm
    summary["best_fixed"]["candidate_regrets"] = fixed_scores
    summary["best_global_all_actions"]["single_action"] = best_all_arm
    summary["gbt_lono"]["predicted_seconds"] = gbt_pred
    summary["pairwise"] = {}
    for reference_name in (
        "best_fixed", "best_global_all_actions", "hand_rule", "oracle"
    ):
        speed, lo, hi = bootstrap_policy_speedup(
            std, gbt, policies[reference_name]
        )
        summary["pairwise"][f"gbt_lono_over_{reference_name}"] = {
            "speedup_gmean": speed,
            "speedup_bootstrap95": [lo, hi],
        }
    summary["scope"] = {
        "policy_cells": [f"n{c[0]}-rpn{c[1]}" for c in policy_cells],
        "excluded_no_communication_control": "n1-rpn1",
    }

    scaling_rows = []
    for rpn in sorted({c[1] for c in cells}):
        base = (1, rpn)
        for arm in STANDARD_ARMS:
            base_s = medians[base][arm]
            for nodes in sorted(c[0] for c in cells if c[1] == rpn):
                value = medians[(nodes, rpn)][arm]
                speedup = base_s / value
                scaling_rows.append(
                    (rpn, nodes, arm, value, speedup, speedup / nodes))

    with (out / "cell-medians.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "nodes", "rpn", "arm",
            "kernel_median_s", "kernel_q1_s", "kernel_q3_s",
            "comm_median_s", "comm_q1_s", "comm_q3_s",
            "comp_median_s", "comp_q1_s", "comp_q3_s",
        ])
        w.writerows(rows)
    with (out / "scaling.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rpn", "nodes", "arm", "kernel_median_s",
            "speedup_from_n1", "parallel_efficiency",
        ])
        w.writerows(scaling_rows)
    (out / "policies.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary, rows


def analyze_o5(samples: list[Sample]):
    o5 = [s for s in samples if s.kind == "o5"]
    by_arm = defaultdict(list)
    by_key = {}
    for s in o5:
        by_arm[s.arm].append(s.kernel_s)
        by_key[(s.rep, s.arm)] = s.kernel_s
    result = {}
    for arm in O5_ARMS:
        med, q1, q3 = quantiles(by_arm[arm])
        result[arm] = {"median_s": med, "q1_s": q1, "q3_s": q3}
    ratios = [
        by_key[(rep, "unmirrored_serial")] /
        by_key[(rep, "mirrored_serial")]
        for rep in range(1, 6)
    ]
    result["unmirrored_over_mirrored"] = {
        "gmean": geometric_mean(ratios),
        "range": [min(ratios), max(ratios)],
        "paired": ratios,
    }
    return result


def write_measurements(samples: list[Sample], out: Path) -> None:
    fields = list(asdict(samples[0]).keys())
    with (out / "measurements.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for sample in sorted(samples, key=lambda s: (s.kind, s.nodes, s.rpn, s.rep, s.arm)):
            w.writerow(asdict(sample))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--standard", type=Path, action="append", default=[])
    ap.add_argument("--o5", type=Path, action="append", default=[])
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--standard-nodes", default="1,2,4,8",
        help="comma-separated node counts required for strict standard analysis",
    )
    ap.add_argument("--allow-incomplete", action="store_true")
    args = ap.parse_args()
    try:
        standard_nodes = tuple(sorted({
            int(value) for value in args.standard_nodes.split(",") if value
        }))
    except ValueError as exc:
        raise SystemExit(f"invalid --standard-nodes: {exc}") from exc
    if not standard_nodes or any(nodes < 1 for nodes in standard_nodes):
        raise SystemExit("--standard-nodes must contain positive integers")
    dirs = args.standard + args.o5
    if not dirs:
        raise SystemExit("provide --standard and/or --o5 result directories")
    samples, errors = load_dirs(dirs, args.allow_incomplete)
    if not samples:
        raise SystemExit("no complete logs found")
    validate_checksums(samples)
    args.output.mkdir(parents=True, exist_ok=True)
    write_measurements(samples, args.output)

    report = {"samples": len(samples), "ignored_incomplete": errors}
    if args.standard and not args.allow_incomplete:
        validate_full(samples, "standard", standard_nodes)
        report["standard_nodes"] = list(standard_nodes)
        report["standard"], _ = analyze_standard(samples, args.output)
    if args.o5 and not args.allow_incomplete:
        validate_full(samples, "o5")
        report["o5"] = analyze_o5(samples)
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
