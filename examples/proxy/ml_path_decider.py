#!/usr/bin/env python3
"""Train a small GBT on measured ctx_bench cells and emit a compiler hint.

The queried application's message size (4 KiB) is excluded wholesale from
training.  The action set is deliberately restricted to the two lowerings the
end-to-end compiler can materialize for ml_path_e2e today:

  * trigger: all K descriptors share one trigger (trig-BK)
  * proxy:   eight issuing blocks and eight worker lanes (proxy-P8-L8)

features.json supplies size and the compiler-derived completion group size.
The output is ordinary gicc-hint-v1 consumed by both device and host lowering.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor


def load_cells(path: Path, heldout_bytes: int):
    rows = []
    with path.open() as f:
        for raw in csv.reader(f):
            if not raw or raw[0] != "GRID" or raw[1] == "path":
                continue
            path_name, cfg = raw[1], raw[2]
            if cfg == "spin-only":
                continue
            b, k, dist = int(raw[3]), int(raw[4]), int(raw[5])
            if b == heldout_bytes:
                continue
            wanted = (path_name == "trigger" and cfg == f"trig-B{k}") or \
                     (path_name == "proxy" and cfg == "proxy-P8-L8")
            if not wanted:
                continue
            median = float(raw[10])
            rows.append((b, k, dist, path_name, median))
    return rows


def feat(b, k, dist, path_name):
    return [math.log2(b), math.log2(k), math.log2(b * k),
            math.log1p(dist), 1.0 if path_name == "proxy" else 0.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--heldout-bytes", type=int, default=4096)
    args = ap.parse_args()

    records = json.loads(args.features.read_text())
    puts = [r for r in records if r.get("op_kind") == "put_no_db"]
    if not puts:
        raise SystemExit("no put_no_db records in features.json")
    size_logs = {r.get("size_log2") for r in puts}
    groups = {r.get("batch_size") for r in puts}
    distance_flops = {r.get("flops_to_first_use") for r in puts}
    distance_exact = {r.get("distance_exact") for r in puts}
    if len(size_logs) != 1 or None in size_logs:
        raise SystemExit(f"need one constant message size, got {size_logs}")
    if len(groups) != 1 or None in groups:
        raise SystemExit(f"need one compiler-derived completion group, got {groups}")
    if distance_exact != {True} or distance_flops != {0}:
        raise SystemExit(
            "this measured model maps only an exact zero compiler distance "
            f"to 0 us, got exact={distance_exact}, flops={distance_flops}")
    b = 1 << int(next(iter(size_logs)))
    k = int(next(iter(groups)))
    dist = 0  # exact compiler result above; no flops-to-us calibration needed

    train = load_cells(args.grid, args.heldout_bytes)
    if not train:
        raise SystemExit("empty training set")
    x = np.asarray([feat(*r[:4]) for r in train], dtype=float)
    y = np.log(np.asarray([r[4] for r in train], dtype=float))
    model = GradientBoostingRegressor(
        random_state=0, n_estimators=200, max_depth=3).fit(x, y)
    candidates = ["trigger", "proxy"]
    pred_log = model.predict(np.asarray([feat(b, k, dist, p)
                                         for p in candidates]))
    predicted = {p: float(math.exp(v)) for p, v in zip(candidates, pred_log)}
    winner = min(predicted, key=predicted.get)
    dispatch = "CPU_PROXY_ENQUEUE" if winner == "proxy" else "DWQ_TRIGGER"

    sites = {}
    for r in puts:
        if winner not in r.get("legal_paths", []):
            raise SystemExit(f"model chose illegal {winner} for {r['site_id']}")
        sites[r["site_id"]] = {
            "dispatch": dispatch,
            "reason": (f"GBT held out bytes={args.heldout_bytes}; "
                       f"pred_trigger={predicted['trigger']:.3f}us "
                       f"pred_proxy={predicted['proxy']:.3f}us"),
        }

    hint = {
        "version": 1,
        "schema_version": "gicc-hint-v1",
        "default_dispatch": "DWQ_TRIGGER",
        "sites": sites,
        "ml_metadata": {
            "model": "GradientBoostingRegressor",
            "training_rows": len(train),
            "heldout_bytes": args.heldout_bytes,
            "query": {"bytes": b, "batch_size": k, "distance_us": dist},
            "predicted_us": predicted,
            "winner": winner,
            "runtime_global": {"GICC_NUM_PROXY_THREADS": 8},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hint, indent=2) + "\n")
    print(json.dumps(hint["ml_metadata"], indent=2))


if __name__ == "__main__":
    main()
