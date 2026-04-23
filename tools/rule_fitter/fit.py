#!/usr/bin/env python3
"""
fit.py — Offline rule-list fitter for GICC-Pilot policy files.

Input: calibration JSON produced by M3 R021/R022. Schema:

    {
      "platform": "tioga",              # or "maple"
      "platform_desc": {                # passthrough to policy file
        "fabric": "ofi_cxi",
        "gpu_family": "mi250x",
        "nic_caps": {...}
      },
      "cells": [
        {
          "workload": "jacobi",
          "scale": 32,
          "features": {"peer_class": "k_from_topology_hint",
                       "size_class": "large_const",
                       "freq_class": "hot_loop"},
          "configs": [
            {"path": "ofi_triggered", "slot_depth": 4, "pool_size": 16,
             "t2s_ms": 276.2, "seeds": 5, "feasible": true},
            ...
          ]
        },
        ...
      ]
    }

Output: policy_<platform>.json with a greedy CN2-style rule list.

Algorithm (simple version, CN2 depth-1 guards):
  1. For each cell, the oracle is argmin median T2S over feasible configs.
  2. The "target label" is the config within 5 % of oracle that minimises a
     NIC-budget tiebreaker  (pool_size * R(P), then slot_depth ascending,
     then prefer ofi_proxy over ofi_triggered). This biases toward scale-
     survival as FINAL_PROPOSAL §Calibration specifies.
  3. Greedy rule induction: repeatedly pick the depth-1 guard covering the
     most uncovered cells whose target-label is feasible across every
     covered cell's scale; emit that rule; remove covered cells. Terminate
     at 20 rules or empty uncovered.
  4. Always append a catch-all conservative rule.

Usage:
    fit.py --calibration r022_tioga.json --out policies/policy_tioga.json

Exit codes:
    0  OK
    1  coverage < 95 %  (warns but still writes the policy)
    2  schema error
"""

import argparse
import json
import math
import os
import sys


# -------- NIC-cap arithmetic (mirrors libgicc_policy::check_feasible) -------

def r_of_p(P):
    return max(1, int(math.ceil(math.log2(P)))) if P > 1 else 1


def is_feasible(cfg, nic_caps, P):
    path = cfg["path"]
    r = r_of_p(P)
    if path == "ib_native":
        return True
    if path == "ofi_triggered":
        if cfg.get("slot_depth", 0) <= 0 or cfg.get("pool_size", 0) <= 0:
            return False
        if cfg["slot_depth"] * r > nic_caps["counter_max"] // nic_caps["ctrs_per_op"]:
            return False
        if cfg["pool_size"] * r > nic_caps["dwq_max"]:
            return False
        return True
    if path == "ofi_proxy":
        if cfg.get("pool_size", 0) <= 0:
            return False
        if cfg["pool_size"] * r > nic_caps["dwq_max"]:
            return False
        return True
    return False


# -------- Oracle + target-label selection -----------------------------------

def nic_budget(cfg, P):
    """Lower is better. Matches FINAL_PROPOSAL §Calibration tiebreaker."""
    r = r_of_p(P)
    return (
        cfg.get("pool_size", 0) * r,
        cfg.get("slot_depth", 0),
        0 if cfg.get("path") == "ofi_proxy" else 1,
    )


def target_label_for_cell(cell, nic_caps, slack=0.05):
    """Return the feasible config to emit for `cell`, per step 2 in the
    module docstring."""
    P = cell["scale"]
    feasibles = [c for c in cell["configs"]
                 if c.get("feasible", True) and is_feasible(c, nic_caps, P)]
    if not feasibles:
        return None
    best_t = min(c["t2s_ms"] for c in feasibles)
    # keep those within slack of oracle
    within = [c for c in feasibles if c["t2s_ms"] <= best_t * (1 + slack)]
    within.sort(key=lambda c: nic_budget(c, P))
    return within[0]


# -------- Greedy rule induction ---------------------------------------------

FEATURE_KEYS = ("peer_class", "size_class", "freq_class")


def guard_matches(guard, features):
    for k, v in guard.items():
        if features.get(k) != v:
            return False
    return True


def cfg_feasible_on_cells(cfg, cells, nic_caps):
    return all(is_feasible(cfg, nic_caps, c["scale"]) for c in cells)


def induce_rules(cells_with_labels, nic_caps, max_rules=20):
    """cells_with_labels: list of (cell, target_label).
    Returns list of (guard_dict, decision_cfg) pairs."""
    uncovered = list(cells_with_labels)
    rules = []
    while uncovered and len(rules) < max_rules:
        best = None
        # Candidate guards: every depth-1 guard seen in the uncovered set.
        # We intentionally DO NOT allow the empty guard here — first-match
        # ordering would make any subsequent specific rule unreachable. The
        # empty catch-all is appended explicitly after induction ends.
        guards = []
        seen = set()
        for cell, _ in uncovered:
            for k in FEATURE_KEYS:
                v = cell["features"][k]
                if (k, v) in seen:
                    continue
                seen.add((k, v))
                guards.append({k: v})
        for g in guards:
            covered = [(c, t) for (c, t) in uncovered if guard_matches(g, c["features"])]
            if not covered:
                continue
            # Majority-vote the decision among covered labels.
            # A "decision tuple" here is a hashable projection of the label.
            buckets = {}
            for _, t in covered:
                key = (t.get("path"), t.get("slot_depth", 0),
                       t.get("pool_size", 0),
                       t.get("channel_map", "static_graph_aware"))
                buckets.setdefault(key, []).append(t)
            maj = max(buckets.values(), key=len)
            decision = maj[0]
            if not cfg_feasible_on_cells(decision, [c for c, _ in covered], nic_caps):
                continue
            score = len(maj)  # only count the majority-voted subset as "covered"
            dec_key = (decision.get("path"), decision.get("slot_depth", 0),
                       decision.get("pool_size", 0),
                       decision.get("channel_map", "static_graph_aware"))
            covered_ids = {id(c) for c, t in covered
                           if (t.get("path"), t.get("slot_depth", 0),
                               t.get("pool_size", 0),
                               t.get("channel_map", "static_graph_aware")) == dec_key}
            if best is None or score > best[0] \
               or (score == best[0] and len(g) > len(best[1])):
                best = (score, g, decision, covered_ids)
        if best is None:
            break
        _, g, decision, covered_ids = best
        rules.append((g, decision))
        uncovered = [(c, t) for (c, t) in uncovered if id(c) not in covered_ids]

    # Emit rules specific-first so first-match ordering respects specificity.
    # Empty guards are never produced here (see induction comment).
    rules.sort(key=lambda rd: -len(rd[0]))
    return rules, len(uncovered)


# -------- JSON emission ------------------------------------------------------

def emit_policy(platform_desc, rules, fallback_cfg, out_path):
    rule_nodes = []
    for i, (guard, decision) in enumerate(rules, start=1):
        then_obj = {"path": decision["path"]}
        if decision["path"] == "ofi_triggered":
            then_obj["slot_depth"] = decision.get("slot_depth", 2)
            then_obj["pool_size"] = decision.get("pool_size", 16)
        elif decision["path"] == "ofi_proxy":
            then_obj["pool_size"] = decision.get("pool_size", 16)
            then_obj["channel_map"] = decision.get("channel_map", "static_graph_aware")
        rule_nodes.append({"id": 10 * i, "if": guard, "then": then_obj})
    # catch-all
    rule_nodes.append({
        "id": 999,
        "_why": "catch-all conservative fallback emitted by fit.py",
        "if": {},
        "then": {**fallback_cfg, "conservative": True}
    })
    doc = {
        "version": 1,
        "_generated_by": "OpenGDA/tools/rule_fitter/fit.py",
        "platform_desc": platform_desc,
        "rules": rule_nodes,
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(doc, f, indent=2)


def default_fallback(platform_desc):
    fab = platform_desc["fabric"]
    if fab == "ib_mlx5":
        return {"path": "ib_native"}
    # ofi_cxi
    return {"path": "ofi_proxy", "pool_size": 16, "channel_map": "static_graph_aware"}


# -------- Main ---------------------------------------------------------------

def main(argv):
    ap = argparse.ArgumentParser(description="Fit a GICC-Pilot rule-list policy from R022 calibration data.")
    ap.add_argument("--calibration", required=True,
                    help="Path to the calibration JSON (schema in fit.py docstring).")
    ap.add_argument("--out", required=True,
                    help="Destination policy_<platform>.json path.")
    ap.add_argument("--slack", type=float, default=0.05,
                    help="Within-X fraction of oracle is considered 'equivalent'. Default 0.05.")
    ap.add_argument("--max-rules", type=int, default=20, help="Cap on rule count (excluding catch-all). Default 20.")
    ap.add_argument("--min-coverage", type=float, default=0.95, help="Warning threshold. Default 0.95.")
    args = ap.parse_args(argv)

    try:
        with open(args.calibration) as f:
            calib = json.load(f)
    except Exception as e:
        print(f"fit.py: cannot read {args.calibration}: {e}", file=sys.stderr)
        return 2

    required_keys = ("platform", "platform_desc", "cells")
    for k in required_keys:
        if k not in calib:
            print(f"fit.py: calibration missing required key '{k}'", file=sys.stderr)
            return 2

    nic_caps = calib["platform_desc"]["nic_caps"]

    # Step 1-2: target label per cell.
    labelled = []
    for cell in calib["cells"]:
        label = target_label_for_cell(cell, nic_caps, slack=args.slack)
        if label is None:
            print(f"fit.py: warning: no feasible config for cell {cell.get('workload')}@N={cell.get('scale')}", file=sys.stderr)
            continue
        labelled.append((cell, label))

    if not labelled:
        print("fit.py: no labelled cells; refusing to write an empty policy", file=sys.stderr)
        return 2

    # Step 3: greedy induction.
    rules, n_uncovered = induce_rules(labelled, nic_caps, max_rules=args.max_rules)
    total = len(labelled)
    coverage = (total - n_uncovered) / total if total else 0.0
    print(f"fit.py: {len(rules)} rule(s) induced, coverage {coverage * 100:.1f}% "
          f"({total - n_uncovered}/{total} cells)", file=sys.stderr)

    emit_policy(calib["platform_desc"], rules,
                default_fallback(calib["platform_desc"]), args.out)
    print(f"fit.py: wrote {args.out}", file=sys.stderr)

    if coverage < args.min_coverage:
        print(f"fit.py: coverage {coverage * 100:.1f}% < {args.min_coverage * 100:.0f}% threshold",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
