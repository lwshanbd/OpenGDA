#!/usr/bin/env python3
"""
test_fit.py — Synthetic test for fit.py. Generates a calibration set where
the right policy is known by construction; asserts fit.py induces it.

The test deliberately uses the same feature vocabulary as policy_tioga.json
so the produced policy can be loaded by libgicc_policy::load_policy without
edits. Exit 0 on success, non-zero on failure.
"""

import json
import os
import subprocess
import sys
import tempfile

THIS = os.path.dirname(os.path.abspath(__file__))
FIT = os.path.join(THIS, "fit.py")


def make_calibration():
    """Three cells: regular stencil hot-loop, matmul-style large-const ring,
    and a dynamic-peer bfs-style cell. Known best configs baked in."""
    def cfg(path, t2s, **kw):
        kw.setdefault("feasible", True)
        return {"path": path, "t2s_ms": t2s, "seeds": 5, **kw}
    cells = []
    # Cell 1: jacobi-style regular, slot_depth=4 wins.
    cells.append({
        "workload": "jacobi", "scale": 32,
        "features": {"peer_class": "k_from_topology_hint",
                     "size_class": "large_const", "freq_class": "hot_loop"},
        "configs": [
            cfg("ofi_triggered", 300.0, slot_depth=2, pool_size=16),
            cfg("ofi_triggered", 260.0, slot_depth=4, pool_size=16),   # oracle
            cfg("ofi_triggered", 270.0, slot_depth=8, pool_size=16),
            cfg("ofi_proxy",     320.0, pool_size=16, channel_map="static_graph_aware"),
            cfg("ofi_proxy",     310.0, pool_size=32, channel_map="static_graph_aware"),
        ],
    })
    # Cell 2: matmul-style compute-heavy, ofi_proxy pool=32 wins.
    cells.append({
        "workload": "matmul", "scale": 32,
        "features": {"peer_class": "k_const",
                     "size_class": "large_const", "freq_class": "outer_loop"},
        "configs": [
            cfg("ofi_triggered", 3200.0, slot_depth=2, pool_size=16),
            cfg("ofi_triggered", 3100.0, slot_depth=4, pool_size=16),
            cfg("ofi_proxy",     2700.0, pool_size=16, channel_map="static_graph_aware"),
            cfg("ofi_proxy",     2400.0, pool_size=32, channel_map="static_graph_aware"),  # oracle
            cfg("ofi_proxy",     2450.0, pool_size=32, channel_map="modular_hash"),
        ],
    })
    # Cell 3: bfs-style irregular, all-dynamic features.
    cells.append({
        "workload": "bfs", "scale": 32,
        "features": {"peer_class": "dynamic",
                     "size_class": "dynamic", "freq_class": "dynamic"},
        "configs": [
            cfg("ofi_proxy",     1000.0, pool_size=16, channel_map="static_graph_aware"),
            cfg("ofi_proxy",      950.0, pool_size=32, channel_map="static_graph_aware"),  # oracle
        ],
    })
    return {
        "platform": "tioga",
        "platform_desc": {
            "fabric": "ofi_cxi",
            "gpu_family": "mi250x",
            "nic_caps": {"counter_max": 2047, "dwq_max": 256, "ctrs_per_op": 2, "qp_max_per_peer": 0},
        },
        "cells": cells,
    }


def main():
    with tempfile.TemporaryDirectory() as td:
        calib_path = os.path.join(td, "calib.json")
        out_path = os.path.join(td, "policy.json")
        with open(calib_path, "w") as f:
            json.dump(make_calibration(), f, indent=2)
        r = subprocess.run([sys.executable, FIT,
                            "--calibration", calib_path,
                            "--out", out_path], capture_output=True, text=True)
        if r.returncode != 0:
            print("FAIL: fit.py exited", r.returncode, file=sys.stderr)
            print("stdout:", r.stdout, file=sys.stderr)
            print("stderr:", r.stderr, file=sys.stderr)
            return 1
        with open(out_path) as f:
            policy = json.load(f)
        # structural checks
        if policy["version"] != 1:
            print("FAIL: version != 1"); return 1
        if policy["platform_desc"]["fabric"] != "ofi_cxi":
            print("FAIL: fabric wrong"); return 1
        rules = policy["rules"]
        if len(rules) < 2:
            print(f"FAIL: expected ≥ 2 rules, got {len(rules)}"); return 1
        if rules[-1]["if"] != {}:
            print("FAIL: last rule is not catch-all"); return 1
        # semantic checks: did we get rules that cover the three target labels?
        decisions = [(r["then"].get("path"), r["then"].get("slot_depth", 0),
                      r["then"].get("pool_size", 0)) for r in rules]
        want = [("ofi_triggered", 4, 16),
                ("ofi_proxy",     0, 32)]
        for w in want:
            if w not in decisions:
                print(f"FAIL: expected decision {w} in rule list, got {decisions}")
                return 1
        print("PASS: synthetic calibration → expected rules")
        print("stderr banner:", r.stderr.strip().replace("\n", " | "))
        print("rule count:", len(rules))
        return 0


if __name__ == "__main__":
    sys.exit(main())
