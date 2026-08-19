#!/usr/bin/env python3
"""Validate and rank all 16 static4 mixed LTO lowering configurations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyze_compiler_lto_eval import OFFSETS, expected_hash  # noqa: E402


FIELDS = re.compile(r"([a-z0-9_]+)=([^ ]+)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()

    rows = []
    for path in args.logs:
        match = re.search(r"static4-mask([0-9]{2})\.log$", path.name)
        if match is None:
            raise SystemExit(f"cannot infer mask from {path}")
        mask = int(match.group(1))
        lines = path.read_text().splitlines()
        run_lines = [line for line in lines if line.startswith("RUN mask=")]
        if len(run_lines) != 1:
            raise SystemExit(f"{path}: expected one binary provenance line")
        run_fields = dict(FIELDS.findall(run_lines[0]))
        if int(run_fields.get("mask", -1)) != mask:
            raise SystemExit(f"{path}: provenance mask mismatch")
        binary_sha256 = run_fields.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", binary_sha256):
            raise SystemExit(f"{path}: invalid binary sha256")
        result_lines = [line for line in lines
                        if line.startswith("COMPILER_LTO_EVAL ")]
        if len(result_lines) != 1:
            raise SystemExit(f"{path}: expected one result line")
        fields = dict(FIELDS.findall(result_lines[0]))
        if fields.get("scenario") != "static4-k4-grid4" or fields.get("data") != "OK":
            raise SystemExit(f"{path}: invalid scenario/data fields")
        runs = int(fields["runs"])
        nproxy = mask.bit_count()
        ntrigger = 4 - nproxy
        active_lanes = mask.bit_length() if mask else 0
        expected_routes = (ntrigger * runs, (nproxy + active_lanes) * runs)
        actual_routes = (int(fields["staged"]), int(fields["pushed"]))
        if actual_routes != expected_routes:
            raise SystemExit(
                f"{path}: routes={actual_routes}, expected={expected_routes}"
            )
        base, size = OFFSETS["static4-k4-grid4"]
        if fields["hash"] != expected_hash(base, size):
            raise SystemExit(f"{path}: payload hash mismatch")
        rows.append({
            "mask": mask,
            "bits": f"{mask:04b}",
            "proxy_sites": nproxy,
            "trigger_sites": ntrigger,
            "active_proxy_lanes": active_lanes,
            "binary_sha256": binary_sha256,
            "median_us": float(fields["median_us"]),
            "p25_us": float(fields["p25_us"]),
            "p75_us": float(fields["p75_us"]),
            "staged": actual_routes[0],
            "pushed": actual_routes[1],
        })
    if {row["mask"] for row in rows} != set(range(16)):
        raise SystemExit("logs do not cover every mask 0..15 exactly once")
    if len({row["binary_sha256"] for row in rows}) != 16:
        raise SystemExit("oracle configurations do not have unique binaries")
    rows.sort(key=lambda row: row["median_us"])
    best = rows[0]
    for row in rows:
        row["slowdown_vs_oracle"] = row["median_us"] / best["median_us"]

    print("rank mask proxy trigger lanes median_us slowdown")
    for rank, row in enumerate(rows, 1):
        print(f"{rank:>4} {row['bits']} {row['proxy_sites']:>5} "
              f"{row['trigger_sites']:>7} {row['active_proxy_lanes']:>5} "
              f"{row['median_us']:>9.3f} {row['slowdown_vs_oracle']:>8.4f}x")
    summary = {
        "scope": "exact 2^4 proxy/trigger oracle for eval_static4_parallel",
        "unique_binary_hashes": 16,
        "oracle": best,
        "configurations": rows,
    }
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
