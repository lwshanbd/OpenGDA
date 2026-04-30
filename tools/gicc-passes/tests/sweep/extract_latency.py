#!/usr/bin/env python3
"""Parse a workload's stdout and emit a single CSV row.

Usage:
  extract_latency.py <workload> <grid> <ranks> <W> <streams> <run_idx>
Reads stdout from stdin. Writes one CSV row to stdout.
"""
import re, sys

if len(sys.argv) != 7:
    sys.exit(f"usage: {sys.argv[0]} <workload> <grid> <ranks> <W> <streams> <run_idx>")
workload, grid, ranks, W, streams, run_idx = sys.argv[1:]

text = sys.stdin.read()

if workload == "minimod":
    m = re.search(r"rank 0 Time comm\s*[:=]?\s*([\d.eE+-]+)", text)
    if not m: sys.exit("no minimod latency found")
    latency_ms = float(m.group(1)) * 1000.0
elif workload == "barrier_bench":
    m = re.search(r"avg=\s*([\d.eE+-]+)\s*(us|ms|s)?", text)
    if not m: sys.exit("no barrier_bench latency found")
    val, unit = float(m.group(1)), (m.group(2) or "us")
    latency_ms = {"us": 1e-3, "ms": 1.0, "s": 1e3}[unit] * val
elif workload == "jacobi":
    # Output: "Done: N iters in Y s, final l2=Z"
    m = re.search(r"Done:.*?in\s+([\d.eE+-]+)\s*s", text)
    if not m: sys.exit("no jacobi latency found")
    latency_ms = float(m.group(1)) * 1000.0
elif workload == "mm_minimal":
    # Output: "gicc::launch average (runs 2-9): Y us"
    m = re.search(r"gicc::launch average[^:]*:\s*([\d.eE+-]+)\s*(us|ms|s)?", text)
    if not m: sys.exit("no mm_minimal latency found")
    val, unit = float(m.group(1)), (m.group(2) or "us")
    latency_ms = {"us": 1e-3, "ms": 1.0, "s": 1e3}[unit] * val
else:
    sys.exit(f"unknown workload: {workload}")

print(f"{workload},{grid},{ranks},{W},{streams},{run_idx},{latency_ms:.4f}")
