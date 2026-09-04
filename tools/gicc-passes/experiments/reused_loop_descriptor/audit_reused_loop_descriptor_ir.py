#!/usr/bin/env python3
"""Fail closed unless final host/device IR matches the frozen two-arm design."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
from typing import Any


KERNEL = "_Z15dwq_loop_kernelPN4gicc9DeviceCtxEiimi"
DEFINE_RE = re.compile(r"^define\b.*@([^ (]+)\(")


class AuditError(RuntimeError):
    """Optimized IR does not implement the frozen compiler schedule."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def functions(ir: str) -> dict[str, str]:
    result: dict[str, str] = {}
    lines = ir.splitlines()
    index = 0
    while index < len(lines):
        match = DEFINE_RE.match(lines[index])
        if not match:
            index += 1
            continue
        name = match.group(1)
        body = [lines[index]]
        index += 1
        while index < len(lines):
            body.append(lines[index])
            if lines[index] == "}":
                break
            index += 1
        else:
            raise AuditError(f"unterminated function {name}")
        require(name not in result, f"duplicate function {name}")
        result[name] = "\n".join(body)
        index += 1
    return result


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def audit_host(ir: str, arm: str) -> dict[str, Any]:
    funcs = functions(ir)
    traces = [name for name in funcs if name.startswith("gicc_trace_dwq_loop_kernel")]
    require(len(traces) <= 1, "expected at most one retained compiler trace")
    # O3 may retain the larger baseline trace but inline the small reused
    # trace into each static launch context. Audit executable call sites over
    # the full module, excluding declarations, instead of requiring one
    # particular inlining outcome.
    calls_batched = len(re.findall(
        r"\b(?:call|invoke)\b[^\n]*@gicc_runtime_dwq_enqueue_batched\(", ir
    ))
    calls_repeated = len(re.findall(
        r"\b(?:call|invoke)\b[^\n]*@gicc_runtime_dwq_enqueue_repeated\(", ir
    ))
    arrays = sum(ir.count(name) for name in (
        "%dwq.peers", "%dwq.dst_bufs", "%dwq.dst_offs",
        "%dwq.src_bufs", "%dwq.src_offs", "%dwq.sizes",
    ))
    if arm == "baseline":
        require(calls_batched >= 1, "baseline must call batched helper")
        require(calls_repeated == 0, "baseline unexpectedly calls repeated helper")
        require(arrays >= 6, "baseline descriptor-array staging is missing")
    else:
        require(calls_repeated >= 1, "reused arm must call repeated helper")
        require(calls_batched == 0, "reused arm unexpectedly calls batched helper")
        require(arrays == 0, "reused arm still constructs descriptor arrays")
        require(
                ("icmp sgt i32" in ir and "select i1" in ir)
                or "@llvm.smax.i32" in ir,
                "reused arm is missing the non-positive-bound clamp")
    kernel_launches = len(re.findall(
        rf"\b(?:call|invoke)\b[^\n]*@hipLaunchKernel\([^\n]*@{re.escape(KERNEL)}",
        ir,
    ))
    require(kernel_launches >= 1, "original kernel launch is missing")
    return {
        "retained_trace": traces[0] if traces else None,
        "batched_helper_calls": calls_batched,
        "repeated_helper_calls": calls_repeated,
        "descriptor_array_name_occurrences": arrays,
        "kernel_launch_calls": kernel_launches,
        "module_sha256": sha256_text(ir),
    }


def audit_device(ir: str) -> dict[str, Any]:
    funcs = functions(ir)
    require(KERNEL in funcs, "device kernel definition is missing")
    body = funcs[KERNEL]
    require("gicc_runtime_dwq_enqueue" not in body,
            "host runtime helper escaped into device kernel")
    require(body.count("store volatile i64") == 1,
            "device kernel must retain exactly one original trigger store")
    return {
        "kernel": KERNEL,
        "trigger_stores": 1,
        "kernel_sha256": sha256_text(body),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-host", required=True, type=Path)
    parser.add_argument("--reused-host", required=True, type=Path)
    parser.add_argument("--baseline-device", required=True, type=Path)
    parser.add_argument("--reused-device", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    baseline_device = args.baseline_device.read_text(encoding="utf-8")
    reused_device = args.reused_device.read_text(encoding="utf-8")
    device_baseline = audit_device(baseline_device)
    device_reused = audit_device(reused_device)
    require(
        device_baseline["kernel_sha256"] == device_reused["kernel_sha256"],
        "host-only transform changed the optimized device kernel",
    )
    baseline_host = audit_host(
        args.baseline_host.read_text(encoding="utf-8"), "baseline"
    )
    reused_host = audit_host(
        args.reused_host.read_text(encoding="utf-8"), "reused"
    )
    require(
        baseline_host["kernel_launch_calls"] ==
            reused_host["kernel_launch_calls"],
        "host-only transform changed the number of original kernel launches",
    )
    result = {
        "schema_version": "gicc-reused-loop-descriptor-ir-audit-v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "baseline": {
            "host": baseline_host,
            "device": device_baseline,
        },
        "reused": {
            "host": reused_host,
            "device": device_reused,
        },
        "host_only_device_identity": True,
        "passed": True,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
