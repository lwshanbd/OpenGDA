#!/usr/bin/env python3
"""Fail closed unless final mm_minimal IR has the frozen guarded shape."""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
import re
from typing import Any


KERNEL = "_Z18matmul_step_kernelPN4gicc9DeviceCtxEPKfS3_Pfiiiiiim"
STUB = "_Z33__device_stub__matmul_step_kernelPN4gicc9DeviceCtxEPKfS3_Pfiiiiiim"
DEFINE_RE = re.compile(r"^define\b.*@([^ (]+)\(")


class AuditError(RuntimeError):
    """The optimized IR does not implement the frozen compiler schedule."""


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
        if name in result:
            raise AuditError(f"duplicate function {name}")
        result[name] = "\n".join(body)
        index += 1
    return result


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def audit_device(ir: str) -> dict[str, Any]:
    funcs = functions(ir)
    require(KERNEL in funcs, "device kernel definition is missing")
    body = funcs[KERNEL]
    stores = [match.start() for match in re.finditer(r"store volatile i64", body)]
    atomics = [match.start() for match in re.finditer(r"atomicrmw fadd", body)]
    require(len(stores) == 2, "device kernel must contain exactly two trigger stores")
    require(len(atomics) == 1, "device kernel must contain exactly one matmul atomic")
    require(stores[0] < atomics[0] < stores[1],
            "early/original trigger stores do not straddle compute")
    require("%gicc.phase" in body and "icmp ne i32 %gicc.phase, 3" in body,
            "device phase-3 dispatch is missing")
    require("gicc.early.synthetic_flush.do" in body,
            "synthetic early trigger block is missing")
    require("gicc.early.original_flush.cont" in body,
            "original trigger fallback block is missing")
    return {
        "kernel": KERNEL,
        "trigger_stores": len(stores),
        "matmul_atomics": len(atomics),
        "ordering": "early_trigger_then_compute_then_original_trigger",
        "phase": 3,
    }


def audit_host(ir: str) -> dict[str, Any]:
    funcs = functions(ir)
    wrappers = [
        name for name in funcs
        if name.startswith("_ZN4gicc6launch") and "matmul_step_kernel" in name
    ]
    require(len(wrappers) == 1, "expected exactly one retained mm launch wrapper")
    wrapper_name = wrappers[0]
    wrapper = funcs[wrapper_name]
    checks = {
        "source_identity_calls": wrapper.count(
            "@gicc_runtime_kernel_arg_matches_local_buffer("),
        "interval_calls": wrapper.count(
            "@gicc_runtime_local_buffer_contains_interval("),
        "write_disjoint_calls": wrapper.count(
            "@gicc_runtime_local_buffer_disjoint_from_kernel_arg_allocation("),
        "phase_set_calls": wrapper.count(
            "@gicc_runtime_set_schedule_phase_from_kernel_args("),
        "guarded_kernel_launches": sum(
            1 for line in wrapper.splitlines()
            if "@hipLaunchKernel(" in line and f"@{KERNEL}" in line
        ),
    }
    require(checks["source_identity_calls"] == 2,
            "wrapper must test both compiler-proved readonly source formals")
    require(checks["interval_calls"] == 1,
            "wrapper must check the exact registered source interval")
    require(checks["write_disjoint_calls"] == 1,
            "wrapper must check the complete write allocation")
    require(checks["phase_set_calls"] == 2,
            "wrapper must set phase 3 and reset phase 0 exactly once")
    require("i32 3, ptr null" in wrapper and "i32 0, ptr null" in wrapper,
            "wrapper phase set/reset values are not 3/0")
    require(checks["guarded_kernel_launches"] == 2,
            "guarded and fallback paths must each launch the kernel once")
    require("gicc.early.guarded" in wrapper and "gicc.early.original" in wrapper,
            "wrapper guarded/fallback CFG is missing")
    require(wrapper.count("@__hipPushCallConfiguration(") == 1,
            "wrapper must preserve one HIP configuration push")

    require(STUB in funcs, "compiler-generated HIP device stub is missing")
    stub_launches = sum(
        1 for line in funcs[STUB].splitlines()
        if "@hipLaunchKernel(" in line and f"@{KERNEL}" in line
    )
    require(stub_launches == 1, "standalone HIP stub shape changed")
    launches_by_function = {
        name: sum(
            1 for line in body.splitlines()
            if "@hipLaunchKernel(" in line and f"@{KERNEL}" in line
        )
        for name, body in funcs.items()
    }
    launches_by_function = {
        name: count for name, count in launches_by_function.items() if count
    }
    require(launches_by_function == {wrapper_name: 2, STUB: 1},
            "a reachable direct kernel-launch bypass may exist")
    call_stub = re.compile(rf"\b(?:call|invoke)\b[^\n]*@{re.escape(STUB)}\(")
    require(not call_stub.search(ir), "optimized host IR still calls the bare stub")
    require("main" in funcs, "host main definition is missing")
    source_launches = funcs["main"].count(f"@{wrapper_name}(")
    require(source_launches == 2,
            "warmup and loop launches must both reach the guarded wrapper")
    return {
        "wrapper": wrapper_name,
        **checks,
        "standalone_stub_launches": stub_launches,
        "source_wrapper_calls": source_launches,
        "direct_stub_calls": 0,
    }


def audit(device_path: Path, host_path: Path) -> dict[str, Any]:
    return {
        "schema_version": "gicc-guarded-early-trigger-ir-audit-v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device_ir": str(device_path.resolve()),
        "host_ir": str(host_path.resolve()),
        "device": audit_device(device_path.read_text(encoding="utf-8")),
        "host": audit_host(host_path.read_text(encoding="utf-8")),
        "passed": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=Path, required=True)
    parser.add_argument("--host", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.device, args.host)
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
