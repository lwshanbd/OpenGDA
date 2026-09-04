#!/usr/bin/env python3
"""Strict retrospective audit of LTO-synthesized vs handwritten DWQ traces.

The historical driver did not freeze binary hashes and always ran the
handwritten arm before the LTO arm.  Consequently this tool reports descriptive
paired measurements only; it deliberately cannot produce a confirmatory or LLM
performance claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from pathlib import Path
from typing import Any


SCHEMA = "gicc-lto-trace-retrospective-v1"
BATCHES = (4, 64)
TRIALS = tuple(range(1, 11))
ARMS = ("handwritten", "lto")
SIZES = (
    "1B", "2B", "4B", "8B", "64B", "256B", "1KB", "4KB",
    "16KB", "64KB", "256KB", "512KB", "1MB", "2MB", "4MB",
    "16MB",
)
ROW_RE = re.compile(
    r"^(1B|2B|4B|8B|64B|256B|1KB|4KB|16KB|64KB|256KB|512KB|"
    r"1MB|2MB|4MB|16MB)\s+(\d+)\s+([0-9]+(?:\.[0-9]+)?)\s+"
    r"([0-9]+(?:\.[0-9]+)?)$",
    re.MULTILINE,
)
LTO_AUDIT_RE = re.compile(
    r"\[enqueue-audit\] mono_total_ops=(\d+)\s+expected=(\d+)\s+match=(\w+)"
)
HAND_AUDIT_RE = re.compile(
    r"\[enqueue-audit mode=dwq\] mono_total_ops=(\d+)\s+"
    r"dwq_expected=(\d+)\s+match=(\w+)"
)


class AuditError(RuntimeError):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True) + "\n").encode()


def geometric_mean(values: list[float]) -> float:
    if not values or any(not math.isfinite(v) or v <= 0 for v in values):
        raise AuditError("geometric mean requires finite positive values")
    return math.exp(sum(math.log(v) for v in values) / len(values))


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def descriptive_trial_bootstrap(values: list[float]) -> dict[str, Any]:
    """Fixed-seed trial-cluster bootstrap; descriptive, not confirmatory."""
    rng = random.Random(0x47544343)
    draws = []
    for _ in range(100_000):
        draws.append(geometric_mean(
            [values[rng.randrange(len(values))] for _ in values]
        ))
    return {
        "method": "fixed-seed trial-cluster bootstrap",
        "replicates": 100_000,
        "lower_95": percentile(draws, 0.025),
        "upper_95": percentile(draws, 0.975),
        "confirmatory": False,
    }


def parse_log(path: Path, arm: str, batch: int) -> dict[str, Any]:
    data = path.read_bytes()
    text = data.decode("utf-8")
    if any(marker in text for marker in
           ("VERIFY-FAIL", "FATAL", "FAILED-AGAIN", "job.exception")):
        raise AuditError(f"failure marker in {path}")
    if arm == "lto":
        if f"batch={batch}" not in text or "LTO-generated host trace" not in text:
            raise AuditError(f"LTO header mismatch in {path}")
        match = LTO_AUDIT_RE.search(text)
        if not match or int(match.group(1)) != int(match.group(2)) or \
                match.group(3) != "YES":
            raise AuditError(f"LTO enqueue audit failed in {path}")
    else:
        if f"batch_per_outer={batch}" not in text or \
                "bench_pingpong (mode=dwq)" not in text:
            raise AuditError(f"handwritten header mismatch in {path}")
        match = HAND_AUDIT_RE.search(text)
        if not match or int(match.group(1)) < int(match.group(2)) or \
                match.group(3) not in ("YES", "MORE"):
            raise AuditError(f"handwritten enqueue audit failed in {path}")

    matches = ROW_RE.findall(text)
    if len(matches) != len(SIZES) or tuple(row[0] for row in matches) != SIZES:
        raise AuditError(f"expected exactly the canonical 16 size rows in {path}")
    rows = {}
    for size, printed_iters, mean, median in matches:
        rows[size] = {
            "printed_iters_total": int(printed_iters),
            "mean_us_per_message": float(mean),
            "median_us_per_message": float(median),
        }
    return {
        "path": str(path),
        "sha256": sha256_bytes(data),
        "bytes": len(data),
        "enqueue_actual": int(match.group(1)),
        "enqueue_expected": int(match.group(2)),
        "enqueue_match_text": match.group(3),
        "rows": rows,
    }


def analyze(input_dir: Path, driver: Path) -> dict[str, Any]:
    if not driver.is_file():
        raise AuditError(f"missing historical driver: {driver}")
    driver_data = driver.read_bytes()
    driver_text = driver_data.decode("utf-8")
    expected_order = (
        "run_one $trial handwritten 4", "run_one $trial lto         4",
        "run_one $trial handwritten 64", "run_one $trial lto         64",
    )
    positions = [driver_text.find(token) for token in expected_order]
    if any(pos < 0 for pos in positions) or positions != sorted(positions):
        raise AuditError("historical driver no longer has the audited fixed order")
    if "for trial in $(seq 1 10)" not in driver_text:
        raise AuditError("historical driver no longer requests ten trials")

    logs: dict[tuple[int, str, int], dict[str, Any]] = {}
    inventory = []
    for batch in BATCHES:
        for trial in TRIALS:
            for arm in ARMS:
                path = input_dir / f"trial_{trial}_{arm}_b{batch}.log"
                if not path.is_file():
                    raise AuditError(f"missing historical log: {path}")
                parsed = parse_log(path, arm, batch)
                logs[(batch, arm, trial)] = parsed
                inventory.append({k: parsed[k] for k in
                                  ("path", "sha256", "bytes")})

    batch_results = {}
    for batch in BATCHES:
        per_size = {}
        for size in SIZES:
            ratios = [
                logs[(batch, "handwritten", trial)]["rows"][size]
                    ["median_us_per_message"] /
                logs[(batch, "lto", trial)]["rows"][size]
                    ["median_us_per_message"]
                for trial in TRIALS
            ]
            per_size[size] = {
                "handwritten_over_lto_geomean": geometric_mean(ratios),
                "lto_over_handwritten_percent":
                    (1.0 / geometric_mean(ratios) - 1.0) * 100.0,
            }
        trial_ratios = []
        for trial in TRIALS:
            trial_ratios.append(geometric_mean([
                logs[(batch, "handwritten", trial)]["rows"][size]
                    ["median_us_per_message"] /
                logs[(batch, "lto", trial)]["rows"][size]
                    ["median_us_per_message"]
                for size in SIZES
            ]))
        aggregate = geometric_mean(trial_ratios)
        batch_results[str(batch)] = {
            "ratio_semantics": (
                "handwritten_median_us_per_message divided by "
                "lto_median_us_per_message; above one favors LTO"
            ),
            "trial_cluster_ratios": trial_ratios,
            "aggregate_geomean": aggregate,
            "lto_over_handwritten_percent": (1.0 / aggregate - 1.0) * 100.0,
            "descriptive_interval": descriptive_trial_bootstrap(trial_ratios),
            "per_size": per_size,
        }

    body = {
        "schema_version": SCHEMA,
        "boundary": {
            "comparison": "LTO-synthesized DWQ trace versus handwritten DWQ trace",
            "lto_arm_application_source_modified_by_compiler_decision": False,
            "provider_or_model_used": False,
            "compiler_selector_evaluated": False,
            "historical_binary_hashes_frozen_at_execution": False,
            "allocation_identity_preserved": False,
            "execution_order": list(expected_order),
            "order_balanced": False,
            "driver_intended_allocations": 1,
            "independent_allocation_count_evidenced": None,
        },
        "provenance": {
            "driver": {
                "path": str(driver),
                "sha256": sha256_bytes(driver_data),
                "bytes": len(driver_data),
            },
            "logs": sorted(inventory, key=lambda item: item["path"]),
        },
        "coverage": {
            "batches": list(BATCHES),
            "trials_per_arm_and_batch": len(TRIALS),
            "message_sizes": list(SIZES),
            "raw_logs": len(inventory),
            "lto_enqueue_audits_exact": True,
            "correctness_failure_markers": 0,
        },
        "results": batch_results,
        "claim_gate": {
            "compiler_generated_trace_executed": True,
            "retrospective_feasibility_evidence": True,
            "confirmatory_performance_equivalence": False,
            "current_graph_runtime_labels_complete": False,
            "llm_performance_measured": False,
            "provider_protocol_permitted": False,
            "reasons": [
                "historical binaries were not content-addressed at execution",
                "the driver intended one allocation but its identity output is absent",
                "the fixed handwritten-then-LTO order is not order-balanced",
                "the comparison does not evaluate a compiler selector or an LLM",
            ],
        },
    }
    body["result_id"] = "sha256:" + sha256_bytes(canonical_bytes(body))
    return body


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("emit", "verify"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--input-dir", type=Path, required=True)
        cmd.add_argument("--driver", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.input_dir, args.driver)
    encoded = canonical_bytes(result)
    if args.command == "emit":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(encoded)
        print(json.dumps({
            "result_id": result["result_id"],
            "serialized_sha256": sha256_bytes(encoded),
            "out": str(args.out),
        }, sort_keys=True))
        return 0
    if not args.out.is_file() or args.out.read_bytes() != encoded:
        raise AuditError("stored report does not match regenerated evidence")
    print(json.dumps({
        "result_id": result["result_id"],
        "serialized_sha256": sha256_bytes(encoded),
        "verified": True,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
