#!/usr/bin/env python3
"""Prove that an interrupted guarded-early scout cannot pass its frozen gate."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any

import monitor_guarded_early_trigger_scout as scout_monitor


common = scout_monitor.common
FLOAT = r"[-+]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][-+]?[0-9]+)?"
RUN_RE = re.compile(
    rf"^Run (?P<run>[0-9]+): (?P<usec>{FLOAT}) us"
    r"(?P<warmup> \(warmup\))?$"
)
AVERAGE_RE = re.compile(
    rf"^gicc::launch average \(runs 2-9\): (?P<usec>{FLOAT}) us$"
)


class PartialNegativeError(RuntimeError):
    """The partial evidence cannot support the one-sided negative audit."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PartialNegativeError(message)


def file_record(path: Path, role: str) -> dict[str, Any]:
    require(path.is_file(), f"missing {role}: {path}")
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": common.sha256(path),
    }


def gate_reachability(observed: list[float]) -> dict[str, Any]:
    require(3 <= len(observed) <= 4,
            "reachability audit requires three or four observed pairs")
    require(all(math.isfinite(value) and value > 0 for value in observed),
            "speedups must be finite and positive")
    missing = 4 - len(observed)
    require(missing <= 1, "at most one pair may be missing")
    wins = sum(value > 1.0 for value in observed)
    maximum_wins = wins + missing
    if missing:
        ordered = sorted(observed)
        maximum_median = (ordered[1] + ordered[2]) / 2
    else:
        maximum_median = statistics.median(observed)
    reachable = maximum_wins >= 3 and maximum_median >= 1.02
    return {
        "observed_speedups": observed,
        "observed_pairs": len(observed),
        "missing_pairs": missing,
        "observed_wins": wins,
        "maximum_possible_wins": maximum_wins,
        "maximum_possible_median_speedup": maximum_median,
        "criterion": ">=3/4 wins and median speedup >=1.02",
        "gate_reachable_under_arbitrarily_favorable_missing_pair": reachable,
    }


def parse_completed_run(stdout: Path, stderr: Path, size: int,
                        arm: str) -> dict[str, Any]:
    require(stdout.is_file() and stderr.is_file(),
            f"missing completed run logs: {stdout}, {stderr}")
    timings: dict[int, float] = {}
    averages = []
    for line in stdout.read_text(encoding="utf-8", errors="replace").splitlines():
        if match := RUN_RE.match(line):
            run = int(match.group("run"))
            require(run not in timings, f"duplicate Run {run} in {stdout}")
            require(bool(match.group("warmup")) == (run < 2),
                    f"warmup label mismatch in {stdout}")
            timings[run] = float(match.group("usec"))
        if match := AVERAGE_RE.match(line):
            averages.append(float(match.group("usec")))
    require(sorted(timings) == list(range(10)) and len(averages) == 1,
            f"incomplete timing output: {stdout}")
    require(all(math.isfinite(value) and value > 0 for value in timings.values()),
            f"invalid timing in {stdout}")
    measured = sum(timings[index] for index in range(2, 10)) / 8
    require(math.isclose(averages[0], measured, rel_tol=5e-5, abs_tol=0.5),
            f"reported average does not match runs 2-9 in {stdout}")

    expected_bytes = size * (size // 16) * 4
    expected_launches = 161 if arm == "baseline" else 483
    checksums: dict[int, str] = {}
    launches: dict[int, int] = {}
    for line in stderr.read_text(encoding="utf-8", errors="replace").splitlines():
        match = scout_monitor.CHECKSUM_RE.match(line)
        if not match or int(match.group("bytes")) != expected_bytes:
            continue
        rank = int(match.group("rank"))
        require(rank not in checksums, f"duplicate checksum rank in {stderr}")
        checksums[rank] = match.group("hash")
        launches[rank] = int(match.group("launches"))
    require(sorted(checksums) == list(range(16)),
            f"incomplete checksum ranks in {stderr}")
    require(set(launches.values()) == {expected_launches},
            f"unexpected {arm} launch count in {stderr}")
    return {
        "requested_size": size,
        "run_us": [timings[index] for index in range(10)],
        "measured_mean_us": averages[0],
        "checksums": {str(rank): checksums[rank] for rank in sorted(checksums)},
        "successful_kernel_launches": {
            str(rank): launches[rank] for rank in sorted(launches)
        },
        "stdout": file_record(stdout, f"{arm}_stdout"),
        "stderr": file_record(stderr, f"{arm}_stderr"),
    }


def parse_pair(output_dir: Path, replicate: int, size: int) -> dict[str, Any]:
    directory = output_dir / f"rep{replicate}" / f"size{size}"
    pair = {
        arm: parse_completed_run(
            directory / f"{arm}.out", directory / f"{arm}.err", size, arm)
        for arm in ("baseline", "guarded")
    }
    require(pair["baseline"]["checksums"] == pair["guarded"]["checksums"],
            f"checksum mismatch in replicate {replicate}, size {size}")
    speedup = (
        pair["baseline"]["measured_mean_us"] /
        pair["guarded"]["measured_mean_us"]
    )
    return {
        "replicate": replicate,
        "size": size,
        "speedup": speedup,
        "correctness": "all_16_rank_checksums_equal",
        "baseline": pair["baseline"],
        "guarded": pair["guarded"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--monitor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    monitor = json.loads(args.monitor.read_text(encoding="utf-8"))
    require(monitor.get("state") == "failed", "scout monitor did not fail")
    require(monitor.get("scheduler", {}).get("exit_code") == 1,
            "scout failure is not the expected clean exit-code-1 case")
    require(not monitor.get("scheduler", {}).get("exception_types"),
            "scout has scheduler exceptions")
    require(monitor.get("jobspec", {}).get("queue") == "pdebug",
            "scout was not submitted to pdebug")
    require(monitor.get("jobspec", {}).get("duration_seconds") == 1800.0,
            "outer scout duration is not the frozen 30 minutes")

    verified_artifacts = []
    for artifact in monitor.get("artifacts", []):
        path = Path(artifact["path"])
        expected = artifact["sha256"]
        require(path.is_file(), f"missing scout artifact: {path}")
        require(common.sha256(path) == expected,
                f"scout artifact changed: {path}")
        verified_artifacts.append({
            "path": str(path.resolve()), "sha256": expected,
        })

    failed_stderr = args.output_dir / "rep4" / "size8192" / "baseline.err"
    failure_text = failed_stderr.read_text(encoding="utf-8", errors="replace")
    require(
        "job duration (8m) exceeds remaining instance lifetime" in failure_text,
        "missing pair was not rejected by the nested-duration guard",
    )
    failed_stdout = args.output_dir / "rep4" / "size8192" / "baseline.out"
    require(failed_stdout.is_file() and failed_stdout.stat().st_size == 0,
            "failed arm unexpectedly produced timing output")
    missing_guarded = args.output_dir / "rep4" / "size8192" / "guarded.out"
    require(not missing_guarded.exists() or missing_guarded.stat().st_size == 0,
            "the declared missing guarded arm contains timing output")

    completed = {
        "4096": [parse_pair(args.output_dir, rep, 4096) for rep in range(1, 5)],
        "8192": [parse_pair(args.output_dir, rep, 8192) for rep in range(1, 4)],
    }
    reachability = {
        size: gate_reachability([pair["speedup"] for pair in pairs])
        for size, pairs in completed.items()
    }
    require(all(
        item["gate_reachable_under_arbitrarily_favorable_missing_pair"] is False
        for item in reachability.values()
    ), "the original positive scout gate remains reachable")

    driver = args.output_dir / f"flux-{monitor['job_id']}.out"
    result = {
        "schema_version": "gicc-guarded-early-partial-negative-audit-v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "negative_gate_mathematically_unreachable",
        "audit_scope": "one_sided_post_failure_reachability_only",
        "original_monitor": file_record(args.monitor, "failed_scout_monitor"),
        "job": {
            "job_id": monitor["job_id"],
            "scheduler": monitor["scheduler"],
            "jobspec": monitor["jobspec"],
            "resource_set": monitor.get("resource_set"),
        },
        "failure": {
            "classification": "nested_step_duration_exceeded_remaining_allocation",
            "failed_arm": "replicate=4 size=8192 arm=baseline",
            "stderr": file_record(failed_stderr, "nested_duration_rejection"),
            "stdout_was_empty": True,
        },
        "completed_pairs": completed,
        "frozen_gate_reachability": reachability,
        "candidate_disposition": {
            "candidate": "guarded_early_trigger",
            "model_visible": False,
            "confirmation_eligible": False,
            "paper_performance_claim": False,
            "action": "keep candidate hidden from model evaluation",
            "reason": (
                "neither size can satisfy the preregistered scout gate even "
                "under an arbitrarily favorable missing pair"
            ),
        },
        "limitations": {
            "post_failure_audit_was_preregistered": False,
            "may_support_positive_result": False,
            "completed_timings_are_paper_performance_evidence": False,
            "rerun_needed_to_estimate_full_eight_pair_performance": True,
            "frozen_monitor_accepts_scientific_timing_notation": False,
        },
        "evidence": [
            file_record(Path(__file__).resolve(), "partial_negative_auditor"),
            file_record(
                Path(__file__).with_name("PARTIAL_NEGATIVE_AUDIT.md"),
                "partial_negative_audit_rules",
            ),
            file_record(driver, "scout_driver_output"),
        ],
        "verified_scout_artifacts": verified_artifacts,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
