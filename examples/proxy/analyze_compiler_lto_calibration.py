#!/usr/bin/env python3
"""Validate paired real-LTO calibration logs and emit training labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tools" / "gicc-passes" / "python"))

import gicc_llm_bridge as bridge  # noqa: E402
from compiler_lto_calibration import SCENARIO_FOR_KERNEL  # noqa: E402


ARMS = ("proxy", "trigger")
KERNEL_FOR_SCENARIO = {
    scenario: kernel for kernel, scenario in SCENARIO_FOR_KERNEL.items()
}
REGION = {
    "single-512-g1": (0 << 20, 512),
    "single-2k-g4": (2 << 20, 2048),
    "single-16k-g1": (4 << 20, 16384),
    "single-64k-g8": (6 << 20, 65536),
    "single-256k-g4": (8 << 20, 262144),
    "single-512k-g8": (10 << 20, 524288),
    "reuse-1k-k8-g1": (12 << 20, 1024),
    "reuse-2k-k24-g1": (14 << 20, 2048),
    "reuse-8k-k48-g4": (16 << 20, 8192),
    "adjacent-1k-k6-g1": (18 << 20, 6 * 1024),
    "adjacent-8k-k12-g4": (20 << 20, 12 * 8192),
    "adjacent-32k-k40-g8": (22 << 20, 40 * 32768),
    "far-2k-k12-i64-g4": (24 << 20, 12 * 2048),
    "far-8k-k48-i256-g8": (26 << 20, 48 * 8192),
    "far-16k-k24-i1024-g8": (28 << 20, 24 * 16384),
    "static2-2k-g2": (30 << 20, 2 * 2048),
    "static3-8k-g4": (32 << 20, 3 * 8192),
    "static6-16k-g8": (34 << 20, 6 * 16384),
}
FIELDS = re.compile(r"([a-z0-9_]+)=([^ ]+)")


class AnalysisError(ValueError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_hash(base: int, size: int) -> str:
    value = 1469598103934665603
    for index in range(base, base + size):
        byte = 1 + ((index // 256) * 17 + index) % 251
        value ^= byte
        value = (value * 1099511628211) & ((1 << 64) - 1)
    return f"{value:016x}"


def groups(dossier: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for site in dossier["sites"]:
        output.setdefault(site["kernel"], []).append(site)
    for sites in output.values():
        sites.sort(key=lambda site: site["site_id"])
    if set(output) != set(SCENARIO_FOR_KERNEL):
        raise AnalysisError("dossier kernel set differs from calibration contract")
    return output


def route(sites: list[dict[str, Any]], action: str) -> tuple[int, int]:
    if len(sites) == 1 and sites[0].get("in_loop") is True:
        batch = sites[0].get("batch_size")
        if not isinstance(batch, int) or batch <= 0:
            raise AnalysisError("loop group lacks a positive batch_size")
        return (batch, 0) if action == "trigger" else (0, batch + 1)
    count = len(sites)
    return (count, 0) if action == "trigger" else (0, 2 * count)


def parse_log(path: Path, group_map: dict[str, list[dict[str, Any]]]) \
        -> tuple[str, int, str, dict[str, dict[str, Any]]]:
    match = re.fullmatch(r"rep([0-9]+)-(proxy|trigger)\.log", path.name)
    if match is None:
        raise AnalysisError(f"cannot infer rep/arm from {path}")
    rep, arm = int(match.group(1)), match.group(2)
    lines = path.read_text().splitlines()
    run = [line for line in lines if line.startswith("RUN rep=")]
    if len(run) != 1:
        raise AnalysisError(f"{path}: expected one RUN line")
    provenance = dict(FIELDS.findall(run[0]))
    if provenance.get("rep") != str(rep) or provenance.get("arm") != arm:
        raise AnalysisError(f"{path}: provenance mismatch")
    binary = Path(provenance.get("binary", ""))
    if binary.name != f"compiler_lto_calibration_{arm}":
        raise AnalysisError(f"{path}: binary/arm mismatch")
    binary_sha = provenance.get("sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", binary_sha):
        raise AnalysisError(f"{path}: invalid binary SHA-256")

    rows: dict[str, dict[str, Any]] = {}
    for line in lines:
        if not line.startswith("COMPILER_LTO_CALIBRATION "):
            continue
        fields = dict(FIELDS.findall(line))
        scenario = fields.get("scenario", "")
        if scenario in rows:
            raise AnalysisError(f"{path}: duplicate scenario {scenario}")
        if scenario not in KERNEL_FOR_SCENARIO:
            raise AnalysisError(f"{path}: unknown scenario {scenario}")
        kernel = fields.get("kernel")
        if kernel != KERNEL_FOR_SCENARIO[scenario]:
            raise AnalysisError(f"{path}: scenario/kernel mismatch")
        row = {
            "runs": int(fields["runs"]),
            "median_us": float(fields["median_us"]),
            "p25_us": float(fields["p25_us"]),
            "p75_us": float(fields["p75_us"]),
            "staged": int(fields["staged"]),
            "pushed": int(fields["pushed"]),
            "hash": fields["hash"],
            "data": fields["data"],
        }
        base, size = REGION[scenario]
        if row["data"] != "OK" or row["hash"] != expected_hash(base, size):
            raise AnalysisError(f"{path}: {scenario} payload validation failed")
        staged, pushed = route(group_map[kernel], arm)
        wanted = staged * row["runs"], pushed * row["runs"]
        if (row["staged"], row["pushed"]) != wanted:
            raise AnalysisError(
                f"{path}: {scenario} route={(row['staged'], row['pushed'])} "
                f"expected={wanted}"
            )
        rows[scenario] = row
    if set(rows) != set(KERNEL_FOR_SCENARIO):
        raise AnalysisError(f"{path}: incomplete scenario set")
    return arm, rep, binary_sha, rows


def geomean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def parse_replicates(value: str) -> list[int]:
    try:
        replicates = [int(item) for item in value.split(",") if item]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("replicates must be comma-separated integers") from exc
    if not replicates or len(set(replicates)) != len(replicates):
        raise argparse.ArgumentTypeError("replicates must be nonempty and unique")
    return replicates


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("logs", nargs="+", type=Path)
    ap.add_argument("--dossier", required=True, type=Path)
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--json", required=True, type=Path)
    ap.add_argument("--expected-reps", type=parse_replicates,
                    default=parse_replicates("1,2,3,4"))
    args = ap.parse_args()
    try:
        dossier = bridge._verified_dossier(read_json(args.dossier))
        manifest = read_json(args.manifest)
        if manifest.get("dossier_id") != dossier["dossier_id"]:
            raise AnalysisError("manifest/dossier ID mismatch")
        group_map = groups(dossier)
        parsed: dict[int, dict[str, dict[str, dict[str, Any]]]] = {}
        binary_hashes = {arm: set() for arm in ARMS}
        for path in args.logs:
            arm, rep, binary_sha, rows = parse_log(path, group_map)
            if arm in parsed.setdefault(rep, {}):
                raise AnalysisError(f"duplicate rep={rep} arm={arm}")
            parsed[rep][arm] = rows
            binary_hashes[arm].add(binary_sha)
        for rep, arms in parsed.items():
            if set(arms) != set(ARMS):
                raise AnalysisError(f"rep={rep}: incomplete arms")
        if set(parsed) != set(args.expected_reps):
            raise AnalysisError(
                f"replicates={sorted(parsed)}, expected={sorted(args.expected_reps)}"
            )
        if any(len(values) != 1 for values in binary_hashes.values()):
            raise AnalysisError("an arm used multiple binaries across replicates")

        aggregate: dict[str, Any] = {}
        print("scenario                        proxy_us trigger_us best stable")
        for kernel, scenario in SCENARIO_FOR_KERNEL.items():
            medians = {
                arm: geomean([
                    parsed[rep][arm][scenario]["median_us"]
                    for rep in sorted(parsed)
                ])
                for arm in ARMS
            }
            best = min(ARMS, key=medians.get)
            winners = [
                min(ARMS, key=lambda arm: parsed[rep][arm][scenario]["median_us"])
                for rep in sorted(parsed)
            ]
            stable = len(set(winners)) == 1
            aggregate[scenario] = {
                "kernel": kernel,
                "geomean_median_us": medians,
                "best_action": best,
                "per_replicate_best_action": winners,
                "stable_winner": stable,
                "proxy_over_trigger": medians["proxy"] / medians["trigger"],
            }
            print(f"{scenario:<31} {medians['proxy']:8.3f} "
                  f"{medians['trigger']:10.3f} {best:>7} {str(stable):>6}")

        summary = {
            "schema_version": "gicc-compiler-lto-calibration-results-v1",
            "scope": "paired real-LTO calibration labels; never frozen evaluation labels",
            "dossier_id": dossier["dossier_id"],
            "dossier_sha256": sha256_file(args.dossier),
            "manifest_sha256": sha256_file(args.manifest),
            "source_sha256": manifest["source"]["sha256"],
            "frozen_evaluation_results_read": False,
            "replicates": sorted(parsed),
            "binary_sha256_by_arm": {
                arm: next(iter(binary_hashes[arm])) for arm in ARMS
            },
            "scenario_count": len(aggregate),
            "stable_winner_count": sum(
                row["stable_winner"] for row in aggregate.values()
            ),
            "aggregate": aggregate,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"stable={summary['stable_winner_count']}/{len(aggregate)}")
        return 0
    except (AnalysisError, bridge.BridgeError, OSError, ValueError) as exc:
        print(f"compiler-lto-calibration-analysis: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
