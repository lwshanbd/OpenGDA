#!/usr/bin/env python3
"""Audit the largest currently schedulable even pdebug collective topology."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "python"))

import gicc_llm_bridge as bridge  # noqa: E402


SCHEMA = "gicc-pdebug-collective-feasibility-audit-v1"
RESOURCE_LIST_COMMAND = ["flux", "resource", "list"]
DRAIN_STATUS_COMMAND = [
    "flux", "resource", "status", "-s", "drained", "-q", "pdebug",
    "-o", "longer",
]


class FeasibilityError(RuntimeError):
    """Scheduler or frozen-bundle evidence cannot support the conclusion."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise FeasibilityError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FeasibilityError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise FeasibilityError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    try:
        display = resolved.relative_to(ROOT).as_posix()
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def run_read_only(command: list[str]) -> str:
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True,
    )
    if completed.returncode != 0:
        raise FeasibilityError(
            f"read-only scheduler command failed: {' '.join(command)}: "
            f"{completed.stderr.strip()}"
        )
    return completed.stdout


def parse_resource_list(text: str) -> dict[str, int]:
    counts = {"free": 0, "allocated": 0, "down": 0}
    rows = 0
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 4 or fields[0] not in counts:
            continue
        state, queues, raw_nodes = fields[:3]
        if "pdebug" not in queues.split(","):
            continue
        try:
            nodes = int(raw_nodes)
        except ValueError as exc:
            raise FeasibilityError("pdebug node count is not an integer") from exc
        require(nodes >= 0, "pdebug node count is negative")
        counts[state] += nodes
        rows += 1
    require(rows > 0, "resource list has no pdebug rows")
    require(counts["free"] + counts["allocated"] > 0,
            "resource list has no usable pdebug nodes")
    return counts


def parse_drain_status(text: str) -> list[dict[str, Any]]:
    records = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[2] != "drained":
            continue
        try:
            nodes = int(fields[3])
        except ValueError as exc:
            raise FeasibilityError("drained node count is not an integer") from exc
        records.append({
            "display_time": " ".join(fields[:2]),
            "state": fields[2],
            "nodes": nodes,
            "reason": " ".join(fields[4:-1]),
            "nodelist": fields[-1],
        })
    require(records, "resource status has no drained pdebug record")
    return records


def scheduler_snapshot() -> dict[str, Any]:
    counts = parse_resource_list(run_read_only(RESOURCE_LIST_COMMAND))
    drains = parse_drain_status(run_read_only(DRAIN_STATUS_COMMAND))
    require(sum(row["nodes"] for row in drains) == counts["down"],
            "resource list and drain status disagree")
    return {"counts": counts, "drained": drains}


def verified_bundle(bundle: Path, expected_nodes: int) -> dict[str, Any]:
    manifest_path = bundle / "FROZEN_V3_MANIFEST.json"
    platform_path = bundle / "inputs/platform.json"
    graph_path = bundle / "discovery/graph.json"
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict), "bundle manifest is not an object")
    payload = dict(manifest)
    manifest_id = payload.pop("manifest_id", None)
    require(manifest_id == bridge._fingerprint(payload),
            "bundle manifest ID does not match content")
    platform = read_json(platform_path)
    graph = read_json(graph_path)
    require(platform.get("topology", {}).get("nodes") == expected_nodes,
            f"bundle is not an N{expected_nodes} topology")
    require(graph.get("graph_id") == manifest.get("graph", {}).get("graph_id"),
            "bundle graph ID changed")
    require(manifest.get("compiler_only") is True
            and manifest.get("application_source_modified") is False,
            "bundle violates compiler-only boundary")
    return {
        "nodes": expected_nodes,
        "manifest_id": manifest_id,
        "graph_id": graph["graph_id"],
        "evidence": {
            "manifest": evidence(manifest_path),
            "platform": evidence(platform_path),
            "graph": evidence(graph_path),
        },
    }


def build_report(n8_bundle: Path, n6_bundle: Path) -> dict[str, Any]:
    snapshot = scheduler_snapshot()
    counts = snapshot["counts"]
    usable = counts["free"] + counts["allocated"]
    nominal = usable + counts["down"]
    largest_even = usable if usable % 2 == 0 else usable - 1
    require(nominal == 8, "pdebug nominal capacity is not eight nodes")
    require(usable == 7 and largest_even == 6,
            "pdebug no longer implies the preregistered N6 replacement")
    require(any(
        row["nodelist"] == "tioga41"
        and row["reason"] == "node falls out consistently."
        for row in snapshot["drained"]
    ), "expected persistent tioga41 drain is absent")
    payload = {
        "schema_version": SCHEMA,
        "boundary": {
            "scheduler_read_only": True,
            "job_submitted_or_cancelled": False,
            "model_or_provider_invoked": False,
            "application_source_visible_or_modified": False,
        },
        "commands": {
            "resource_list": RESOURCE_LIST_COMMAND,
            "drain_status": DRAIN_STATUS_COMMAND,
        },
        "pdebug": {
            "nominal_nodes": nominal,
            "usable_nodes": usable,
            "drained_nodes": counts["down"],
            "drain_records": snapshot["drained"],
        },
        "conclusion": {
            "n8_currently_schedulable": False,
            "largest_currently_schedulable_even_node_count": largest_even,
            "replacement_topology": "n6",
            "n8_controller_failure_is_performance_evidence": False,
            "n6_offline_freeze_is_runtime_evidence": False,
        },
        "bundles": {
            "n8_unexecuted": verified_bundle(n8_bundle, 8),
            "n6_replacement": verified_bundle(n6_bundle, 6),
        },
    }
    result = dict(payload)
    result["audit_id"] = bridge._fingerprint(payload)
    return result


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("emit", "verify"))
    parser.add_argument("--n8-bundle", type=Path, required=True)
    parser.add_argument("--n6-bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(
            args.n8_bundle.resolve(), args.n6_bundle.resolve()
        )
        if args.command == "emit":
            if args.report.exists():
                raise FeasibilityError(
                    f"refusing to overwrite report: {args.report}"
                )
            write_json_atomic(args.report, report)
            action = "wrote"
        else:
            require(read_json(args.report) == report,
                    "feasibility report does not match live scheduler state")
            action = "verified"
        largest_even = report["conclusion"][
            "largest_currently_schedulable_even_node_count"
        ]
        print(
            f"pdebug-collective-feasibility: {action}; "
            f"usable={report['pdebug']['usable_nodes']}; "
            f"largest_even={largest_even}; "
            f"audit_id={report['audit_id']}"
        )
        return 0
    except (FeasibilityError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"pdebug-collective-feasibility: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
