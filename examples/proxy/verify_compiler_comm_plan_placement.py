#!/usr/bin/env python3
"""Verify the complete frozen compiler-only trigger-placement bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "examples" / "proxy"))

import compiler_comm_plan_eval as controls  # noqa: E402


class BundleError(ValueError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise BundleError(f"cannot read artifact {path}: {exc}") from exc


def require_hash(label: str, path: Path, expected: Any) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise BundleError(f"{label}: protocol SHA-256 is invalid")
    actual = sha256_file(path)
    if actual != expected:
        raise BundleError(
            f"{label}: SHA-256 mismatch for {path}: {actual}, expected {expected}"
        )


def verify_bundle(protocol_path: Path, root: Path, build: Path) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    if protocol.get("schema_version") != (
        "gicc-communication-plan-placement-protocol-v1"
    ):
        raise BundleError("unexpected placement protocol schema")
    if protocol.get("status") != (
        "preregistered-before-first-trigger-placement-runtime-job"
    ):
        raise BundleError("placement protocol is not in preregistered state")
    artifacts = protocol.get("artifacts")
    if not isinstance(artifacts, dict):
        raise BundleError("protocol artifacts are missing")

    generated = build / "generated"
    graph_path = generated / "opportunity-graph.json"
    dossier_path = generated / "dossier.json"
    manifest_path = generated / "controls" / "manifest.json"
    prompt_path = generated / "prompt.txt"
    profile_path = root / "examples/proxy/compiler_comm_plan_placement_profile.json"
    source_path = root / "examples/proxy/compiler_lto_calibration.cpp"
    plugin_path = root / "tools/gicc-passes/build/libgicc-passes.so"

    require_hash("source", source_path, artifacts.get("source_sha256"))
    require_hash("pass plugin", plugin_path, artifacts.get("pass_plugin_sha256"))
    require_hash(
        "platform profile", profile_path,
        artifacts.get("platform_profile_sha256"),
    )
    require_hash(
        "features", generated / "features.json",
        artifacts.get("features_sha256"),
    )
    require_hash(
        "compiler dossier", dossier_path,
        artifacts.get("compiler_dossier", {}).get("sha256"),
    )
    require_hash(
        "opportunity graph", graph_path,
        artifacts.get("opportunity_graph", {}).get("sha256"),
    )
    require_hash("prompt", prompt_path, artifacts.get("prompt_sha256"))
    require_hash(
        "control manifest", manifest_path,
        artifacts.get("control_manifest", {}).get("sha256"),
    )

    graph = read_json(graph_path)
    dossier = read_json(dossier_path)
    manifest = read_json(manifest_path)
    expected_ids = {
        "dossier_id": artifacts.get("compiler_dossier", {}).get("id"),
        "graph_id": artifacts.get("opportunity_graph", {}).get("id"),
        "manifest_id": artifacts.get("control_manifest", {}).get("id"),
    }
    actual_ids = {
        "dossier_id": dossier.get("dossier_id"),
        "graph_id": graph.get("graph_id"),
        "manifest_id": manifest.get("manifest_id"),
    }
    if actual_ids != expected_ids:
        raise BundleError(
            f"content IDs do not match protocol: {actual_ids}, expected {expected_ids}"
        )
    if manifest.get("source_sha256") != artifacts.get("source_sha256"):
        raise BundleError("manifest source hash does not match protocol")
    opportunities = graph.get("opportunities")
    if (
        not isinstance(opportunities, list)
        or len(opportunities) != 6
        or any(len(item.get("candidates", [])) != 4 for item in opportunities)
    ):
        raise BundleError("graph is not the frozen six-by-four action space")
    boundary = graph.get("boundary", {})
    if (
        boundary.get("source_visible") is not False
        or boundary.get("model_may_generate_code") is not False
        or boundary.get("model_may_generate_ir") is not False
    ):
        raise BundleError("compiler-only graph boundary is not enforced")
    if manifest.get("model_invoked") is not False:
        raise BundleError("control manifest unexpectedly contains model output")

    binary_hashes = artifacts.get("binaries", {})
    host_hashes = artifacts.get("host_lto_ir", {})
    device_hashes = artifacts.get("device_ir", {})
    for arm in ("uniform_p", "uniform_t", "uniform_c", "uniform_e"):
        require_hash(
            f"{arm} binary",
            build / f"compiler_comm_plan_placement_{arm}",
            binary_hashes.get(arm),
        )
        require_hash(
            f"{arm} host IR", generated / "ir" / f"{arm}.ll",
            host_hashes.get(arm),
        )
        require_hash(
            f"{arm} device IR", generated / "ir" / f"{arm}.device.ll",
            device_hashes.get(arm),
        )

    controls.verify_manifest(graph, manifest, manifest_path.parent)
    controls.verify_ir(graph, manifest, manifest_path.parent, generated / "ir")
    controls.verify_placement_device_ir(
        graph, manifest, manifest_path.parent, generated / "ir"
    )
    return {
        "graph_id": graph["graph_id"],
        "manifest_id": manifest["manifest_id"],
        "source_sha256": manifest["source_sha256"],
        "arms": [arm["name"] for arm in manifest["arms"]],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=ROOT,
        help="repository root",
    )
    parser.add_argument(
        "--build", type=Path,
        default=ROOT / "build_ofi/compiler_comm_plan_placement",
    )
    parser.add_argument(
        "--protocol", type=Path,
        default=(
            ROOT / "docs/experiments/compiler-comm-plan-capacity/"
            "protocol-placement-v1.json"
        ),
    )
    args = parser.parse_args()
    try:
        result = verify_bundle(
            args.protocol.resolve(), args.root.resolve(), args.build.resolve()
        )
        print(
            "compiler-comm-placement-bundle: verified "
            f"graph={result['graph_id']} manifest={result['manifest_id']} "
            f"arms={','.join(result['arms'])}"
        )
        return 0
    except (BundleError, controls.ControlError, OSError, ValueError) as exc:
        print(f"compiler-comm-placement-bundle: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
