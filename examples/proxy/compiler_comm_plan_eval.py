#!/usr/bin/env python3
"""Generate and audit exact compiler-only communication-plan controls.

This tool never reads or rewrites benchmark source.  It consumes the verified
compiler opportunity graph, enumerates the finite candidate-ID product, and
asks ``gicc_comm_plan_bridge`` to translate each selection into a narrow LTO
hint.  The resulting controls establish the exact oracle for the exposed
compiler action space before any model is evaluated.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PASS_PYTHON = ROOT / "tools" / "gicc-passes" / "python"
sys.path.insert(0, str(PASS_PYTHON))

import gicc_comm_plan_bridge as plans  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


MANIFEST_SCHEMA = "gicc-communication-plan-controls-v1"
KIND_CODES = {
    "proxy_device": "p",
    "trigger_descriptor_batch": "t",
    "trigger_coalesced_loop": "c",
}
EXPECTED_KINDS = tuple(KIND_CODES)


class ControlError(ValueError):
    """The exact-control contract is malformed or incomplete."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ControlError(f"cannot read JSON {path}: {exc}") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    bridge._write_json_atomic(path, value)


def _ordered_opportunities(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        graph["opportunities"],
        key=lambda opportunity: (
            opportunity.get("compiler_facts", {}).get("kernel", ""),
            opportunity["opportunity_id"],
        ),
    )


def _candidate_by_kind(
    opportunity: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    candidates = {
        candidate["kind"]: candidate
        for candidate in opportunity["candidates"]
    }
    if set(candidates) != set(EXPECTED_KINDS):
        raise ControlError(
            f"{opportunity['opportunity_id']}: candidate kinds "
            f"{sorted(candidates)}, expected {list(EXPECTED_KINDS)}"
        )
    return candidates


def generate_controls(
    graph_value: Any, outdir: Path, source_sha256: str,
) -> dict[str, Any]:
    graph = plans.verified_graph(graph_value)
    opportunities = _ordered_opportunities(graph)
    if len(opportunities) != 2:
        raise ControlError(
            "v1 capacity gate requires exactly two structural opportunities; "
            f"got {len(opportunities)}"
        )
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ControlError("source_sha256 must be 64 lowercase hex characters")

    outdir.mkdir(parents=True, exist_ok=True)
    choices = [_candidate_by_kind(item) for item in opportunities]
    arms: list[dict[str, Any]] = []
    for kinds in itertools.product(EXPECTED_KINDS, repeat=len(opportunities)):
        code = "".join(KIND_CODES[kind] for kind in kinds)
        name = f"plan_{code}"
        response = {
            "schema_version": plans.DECISION_SCHEMA,
            "graph_id": graph["graph_id"],
            "selections": {},
        }
        selected: list[dict[str, Any]] = []
        for opportunity, candidates, kind in zip(
            opportunities, choices, kinds, strict=True
        ):
            candidate = candidates[kind]
            opportunity_id = opportunity["opportunity_id"]
            response["selections"][opportunity_id] = {
                "candidate_id": candidate["candidate_id"],
                "confidence": 1.0,
                "rationale": "compiler-generated exact action-space control",
            }
            selected.append({
                "opportunity_id": opportunity_id,
                "kernel": opportunity["compiler_facts"]["kernel"],
                "kind": kind,
                "candidate_id": candidate["candidate_id"],
                "materializer": candidate["materializer"],
                "effects": candidate["effects"],
            })

        hint, accepted, errors = plans.plan_to_hint(graph, response)
        if not accepted:
            raise ControlError(f"internally generated {name} rejected: {errors}")
        hint["llm_metadata"].update({
            "producer": "compiler-generated exact control",
            "model_invoked": False,
        })
        response_path = outdir / f"{name}-response.json"
        hint_path = outdir / f"{name}-hint.json"
        _write_json(response_path, response)
        _write_json(hint_path, hint)
        arms.append({
            "name": name,
            "response": response_path.name,
            "response_sha256": _sha256(response_path),
            "hint": hint_path.name,
            "hint_sha256": _sha256(hint_path),
            "selections": selected,
        })

    if len(arms) != 9 or len({item["name"] for item in arms}) != 9:
        raise ControlError("exact 3x3 product did not produce nine unique arms")
    payload = {
        "schema_version": MANIFEST_SCHEMA,
        "graph_id": graph["graph_id"],
        "source_sha256": source_sha256,
        "model_invoked": False,
        "model_output_scope": "compiler candidate IDs only",
        "opportunity_order": [
            {
                "opportunity_id": item["opportunity_id"],
                "kernel": item["compiler_facts"]["kernel"],
            }
            for item in opportunities
        ],
        "arms": arms,
    }
    manifest = dict(payload)
    manifest["manifest_id"] = bridge._fingerprint(payload)
    _write_json(outdir / "manifest.json", manifest)
    return manifest


def verify_manifest(graph_value: Any, manifest_value: Any, root: Path) -> None:
    graph = plans.verified_graph(graph_value)
    if (
        not isinstance(manifest_value, dict)
        or manifest_value.get("schema_version") != MANIFEST_SCHEMA
    ):
        raise ControlError(f"expected manifest schema {MANIFEST_SCHEMA}")
    manifest_id = manifest_value.get("manifest_id")
    payload = dict(manifest_value)
    payload.pop("manifest_id", None)
    if manifest_id != bridge._fingerprint(payload):
        raise ControlError("manifest_id does not match manifest content")
    if manifest_value.get("graph_id") != graph["graph_id"]:
        raise ControlError("manifest graph_id does not match graph")
    arms = manifest_value.get("arms")
    if not isinstance(arms, list) or len(arms) != 9:
        raise ControlError("manifest must contain exactly nine arms")
    if manifest_value.get("model_invoked") is not False:
        raise ControlError("exact controls must record model_invoked=false")
    names: set[str] = set()
    combinations: set[tuple[str, ...]] = set()
    for arm in arms:
        if not isinstance(arm, dict) or not isinstance(arm.get("name"), str):
            raise ControlError("invalid arm entry")
        if arm["name"] in names:
            raise ControlError(f"duplicate arm {arm['name']}")
        names.add(arm["name"])
        response_path = root / arm["response"]
        hint_path = root / arm["hint"]
        if _sha256(response_path) != arm["response_sha256"]:
            raise ControlError(f"{response_path}: response hash mismatch")
        if _sha256(hint_path) != arm["hint_sha256"]:
            raise ControlError(f"{hint_path}: hint hash mismatch")
        hint, accepted, errors = plans.plan_to_hint(
            graph, _read_json(response_path)
        )
        if not accepted:
            raise ControlError(f"{response_path}: rejected: {errors}")
        stored_hint = _read_json(hint_path)
        stored_core = dict(stored_hint)
        stored_core["llm_metadata"] = dict(stored_core["llm_metadata"])
        stored_core["llm_metadata"].pop("producer", None)
        stored_core["llm_metadata"].pop("model_invoked", None)
        if hint != stored_core:
            raise ControlError(f"{hint_path}: hint does not match response")
        kinds = tuple(item["kind"] for item in arm.get("selections", []))
        if len(kinds) != 2 or any(kind not in EXPECTED_KINDS for kind in kinds):
            raise ControlError(f"{arm['name']}: invalid selection summary")
        combinations.add(kinds)
    if len(combinations) != 9:
        raise ControlError("manifest does not cover the exact 3x3 action product")


def verify_ir(
    graph_value: Any, manifest_value: Any, manifest_root: Path, ir_dir: Path,
) -> None:
    """Prove that every exact-control hint reached the expected host LTO IR."""
    graph = plans.verified_graph(graph_value)
    opportunities = _ordered_opportunities(graph)
    manifest = manifest_value
    verify_manifest(graph, manifest, manifest_root)
    facts_by_id = {
        item["opportunity_id"]: item["compiler_facts"]
        for item in opportunities
    }
    signatures: dict[str, tuple[int, int]] = {}
    for opportunity_id, facts in facts_by_id.items():
        trips = facts.get("trip_count")
        size = facts.get("size_bytes")
        if not isinstance(trips, int) or not isinstance(size, int):
            raise ControlError(f"{opportunity_id}: missing constant IR signature")
        signatures[opportunity_id] = (trips, trips * size)
    if len(set(signatures.values())) != len(signatures):
        raise ControlError("v1 IR verifier requires unique opportunity signatures")

    for arm in manifest["arms"]:
        path = ir_dir / f"{arm['name']}.ll"
        try:
            text = path.read_text()
        except OSError as exc:
            raise ControlError(f"cannot read LLVM IR {path}: {exc}") from exc
        for selection in arm["selections"]:
            opportunity_id = selection["opportunity_id"]
            trips, total_bytes = signatures[opportunity_id]
            batch = len(re.findall(
                rf"(?:tail )?call void @gicc_runtime_dwq_enqueue_batched\("
                rf"[^\n]*, i32 {trips},",
                text,
            ))
            coalesced = len(re.findall(
                rf"(?:tail )?call void @gicc_runtime_dwq_enqueue\("
                rf"[^\n]*, i64 {total_bytes}\)",
                text,
            ))
            wanted = {
                "proxy_device": (0, 0),
                "trigger_descriptor_batch": (1, 0),
                "trigger_coalesced_loop": (0, 1),
            }[selection["kind"]]
            if (batch, coalesced) != wanted:
                raise ControlError(
                    f"{path}: {selection['kernel']} materialized "
                    f"batch/coalesced={(batch, coalesced)}, expected={wanted}"
                )
        # Reuse is outside the LLM opportunity graph and must stay on the
        # compiler-fixed 32-descriptor trigger path in every arm.
        fixed_reuse = len(re.findall(
            r"(?:tail )?call void @gicc_runtime_dwq_enqueue_batched\("
            r"[^\n]*, i32 32,",
            text,
        ))
        if fixed_reuse != 1:
            raise ControlError(
                f"{path}: compiler-fixed reuse path count={fixed_reuse}, expected=1"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    controls = sub.add_parser("controls")
    controls.add_argument("--graph", required=True, type=Path)
    controls.add_argument("--out", required=True, type=Path)
    controls.add_argument("--source-sha256", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--graph", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    verify_ir_parser = sub.add_parser("verify-ir")
    verify_ir_parser.add_argument("--graph", required=True, type=Path)
    verify_ir_parser.add_argument("--manifest", required=True, type=Path)
    verify_ir_parser.add_argument("--ir", required=True, type=Path)
    args = parser.parse_args()
    try:
        graph = _read_json(args.graph)
        if args.command == "controls":
            manifest = generate_controls(graph, args.out, args.source_sha256)
            print(
                f"compiler-comm-plan-eval: generated {len(manifest['arms'])} "
                f"exact controls; manifest_id={manifest['manifest_id']}"
            )
        elif args.command == "verify":
            manifest = _read_json(args.manifest)
            verify_manifest(graph, manifest, args.manifest.parent)
            print(
                f"compiler-comm-plan-eval: verified {len(manifest['arms'])} "
                "exact controls"
            )
        else:
            manifest = _read_json(args.manifest)
            verify_ir(graph, manifest, args.manifest.parent, args.ir)
            print(
                f"compiler-comm-plan-eval: verified LTO IR for "
                f"{len(manifest['arms'])} exact controls"
            )
        return 0
    except (ControlError, plans.PlanBridgeError, OSError, ValueError) as exc:
        print(f"compiler-comm-plan-eval: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
