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
from collections import Counter
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
UNIFORM_MANIFEST_SCHEMA = "gicc-communication-plan-uniform-controls-v1"
PLACEMENT_MANIFEST_SCHEMA = "gicc-communication-plan-placement-controls-v1"
KIND_CODES = {
    "proxy_device": "p",
    "trigger_descriptor_batch": "t",
    "trigger_coalesced_loop": "c",
    "trigger_coalesced_early": "e",
}
EXPECTED_KINDS = (
    "proxy_device",
    "trigger_descriptor_batch",
    "trigger_coalesced_loop",
)
PLACEMENT_KINDS = EXPECTED_KINDS + ("trigger_coalesced_early",)


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
    opportunity: dict[str, Any], expected_kinds: tuple[str, ...] = EXPECTED_KINDS,
) -> dict[str, dict[str, Any]]:
    candidates = {
        candidate["kind"]: candidate
        for candidate in opportunity["candidates"]
    }
    if set(candidates) != set(expected_kinds):
        raise ControlError(
            f"{opportunity['opportunity_id']}: candidate kinds "
            f"{sorted(candidates)}, expected {list(expected_kinds)}"
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


def generate_uniform_controls(
    graph_value: Any, outdir: Path, source_sha256: str, *,
    kinds: tuple[str, ...] = EXPECTED_KINDS,
    schema: str = UNIFORM_MANIFEST_SCHEMA,
) -> dict[str, Any]:
    """Generate one uniform arm per requested compiler candidate kind."""
    graph = plans.verified_graph(graph_value)
    opportunities = _ordered_opportunities(graph)
    if len(source_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in source_sha256
    ):
        raise ControlError("source_sha256 must be 64 lowercase hex characters")
    choices = [_candidate_by_kind(item, kinds) for item in opportunities]
    outdir.mkdir(parents=True, exist_ok=True)
    arms: list[dict[str, Any]] = []
    for kind in kinds:
        name = f"uniform_{KIND_CODES[kind]}"
        response = {
            "schema_version": plans.DECISION_SCHEMA,
            "graph_id": graph["graph_id"],
            "selections": {},
        }
        selected: list[dict[str, Any]] = []
        for opportunity, candidates in zip(opportunities, choices, strict=True):
            candidate = candidates[kind]
            opportunity_id = opportunity["opportunity_id"]
            response["selections"][opportunity_id] = {
                "candidate_id": candidate["candidate_id"],
                "confidence": 1.0,
                "rationale": "compiler-generated uniform capacity control",
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
            "producer": "compiler-generated uniform control",
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
    payload = {
        "schema_version": schema,
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


def generate_placement_controls(
    graph_value: Any, outdir: Path, source_sha256: str,
) -> dict[str, Any]:
    return generate_uniform_controls(
        graph_value,
        outdir,
        source_sha256,
        kinds=PLACEMENT_KINDS,
        schema=PLACEMENT_MANIFEST_SCHEMA,
    )


def verify_manifest(graph_value: Any, manifest_value: Any, root: Path) -> None:
    graph = plans.verified_graph(graph_value)
    if not isinstance(manifest_value, dict):
        raise ControlError("control manifest must be a JSON object")
    schema = manifest_value.get("schema_version")
    if schema not in (
        MANIFEST_SCHEMA, UNIFORM_MANIFEST_SCHEMA, PLACEMENT_MANIFEST_SCHEMA,
    ):
        raise ControlError(
            f"expected manifest schema {MANIFEST_SCHEMA} or "
            f"{UNIFORM_MANIFEST_SCHEMA}, or {PLACEMENT_MANIFEST_SCHEMA}"
        )
    manifest_id = manifest_value.get("manifest_id")
    payload = dict(manifest_value)
    payload.pop("manifest_id", None)
    if manifest_id != bridge._fingerprint(payload):
        raise ControlError("manifest_id does not match manifest content")
    if manifest_value.get("graph_id") != graph["graph_id"]:
        raise ControlError("manifest graph_id does not match graph")
    arms = manifest_value.get("arms")
    expected_arm_count = {
        MANIFEST_SCHEMA: 9,
        UNIFORM_MANIFEST_SCHEMA: 3,
        PLACEMENT_MANIFEST_SCHEMA: 4,
    }[schema]
    expected_kinds = (
        PLACEMENT_KINDS
        if schema == PLACEMENT_MANIFEST_SCHEMA else EXPECTED_KINDS
    )
    if not isinstance(arms, list) or len(arms) != expected_arm_count:
        raise ControlError(
            f"manifest must contain exactly {expected_arm_count} arms"
        )
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
        if (
            len(kinds) != len(graph["opportunities"])
            or any(kind not in expected_kinds for kind in kinds)
        ):
            raise ControlError(f"{arm['name']}: invalid selection summary")
        combinations.add(kinds)
    if schema == MANIFEST_SCHEMA:
        if len(graph["opportunities"]) != 2 or len(combinations) != 9:
            raise ControlError(
                "manifest does not cover the exact 3x3 action product"
            )
    else:
        uniform_kinds = {
            kinds[0] for kinds in combinations
            if kinds and len(set(kinds)) == 1
        }
        if (
            uniform_kinds != set(expected_kinds)
            or len(combinations) != expected_arm_count
        ):
            raise ControlError(
                "uniform manifest must cover one all-site arm per action kind"
            )


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
    if manifest["schema_version"] in (
        UNIFORM_MANIFEST_SCHEMA, PLACEMENT_MANIFEST_SCHEMA,
    ):
        texts: dict[str, str] = {}
        kind_for_arm: dict[str, str] = {}
        for arm in manifest["arms"]:
            path = ir_dir / f"{arm['name']}.ll"
            try:
                texts[arm["name"]] = path.read_text()
            except OSError as exc:
                raise ControlError(f"cannot read LLVM IR {path}: {exc}") from exc
            kinds = {selection["kind"] for selection in arm["selections"]}
            if len(kinds) != 1:
                raise ControlError(f"{arm['name']}: IR arm is not uniform")
            kind_for_arm[arm["name"]] = next(iter(kinds))
        arm_for_kind = {kind: arm for arm, kind in kind_for_arm.items()}
        expected_kinds = (
            PLACEMENT_KINDS
            if manifest["schema_version"] == PLACEMENT_MANIFEST_SCHEMA
            else EXPECTED_KINDS
        )
        if set(arm_for_kind) != set(expected_kinds):
            raise ControlError("uniform IR set is missing an action kind")
        baseline = texts[arm_for_kind["proxy_device"]]
        trip_multiplicity = Counter(trips for trips, _ in signatures.values())
        byte_multiplicity = Counter(total for _, total in signatures.values())

        def batch_count(text: str, trips: int) -> int:
            return len(re.findall(
                rf"(?:tail )?call void @gicc_runtime_dwq_enqueue_batched\("
                rf"[^\n]*, i32 {trips},",
                text,
            ))

        def coalesced_count(text: str, total: int) -> int:
            return len(re.findall(
                rf"(?:tail )?call void @gicc_runtime_dwq_enqueue\("
                rf"[^\n]*, i64 {total}\)",
                text,
            ))

        for kind, arm_name in arm_for_kind.items():
            text = texts[arm_name]
            for trips, multiplicity in trip_multiplicity.items():
                delta = batch_count(text, trips) - batch_count(baseline, trips)
                wanted = multiplicity if kind == "trigger_descriptor_batch" else 0
                if delta != wanted:
                    raise ControlError(
                        f"{arm_name}: batch-count delta for trip_count={trips} "
                        f"is {delta}, expected {wanted}"
                    )
            for total, multiplicity in byte_multiplicity.items():
                delta = (
                    coalesced_count(text, total)
                    - coalesced_count(baseline, total)
                )
                wanted = (
                    multiplicity
                    if kind in (
                        "trigger_coalesced_loop", "trigger_coalesced_early",
                    )
                    else 0
                )
                if delta != wanted:
                    raise ControlError(
                        f"{arm_name}: coalesced-call delta for bytes={total} "
                        f"is {delta}, expected {wanted}"
                    )
        return

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
        if manifest["schema_version"] == MANIFEST_SCHEMA:
            # Reuse is outside the two-op exact graph and must stay on the
            # compiler-fixed 32-descriptor trigger path in every arm.
            fixed_reuse = len(re.findall(
                r"(?:tail )?call void @gicc_runtime_dwq_enqueue_batched\("
                r"[^\n]*, i32 32,",
                text,
            ))
            if fixed_reuse != 1:
                raise ControlError(
                    f"{path}: compiler-fixed reuse path count={fixed_reuse}, "
                    "expected=1"
                )


def verify_placement_device_ir(
    graph_value: Any, manifest_value: Any, root: Path, ir_dir: Path,
) -> None:
    """Prove the real device lowering moved only early-plan triggers."""
    graph = plans.verified_graph(graph_value)
    verify_manifest(graph, manifest_value, root)
    if manifest_value["schema_version"] != PLACEMENT_MANIFEST_SCHEMA:
        raise ControlError("device placement verifier requires placement controls")
    opportunity_count = len(graph["opportunities"])
    for arm in manifest_value["arms"]:
        path = ir_dir / f"{arm['name']}.device.ll"
        try:
            text = path.read_text()
        except OSError as exc:
            raise ControlError(f"cannot read device LLVM IR {path}: {exc}") from exc
        kinds = {selection["kind"] for selection in arm["selections"]}
        if len(kinds) != 1:
            raise ControlError(f"{arm['name']}: device IR arm is not uniform")
        kind = next(iter(kinds))
        moved = len(re.findall(
            r"store volatile i64[^\n]*!gicc\.communication_transform ![0-9]+",
            text,
        ))
        wanted = opportunity_count if kind == "trigger_coalesced_early" else 0
        if moved != wanted:
            raise ControlError(
                f"{path}: early-trigger store count={moved}, expected={wanted}"
            )
        has_marker = '!{!"COALESCE_LOOP_EARLY"}' in text
        if has_marker != (wanted > 0):
            raise ControlError(
                f"{path}: early-trigger metadata marker does not match plan"
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    controls = sub.add_parser("controls")
    controls.add_argument("--graph", required=True, type=Path)
    controls.add_argument("--out", required=True, type=Path)
    controls.add_argument("--source-sha256", required=True)
    uniform = sub.add_parser("uniform-controls")
    uniform.add_argument("--graph", required=True, type=Path)
    uniform.add_argument("--out", required=True, type=Path)
    uniform.add_argument("--source-sha256", required=True)
    placement = sub.add_parser("placement-controls")
    placement.add_argument("--graph", required=True, type=Path)
    placement.add_argument("--out", required=True, type=Path)
    placement.add_argument("--source-sha256", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--graph", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    verify_ir_parser = sub.add_parser("verify-ir")
    verify_ir_parser.add_argument("--graph", required=True, type=Path)
    verify_ir_parser.add_argument("--manifest", required=True, type=Path)
    verify_ir_parser.add_argument("--ir", required=True, type=Path)
    verify_device = sub.add_parser("verify-placement-device-ir")
    verify_device.add_argument("--graph", required=True, type=Path)
    verify_device.add_argument("--manifest", required=True, type=Path)
    verify_device.add_argument("--ir", required=True, type=Path)
    args = parser.parse_args()
    try:
        graph = _read_json(args.graph)
        if args.command == "controls":
            manifest = generate_controls(graph, args.out, args.source_sha256)
            print(
                f"compiler-comm-plan-eval: generated {len(manifest['arms'])} "
                f"exact controls; manifest_id={manifest['manifest_id']}"
            )
        elif args.command == "uniform-controls":
            manifest = generate_uniform_controls(
                graph, args.out, args.source_sha256
            )
            print(
                f"compiler-comm-plan-eval: generated {len(manifest['arms'])} "
                f"uniform controls over "
                f"{len(manifest['opportunity_order'])} opportunities; "
                f"manifest_id={manifest['manifest_id']}"
            )
        elif args.command == "placement-controls":
            manifest = generate_placement_controls(
                graph, args.out, args.source_sha256
            )
            print(
                f"compiler-comm-plan-eval: generated {len(manifest['arms'])} "
                f"placement controls over "
                f"{len(manifest['opportunity_order'])} opportunities; "
                f"manifest_id={manifest['manifest_id']}"
            )
        elif args.command == "verify":
            manifest = _read_json(args.manifest)
            verify_manifest(graph, manifest, args.manifest.parent)
            print(
                f"compiler-comm-plan-eval: verified {len(manifest['arms'])} "
                "compiler-generated controls"
            )
        elif args.command == "verify-ir":
            manifest = _read_json(args.manifest)
            verify_ir(graph, manifest, args.manifest.parent, args.ir)
            print(
                f"compiler-comm-plan-eval: verified LTO IR for "
                f"{len(manifest['arms'])} compiler-generated controls"
            )
        else:
            manifest = _read_json(args.manifest)
            verify_placement_device_ir(
                graph, manifest, args.manifest.parent, args.ir
            )
            print(
                "compiler-comm-plan-eval: verified device trigger placement "
                f"for {len(manifest['arms'])} controls"
            )
        return 0
    except (ControlError, plans.PlanBridgeError, OSError, ValueError) as exc:
        print(f"compiler-comm-plan-eval: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
