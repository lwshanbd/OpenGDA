#!/usr/bin/env python3
"""Bridge compiler-derived LTO facts to a schema-checked LLM decision.

This module deliberately does not call a model provider.  It creates a
replayable dossier and prompt from ``features.json``, then validates a model's
JSON response before translating it to ``gicc-hint-v1``.  Program source is
never an input and the model never emits code.

Typical two-phase build::

  python3 gicc_llm_bridge.py emit \
      --features build/features.json \
      --platform python/profiles/tioga-mi250x-slingshot11.json \
      --dossier build/llm-dossier.json --prompt build/llm-prompt.txt

  # Run the provider outside the linker and save JSON as llm-response.json.

  python3 gicc_llm_bridge.py accept \
      --dossier build/llm-dossier.json \
      --response build/llm-response.json --hint build/llm-hint.json

  GICC_MODE=lower GICC_HINT_IN=build/llm-hint.json ...

``accept`` fails closed by default: an invalid, stale, incomplete, or illegal
response produces a deterministic baseline hint.  ``--strict`` instead exits
without writing a hint, which is useful for experiments that must reject a
sample rather than execute the baseline.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


FEATURE_SCHEMA = 5
DOSSIER_SCHEMA = "gicc-llm-dossier-v1"
DECISION_SCHEMA = "gicc-llm-decision-v1"
PLATFORM_SCHEMA = "gicc-platform-profile-v1"
HINT_SCHEMA = "gicc-hint-v1"

SUPPORTED_OPS = {"put_no_db", "get_no_db"}
PATH_ORDER = ("proxy", "trigger", "ipc")
ACTION_TO_DISPATCH = {
    "proxy": "CPU_PROXY_ENQUEUE",
    "trigger": "DWQ_TRIGGER",
    "ipc": "IPC_PUSH",
}
DEFAULT_DISPATCH = "IPC_OR_DWQ"

# Only pass-derived fields enter the model dossier.  Keeping this allowlist
# makes it hard for a later producer to accidentally attach source text.
FACT_FIELDS = (
    "site_id",
    "kernel",
    "op_kind",
    "hk_capable",
    "size_kind",
    "size_log2",
    "peer_kind",
    "peer_locality",
    "in_loop",
    "loop",
    "guard_density",
    "fan_out",
    "launch_grid",
    "launch_block",
    "grid_blocks",
    "threads_per_block",
    "compute_before_flops",
    "flops_to_first_use",
    "trip_count",
    "distance_exact",
    "iter_estimate",
    "descriptor_reusable",
    "buffer_reusable",
    "coalescable",
    "max_vector_bytes",
    "batch_size",
)


class BridgeError(ValueError):
    """The compiler facts or model response violate the bridge contract."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")


def _fingerprint(payload: dict[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(payload)).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeError(f"cannot read JSON {path}: {exc}") from exc


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def _write_json_atomic(path: Path, value: Any) -> None:
    _write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _verified_dossier(dossier: Any) -> dict[str, Any]:
    if not isinstance(dossier, dict):
        raise BridgeError("dossier must be a JSON object")
    if dossier.get("schema_version") != DOSSIER_SCHEMA:
        raise BridgeError(f"expected dossier schema {DOSSIER_SCHEMA}")
    dossier_id = dossier.get("dossier_id")
    if not isinstance(dossier_id, str):
        raise BridgeError("dossier_id is missing")
    payload = dict(dossier)
    del payload["dossier_id"]
    expected = _fingerprint(payload)
    if dossier_id != expected:
        raise BridgeError(
            f"dossier_id does not match content: {dossier_id!r} != {expected!r}"
        )
    return dossier


def _legal_actions(record: dict[str, Any]) -> list[str]:
    paths = record.get("legal_paths")
    if not isinstance(paths, list) or not paths:
        raise BridgeError(f"site {record.get('site_id')!r} has no legal_paths")
    if any(not isinstance(path, str) for path in paths):
        raise BridgeError(f"site {record.get('site_id')!r} has malformed legal_paths")
    unknown = set(paths) - set(PATH_ORDER)
    if unknown:
        raise BridgeError(
            f"site {record.get('site_id')!r} has unknown legal path(s): "
            f"{sorted(unknown)}"
        )

    legal = [path for path in PATH_ORDER if path in paths]
    # IPC_PUSH dereferences a mapped peer base.  Until the topology side-band
    # fills peer_locality, the hybrid default is safe but forced IPC is not.
    if record.get("peer_locality") != "same_node" and "ipc" in legal:
        legal.remove("ipc")

    host_can_stage = record.get("hk_capable") is True
    loop = record.get("loop")
    if isinstance(loop, dict) and loop.get("degraded") is True:
        host_can_stage = False
    if not host_can_stage:
        unsafe = set(legal) - {"proxy"}
        if unsafe:
            raise BridgeError(
                f"site {record.get('site_id')!r} exposes host-only paths "
                f"despite failed compiler legality: {sorted(unsafe)}"
            )
    else:
        # Abstention means the pass's hybrid IPC_OR_DWQ default.  It requires
        # host staging and is therefore unavailable to a proxy-only site.
        legal.append("default")
    return legal


def make_dossier(features: Any, platform: Any) -> dict[str, Any]:
    """Validate compiler output and construct a content-addressed dossier."""
    if not isinstance(features, list):
        raise BridgeError("features.json must contain a JSON array")
    if not isinstance(platform, dict):
        raise BridgeError("platform profile must be a JSON object")
    if platform.get("schema_version") != PLATFORM_SCHEMA:
        raise BridgeError(f"expected platform schema {PLATFORM_SCHEMA}")

    sites: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in features:
        if not isinstance(record, dict):
            raise BridgeError("every feature record must be a JSON object")
        if record.get("schema_version") != FEATURE_SCHEMA:
            raise BridgeError(
                f"site {record.get('site_id')!r} has feature schema "
                f"{record.get('schema_version')!r}; expected {FEATURE_SCHEMA}"
            )
        if record.get("op_kind") not in SUPPORTED_OPS:
            continue
        site_id = record.get("site_id")
        if not isinstance(site_id, str) or not site_id:
            raise BridgeError("a decision site is missing site_id")
        if site_id in seen:
            raise BridgeError(f"duplicate decision site_id {site_id!r}")
        seen.add(site_id)

        facts = {key: record.get(key) for key in FACT_FIELDS}
        facts["legal_actions"] = _legal_actions(record)
        sites.append(facts)

    if not sites:
        raise BridgeError("features contain no put_no_db/get_no_db decision sites")
    sites.sort(key=lambda site: site["site_id"])

    payload: dict[str, Any] = {
        "schema_version": DOSSIER_SCHEMA,
        "feature_schema_version": FEATURE_SCHEMA,
        "objective": {
            "metric": "end_to_end_wall_time",
            "instruction": (
                "Choose one legal action per site. Prefer default when the "
                "platform evidence does not justify pinning a lowering."
            ),
        },
        "action_semantics": {
            "default": "abstain; keep the pass's IPC_OR_DWQ baseline",
            "proxy": "preserve the device operation and enqueue CPU proxy work",
            "trigger": "host-stage the descriptor and release it through DWQ",
            "ipc": "force same-node IPC; offered only for proven local peers",
        },
        "platform_profile": platform,
        "sites": sites,
    }
    dossier = dict(payload)
    dossier["dossier_id"] = _fingerprint(payload)
    return dossier


def render_prompt(dossier: dict[str, Any]) -> str:
    dossier = _verified_dossier(dossier)
    example = {
        "schema_version": DECISION_SCHEMA,
        "dossier_id": dossier["dossier_id"],
        "decisions": {
            "<site_id>": {
                "action": "<one of that site's legal_actions>",
                "confidence": 0.0,
                "rationale": "<brief compiler/platform-fact rationale>",
            }
        },
    }
    return (
        "You are the decision component of an LTO communication pass.\n"
        "The dossier below contains only compiler-derived facts and a measured "
        "platform profile. Choose exactly one action for every site. Do not "
        "invent actions, edit code, request source, or assert new legality. "
        "Use action=default when evidence is weak.\n\n"
        "Return ONLY one JSON object with this shape:\n"
        + json.dumps(example, indent=2, sort_keys=True)
        + "\n\nDOSSIER:\n"
        + json.dumps(dossier, indent=2, sort_keys=True)
        + "\n"
    )


def _baseline_hint(dossier: dict[str, Any], errors: list[str]) -> dict[str, Any]:
    sites: dict[str, Any] = {}
    for site in dossier.get("sites", []):
        legal = site.get("legal_actions", [])
        if "default" not in legal and "proxy" in legal:
            sites[site["site_id"]] = {
                "dispatch": ACTION_TO_DISPATCH["proxy"],
                "reason": "deterministic legality baseline for proxy-only site",
            }
    return {
        "version": 1,
        "schema_version": HINT_SCHEMA,
        "default_dispatch": DEFAULT_DISPATCH,
        "sites": sites,
        "llm_metadata": {
            "accepted": False,
            "dossier_id": dossier.get("dossier_id"),
            "errors": errors,
            "fallback": "deterministic compiler baseline",
        },
    }


def decision_to_hint(
    dossier_value: Any, decision: Any
) -> tuple[dict[str, Any], bool, list[str]]:
    """Translate a model decision, or return the deterministic fallback."""
    try:
        dossier = _verified_dossier(dossier_value)
    except BridgeError as exc:
        # A corrupted dossier cannot safely describe even proxy-only sites.
        raise BridgeError(f"cannot build fallback from invalid dossier: {exc}") from exc

    errors: list[str] = []
    if not isinstance(decision, dict):
        errors.append("decision must be a JSON object")
    else:
        if decision.get("schema_version") != DECISION_SCHEMA:
            errors.append(f"expected decision schema {DECISION_SCHEMA}")
        if decision.get("dossier_id") != dossier["dossier_id"]:
            errors.append("decision dossier_id is stale or does not match")
        if not isinstance(decision.get("decisions"), dict):
            errors.append("decisions must be a JSON object")

    expected = {site["site_id"]: site for site in dossier["sites"]}
    decisions = decision.get("decisions", {}) if isinstance(decision, dict) else {}
    if not isinstance(decisions, dict):
        decisions = {}
    missing = sorted(set(expected) - set(decisions))
    unknown = sorted(set(decisions) - set(expected))
    if missing:
        errors.append(f"missing decision site(s): {missing}")
    if unknown:
        errors.append(f"unknown decision site(s): {unknown}")

    accepted_sites: dict[str, Any] = {}
    for site_id in sorted(set(expected) & set(decisions)):
        entry = decisions[site_id]
        if not isinstance(entry, dict):
            errors.append(f"site {site_id!r}: decision must be an object")
            continue
        action = entry.get("action")
        if action not in expected[site_id]["legal_actions"]:
            errors.append(
                f"site {site_id!r}: illegal action {action!r}; legal actions are "
                f"{expected[site_id]['legal_actions']}"
            )
            continue
        confidence = entry.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0.0 <= float(confidence) <= 1.0
        ):
            errors.append(f"site {site_id!r}: confidence must be finite in [0, 1]")
            continue
        rationale = entry.get("rationale", "")
        if not isinstance(rationale, str) or len(rationale) > 512:
            errors.append(f"site {site_id!r}: rationale must be at most 512 characters")
            continue
        if action != "default":
            accepted_sites[site_id] = {
                "dispatch": ACTION_TO_DISPATCH[action],
                "reason": f"LLM confidence={float(confidence):.3f}: {rationale}",
            }

    if errors:
        return _baseline_hint(dossier, errors), False, errors

    hint = {
        "version": 1,
        "schema_version": HINT_SCHEMA,
        "default_dispatch": DEFAULT_DISPATCH,
        "sites": accepted_sites,
        "llm_metadata": {
            "accepted": True,
            "decision_schema": DECISION_SCHEMA,
            "dossier_id": dossier["dossier_id"],
            "decision_sites": len(decisions),
            "pinned_sites": len(accepted_sites),
        },
    }
    return hint, True, []


def _strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if len(lines) < 3 or lines[-1].strip() != "```":
        return stripped
    return "\n".join(lines[1:-1]).strip()


def _parse_decision(path: Path) -> Any:
    try:
        return json.loads(_strip_code_fence(path.read_text()))
    except (OSError, json.JSONDecodeError) as exc:
        raise BridgeError(f"cannot read model decision {path}: {exc}") from exc


def _emit(args: argparse.Namespace) -> int:
    dossier = make_dossier(_read_json(args.features), _read_json(args.platform))
    _write_json_atomic(args.dossier, dossier)
    _write_text_atomic(args.prompt, render_prompt(dossier))
    print(
        f"gicc-llm-bridge: wrote {len(dossier['sites'])} compiler sites; "
        f"dossier_id={dossier['dossier_id']}",
        file=sys.stderr,
    )
    return 0


def _accept(args: argparse.Namespace) -> int:
    dossier = _read_json(args.dossier)
    try:
        decision = _parse_decision(args.response)
    except BridgeError as exc:
        verified = _verified_dossier(dossier)
        errors = [str(exc)]
        if args.strict:
            print(f"gicc-llm-bridge: REJECTED: {errors[0]}", file=sys.stderr)
            return 2
        _write_json_atomic(args.hint, _baseline_hint(verified, errors))
        print(
            f"gicc-llm-bridge: REJECTED; wrote deterministic fallback to {args.hint}: "
            f"{errors[0]}",
            file=sys.stderr,
        )
        return 0

    hint, accepted, errors = decision_to_hint(dossier, decision)
    if not accepted and args.strict:
        print("gicc-llm-bridge: REJECTED: " + "; ".join(errors), file=sys.stderr)
        return 2
    _write_json_atomic(args.hint, hint)
    state = "accepted" if accepted else "REJECTED; deterministic fallback"
    print(
        f"gicc-llm-bridge: {state}; wrote {len(hint['sites'])} site hint(s) "
        f"to {args.hint}",
        file=sys.stderr,
    )
    if errors:
        for error in errors:
            print(f"gicc-llm-bridge:   {error}", file=sys.stderr)
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    emit = sub.add_parser("emit", help="create a dossier and provider prompt")
    emit.add_argument("--features", type=Path, required=True)
    emit.add_argument("--platform", type=Path, required=True)
    emit.add_argument("--dossier", type=Path, required=True)
    emit.add_argument("--prompt", type=Path, required=True)
    emit.set_defaults(run=_emit)

    accept = sub.add_parser("accept", help="validate a response and emit an LTO hint")
    accept.add_argument("--dossier", type=Path, required=True)
    accept.add_argument("--response", type=Path, required=True)
    accept.add_argument("--hint", type=Path, required=True)
    accept.add_argument(
        "--strict",
        action="store_true",
        help="reject without writing a fallback hint",
    )
    accept.set_defaults(run=_accept)
    return parser


def main() -> int:
    try:
        args = _parser().parse_args()
        return int(args.run(args))
    except BridgeError as exc:
        print(f"gicc-llm-bridge: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
