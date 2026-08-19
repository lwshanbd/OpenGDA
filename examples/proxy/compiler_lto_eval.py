#!/usr/bin/env python3
"""Freeze and validate the compiler-only LLM evaluation boundary.

The benchmark source is never rendered into a prompt.  This tool consumes the
LTO pass's features, verifies that the intended heterogeneous decision sites
actually survived the real HIP compilation, and creates deterministic control
responses that exercise the same decision-schema -> hint bridge as a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
BRIDGE_DIR = ROOT / "tools" / "gicc-passes" / "python"
sys.path.insert(0, str(BRIDGE_DIR))

import gicc_llm_bridge as bridge  # noqa: E402


FREEZE_SCHEMA = "gicc-compiler-lto-eval-freeze-v1"

EXPECTED: dict[str, dict[str, Any]] = {
    "eval_tiny_single": {
        "count": 1, "size_bytes": 256, "batch_size": 1,
        "trip_count": None, "grid_blocks": 1, "hk_capable": True,
        "descriptor_reusable": False, "coalescable": False,
    },
    "eval_reuse_batch": {
        "count": 1, "size_bytes": 4096, "batch_size": 32,
        "trip_count": 32, "grid_blocks": 1, "hk_capable": True,
        "descriptor_reusable": True, "coalescable": False,
        "batched_loop": True,
    },
    "eval_adjacent_batch": {
        "count": 1, "size_bytes": 4096, "batch_size": 16,
        "trip_count": 16, "grid_blocks": 1, "hk_capable": True,
        "descriptor_reusable": False, "coalescable": True,
        "batched_loop": True,
    },
    "eval_far_batch": {
        "count": 1, "size_bytes": 4096, "batch_size": 64,
        "trip_count": 64, "grid_blocks": 8, "hk_capable": True,
        "descriptor_reusable": False, "coalescable": True,
        "batched_loop": True,
        "minimum_flops_to_first_use": 100,
    },
    "eval_large_single": {
        "count": 1, "size_bytes": 1 << 20, "batch_size": 1,
        "trip_count": None, "grid_blocks": 8, "hk_capable": True,
        "descriptor_reusable": False, "coalescable": False,
    },
    "eval_dynamic_offset": {
        "count": 1, "size_bytes": 4096, "batch_size": 1,
        "trip_count": None, "grid_blocks": 1, "hk_capable": False,
        "descriptor_reusable": False, "coalescable": False,
    },
    "eval_static4_parallel": {
        "count": 4, "size_bytes": 4096, "batch_size": 4,
        "trip_count": None, "grid_blocks": 4, "hk_capable": True,
        "descriptor_reusable": False, "coalescable": False,
    },
}


class EvalError(ValueError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvalError(f"cannot read JSON {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def decision_for(dossier: dict[str, Any], policy: str) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for site in dossier["sites"]:
        legal = site["legal_actions"]
        if policy == "default":
            if "default" in legal:
                action = "default"
            elif "trigger" in legal:
                action = "trigger"
            else:
                action = "proxy"
        elif policy == "proxy":
            action = "proxy"
        elif policy == "trigger":
            action = "trigger" if "trigger" in legal else "proxy"
        else:
            raise EvalError(f"unknown control policy {policy}")
        if action not in legal:
            raise EvalError(
                f"control {policy} chose illegal {action} for {site['site_id']}"
            )
        decisions[site["site_id"]] = {
            "action": action,
            "confidence": 1.0,
            "rationale": f"deterministic {policy} control; not a model trial",
        }
    return {
        "schema_version": bridge.DECISION_SCHEMA,
        "dossier_id": dossier["dossier_id"],
        "producer": {"kind": "deterministic_control", "policy": policy},
        "decisions": decisions,
    }


def make_controls(dossier_path: Path, output: Path) -> list[Path]:
    dossier = bridge._verified_dossier(read_json(dossier_path))
    written: list[Path] = []
    for policy in ("default", "proxy", "trigger"):
        response = decision_for(dossier, policy)
        hint, accepted, errors = bridge.decision_to_hint(dossier, response)
        if not accepted or errors:
            raise EvalError(f"bridge rejected {policy} control: {errors}")
        response_path = output / f"{policy}-response.json"
        hint_path = output / f"{policy}-hint.json"
        write_json(response_path, response)
        write_json(hint_path, hint)
        written.extend((response_path, hint_path))
    return written


def make_static4_oracle(dossier_path: Path, output: Path) -> list[Path]:
    """Enumerate every proxy/trigger assignment for the four-site group."""
    dossier = bridge._verified_dossier(read_json(dossier_path))
    group = sorted(
        (site for site in dossier["sites"]
         if site.get("kernel") == "eval_static4_parallel"),
        key=lambda site: site["site_id"],
    )
    if len(group) != 4:
        raise EvalError(f"expected four static-group sites, got {len(group)}")
    for site in group:
        if not {"proxy", "trigger"}.issubset(site["legal_actions"]):
            raise EvalError(f"static-group site lacks both actions: {site['site_id']}")

    output.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for mask in range(16):
        response = decision_for(dossier, "default")
        response["producer"] = {
            "kind": "exhaustive_oracle_control",
            "group": "eval_static4_parallel",
            "mask": mask,
            "bit_semantics": "1=proxy, 0=trigger in sorted site_id order",
        }
        for bit, site in enumerate(group):
            action = "proxy" if mask & (1 << bit) else "trigger"
            response["decisions"][site["site_id"]] = {
                "action": action,
                "confidence": 1.0,
                "rationale": f"exhaustive static4 oracle mask={mask:04b}",
            }
        hint, accepted, errors = bridge.decision_to_hint(dossier, response)
        if not accepted or errors:
            raise EvalError(f"bridge rejected static4 mask {mask:04b}: {errors}")
        response_path = output / f"static4-mask{mask:02d}-response.json"
        hint_path = output / f"static4-mask{mask:02d}-hint.json"
        write_json(response_path, response)
        write_json(hint_path, hint)
        written.extend((response_path, hint_path))
    return written


def decision_sites(features: Any) -> list[dict[str, Any]]:
    if not isinstance(features, list):
        raise EvalError("features must be a JSON array")
    return [
        row for row in features
        if isinstance(row, dict)
        and row.get("op_kind") in bridge.SUPPORTED_OPS
    ]


def verify(source: Path, features_path: Path, dossier_path: Path,
           profile_path: Path, prompt_path: Path) -> dict[str, Any]:
    if not source.is_file():
        raise EvalError(f"missing benchmark source {source}")
    features = read_json(features_path)
    dossier = bridge._verified_dossier(read_json(dossier_path))
    profile = read_json(profile_path)
    rebuilt = bridge.make_dossier(features, profile)
    if rebuilt != dossier:
        raise EvalError("dossier is not the canonical projection of features/profile")
    rendered = bridge.render_prompt(dossier)
    if prompt_path.read_text() != rendered:
        raise EvalError("prompt is not the canonical rendering of the dossier")

    sites = decision_sites(features)
    if len(sites) != 10:
        raise EvalError(f"expected 10 decision sites, found {len(sites)}")
    by_kernel: dict[str, list[dict[str, Any]]] = {}
    for site in sites:
        by_kernel.setdefault(site.get("kernel"), []).append(site)
    if set(by_kernel) != set(EXPECTED):
        raise EvalError(
            f"kernel set mismatch: got {sorted(by_kernel)}, "
            f"expected {sorted(EXPECTED)}"
        )

    summary: dict[str, Any] = {}
    seen_ids: set[str] = set()
    dossier_by_id = {site["site_id"]: site for site in dossier["sites"]}
    for kernel, expected in EXPECTED.items():
        rows = sorted(by_kernel[kernel], key=lambda row: row["site_id"])
        if len(rows) != expected["count"]:
            raise EvalError(
                f"{kernel}: expected {expected['count']} sites, got {len(rows)}"
            )
        for row in rows:
            site_id = row.get("site_id")
            if not isinstance(site_id, str) or not site_id or site_id in seen_ids:
                raise EvalError(f"invalid or duplicate site_id {site_id!r}")
            seen_ids.add(site_id)
            if row.get("schema_version") != bridge.FEATURE_SCHEMA:
                raise EvalError(f"{site_id}: feature schema is not v6")
            for key in (
                "size_bytes", "batch_size", "trip_count", "grid_blocks",
                "hk_capable", "descriptor_reusable", "coalescable",
            ):
                if row.get(key) != expected[key]:
                    raise EvalError(
                        f"{site_id}: {key}={row.get(key)!r}, "
                        f"expected {expected[key]!r}"
                    )
            if row.get("static_launch_sites") != 1:
                raise EvalError(
                    f"{site_id}: expected one static launch site, got "
                    f"{row.get('static_launch_sites')!r}"
                )
            contexts = row.get("launch_contexts")
            if not isinstance(contexts, list) or len(contexts) != 1:
                raise EvalError(f"{site_id}: expected one launch context")
            minimum = expected.get("minimum_flops_to_first_use")
            if minimum is not None and (
                not isinstance(row.get("flops_to_first_use"), int)
                or row["flops_to_first_use"] < minimum
            ):
                raise EvalError(
                    f"{site_id}: flops_to_first_use did not preserve far distance"
                )

            dossier_site = dossier_by_id.get(site_id)
            if dossier_site is None:
                raise EvalError(f"{site_id}: missing from dossier")
            legal = dossier_site.get("legal_actions")
            if expected["hk_capable"] is False:
                wanted_legal = ["proxy"]
            elif expected.get("batched_loop"):
                wanted_legal = ["proxy", "trigger"]
            else:
                wanted_legal = ["proxy", "trigger", "default"]
            if legal != wanted_legal:
                raise EvalError(
                    f"{site_id}: legal_actions={legal!r}, expected {wanted_legal!r}"
                )
        summary[kernel] = {
            "sites": len(rows),
            "site_ids": [row["site_id"] for row in rows],
            "size_bytes": expected["size_bytes"],
            "batch_size": expected["batch_size"],
            "trip_count": expected["trip_count"],
            "grid_blocks": expected["grid_blocks"],
            "hk_capable": expected["hk_capable"],
            "descriptor_reusable": expected["descriptor_reusable"],
            "coalescable": expected["coalescable"],
        }

    allowed = set(bridge.FACT_FIELDS) | {"legal_actions"}
    for site in dossier["sites"]:
        extra = set(site) - allowed
        if extra:
            raise EvalError(
                f"dossier site {site.get('site_id')}: non-compiler keys {sorted(extra)}"
            )

    return {
        "schema_version": FREEZE_SCHEMA,
        "source": {
            "path": "examples/proxy/compiler_lto_eval.cpp",
            "sha256": sha256_file(source),
        },
        "feature_schema_version": bridge.FEATURE_SCHEMA,
        "decision_sites": len(sites),
        "dossier_id": dossier["dossier_id"],
        "artifacts": {
            "features.json": sha256_file(features_path),
            "dossier.json": sha256_file(dossier_path),
            "prompt.txt": sha256_file(prompt_path),
            "profile.json": sha256_file(profile_path),
        },
        "kernels": summary,
        "boundary": {
            "model_input": "compiler facts plus hash-bound deployment profile",
            "source_in_prompt": False,
            "model_output": bridge.DECISION_SCHEMA,
            "pass_input": bridge.HINT_SCHEMA,
        },
    }


def add_control_hashes(manifest: dict[str, Any], controls: Path) -> None:
    for path in sorted(controls.glob("*-response.json")):
        manifest["artifacts"][f"controls/{path.name}"] = sha256_file(path)
    for path in sorted(controls.glob("*-hint.json")):
        manifest["artifacts"][f"controls/{path.name}"] = sha256_file(path)


def freeze(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    args.frozen.mkdir(parents=True, exist_ok=True)
    controls_out = args.frozen / "controls"
    controls_out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.features, args.frozen / "features.json")
    shutil.copyfile(args.dossier, args.frozen / "dossier.json")
    shutil.copyfile(args.prompt, args.frozen / "prompt.txt")
    shutil.copyfile(args.profile, args.frozen / "profile.json")
    for path in make_controls(args.dossier, controls_out):
        if not path.is_file():
            raise EvalError(f"failed to create control artifact {path}")
    add_control_hashes(manifest, controls_out)
    write_json(args.frozen / "manifest.json", manifest)


def check_frozen(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    expected_manifest = args.frozen / "manifest.json"
    if not expected_manifest.is_file():
        raise EvalError(f"missing frozen manifest {expected_manifest}")
    controls_tmp = args.controls
    make_controls(args.dossier, controls_tmp)
    add_control_hashes(manifest, controls_tmp)
    if read_json(expected_manifest) != manifest:
        raise EvalError("frozen manifest differs from current compiler output")

    pairs = [
        (args.features, args.frozen / "features.json"),
        (args.dossier, args.frozen / "dossier.json"),
        (args.prompt, args.frozen / "prompt.txt"),
        (args.profile, args.frozen / "profile.json"),
    ]
    for current in sorted(controls_tmp.glob("*.json")):
        pairs.append((current, args.frozen / "controls" / current.name))
    for current, frozen_path in pairs:
        if not frozen_path.is_file() or current.read_bytes() != frozen_path.read_bytes():
            raise EvalError(f"frozen artifact differs: {frozen_path}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "command",
        choices=("verify", "controls", "oracle", "freeze", "check-frozen"),
    )
    ap.add_argument("--source", type=Path,
                    default=ROOT / "examples/proxy/compiler_lto_eval.cpp")
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--dossier", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--profile", type=Path,
                    default=ROOT / "examples/proxy/compiler_lto_eval_profile.json")
    ap.add_argument("--controls", type=Path, required=True)
    ap.add_argument("--frozen", type=Path,
                    default=ROOT / "docs/experiments/compiler-lto-eval/frozen-v1")
    return ap


def main() -> int:
    args = parser().parse_args()
    try:
        # Every artifact generator is tied to the same real compiler output.
        # This prevents a convenient standalone `controls`/`oracle` invocation
        # from silently accepting a stale or hand-authored dossier.
        manifest = verify(args.source, args.features, args.dossier,
                          args.profile, args.prompt)
        if args.command == "controls":
            paths = make_controls(args.dossier, args.controls)
            print(f"wrote {len(paths)} deterministic control artifacts")
            return 0
        if args.command == "oracle":
            paths = make_static4_oracle(args.dossier, args.controls)
            print(f"wrote {len(paths)} exhaustive static4 oracle artifacts")
            return 0
        if args.command == "freeze":
            freeze(args, manifest)
            print(f"froze {manifest['decision_sites']} compiler sites in {args.frozen}")
        elif args.command == "check-frozen":
            check_frozen(args, manifest)
            print(f"frozen compiler dossier verified: {manifest['dossier_id']}")
        else:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except (EvalError, bridge.BridgeError, OSError) as exc:
        print(f"compiler-lto-eval: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
