#!/usr/bin/env python3
"""Freeze and validate a disjoint compiler-path calibration dossier."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "gicc-passes" / "python"))

import gicc_llm_bridge as bridge  # noqa: E402


FREEZE_SCHEMA = "gicc-compiler-lto-calibration-freeze-v1"
FROZEN_EVAL_SIZES = {256, 4096, 1 << 20}


def spec(count: int, size: int, batch: int, grid: int, *, loop: bool = False,
         reuse: bool = False, coal: bool = False, min_flops: int = 0) \
        -> dict[str, Any]:
    return {
        "count": count,
        "size_bytes": size,
        "batch_size": batch,
        "trip_count": batch if loop else None,
        "grid_blocks": grid,
        "in_loop": loop,
        "descriptor_reusable": reuse,
        "coalescable": coal,
        "minimum_flops_to_first_use": min_flops,
    }


EXPECTED: dict[str, dict[str, Any]] = {
    "cal_single_512_g1": spec(1, 512, 1, 1),
    "cal_single_2k_g4": spec(1, 2048, 1, 4),
    "cal_single_16k_g1": spec(1, 16384, 1, 1),
    "cal_single_64k_g8": spec(1, 65536, 1, 8),
    "cal_single_256k_g4": spec(1, 262144, 1, 4),
    "cal_single_512k_g8": spec(1, 524288, 1, 8),
    "cal_reuse_1k_k8_g1": spec(1, 1024, 8, 1, loop=True, reuse=True),
    "cal_reuse_2k_k24_g1": spec(1, 2048, 24, 1, loop=True, reuse=True),
    "cal_reuse_8k_k48_g4": spec(1, 8192, 48, 4, loop=True, reuse=True),
    "cal_adjacent_1k_k6_g1": spec(1, 1024, 6, 1, loop=True, coal=True),
    "cal_adjacent_8k_k12_g4": spec(1, 8192, 12, 4, loop=True, coal=True),
    "cal_adjacent_32k_k40_g8": spec(1, 32768, 40, 8, loop=True, coal=True),
    "cal_far_2k_k12_i64_g4": spec(
        1, 2048, 12, 4, loop=True, coal=True, min_flops=100,
    ),
    "cal_far_8k_k48_i256_g8": spec(
        1, 8192, 48, 8, loop=True, coal=True, min_flops=500,
    ),
    "cal_far_16k_k24_i1024_g8": spec(
        1, 16384, 24, 8, loop=True, coal=True, min_flops=2000,
    ),
    "cal_static2_2k_g2": spec(2, 2048, 2, 2),
    "cal_static3_8k_g4": spec(3, 8192, 3, 4),
    "cal_static6_16k_g8": spec(6, 16384, 6, 8),
}

SCENARIO_FOR_KERNEL = {
    "cal_single_512_g1": "single-512-g1",
    "cal_single_2k_g4": "single-2k-g4",
    "cal_single_16k_g1": "single-16k-g1",
    "cal_single_64k_g8": "single-64k-g8",
    "cal_single_256k_g4": "single-256k-g4",
    "cal_single_512k_g8": "single-512k-g8",
    "cal_reuse_1k_k8_g1": "reuse-1k-k8-g1",
    "cal_reuse_2k_k24_g1": "reuse-2k-k24-g1",
    "cal_reuse_8k_k48_g4": "reuse-8k-k48-g4",
    "cal_adjacent_1k_k6_g1": "adjacent-1k-k6-g1",
    "cal_adjacent_8k_k12_g4": "adjacent-8k-k12-g4",
    "cal_adjacent_32k_k40_g8": "adjacent-32k-k40-g8",
    "cal_far_2k_k12_i64_g4": "far-2k-k12-i64-g4",
    "cal_far_8k_k48_i256_g8": "far-8k-k48-i256-g8",
    "cal_far_16k_k24_i1024_g8": "far-16k-k24-i1024-g8",
    "cal_static2_2k_g2": "static2-2k-g2",
    "cal_static3_8k_g4": "static3-8k-g4",
    "cal_static6_16k_g8": "static6-16k-g8",
}


class CalibrationError(ValueError):
    pass


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CalibrationError(f"cannot read JSON {path}: {exc}") from exc


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def control_response(dossier: dict[str, Any], action: str) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for site in dossier["sites"]:
        if action not in site["legal_actions"]:
            raise CalibrationError(
                f"{site['site_id']}: calibration action {action} is illegal"
            )
        decisions[site["site_id"]] = {
            "action": action,
            "confidence": 1.0,
            "rationale": f"uniform {action} calibration control; not a model",
        }
    return {
        "schema_version": bridge.DECISION_SCHEMA,
        "dossier_id": dossier["dossier_id"],
        "producer": {
            "kind": "compiler_path_calibration_control",
            "policy": action,
            "model_output": False,
        },
        "decisions": decisions,
    }


def make_controls(dossier_path: Path, output: Path) -> list[Path]:
    dossier = bridge._verified_dossier(read_json(dossier_path))
    written: list[Path] = []
    for action in ("proxy", "trigger"):
        response = control_response(dossier, action)
        hint, accepted, errors = bridge.decision_to_hint(dossier, response)
        if not accepted or errors:
            raise CalibrationError(f"bridge rejected {action}: {errors}")
        response_path = output / f"{action}-response.json"
        hint_path = output / f"{action}-hint.json"
        write_json(response_path, response)
        write_json(hint_path, hint)
        written.extend((response_path, hint_path))
    return written


def verify(source: Path, features_path: Path, dossier_path: Path,
           profile_path: Path, prompt_path: Path) -> dict[str, Any]:
    features = read_json(features_path)
    profile = read_json(profile_path)
    dossier = bridge._verified_dossier(read_json(dossier_path))
    if bridge.make_dossier(features, profile) != dossier:
        raise CalibrationError("dossier is not canonical features/profile")
    if bridge.render_prompt(dossier) != prompt_path.read_text():
        raise CalibrationError("prompt is not canonical")

    sites = [
        row for row in features
        if isinstance(row, dict) and row.get("op_kind") in bridge.SUPPORTED_OPS
    ]
    by_kernel: dict[str, list[dict[str, Any]]] = {}
    for site in sites:
        by_kernel.setdefault(site.get("kernel"), []).append(site)
    if set(by_kernel) != set(EXPECTED):
        raise CalibrationError(
            f"kernel set differs: {sorted(by_kernel)} != {sorted(EXPECTED)}"
        )
    if any(row["size_bytes"] in FROZEN_EVAL_SIZES for row in sites):
        raise CalibrationError("calibration reuses a frozen evaluation size")

    dossier_by_id = {site["site_id"]: site for site in dossier["sites"]}
    summary: dict[str, Any] = {}
    seen: set[str] = set()
    for kernel, expected in EXPECTED.items():
        rows = sorted(by_kernel[kernel], key=lambda row: row["site_id"])
        if len(rows) != expected["count"]:
            raise CalibrationError(
                f"{kernel}: {len(rows)} sites != {expected['count']}"
            )
        for row in rows:
            site_id = row.get("site_id")
            if not isinstance(site_id, str) or not site_id or site_id in seen:
                raise CalibrationError(f"invalid/duplicate site_id {site_id!r}")
            seen.add(site_id)
            if row.get("schema_version") != bridge.FEATURE_SCHEMA:
                raise CalibrationError(f"{site_id}: wrong feature schema")
            for field in (
                "size_bytes", "batch_size", "trip_count", "grid_blocks",
                "in_loop", "descriptor_reusable", "coalescable",
            ):
                if row.get(field) != expected[field]:
                    raise CalibrationError(
                        f"{site_id}: {field}={row.get(field)!r}, "
                        f"expected {expected[field]!r}"
                    )
            flops = row.get("flops_to_first_use")
            if not isinstance(flops, int) or flops < expected["minimum_flops_to_first_use"]:
                raise CalibrationError(f"{site_id}: insufficient distance facts")
            if row.get("hk_capable") is not True:
                raise CalibrationError(f"{site_id}: calibration site not host-known")
            dossier_site = dossier_by_id.get(site_id)
            if dossier_site is None:
                raise CalibrationError(f"{site_id}: absent from dossier")
            legal = dossier_site["legal_actions"]
            if not {"proxy", "trigger"}.issubset(legal):
                raise CalibrationError(f"{site_id}: missing calibration actions")
        summary[kernel] = {
            **expected,
            "scenario": SCENARIO_FOR_KERNEL[kernel],
            "site_ids": [row["site_id"] for row in rows],
        }

    allowed = set(bridge.FACT_FIELDS) | {"legal_actions"}
    for site in dossier["sites"]:
        extra = set(site) - allowed
        if extra:
            raise CalibrationError(
                f"{site['site_id']}: non-compiler dossier keys {sorted(extra)}"
            )

    return {
        "schema_version": FREEZE_SCHEMA,
        "source": {
            "path": "examples/proxy/compiler_lto_calibration.cpp",
            "sha256": sha256_file(source),
        },
        "feature_schema_version": bridge.FEATURE_SCHEMA,
        "decision_sites": len(sites),
        "scenario_count": len(EXPECTED),
        "dossier_id": dossier["dossier_id"],
        "frozen_evaluation_sizes_excluded": sorted(FROZEN_EVAL_SIZES),
        "artifacts": {
            "features.json": sha256_file(features_path),
            "dossier.json": sha256_file(dossier_path),
            "prompt.txt": sha256_file(prompt_path),
            "profile.json": sha256_file(profile_path),
        },
        "kernels": summary,
        "boundary": {
            "role": "training/calibration only",
            "source_in_model_input": False,
            "frozen_runtime_or_oracle_read": False,
            "model_output": bridge.DECISION_SCHEMA,
            "pass_input": bridge.HINT_SCHEMA,
        },
    }


def add_control_hashes(manifest: dict[str, Any], controls: Path) -> None:
    for path in sorted(controls.glob("*.json")):
        manifest["artifacts"][f"controls/{path.name}"] = sha256_file(path)


def freeze(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    args.frozen.mkdir(parents=True, exist_ok=True)
    controls = args.frozen / "controls"
    controls.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.features, args.frozen / "features.json")
    shutil.copyfile(args.dossier, args.frozen / "dossier.json")
    shutil.copyfile(args.prompt, args.frozen / "prompt.txt")
    shutil.copyfile(args.profile, args.frozen / "profile.json")
    make_controls(args.dossier, controls)
    add_control_hashes(manifest, controls)
    write_json(args.frozen / "manifest.json", manifest)


def check_frozen(args: argparse.Namespace, manifest: dict[str, Any]) -> None:
    make_controls(args.dossier, args.controls)
    add_control_hashes(manifest, args.controls)
    if read_json(args.frozen / "manifest.json") != manifest:
        raise CalibrationError("frozen manifest differs from current output")
    pairs = [
        (args.features, args.frozen / "features.json"),
        (args.dossier, args.frozen / "dossier.json"),
        (args.prompt, args.frozen / "prompt.txt"),
        (args.profile, args.frozen / "profile.json"),
    ]
    pairs.extend(
        (path, args.frozen / "controls" / path.name)
        for path in sorted(args.controls.glob("*.json"))
    )
    for current, frozen in pairs:
        if not frozen.is_file() or current.read_bytes() != frozen.read_bytes():
            raise CalibrationError(f"frozen artifact differs: {frozen}")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("verify", "controls", "freeze", "check-frozen"))
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--dossier", type=Path, required=True)
    ap.add_argument("--prompt", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--controls", type=Path, required=True)
    ap.add_argument("--frozen", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    try:
        manifest = verify(
            args.source, args.features, args.dossier, args.profile, args.prompt,
        )
        if args.command == "controls":
            paths = make_controls(args.dossier, args.controls)
            print(f"wrote {len(paths)} calibration control artifacts")
        elif args.command == "freeze":
            freeze(args, manifest)
            print(f"froze {manifest['decision_sites']} calibration sites")
        elif args.command == "check-frozen":
            check_frozen(args, manifest)
            print(f"verified calibration dossier {manifest['dossier_id']}")
        else:
            print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0
    except (CalibrationError, bridge.BridgeError, OSError, ValueError) as exc:
        print(f"compiler-lto-calibration: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
