#!/usr/bin/env python3
"""Refreeze a suite with the confirmed reused-descriptor loop graph.

The shared suite-transition checker preserves every non-loop_lto entry and
replaces only the compiler-bound loop_lto graph.  This tool has no model,
provider, scheduler, compiler, or source-edit path and grants no provider-call
authorization.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PASS_PYTHON = ROOT / "tools/gicc-passes/python"
PRODUCER_DIR = HERE.parent / "producer_fission"
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PRODUCER_DIR))

import prepare_confirmed_reused_loop_descriptor_graph as expansion  # noqa: E402
import prepare_producer_fission_suite_refreeze as base  # noqa: E402


REFREEZE_SCHEMA = "gicc-reused-loop-descriptor-suite-refreeze-v1"
RefreezeError = base.RefreezeError


def make_refrozen_suite(
    current_suite_path: Path, current_prompt_dir: Path,
    expansion_manifest_path: Path,
    communication: list[tuple[str, Path]],
    collective: list[tuple[str, Path]],
    structural: list[tuple[str, Path]],
    *, prompt_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    return base.make_refrozen_entry_suite(
        current_suite_path, current_prompt_dir, expansion_manifest_path,
        communication, collective, structural,
        expansion_verifier=expansion.verify_contained,
        target_label="loop_lto", expected_mask_delta=0,
        prompt_dir=prompt_dir,
    )


def materialize_bundle(
    output_dir: Path, current_suite_path: Path, current_prompt_dir: Path,
    expansion_manifest_path: Path,
    communication: list[tuple[str, Path]],
    collective: list[tuple[str, Path]],
    structural: list[tuple[str, Path]],
) -> dict[str, Any]:
    refrozen, expansion_manifest, delta = make_refrozen_suite(
        current_suite_path, current_prompt_dir, expansion_manifest_path,
        communication, collective, structural,
        prompt_dir=output_dir / "prompts",
    )
    suite_path = output_dir / "suite.json"
    suite_path.write_text(base.json_text(refrozen), encoding="utf-8")
    specs = base._spec_maps(
        communication, collective, structural, target_label="loop_lto",
    )
    inputs = [
        base.file_record(current_suite_path, "current_suite"),
        base.file_record(
            expansion_manifest_path, "reused_expansion_manifest",
        ),
    ]
    inputs.extend(
        base.file_record(path, f"graph:{family}:{label}")
        for label, (family, path) in sorted(specs.items())
    )
    output_records = []
    for path in sorted(
        item for item in output_dir.rglob("*") if item.is_file()
    ):
        relative = path.relative_to(output_dir).as_posix()
        output_records.append({
            "role": "refrozen_suite" if relative == "suite.json"
            else "refrozen_prompt_or_schema",
            "path": relative,
            "sha256": base.sha256_file(path),
            "bytes": path.stat().st_size,
        })
    payload = {
        "schema_version": REFREEZE_SCHEMA,
        "status": "refrozen_suite_ready_for_readiness_audit",
        "current_suite_id": base.read_json(current_suite_path)["suite_id"],
        "refrozen_suite_id": refrozen["suite_id"],
        "reused_expansion_id": expansion_manifest["expansion_id"],
        "entry_transition": delta,
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_hash_verified": True,
            "application_source_visible_to_model": False,
            "application_source_modified": False,
            "model_invoked": False,
            "provider_call_authorized": False,
            "scheduler_job_submitted": False,
            "current_suite_modified": False,
        },
        "next_stage": (
            "run the readiness audit on this refrozen suite; freeze an exact "
            "provider request only if that audit permits the protocol"
        ),
        "current_prompt_dir": base.display_path(current_prompt_dir),
        "inputs": sorted(inputs, key=lambda item: item["role"]),
        "outputs": output_records,
    }
    manifest = {**payload, "refreeze_id": base.bridge._fingerprint(payload)}
    (output_dir / "manifest.json").write_text(
        base.json_text(manifest), encoding="utf-8",
    )
    return manifest


def prepare_bundle(
    output_dir: Path, current_suite_path: Path, current_prompt_dir: Path,
    expansion_manifest_path: Path,
    communication: list[tuple[str, Path]],
    collective: list[tuple[str, Path]],
    structural: list[tuple[str, Path]],
) -> dict[str, Any]:
    output = output_dir.resolve()
    if output.exists():
        raise RefreezeError(f"refusing to overwrite reused refreeze: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
    ))
    try:
        manifest = materialize_bundle(
            temporary, current_suite_path, current_prompt_dir,
            expansion_manifest_path, communication, collective, structural,
        )
        if output.exists():
            raise RefreezeError(
                f"refusing to overwrite reused refreeze: {output}"
            )
        os.rename(temporary, output)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def verify_contained(manifest_path: Path) -> dict[str, Any]:
    manifest = base.read_json(manifest_path)
    if (not isinstance(manifest, dict)
            or manifest.get("schema_version") != REFREEZE_SCHEMA):
        raise RefreezeError("unexpected reused-refreeze manifest schema")
    payload = dict(manifest)
    refreeze_id = payload.pop("refreeze_id", None)
    if refreeze_id != base.bridge._fingerprint(payload):
        raise RefreezeError("reused-refreeze ID changed")
    records = manifest.get("inputs")
    if not isinstance(records, list):
        raise RefreezeError("reused refreeze lacks inputs")
    paths = {}
    specs = {"communication": [], "collective": [], "structural": []}
    for record in records:
        if (not isinstance(record, dict)
                or not isinstance(record.get("role"), str)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)):
            raise RefreezeError("reused refreeze has an invalid input record")
        role = record["role"]
        path = base.recorded_path(record["path"])
        if role in paths or not path.is_file():
            raise RefreezeError(f"missing or duplicate reused input: {role}")
        if (base.sha256_file(path) != record["sha256"]
                or path.stat().st_size != record["bytes"]):
            raise RefreezeError(f"reused-refreeze input changed: {path}")
        paths[role] = path
        if role.startswith("graph:"):
            parts = role.split(":", 2)
            if len(parts) != 3 or parts[1] not in specs:
                raise RefreezeError(f"invalid graph role: {role}")
            specs[parts[1]].append((parts[2], path))
    required = {"current_suite", "reused_expansion_manifest"}
    graph_roles = {
        f"graph:{family}:{label}"
        for family, values in specs.items() for label, _ in values
    }
    if not required.issubset(paths) or set(paths) - required != graph_roles:
        raise RefreezeError("reused refreeze input roles changed")
    prompt_value = manifest.get("current_prompt_dir")
    if not isinstance(prompt_value, str):
        raise RefreezeError("reused refreeze lacks current prompt directory")
    current_prompt_dir = base.recorded_path(prompt_value)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        regenerated = materialize_bundle(
            temporary, paths["current_suite"], current_prompt_dir,
            paths["reused_expansion_manifest"],
            specs["communication"], specs["collective"], specs["structural"],
        )
        if regenerated != manifest:
            raise RefreezeError("reused-refreeze manifest does not regenerate")
        output = manifest_path.resolve().parent
        expected_names = {
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*") if path.is_file()
        }
        observed_names = {
            path.relative_to(output).as_posix()
            for path in output.rglob("*") if path.is_file()
        }
        if expected_names != observed_names:
            raise RefreezeError("reused-refreeze output set changed")
        for name in expected_names:
            expected = temporary / name
            observed = output / name
            if (base.sha256_file(expected) != base.sha256_file(observed)
                    or expected.stat().st_size != observed.stat().st_size):
                raise RefreezeError(f"reused-refreeze output changed: {observed}")
    return manifest


def _spec(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("expected LABEL=GRAPH.json")
    return label, Path(raw_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--current-suite", type=Path, required=True)
    prepare.add_argument("--current-prompt-dir", type=Path, required=True)
    prepare.add_argument("--expansion-manifest", type=Path, required=True)
    prepare.add_argument("--communication", type=_spec, action="append", default=[])
    prepare.add_argument("--collective", type=_spec, action="append", default=[])
    prepare.add_argument("--structural", type=_spec, action="append", default=[])
    prepare.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify-contained")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            manifest = prepare_bundle(
                args.output_dir, args.current_suite, args.current_prompt_dir,
                args.expansion_manifest, args.communication,
                args.collective, args.structural,
            )
            action = "prepared"
        else:
            manifest = verify_contained(args.manifest)
            action = "verified-contained"
        print(
            f"reused-descriptor-suite-refreeze: {action}; "
            f"model_invoked=false; provider_call_authorized=false; "
            f"scheduler_job_submitted=false; "
            f"refreeze_id={manifest['refreeze_id']}"
        )
        return 0
    except (
        RefreezeError, expansion.ExpansionError,
        expansion.confirmation.ConfirmError,
        expansion.confirmation.common.MonitorError,
        base.suites.SuiteError, base.suites.communication.GroupPlanError,
        base.suites.collective.CollectivePlanError,
        base.suites.structural.PlanBridgeError,
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"reused-descriptor-suite-refreeze: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
