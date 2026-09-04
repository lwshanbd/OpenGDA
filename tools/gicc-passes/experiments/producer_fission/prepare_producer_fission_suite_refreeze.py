#!/usr/bin/env python3
"""Refreeze a decision suite with one confirmed producer-fission graph.

The tool verifies the old suite and its prompts, replays the graph-expansion
bundle, preserves every non-Jacobi suite entry byte-for-byte at the semantic
JSON level, and replaces only the Jacobi graph.  It performs no provider,
scheduler, compiler, or application-source action and does not itself grant
provider authorization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PASS_PYTHON = ROOT / "tools/gicc-passes/python"
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(HERE))

import gicc_compiler_decision_suite as suites  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import prepare_confirmed_producer_fission_graph as expansion  # noqa: E402


REFREEZE_SCHEMA = "gicc-producer-fission-suite-refreeze-v1"


class RefreezeError(RuntimeError):
    """The expanded graph cannot safely replace the frozen suite entry."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RefreezeError(f"cannot read JSON {path}: {exc}") from exc


def json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise RefreezeError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def recorded_path(value: str) -> Path:
    raw = Path(value)
    return raw.resolve() if raw.is_absolute() else (ROOT / raw).resolve()


def file_record(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise RefreezeError(f"missing {role}: {resolved}")
    return {
        "role": role,
        "path": display_path(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _expansion_output(
    manifest_path: Path, manifest: dict[str, Any], role: str,
) -> Path:
    records = manifest.get("outputs")
    matches = [
        record for record in records if record.get("role") == role
    ] if isinstance(records, list) else []
    if len(matches) != 1 or not isinstance(matches[0].get("path"), str):
        raise RefreezeError(f"graph expansion lacks output role {role}")
    path = manifest_path.resolve().parent / matches[0]["path"]
    if (not path.is_file() or sha256_file(path) != matches[0].get("sha256")
            or path.stat().st_size != matches[0].get("bytes")):
        raise RefreezeError(f"graph-expansion output changed: {path}")
    return path


def _expansion_input(
    manifest: dict[str, Any], role: str,
) -> dict[str, Any]:
    records = manifest.get("inputs")
    matches = [
        record for record in records if record.get("role") == role
    ] if isinstance(records, list) else []
    if len(matches) != 1:
        raise RefreezeError(f"graph expansion lacks input role {role}")
    return matches[0]


def _spec_maps(
    communication: list[tuple[str, Path]],
    collective: list[tuple[str, Path]],
    structural: list[tuple[str, Path]],
) -> dict[str, tuple[str, Path]]:
    records = {}
    for family, specs in (
        ("communication", communication),
        ("collective", collective),
        ("structural", structural),
    ):
        for label, path in specs:
            if label in records:
                raise RefreezeError(f"duplicate suite graph label: {label}")
            if label == "jacobi":
                raise RefreezeError(
                    "the Jacobi graph must come from the verified expansion"
                )
            records[label] = (family, path.resolve())
    return records


def make_refrozen_suite(
    current_suite_path: Path, current_prompt_dir: Path,
    expansion_manifest_path: Path,
    communication: list[tuple[str, Path]],
    collective: list[tuple[str, Path]],
    structural: list[tuple[str, Path]],
    *, prompt_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    current = suites.verified_suite(
        read_json(current_suite_path), current_prompt_dir,
    )
    expansion_manifest = expansion.verify_contained(
        expansion_manifest_path
    )
    expanded_graph_path = _expansion_output(
        expansion_manifest_path, expansion_manifest, "expanded_graph",
    )
    current_graph_record = _expansion_input(
        expansion_manifest, "frozen_current_graph",
    )
    entries = {entry["label"]: entry for entry in current["entries"]}
    if "jacobi" not in entries:
        raise RefreezeError("current suite lacks the Jacobi entry")
    jacobi = entries["jacobi"]
    transition = expansion_manifest.get("graph_transition", {})
    if (jacobi.get("decision_family") != "communication_route_or_schedule"
            or jacobi.get("graph_id") != transition.get("current_graph_id")
            or jacobi.get("graph_file_sha256")
            != current_graph_record.get("sha256")):
        raise RefreezeError(
            "current suite does not bind the expansion's frozen Jacobi graph"
        )

    specs = _spec_maps(communication, collective, structural)
    if set(specs) != set(entries) - {"jacobi"}:
        raise RefreezeError(
            "refreeze must supply exactly every non-Jacobi suite graph"
        )
    expected_families = {
        "communication_route_or_schedule": "communication",
        "collective_algorithm_and_size_policy": "collective",
        "communication_coalescing_and_trigger_placement": "structural",
    }
    for label, entry in entries.items():
        if label == "jacobi":
            continue
        family = expected_families.get(entry.get("decision_family"))
        if family is None or specs[label][0] != family:
            raise RefreezeError(f"suite graph family changed for {label}")

    communication_specs = [
        (label, path) for label, (family, path) in specs.items()
        if family == "communication"
    ] + [("jacobi", expanded_graph_path)]
    collective_specs = [
        (label, path) for label, (family, path) in specs.items()
        if family == "collective"
    ]
    structural_specs = [
        (label, path) for label, (family, path) in specs.items()
        if family == "structural"
    ]
    refrozen = suites.make_suite(
        communication_specs, collective_specs, structural_specs,
        prompt_dir=prompt_dir,
    )
    refrozen_entries = {
        entry["label"]: entry for entry in refrozen["entries"]
    }
    if set(refrozen_entries) != set(entries):
        raise RefreezeError("refrozen suite labels changed")
    for label in set(entries) - {"jacobi"}:
        if refrozen_entries[label] != entries[label]:
            raise RefreezeError(f"refreeze changed unrelated entry {label}")
    updated = refrozen_entries["jacobi"]
    expanded_graph_record = next(
        record for record in expansion_manifest["outputs"]
        if record["role"] == "expanded_graph"
    )
    if (updated["graph_id"] != transition.get("expanded_graph_id")
            or updated["graph_file_sha256"]
            != expanded_graph_record["sha256"]):
        raise RefreezeError("refrozen suite does not bind the expanded graph")
    old_space = jacobi["decision_space"]
    new_space = updated["decision_space"]
    if (new_space.get("selectable_candidate_id_count")
            != old_space.get("selectable_candidate_id_count", 0) + 1
            or new_space.get("independent_policy_count")
            != old_space.get("independent_policy_count", 0) + 1
            or new_space.get("masked_candidate_count")
            != old_space.get("masked_candidate_count", 0) - 1
            or transition.get("candidate_id") is None):
        raise RefreezeError("Jacobi decision-space delta is not exactly +1/-1")
    delta = {
        "label": "jacobi",
        "old_entry_id": jacobi["entry_id"],
        "new_entry_id": updated["entry_id"],
        "old_graph_id": jacobi["graph_id"],
        "new_graph_id": updated["graph_id"],
        "new_candidate_id": transition["candidate_id"],
        "new_candidate_kind": transition["candidate_kind"],
        "old_policy_count": old_space["independent_policy_count"],
        "new_policy_count": new_space["independent_policy_count"],
        "old_masked_candidate_count": old_space["masked_candidate_count"],
        "new_masked_candidate_count": new_space["masked_candidate_count"],
        "all_other_entries_preserved": True,
    }
    return refrozen, expansion_manifest, delta


def materialize_bundle(
    output_dir: Path, current_suite_path: Path, current_prompt_dir: Path,
    expansion_manifest_path: Path,
    communication: list[tuple[str, Path]],
    collective: list[tuple[str, Path]],
    structural: list[tuple[str, Path]],
) -> dict[str, Any]:
    prompts = output_dir / "prompts"
    refrozen, expansion_manifest, delta = make_refrozen_suite(
        current_suite_path, current_prompt_dir, expansion_manifest_path,
        communication, collective, structural, prompt_dir=prompts,
    )
    suite_path = output_dir / "suite.json"
    suite_path.write_text(json_text(refrozen), encoding="utf-8")
    specs = _spec_maps(communication, collective, structural)
    inputs = [
        file_record(current_suite_path, "current_suite"),
        file_record(expansion_manifest_path, "producer_expansion_manifest"),
    ]
    inputs.extend(
        file_record(path, f"graph:{family}:{label}")
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
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    payload = {
        "schema_version": REFREEZE_SCHEMA,
        "status": "refrozen_suite_ready_for_readiness_audit",
        "current_suite_id": read_json(current_suite_path)["suite_id"],
        "refrozen_suite_id": refrozen["suite_id"],
        "producer_expansion_id": expansion_manifest["expansion_id"],
        "entry_transition": delta,
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_input": False,
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
        "current_prompt_dir": display_path(current_prompt_dir),
        "inputs": sorted(inputs, key=lambda item: item["role"]),
        "outputs": output_records,
    }
    manifest = {**payload, "refreeze_id": bridge._fingerprint(payload)}
    (output_dir / "manifest.json").write_text(
        json_text(manifest), encoding="utf-8",
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
        raise RefreezeError(f"refusing to overwrite suite refreeze: {output}")
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
                f"refusing to overwrite suite refreeze: {output}"
            )
        os.rename(temporary, output)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def verify_contained(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if (not isinstance(manifest, dict)
            or manifest.get("schema_version") != REFREEZE_SCHEMA):
        raise RefreezeError("unexpected suite-refreeze manifest schema")
    payload = dict(manifest)
    refreeze_id = payload.pop("refreeze_id", None)
    if refreeze_id != bridge._fingerprint(payload):
        raise RefreezeError("suite-refreeze ID changed")
    records = manifest.get("inputs")
    if not isinstance(records, list):
        raise RefreezeError("suite refreeze lacks inputs")
    paths = {}
    specs = {"communication": [], "collective": [], "structural": []}
    for record in records:
        if (not isinstance(record, dict)
                or not isinstance(record.get("role"), str)
                or not isinstance(record.get("path"), str)
                or not isinstance(record.get("sha256"), str)
                or isinstance(record.get("bytes"), bool)
                or not isinstance(record.get("bytes"), int)):
            raise RefreezeError("suite refreeze has an invalid input record")
        role = record["role"]
        path = recorded_path(record["path"])
        if role in paths or not path.is_file():
            raise RefreezeError(f"missing or duplicate refreeze input: {role}")
        if (sha256_file(path) != record["sha256"]
                or path.stat().st_size != record["bytes"]):
            raise RefreezeError(f"suite-refreeze input changed: {path}")
        paths[role] = path
        if role.startswith("graph:"):
            parts = role.split(":", 2)
            if len(parts) != 3 or parts[1] not in specs:
                raise RefreezeError(f"invalid graph role: {role}")
            specs[parts[1]].append((parts[2], path))
    if set(paths) - {"current_suite", "producer_expansion_manifest"} != {
            f"graph:{family}:{label}"
            for family, values in specs.items() for label, _ in values}:
        raise RefreezeError("suite refreeze has unexpected input roles")
    if not {"current_suite", "producer_expansion_manifest"}.issubset(paths):
        raise RefreezeError("suite refreeze lacks required input roles")
    prompt_value = manifest.get("current_prompt_dir")
    if not isinstance(prompt_value, str):
        raise RefreezeError("suite refreeze lacks the current prompt directory")
    current_prompt_dir = recorded_path(prompt_value)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        regenerated = materialize_bundle(
            temporary, paths["current_suite"], current_prompt_dir,
            paths["producer_expansion_manifest"],
            specs["communication"], specs["collective"], specs["structural"],
        )
        if regenerated != manifest:
            raise RefreezeError("suite-refreeze manifest does not regenerate")
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
            raise RefreezeError("suite-refreeze output set changed")
        for name in expected_names:
            expected = temporary / name
            observed = output / name
            if (sha256_file(expected) != sha256_file(observed)
                    or expected.stat().st_size != observed.stat().st_size):
                raise RefreezeError(f"suite-refreeze output changed: {observed}")
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
            f"producer-fission-suite-refreeze: {action}; model_invoked=false; "
            f"provider_call_authorized=false; scheduler_job_submitted=false; "
            f"refreeze_id={manifest['refreeze_id']}"
        )
        return 0
    except (
        RefreezeError, expansion.ExpansionError,
        expansion.confirmation.ConfirmError,
        expansion.confirmation.common.MonitorError,
        suites.SuiteError, suites.communication.GroupPlanError,
        suites.collective.CollectivePlanError,
        suites.structural.PlanBridgeError,
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"producer-fission-suite-refreeze: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
