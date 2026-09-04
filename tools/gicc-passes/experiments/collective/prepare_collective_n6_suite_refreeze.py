#!/usr/bin/env python3
"""Replace collective_n8 with a confirmed compiler-only collective_n6 entry.

This transition is deliberately downstream of the topology-matched N6 runtime
confirmation.  It replays that confirmation, verifies the exact N6 graph, and
regenerates the suite while preserving every non-collective entry semantically.
It has no provider, scheduler, compiler, or application-source action.
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
ROOT = HERE.parents[4]
PASS_PYTHON = HERE.parents[1] / "python"
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(HERE))

import analyze_compiler_collective_n6_confirmation as n6_confirmation  # noqa: E402
import gicc_compiler_decision_suite as suites  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


REFREEZE_SCHEMA = "gicc-collective-n6-suite-refreeze-v1"
OLD_LABEL = "collective_n8"
NEW_LABEL = "collective_n6"
BOUNDARY = {
    "compiler_lto_decisions_only": True,
    "application_source_input": False,
    "application_source_visible_to_model": False,
    "application_source_modified": False,
    "model_invoked": False,
    "provider_call_authorized": False,
    "scheduler_job_submitted": False,
    "current_suite_modified": False,
}


class RefreezeError(RuntimeError):
    """The N6 confirmation cannot safely replace the N8 suite entry."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RefreezeError(message)


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
    require(resolved.is_file(), f"missing {role}: {resolved}")
    return {
        "role": role,
        "path": display_path(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def verify_confirmation(
    confirmation_path: Path, n6_graph_path: Path,
) -> dict[str, Any]:
    """Replay the exact N6 confirmation and bind its compiler graph."""
    value = read_json(confirmation_path)
    require(
        isinstance(value, dict)
        and value.get("schema_version") == n6_confirmation.base.RESULT_SCHEMA,
        "N6 suite refreeze requires an N6 confirmation",
    )
    payload = dict(value)
    result_id = payload.pop("result_id", None)
    require(result_id == bridge._fingerprint(payload),
            "N6 confirmation ID does not match content")
    require(value.get("confirmation_gate", {}).get("passed") is True,
            "N6 confirmation gate did not pass")
    for key, expected in {
        "model_invoked": False,
        "application_source_modified": False,
        "provider_call_authorized": False,
    }.items():
        require(value.get(key) is expected,
                f"N6 confirmation boundary changed: {key}")
    transition_path = Path(value.get("transition", ""))
    require(
        transition_path.is_absolute() and transition_path.is_file()
        and sha256_file(transition_path) == value.get("transition_sha256"),
        "N6 confirmation transition changed",
    )
    transition, files = n6_confirmation.base.validate_transition(
        transition_path
    )
    confirmed_graph_path = files["compiler_graph"]
    require(
        confirmed_graph_path.resolve() == n6_graph_path.resolve()
        and sha256_file(confirmed_graph_path) == sha256_file(n6_graph_path),
        "N6 confirmation binds another graph file",
    )
    graph = suites.collective.verified_graph(read_json(n6_graph_path))
    require(graph["graph_id"] == transition.get("graph_id"),
            "N6 confirmation binds another graph ID")
    topology = graph.get("platform_profile", {}).get("topology", {})
    require(
        topology.get("nodes") == 6
        and topology.get("ranks_per_node") == 8
        and topology.get("gpus_per_node") == 8,
        "replacement collective graph is not the frozen N6 topology",
    )
    monitors = value.get("allocation_monitors")
    require(isinstance(monitors, list) and len(monitors) == 3,
            "N6 confirmation lacks three allocation monitors")
    monitor_paths = [Path(item.get("monitor", "")) for item in monitors]
    require(all(path.is_absolute() for path in monitor_paths),
            "N6 confirmation monitor path is not absolute")
    require(
        n6_confirmation.base.analyze_monitors(
            transition_path, monitor_paths
        ) == value,
        "N6 confirmation does not replay from raw evidence",
    )
    require(transition.get("transition_id") == value.get("transition_id"),
            "N6 confirmation transition identity changed")
    return value


def _spec_maps(
    communication: list[tuple[str, Path]],
    structural: list[tuple[str, Path]],
) -> dict[str, tuple[str, Path]]:
    records: dict[str, tuple[str, Path]] = {}
    for family, specs in (
        ("communication", communication),
        ("structural", structural),
    ):
        for label, path in specs:
            require(label not in {OLD_LABEL, NEW_LABEL},
                    "collective replacement graph is not caller-selectable")
            require(label not in records, f"duplicate suite graph label: {label}")
            records[label] = (family, path.resolve())
    return records


def make_refrozen_suite(
    current_suite_path: Path, current_prompt_dir: Path,
    confirmation_path: Path, n8_graph_path: Path, n6_graph_path: Path,
    communication: list[tuple[str, Path]],
    structural: list[tuple[str, Path]],
    *, prompt_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    current = suites.verified_suite(
        read_json(current_suite_path), current_prompt_dir,
    )
    confirmation = verify_confirmation(confirmation_path, n6_graph_path)
    entries = {entry["label"]: entry for entry in current["entries"]}
    require(OLD_LABEL in entries and NEW_LABEL not in entries,
            "current suite must contain collective_n8 and not collective_n6")
    old = entries[OLD_LABEL]
    require(old.get("decision_family")
            == "collective_algorithm_and_size_policy",
            "collective_n8 is not a collective-policy entry")
    require(sum(
        entry.get("decision_family")
        == "collective_algorithm_and_size_policy"
        for entry in entries.values()
    ) == 1, "current suite must contain exactly one collective entry")
    old_graph = suites.collective.verified_graph(read_json(n8_graph_path))
    old_topology = old_graph.get("platform_profile", {}).get("topology", {})
    require(
        old.get("graph_id") == old_graph.get("graph_id")
        and old.get("graph_file_sha256") == sha256_file(n8_graph_path)
        and old_topology.get("nodes") == 8
        and old_topology.get("ranks_per_node") == 8
        and old_topology.get("gpus_per_node") == 8,
        "current collective_n8 entry does not bind the frozen N8 topology",
    )

    specs = _spec_maps(communication, structural)
    require(set(specs) == set(entries) - {OLD_LABEL},
            "refreeze must supply exactly every non-collective suite graph")
    expected_families = {
        "communication_route_or_schedule": "communication",
        "communication_coalescing_and_trigger_placement": "structural",
    }
    for label, entry in entries.items():
        if label == OLD_LABEL:
            continue
        require(expected_families.get(entry.get("decision_family"))
                == specs[label][0],
                f"suite graph family changed for {label}")

    refrozen = suites.make_suite(
        [
            (label, path) for label, (family, path) in specs.items()
            if family == "communication"
        ],
        [(NEW_LABEL, n6_graph_path.resolve())],
        [
            (label, path) for label, (family, path) in specs.items()
            if family == "structural"
        ],
        prompt_dir=prompt_dir,
    )
    new_entries = {entry["label"]: entry for entry in refrozen["entries"]}
    require(set(new_entries) == (set(entries) - {OLD_LABEL}) | {NEW_LABEL},
            "N6 refreeze changed the suite label set")
    for label in set(entries) - {OLD_LABEL}:
        require(new_entries[label] == entries[label],
                f"N6 refreeze changed unrelated entry {label}")
    new = new_entries[NEW_LABEL]
    verified_n6_graph = suites.collective.verified_graph(
        read_json(n6_graph_path)
    )
    require(
        new.get("graph_id") == verified_n6_graph.get("graph_id")
        and new.get("graph_file_sha256") == sha256_file(n6_graph_path),
        "collective_n6 entry does not bind the confirmed graph",
    )
    old_space = old.get("decision_space", {})
    new_space = new.get("decision_space", {})
    require(
        old_space.get("independent_policy_count") == 4096
        and new_space.get("independent_policy_count") == 4096
        and old_space.get("decision_slot_count") == 4
        and new_space.get("decision_slot_count") == 4
        and old_space.get("selectable_option_id_count") == 32
        and new_space.get("selectable_option_id_count") == 32,
        "N8-to-N6 transition changed the frozen compiler action capacity",
    )
    transition = {
        "old_label": OLD_LABEL,
        "new_label": NEW_LABEL,
        "old_entry_id": old["entry_id"],
        "new_entry_id": new["entry_id"],
        "old_graph_id": old["graph_id"],
        "new_graph_id": new["graph_id"],
        "old_policy_count": 4096,
        "new_policy_count": 4096,
        "old_topology_nodes": 8,
        "new_topology_nodes": 6,
        "all_noncollective_entries_preserved": True,
        "action_capacity_preserved": True,
    }
    return refrozen, confirmation, transition


def materialize_bundle(
    output_dir: Path, current_suite_path: Path, current_prompt_dir: Path,
    confirmation_path: Path, n8_graph_path: Path, n6_graph_path: Path,
    communication: list[tuple[str, Path]],
    structural: list[tuple[str, Path]],
) -> dict[str, Any]:
    prompts = output_dir / "prompts"
    refrozen, confirmation, transition = make_refrozen_suite(
        current_suite_path, current_prompt_dir, confirmation_path,
        n8_graph_path, n6_graph_path, communication, structural,
        prompt_dir=prompts,
    )
    suite_path = output_dir / "suite.json"
    suite_path.write_text(json_text(refrozen), encoding="utf-8")
    specs = _spec_maps(communication, structural)
    inputs = [
        file_record(current_suite_path, "current_suite"),
        file_record(confirmation_path, "passed_n6_confirmation"),
        file_record(n8_graph_path, "frozen_n8_graph"),
        file_record(n6_graph_path, "confirmed_n6_graph"),
        file_record(Path(__file__), "n6_suite_refreezer"),
        file_record(
            Path(n6_confirmation.__file__), "n6_confirmation_analyzer"
        ),
        file_record(
            n6_confirmation.BASE_SCRIPT, "n6_confirmation_analyzer_base"
        ),
    ]
    inputs.extend(
        file_record(path, f"graph:{family}:{label}")
        for label, (family, path) in sorted(specs.items())
    )
    outputs = []
    for path in sorted(item for item in output_dir.rglob("*") if item.is_file()):
        relative = path.relative_to(output_dir).as_posix()
        outputs.append({
            "role": (
                "refrozen_suite" if relative == "suite.json"
                else "refrozen_prompt_or_schema"
            ),
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        })
    payload = {
        "schema_version": REFREEZE_SCHEMA,
        "status": "refrozen_suite_ready_for_readiness_audit",
        "current_suite_id": read_json(current_suite_path)["suite_id"],
        "refrozen_suite_id": refrozen["suite_id"],
        "confirmation_result_id": confirmation["result_id"],
        "entry_transition": transition,
        "boundary": dict(BOUNDARY),
        "next_stage": (
            "run the N6-aware readiness and input-separation audits; freeze "
            "an exact provider request only if those audits permit it"
        ),
        "current_prompt_dir": display_path(current_prompt_dir),
        "inputs": sorted(inputs, key=lambda item: item["role"]),
        "outputs": outputs,
    }
    manifest = {**payload, "refreeze_id": bridge._fingerprint(payload)}
    (output_dir / "manifest.json").write_text(
        json_text(manifest), encoding="utf-8"
    )
    return manifest


def prepare_bundle(
    output_dir: Path, current_suite_path: Path, current_prompt_dir: Path,
    confirmation_path: Path, n8_graph_path: Path, n6_graph_path: Path,
    communication: list[tuple[str, Path]],
    structural: list[tuple[str, Path]],
) -> dict[str, Any]:
    output = output_dir.resolve()
    require(not output.exists(), f"refusing to overwrite suite refreeze: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent,
    ))
    try:
        manifest = materialize_bundle(
            temporary, current_suite_path, current_prompt_dir,
            confirmation_path, n8_graph_path, n6_graph_path,
            communication, structural,
        )
        require(not output.exists(),
                f"refusing to overwrite suite refreeze: {output}")
        os.rename(temporary, output)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def verify_contained(manifest_path: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    require(
        isinstance(manifest, dict)
        and manifest.get("schema_version") == REFREEZE_SCHEMA,
        "unexpected N6 suite-refreeze manifest schema",
    )
    payload = dict(manifest)
    require(payload.pop("refreeze_id", None) == bridge._fingerprint(payload),
            "N6 suite-refreeze ID changed")
    require(manifest.get("boundary") == BOUNDARY,
            "N6 suite refreeze crossed its compiler-only boundary")
    records = manifest.get("inputs")
    require(isinstance(records, list), "N6 suite refreeze lacks inputs")
    paths: dict[str, Path] = {}
    specs: dict[str, list[tuple[str, Path]]] = {
        "communication": [], "structural": [],
    }
    for record in records:
        require(
            isinstance(record, dict)
            and isinstance(record.get("role"), str)
            and isinstance(record.get("path"), str)
            and isinstance(record.get("sha256"), str)
            and isinstance(record.get("bytes"), int)
            and not isinstance(record.get("bytes"), bool),
            "N6 suite refreeze has an invalid input record",
        )
        role = record["role"]
        path = recorded_path(record["path"])
        require(role not in paths and path.is_file(),
                f"missing or duplicate N6 refreeze input: {role}")
        require(
            sha256_file(path) == record["sha256"]
            and path.stat().st_size == record["bytes"],
            f"N6 suite-refreeze input changed: {path}",
        )
        paths[role] = path
        if role.startswith("graph:"):
            parts = role.split(":", 2)
            require(len(parts) == 3 and parts[1] in specs,
                    f"invalid N6 graph role: {role}")
            specs[parts[1]].append((parts[2], path))
    fixed_roles = {
        "current_suite", "passed_n6_confirmation", "frozen_n8_graph",
        "confirmed_n6_graph",
        "n6_suite_refreezer", "n6_confirmation_analyzer",
        "n6_confirmation_analyzer_base",
    }
    graph_roles = {
        f"graph:{family}:{label}"
        for family, values in specs.items() for label, _ in values
    }
    require(set(paths) == fixed_roles | graph_roles,
            "N6 suite refreeze input roles changed")
    require(paths["n6_suite_refreezer"] == Path(__file__).resolve(),
            "N6 suite refreeze names another refreezer")
    require(paths["n6_confirmation_analyzer"]
            == Path(n6_confirmation.__file__).resolve(),
            "N6 suite refreeze names another confirmation analyzer")
    require(paths["n6_confirmation_analyzer_base"]
            == n6_confirmation.BASE_SCRIPT.resolve(),
            "N6 suite refreeze names another analyzer base")
    prompt_value = manifest.get("current_prompt_dir")
    require(isinstance(prompt_value, str),
            "N6 suite refreeze lacks the current prompt directory")
    current_prompt_dir = recorded_path(prompt_value)
    with tempfile.TemporaryDirectory() as directory:
        temporary = Path(directory)
        regenerated = materialize_bundle(
            temporary, paths["current_suite"], current_prompt_dir,
            paths["passed_n6_confirmation"], paths["frozen_n8_graph"],
            paths["confirmed_n6_graph"],
            specs["communication"], specs["structural"],
        )
        require(regenerated == manifest,
                "N6 suite-refreeze manifest does not regenerate")
        output = manifest_path.resolve().parent
        expected_names = {
            path.relative_to(temporary).as_posix()
            for path in temporary.rglob("*") if path.is_file()
        }
        observed_names = {
            path.relative_to(output).as_posix()
            for path in output.rglob("*") if path.is_file()
        }
        require(expected_names == observed_names,
                "N6 suite-refreeze output set changed")
        for name in expected_names:
            expected = temporary / name
            observed = output / name
            require(
                sha256_file(expected) == sha256_file(observed)
                and expected.stat().st_size == observed.stat().st_size,
                f"N6 suite-refreeze output changed: {observed}",
            )
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
    prepare.add_argument("--confirmation-analysis", type=Path, required=True)
    prepare.add_argument("--n8-graph", type=Path, required=True)
    prepare.add_argument("--n6-graph", type=Path, required=True)
    prepare.add_argument(
        "--communication", type=_spec, action="append", default=[]
    )
    prepare.add_argument("--structural", type=_spec, action="append", default=[])
    prepare.add_argument("--output-dir", type=Path, required=True)
    verify = subparsers.add_parser("verify-contained")
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            manifest = prepare_bundle(
                args.output_dir, args.current_suite, args.current_prompt_dir,
                args.confirmation_analysis, args.n8_graph, args.n6_graph,
                args.communication, args.structural,
            )
            action = "prepared"
        else:
            manifest = verify_contained(args.manifest)
            action = "verified"
        print(
            f"collective-n6-suite-refreeze: {action}; "
            f"model_invoked=false; provider_call_authorized=false; "
            f"scheduler_job_submitted=false; "
            f"refreeze_id={manifest['refreeze_id']}"
        )
        return 0
    except (
        RefreezeError, suites.SuiteError, OSError, KeyError, TypeError,
        ValueError, n6_confirmation.base.ConfirmError,
    ) as exc:
        print(f"collective-n6-suite-refreeze: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
