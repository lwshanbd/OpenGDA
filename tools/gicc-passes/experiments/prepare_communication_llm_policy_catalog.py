#!/usr/bin/env python3
"""Freeze every graph-bound communication policy as a private LTO hint.

This post-archive adapter is intentionally offline.  It verifies the exact
source-free request and complete provider archive, enumerates the compiler's
entire communication route/schedule action space, and asks the compiler bridge
to revalidate every selection before writing a private hint.  It never reads an
application source path and never invokes a provider, compiler, or scheduler.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PASS_PYTHON = HERE.parent / "python"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(PASS_PYTHON))

import analyze_compiler_llm_capability_trials as capability  # noqa: E402
import gicc_comm_group_plan_bridge as communication  # noqa: E402
import gicc_comm_structural_heuristic as heuristic  # noqa: E402
import gicc_compiler_policy_bridge as policy_bridge  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import run_compiler_llm_capability_trials as trials  # noqa: E402


CATALOG_SCHEMA = "gicc-communication-llm-policy-catalog-v1"
DECISION_FAMILY = "communication_route_or_schedule"
BOUNDARY = {
    "application_source_read_or_modified": False,
    "provider_invoked": False,
    "compiler_invoked": False,
    "scheduler_invoked": False,
    "runtime_or_oracle_labels_read": False,
    "model_output_authority": "existing graph-bound candidate IDs only",
    "private_hints_visible_to_model": False,
    "compiler_revalidates_every_catalog_policy": True,
}


class CommunicationPolicyCatalogError(RuntimeError):
    """The request/archive cannot produce an exact compiler policy catalog."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CommunicationPolicyCatalogError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CommunicationPolicyCatalogError(
            f"cannot read JSON {path}: {exc}"
        ) from exc


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


def sha256_file(path: Path) -> str:
    return trials.sha256_file(path)


def display_path(path: Path, repo_root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(repo_root.resolve()).as_posix()
    except ValueError as exc:
        raise CommunicationPolicyCatalogError(
            f"catalog evidence escapes repository: {resolved}"
        ) from exc


def evidence(path: Path, repo_root: Path) -> dict[str, Any]:
    require(path.is_file(), f"catalog evidence is absent: {path}")
    return {
        "path": display_path(path, repo_root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def verify_evidence(
    record: Any, repo_root: Path, role: str,
) -> Path:
    require(
        isinstance(record, dict)
        and set(record) == {"path", "sha256", "bytes"}
        and isinstance(record.get("path"), str),
        f"{role}: malformed evidence record",
    )
    path = (repo_root / record["path"]).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise CommunicationPolicyCatalogError(
            f"{role}: evidence escapes repository"
        ) from exc
    require(
        path.is_file()
        and sha256_file(path) == record["sha256"]
        and path.stat().st_size == record["bytes"],
        f"{role}: evidence changed",
    )
    return path


def _request_payload(value: Any) -> dict[str, Any]:
    require(
        isinstance(value, dict)
        and value.get("schema_version")
        == trials.request_freezer.REQUEST_SCHEMA,
        "wrong compiler LLM request schema",
    )
    payload = dict(value)
    request_id = payload.pop("request_id", None)
    require(request_id == bridge._fingerprint(payload),
            "request ID does not match content")
    return payload


def verify_request_archive(
    graph: dict[str, Any], request_dir: Path, archive_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_path = request_dir / "request.json"
    request = read_json(request_path)
    _request_payload(request)
    require(request.get("decision_family") == DECISION_FAMILY,
            "request is not a communication route/schedule request")
    require(request.get("compiler_graph_id") == graph["graph_id"],
            "request binds another compiler graph")
    boundary = request.get("boundary", {})
    for key, expected in {
        "compiler_lto_decisions_only": True,
        "application_source_visible": False,
        "llvm_ir_visible": False,
        "evaluation_runtime_labels_visible": False,
        "evaluation_oracle_visible": False,
        "model_may_generate_code_or_ir": False,
        "model_output_is_existing_graph_bound_ids_only": True,
        "compiler_revalidates_every_response": True,
    }.items():
        require(boundary.get(key) is expected,
                f"request boundary changed: {key}")
    private_graph = request_dir / "private/compiler-graph.json"
    require(
        private_graph.is_file() and read_json(private_graph) == graph,
        "request private graph differs from the catalog graph",
    )
    authorization = read_json(archive_dir / "authorization.json")
    index, _ = trials.verify_complete_archive(
        request=request, authorization_value=authorization, graph=graph,
        output_dir=archive_dir,
    )
    require(index.get("status") == "complete",
            "provider archive is not complete")
    capability.observed_policies(index, graph)
    return request, index


def _decision(
    graph: dict[str, Any], selected: dict[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": communication.DECISION_SCHEMA,
        "graph_id": graph["graph_id"],
        "selections": {
            opportunity_id: {
                "candidate_id": candidate_id,
                "confidence": 1.0,
                "rationale": (
                    "compiler-enumerated catalog policy; no model authority"
                ),
            }
            for opportunity_id, candidate_id in selected.items()
        },
    }


def enumerate_catalog(graph_value: Any) -> list[dict[str, Any]]:
    """Enumerate and revalidate the complete independent policy product."""
    graph = policy_bridge.verified_graph(graph_value)
    require(graph.get("schema_version") == communication.GRAPH_SCHEMA,
            "policy catalog requires a communication route/schedule graph")
    opportunities = graph.get("opportunities", [])
    require(opportunities, "communication graph has no opportunities")
    identifiers = [item["opportunity_id"] for item in opportunities]
    choices = [
        [candidate["candidate_id"] for candidate in item["candidates"]]
        for item in opportunities
    ]
    require(all(choice for choice in choices),
            "communication opportunity has no compiler candidate")
    records = []
    for combination in itertools.product(*choices):
        selected = dict(zip(identifiers, combination, strict=True))
        verified = policy_bridge.verified_policy(graph, selected)
        bridged = policy_bridge.decision_to_policy(
            graph, _decision(graph, selected)
        )
        require(
            bridged.get("bridge_accepted") is True
            and bridged.get("fallback_applied") is False
            and bridged.get("selected_ids_by_slot")
            == verified["selected_ids_by_slot"]
            and bridged.get("policy_id") == verified["policy_id"],
            "compiler bridge did not reproduce an enumerated policy",
        )
        records.append({
            "policy_id": verified["policy_id"],
            "selected_ids_by_slot": verified["selected_ids_by_slot"],
            "compiler_hint_id": bridged["compiler_hint_id"],
            "compiler_hint": bridged["compiler_hint"],
        })
    records.sort(key=lambda item: item["policy_id"])
    require(
        len({item["policy_id"] for item in records}) == len(records),
        "compiler policy enumeration produced duplicate IDs",
    )
    expected_count = 1
    for choice in choices:
        expected_count *= len(choice)
    require(len(records) == expected_count,
            "compiler policy enumeration is incomplete")
    return records


def catalog_with_roles(
    graph: dict[str, Any], index: dict[str, Any],
) -> tuple[list[dict[str, Any]], str, str]:
    catalog = enumerate_catalog(graph)
    by_id = {item["policy_id"]: item for item in catalog}
    observed = capability.observed_policies(index, graph)
    require(set(observed).issubset(by_id),
            "provider archive contains a policy outside the compiler catalog")
    occurrences: dict[str, list[dict[str, Any]]] = {
        policy_id: [] for policy_id in by_id
    }
    for row in index["runs"]:
        occurrences[row["policy_id"]].append({
            "view": row["view"], "trial": row["trial"],
            "bridge_accepted": row["bridge_accepted"],
        })
    anchor = policy_bridge.decision_to_policy(graph, None)
    deterministic_decision, _ = heuristic.make_decision(graph)
    deterministic = policy_bridge.decision_to_policy(
        graph, deterministic_decision
    )
    require(deterministic.get("bridge_accepted") is True,
            "compiler rejected its deterministic communication control")
    for ordinal, record in enumerate(catalog, start=1):
        record["name"] = f"policy-{ordinal:04d}"
        record["roles"] = {
            "semantic_anchor": record["policy_id"] == anchor["policy_id"],
            "deterministic_compiler_control": (
                record["policy_id"] == deterministic["policy_id"]
            ),
            "observed_in_provider_archive": bool(
                occurrences[record["policy_id"]]
            ),
        }
        record["archive_occurrences"] = occurrences[record["policy_id"]]
    return catalog, anchor["policy_id"], deterministic["policy_id"]


def implementation_records(repo_root: Path) -> dict[str, dict[str, Any]]:
    return {
        "catalog_preparer": evidence(Path(__file__), repo_root),
        "capability_analyzer": evidence(Path(capability.__file__), repo_root),
        "trial_runner": evidence(Path(trials.__file__), repo_root),
        "unified_policy_bridge": evidence(
            Path(policy_bridge.__file__), repo_root
        ),
        "communication_bridge": evidence(
            Path(communication.__file__), repo_root
        ),
        "deterministic_control": evidence(Path(heuristic.__file__), repo_root),
    }


def prepare(
    graph_path: Path, request_dir: Path, archive_dir: Path,
    output_dir: Path, repo_root: Path,
) -> dict[str, Any]:
    require(not output_dir.exists(),
            f"refusing to overwrite policy catalog: {output_dir}")
    graph = policy_bridge.verified_graph(read_json(graph_path))
    request, index = verify_request_archive(graph, request_dir, archive_dir)
    catalog, anchor_id, deterministic_id = catalog_with_roles(graph, index)
    output_dir.mkdir(parents=True)
    try:
        policies = []
        for record in catalog:
            policy_dir = output_dir / "policies" / record["name"]
            hint_path = policy_dir / "hint.json"
            selection_path = policy_dir / "selection.json"
            write_json_atomic(hint_path, record["compiler_hint"])
            write_json_atomic(selection_path, {
                "policy_id": record["policy_id"],
                "selected_ids_by_slot": record["selected_ids_by_slot"],
            })
            policies.append({
                key: value for key, value in record.items()
                if key != "compiler_hint"
            } | {
                "hint": evidence(hint_path, repo_root),
                "selection": evidence(selection_path, repo_root),
            })
        payload = {
            "schema_version": CATALOG_SCHEMA,
            "graph_id": graph["graph_id"],
            "request_id": request["request_id"],
            "run_index_sha256": sha256_file(
                archive_dir / "run-index.json"
            ),
            "decision_family": DECISION_FAMILY,
            "boundary": dict(BOUNDARY),
            "complete_policy_count": len(policies),
            "observed_policy_count": sum(
                row["roles"]["observed_in_provider_archive"]
                for row in policies
            ),
            "semantic_anchor_policy_id": anchor_id,
            "deterministic_compiler_policy_id": deterministic_id,
            "policies": policies,
            "inputs": {
                "graph": evidence(graph_path, repo_root),
                "request": evidence(request_dir / "request.json", repo_root),
                "authorization": evidence(
                    archive_dir / "authorization.json", repo_root
                ),
                "run_index": evidence(
                    archive_dir / "run-index.json", repo_root
                ),
            },
            "implementation": implementation_records(repo_root),
        }
        manifest = {"catalog_id": bridge._fingerprint(payload), **payload}
        write_json_atomic(output_dir / "manifest.json", manifest)
        return manifest
    except BaseException:
        # A partial directory is forensic evidence and is never resumed.
        raise


def _manifest_payload(value: Any) -> dict[str, Any]:
    require(
        isinstance(value, dict)
        and value.get("schema_version") == CATALOG_SCHEMA,
        f"expected {CATALOG_SCHEMA}",
    )
    require(set(value) == {
        "catalog_id", "schema_version", "graph_id", "request_id",
        "run_index_sha256", "decision_family", "boundary",
        "complete_policy_count", "observed_policy_count",
        "semantic_anchor_policy_id", "deterministic_compiler_policy_id",
        "policies", "inputs", "implementation",
    }, "communication policy catalog fields do not match schema")
    payload = dict(value)
    catalog_id = payload.pop("catalog_id", None)
    require(catalog_id == bridge._fingerprint(payload),
            "communication policy catalog ID does not match content")
    return payload


def verify_contained(
    manifest_path: Path, repo_root: Path,
) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    _manifest_payload(manifest)
    require(manifest.get("decision_family") == DECISION_FAMILY,
            "policy catalog has the wrong decision family")
    require(manifest.get("boundary") == BOUNDARY,
            "policy catalog crossed its offline compiler-only boundary")
    inputs = manifest.get("inputs")
    require(isinstance(inputs, dict) and set(inputs) == {
        "graph", "request", "authorization", "run_index",
    }, "policy catalog input records changed")
    graph_path = verify_evidence(inputs["graph"], repo_root, "graph")
    request_path = verify_evidence(inputs["request"], repo_root, "request")
    authorization_path = verify_evidence(
        inputs["authorization"], repo_root, "authorization"
    )
    index_path = verify_evidence(inputs["run_index"], repo_root, "run index")
    implementation = manifest.get("implementation")
    require(
        isinstance(implementation, dict)
        and set(implementation) == set(implementation_records(repo_root)),
        "policy catalog implementation records changed",
    )
    for role, record in implementation.items():
        verify_evidence(record, repo_root, role)
    graph = policy_bridge.verified_graph(read_json(graph_path))
    request, index = verify_request_archive(
        graph, request_path.parent, index_path.parent
    )
    require(authorization_path == index_path.parent / "authorization.json",
            "authorization and run index are from different archives")
    rebuilt, anchor_id, deterministic_id = catalog_with_roles(graph, index)
    policies = manifest.get("policies")
    require(isinstance(policies, list) and len(policies) == len(rebuilt),
            "policy catalog is incomplete")
    for expected, observed in zip(rebuilt, policies, strict=True):
        require(isinstance(observed, dict) and set(observed) == {
            "policy_id", "selected_ids_by_slot", "compiler_hint_id", "name",
            "roles", "archive_occurrences", "hint", "selection",
        }, "catalog policy fields do not match schema")
        require(
            all(observed.get(key) == value for key, value in expected.items()
                if key != "compiler_hint"),
            f"catalog policy changed: {expected['policy_id']}",
        )
        hint = verify_evidence(observed.get("hint"), repo_root, "policy hint")
        selection = verify_evidence(
            observed.get("selection"), repo_root, "policy selection"
        )
        expected_dir = (
            manifest_path.parent / "policies" / expected["name"]
        ).resolve()
        require(
            hint == expected_dir / "hint.json"
            and selection == expected_dir / "selection.json",
            f"catalog policy escapes its contained path: {expected['name']}",
        )
        require(read_json(hint) == expected["compiler_hint"],
                f"compiler hint changed: {expected['policy_id']}")
        require(read_json(selection) == {
            "policy_id": expected["policy_id"],
            "selected_ids_by_slot": expected["selected_ids_by_slot"],
        }, f"policy selection changed: {expected['policy_id']}")
    require(
        manifest.get("graph_id") == graph["graph_id"]
        and manifest.get("request_id") == request["request_id"]
        and manifest.get("run_index_sha256") == sha256_file(index_path)
        and manifest.get("complete_policy_count") == len(rebuilt)
        and manifest.get("observed_policy_count")
        == sum(bool(item["archive_occurrences"]) for item in rebuilt)
        and manifest.get("semantic_anchor_policy_id") == anchor_id
        and manifest.get("deterministic_compiler_policy_id")
        == deterministic_id,
        "policy catalog summary does not regenerate",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    emit = children.add_parser("emit")
    emit.add_argument("--graph", type=Path, required=True)
    emit.add_argument("--request-dir", type=Path, required=True)
    emit.add_argument("--archive-dir", type=Path, required=True)
    emit.add_argument("--output-dir", type=Path, required=True)
    emit.add_argument("--repo-root", type=Path, default=ROOT)
    verify = children.add_parser("verify-contained")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--repo-root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        if args.command == "emit":
            result = prepare(
                args.graph.resolve(), args.request_dir.resolve(),
                args.archive_dir.resolve(), args.output_dir.resolve(),
                args.repo_root.resolve(),
            )
            action = "wrote"
        else:
            result = verify_contained(
                args.manifest.resolve(), args.repo_root.resolve()
            )
            action = "verified"
        print(
            f"communication-llm-policy-catalog: {action}; "
            f"catalog_id={result['catalog_id']}; "
            f"policies={result['complete_policy_count']}; "
            "model_invoked=false; source_read_or_modified=false"
        )
        return 0
    except (
        CommunicationPolicyCatalogError, capability.CapabilityAnalysisError,
        trials.CapabilityTrialError, policy_bridge.CompilerPolicyBridgeError,
        communication.GroupPlanError, heuristic.HeuristicError,
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"communication-llm-policy-catalog: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
