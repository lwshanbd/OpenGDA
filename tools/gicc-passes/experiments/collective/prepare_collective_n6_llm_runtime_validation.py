#!/usr/bin/env python3
"""Freeze deduplicated compiler policies for N6 collective runtime validation.

The preparer replays the complete offline capability analysis, extracts the
preregistered modal and post-hoc representatives plus compiler controls, and
maps only their existing graph-bound option IDs to private compiler hints.  It
does not invoke a model, compiler, scheduler, runtime benchmark, or source edit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
EXPERIMENTS = HERE.parent
PASS_PYTHON = HERE.parents[1] / "python"
ROOT = HERE.parents[4]
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(EXPERIMENTS))
sys.path.insert(0, str(HERE))

import analyze_compiler_llm_capability_trials as capability_analysis  # noqa: E402
import gicc_collective_plan_bridge as plans  # noqa: E402
import gicc_compiler_policy_bridge as policy_bridge  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import prepare_compiler_llm_capability_request as request_freezer  # noqa: E402


PLAN_SCHEMA = "gicc-collective-n6-llm-runtime-validation-plan-v1"
CONTROL_ROLES = ("anchor", "deterministic", "oracle")
REPRESENTATIVE_ROLES = (
    "primary_modal_representative",
    "posthoc_capability_upper_bound",
)
BOUNDARY = {
    "compiler_lto_decisions_only": True,
    "application_source_read": False,
    "application_source_modified": False,
    "model_invoked": False,
    "provider_invoked": False,
    "scheduler_invoked": False,
    "compiler_invoked": False,
    "runtime_benchmark_invoked": False,
    "model_output_materializes_only_existing_graph_bound_ids": True,
}


class RuntimeValidationError(RuntimeError):
    """The offline analysis cannot define exact runtime representatives."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeValidationError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeValidationError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise RuntimeValidationError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def display_path(path: Path, root: Path = ROOT) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    require(resolved.is_file(), f"runtime-plan evidence is absent: {resolved}")
    return {
        "path": display_path(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


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


def verified_analysis(value: Any) -> dict[str, Any]:
    require(
        isinstance(value, dict)
        and value.get("schema_version") == capability_analysis.ANALYSIS_SCHEMA,
        "wrong capability-analysis schema",
    )
    payload = dict(value)
    observed = payload.pop("analysis_id", None)
    require(observed == bridge._fingerprint(payload),
            "capability analysis ID does not match content")
    require(
        value.get("decision_family") == "collective_algorithm_and_size_policy"
        and value.get("status")
        == "offline_screen_complete_runtime_validation_required",
        "capability analysis is not awaiting collective runtime validation",
    )
    boundary = value.get("boundary", {})
    require(
        boundary.get("compiler_lto_decisions_only") is True
        and boundary.get("application_source_visible_or_modified") is False
        and boundary.get("representative_runtime_validation_still_required")
        is True,
        "capability analysis boundary changed",
    )
    return value


def representative_policies(
    graph_value: Any, analysis_value: Any, screen_value: Any,
) -> list[dict[str, Any]]:
    graph = policy_bridge.verified_graph(graph_value)
    topology = graph.get("platform_profile", {}).get("topology", {})
    require(
        topology.get("nodes") == 6
        and topology.get("ranks_per_node") == 8,
        "runtime validation requires the frozen N6 topology",
    )
    analysis = verified_analysis(analysis_value)
    require(analysis.get("compiler_graph_id") == graph["graph_id"],
            "capability analysis binds another graph")
    capability_analysis._screen_payload(screen_value)
    require(
        screen_value.get("graph_id") == graph["graph_id"]
        and analysis.get("screen_id") == screen_value.get("screen_id"),
        "capability analysis and policy screen disagree",
    )

    roles: list[tuple[str, Any]] = []
    controls = screen_value.get("controls")
    require(isinstance(controls, dict), "policy screen lacks controls")
    for name in CONTROL_ROLES:
        control = controls.get(name)
        require(isinstance(control, dict), f"policy screen lacks {name}")
        roles.append((f"control:{name}", control.get("selected_ids_by_slot")))

    views = analysis.get("results", {}).get("metrics", {}).get("views")
    require(
        isinstance(views, dict)
        and set(views) == set(request_freezer.VIEWS),
        "capability analysis lacks exact information views",
    )
    for view in request_freezer.VIEWS:
        for role in REPRESENTATIVE_ROLES:
            record = views[view].get(role)
            require(isinstance(record, dict), f"{view}/{role} is absent")
            roles.append((f"{view}:{role}", record.get("selected_ids_by_slot")))

    by_id: dict[str, dict[str, Any]] = {}
    for role, selected in roles:
        policy = policy_bridge.verified_policy(graph, selected)
        record = by_id.setdefault(policy["policy_id"], {
            "policy_id": policy["policy_id"],
            "selected_ids_by_slot": policy["selected_ids_by_slot"],
            "roles": [],
        })
        require(
            record["selected_ids_by_slot"] == policy["selected_ids_by_slot"],
            "one policy ID maps to inconsistent selections",
        )
        record["roles"].append(role)

    result = []
    for index, policy_id in enumerate(sorted(by_id), start=1):
        record = by_id[policy_id]
        result.append({
            "name": f"policy{index:02d}",
            "policy_id": policy_id,
            "selected_ids_by_slot": record["selected_ids_by_slot"],
            "roles": sorted(record["roles"]),
        })
    require(1 <= len(result) <= 9,
            "deduplicated representative count is outside 1..9")
    return result


def decision_and_hint(
    graph_value: Any, policy: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    graph = policy_bridge.verified_graph(graph_value)
    require(len(graph["opportunities"]) == 1,
            "collective runtime validation requires one opportunity")
    opportunity = graph["opportunities"][0]
    prefix = opportunity["opportunity_id"] + "/"
    selected = policy_bridge.verified_policy(
        graph, policy["selected_ids_by_slot"]
    )
    require(selected["policy_id"] == policy["policy_id"],
            "runtime policy ID changed")
    slots = {}
    for key, option_id in selected["selected_ids_by_slot"].items():
        require(key.startswith(prefix), "runtime policy has another opportunity")
        slots[key[len(prefix):]] = option_id
    decision = {
        "schema_version": plans.DECISION_SCHEMA,
        "graph_id": graph["graph_id"],
        "selections": {
            opportunity["opportunity_id"]: {
                "slot_candidate_ids": slots,
                "confidence": 1.0,
                "rationale": (
                    "archived compiler-policy representative " + policy["name"]
                ),
            },
        },
    }
    hint, accepted, errors = plans.decision_to_hint(graph, decision)
    require(accepted and not errors,
            "compiler bridge rejected a verified representative")
    hint["llm_metadata"].update({
        "model_invoked": False,
        "decision_origin": "verified_offline_capability_analysis",
        "policy_id": policy["policy_id"],
    })
    return decision, hint


def plan_payload(
    *, graph: dict[str, Any], analysis: dict[str, Any],
    screen: dict[str, Any], policies: list[dict[str, Any]],
    output_dir: Path, input_paths: dict[str, Path],
) -> dict[str, Any]:
    policy_records = []
    for policy in policies:
        root = output_dir / "policies" / policy["name"]
        policy_records.append({
            **policy,
            "decision": evidence(root / "decision.json"),
            "compiler_hint": evidence(root / "hint.json"),
        })
    return {
        "schema_version": PLAN_SCHEMA,
        "status": "ready_for_compiler_lto_materialization",
        "compiler_graph_id": graph["graph_id"],
        "capability_analysis_id": analysis["analysis_id"],
        "policy_screen_id": screen["screen_id"],
        "boundary": dict(BOUNDARY),
        "representative_contract": {
            "primary": "modal accepted policy per information view",
            "capability_ceiling": "post-hoc best-of-20 per information view",
            "compiler_controls": list(CONTROL_ROLES),
            "deduplicated_before_build_and_runtime": True,
            "all_policies_are_existing_graph_bound_option_ids": True,
            "posthoc_policies_are_not_primary_stability_evidence": True,
        },
        "unique_policy_count": len(policy_records),
        "policies": policy_records,
        "runtime_contract": {
            "queue": "pdebug",
            "nodes": 6,
            "ranks": 48,
            "ranks_per_node": 8,
            "independent_allocations": 3,
            "maximum_active_or_queued_jobs": 1,
            "all_policies_run_sequentially_with_rotated_order": True,
            "timed_calls": 7,
            "warmup_calls": 2,
            "point_estimate_threshold": 1.03,
            "paired_cluster_bootstrap_lower_95_threshold": 1.0,
        },
        "evidence": {
            role: evidence(path) for role, path in sorted(input_paths.items())
        },
    }


def build_inputs(args: argparse.Namespace) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, Any], list[dict[str, Any]]
]:
    expected = capability_analysis.build_report(
        suite_path=args.suite.resolve(),
        prompt_dir=args.prompt_dir.resolve(),
        readiness_path=args.readiness.resolve(),
        separation_path=args.input_separation.resolve(),
        sampling_null_path=args.sampling_null.resolve(),
        protocol_path=args.capability_protocol.resolve(),
        label=args.label, graph_path=args.graph.resolve(),
        request_dir=args.request_dir.resolve(),
        authorization_path=args.authorization.resolve(),
        archive_dir=args.archive_dir.resolve(),
        screen_path=args.policy_screen.resolve(),
        repo_root=args.repo_root.resolve(),
    )
    analysis = read_json(args.analysis.resolve())
    require(analysis == expected,
            "capability analysis does not regenerate from its archive")
    analysis = verified_analysis(analysis)
    graph = policy_bridge.verified_graph(read_json(args.graph.resolve()))
    screen = read_json(args.policy_screen.resolve())
    policies = representative_policies(graph, analysis, screen)
    return graph, analysis, screen, policies


def prepare(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    require(not output_dir.exists(), f"refusing to overwrite {output_dir}")
    graph, analysis, screen, policies = build_inputs(args)
    output_dir.mkdir(parents=True)
    for policy in policies:
        root = output_dir / "policies" / policy["name"]
        decision, hint = decision_and_hint(graph, policy)
        write_json_atomic(root / "decision.json", decision)
        write_json_atomic(root / "hint.json", hint)
    payload = plan_payload(
        graph=graph, analysis=analysis, screen=screen, policies=policies,
        output_dir=output_dir,
        input_paths={
            "preparer": Path(__file__),
            "compiler_graph": args.graph.resolve(),
            "capability_analysis": args.analysis.resolve(),
            "policy_screen": args.policy_screen.resolve(),
            "compiler_build_script": HERE / "build_compiler_collective_eval.sh",
            "compiler_evaluator": HERE / "compiler_collective_eval.py",
            "collective_bridge": PASS_PYTHON / "gicc_collective_plan_bridge.py",
        },
    )
    plan = {"plan_id": bridge._fingerprint(payload), **payload}
    write_json_atomic(output_dir / "plan.json", plan)
    return plan


def verify(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    graph, analysis, screen, policies = build_inputs(args)
    for policy in policies:
        root = output_dir / "policies" / policy["name"]
        decision, hint = decision_and_hint(graph, policy)
        require(read_json(root / "decision.json") == decision,
                f"{policy['name']}: decision changed")
        require(read_json(root / "hint.json") == hint,
                f"{policy['name']}: compiler hint changed")
    payload = plan_payload(
        graph=graph, analysis=analysis, screen=screen, policies=policies,
        output_dir=output_dir,
        input_paths={
            "preparer": Path(__file__),
            "compiler_graph": args.graph.resolve(),
            "capability_analysis": args.analysis.resolve(),
            "policy_screen": args.policy_screen.resolve(),
            "compiler_build_script": HERE / "build_compiler_collective_eval.sh",
            "compiler_evaluator": HERE / "compiler_collective_eval.py",
            "collective_bridge": PASS_PYTHON / "gicc_collective_plan_bridge.py",
        },
    )
    expected = {"plan_id": bridge._fingerprint(payload), **payload}
    require(read_json(output_dir / "plan.json") == expected,
            "runtime validation plan does not regenerate")
    return expected


def add_inputs(parser: argparse.ArgumentParser) -> None:
    capability_analysis.add_inputs(parser)
    parser.add_argument("--analysis", type=Path, required=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    prepare_parser = children.add_parser("prepare")
    add_inputs(prepare_parser)
    prepare_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser = children.add_parser("verify")
    add_inputs(verify_parser)
    verify_parser.add_argument("--plan-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            plan = prepare(args, args.output_dir.resolve())
            action = "prepared"
        else:
            plan = verify(args, args.plan_dir.resolve())
            action = "verified"
        print(
            f"collective-n6-llm-runtime-validation: {action}; "
            f"policies={plan['unique_policy_count']}; "
            f"compiler_invoked=false scheduler_invoked=false; "
            f"plan_id={plan['plan_id']}"
        )
        return 0
    except (
        RuntimeValidationError, capability_analysis.CapabilityAnalysisError,
        policy_bridge.CompilerPolicyBridgeError, plans.CollectivePlanError,
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(
            f"collective-n6-llm-runtime-validation: ERROR: {exc}",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
