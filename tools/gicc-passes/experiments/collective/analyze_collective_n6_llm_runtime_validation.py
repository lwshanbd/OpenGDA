#!/usr/bin/env python3
"""Audit three paired N6 allocations of materialized LLM compiler policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
PASS_PYTHON = HERE.parents[1] / "python"
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(HERE))

import analyze_compiler_collective_n6_confirmation as n6_confirmation  # noqa: E402
import gicc_compiler_policy_bridge as policy_bridge  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import monitor_compiler_collective_job as monitor_base  # noqa: E402
import prepare_collective_n6_llm_runtime_validation as preparation  # noqa: E402


RESULT_SCHEMA = "gicc-collective-n6-llm-runtime-validation-v1"
SIZES = (1024, 4096, 8192, 65536, 262144, 1048576, 4194304,
         8388608, 16777216)
NODES = 6
RANKS = 48
PPN = 8
RUNS = 7
WARMUP = 2
DURATION_SECONDS = 3540.0
RUNNER = HERE / "run_collective_n6_llm_runtime_validation.sh"
CONTROLLER = HERE / "continue_collective_n6_llm_runtime_validation.sh"
MONITOR = HERE / "monitor_compiler_collective_replicate.py"


class RuntimeAnalysisError(RuntimeError):
    """The paired N6 runtime evidence is incomplete or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeAnalysisError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeAnalysisError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise RuntimeAnalysisError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _expected_resources() -> list[dict[str, Any]]:
    return [{
        "type": "node", "count": NODES,
        "with": [{
            "type": "slot", "count": PPN, "label": "task",
            "with": [
                {"type": "core", "count": 8},
                {"type": "gpu", "count": 1},
            ],
        }],
    }]


def _driver_lines(replicate: int, order: list[str]) -> list[str]:
    lines = [
        f"COLLECTIVE_N6_LLM_RUNTIME_CONFIG replicate={replicate} "
        f"policies={' '.join(order)}",
    ]
    for name in order:
        lines.extend([
            f"COLLECTIVE_N6_LLM_RUNTIME_START replicate={replicate} "
            f"policy={name}",
            f"COLLECTIVE_N6_LLM_RUNTIME_DONE replicate={replicate} "
            f"policy={name}",
        ])
    lines.append(f"COLLECTIVE_N6_LLM_RUNTIME_COMPLETE replicate={replicate}")
    return lines


def _required_artifacts(plan_dir: Path,
                        policies: list[dict[str, Any]]) -> set[Path]:
    result = {
        (plan_dir / "plan.json").resolve(),
        (plan_dir / "manifest.json").resolve(),
        Path(preparation.__file__).resolve(),
        (HERE / "build_collective_n6_llm_runtime_validation.sh").resolve(),
        (HERE / "build_compiler_collective_eval.sh").resolve(),
        (HERE / "compiler_collective_eval.py").resolve(),
        RUNNER.resolve(), CONTROLLER.resolve(), MONITOR.resolve(),
        Path(__file__).resolve(),
    }
    for policy in policies:
        root = plan_dir / "policies" / policy["name"]
        result.update({
            (root / "hint.json").resolve(),
            (root / "build/compiler_collective_eval").resolve(),
            (root / "build/build-provenance.json").resolve(),
        })
    return result


def _artifact_paths(value: dict[str, Any]) -> set[Path]:
    records = value.get("artifacts")
    require(isinstance(records, list), "runtime monitor lacks artifacts")
    result = set()
    for record in records:
        require(
            isinstance(record, dict)
            and isinstance(record.get("path"), str)
            and isinstance(record.get("sha256"), str),
            "runtime monitor has malformed artifact record",
        )
        path = Path(record["path"]).resolve()
        require(
            path.is_file() and sha256_file(path) == record["sha256"],
            f"runtime artifact changed: {path}",
        )
        require(path not in result, "runtime monitor repeats an artifact")
        result.add(path)
    return result


def validate_monitor(
    path: Path, *, replicate: int, plan_dir: Path,
    policies: list[dict[str, Any]], order: list[str],
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    value = read_json(path)
    names = [policy["name"] for policy in policies]
    require(
        isinstance(value, dict)
        and value.get("schema_version")
        == "gicc-collective-replicate-job-monitor-v1"
        and value.get("state") == "passed"
        and value.get("replicate") == replicate,
        f"allocation {replicate} monitor did not pass",
    )
    expected = {
        "sizes": list(SIZES), "nodes": NODES, "ranks": RANKS,
        "ppn": PPN, "runs": RUNS, "warmup": WARMUP,
    }
    jobspec = value.get("jobspec", {})
    scheduler = value.get("scheduler", {})
    require(
        value.get("expected") == expected
        and scheduler.get("exit_code") == 0
        and scheduler.get("exception_types") == []
        and jobspec.get("queue") == "pdebug"
        and jobspec.get("duration_seconds") == DURATION_SECONDS
        and jobspec.get("resources") == _expected_resources(),
        f"allocation {replicate} changed the N6 pdebug contract",
    )
    require(
        _artifact_paths(value) == _required_artifacts(plan_dir, policies),
        f"allocation {replicate} artifact closure changed",
    )
    require(jobspec.get("embedded_script_sha256") == sha256_file(RUNNER),
            f"allocation {replicate} embedded another runner")
    command = jobspec.get("command")
    require(isinstance(command, list), "runtime jobspec lacks a command")
    try:
        script_index = command.index("{{tmpdir}}/script")
    except ValueError:
        raise RuntimeAnalysisError("runtime jobspec lacks its batch script") from None
    driver = Path(value.get("driver_stdout", {}).get("path", ""))
    driver_err = Path(value.get("driver_stderr", {}).get("path", ""))
    rep_dir = driver.parent.resolve()
    require(
        command[script_index + 1:] == [
            str(plan_dir.resolve()), str(rep_dir), str(replicate), *order,
        ],
        f"allocation {replicate} command changed",
    )
    require(
        driver.is_file()
        and sha256_file(driver)
        == value["driver_stdout"].get("sha256")
        and driver.stat().st_size == value["driver_stdout"].get("bytes")
        and driver_err.is_file()
        and sha256_file(driver_err)
        == value["driver_stderr"].get("sha256")
        and driver_err.stat().st_size
        == value["driver_stderr"].get("bytes") == 0,
        f"allocation {replicate} driver evidence changed",
    )
    require(
        driver.read_text(encoding="utf-8").splitlines()
        == _driver_lines(replicate, order),
        f"allocation {replicate} did not finish its frozen order",
    )
    benchmarks = value.get("benchmarks")
    require(isinstance(benchmarks, dict) and set(benchmarks) == set(names),
            f"allocation {replicate} policy coverage changed")
    rows = {}
    for name in names:
        record = benchmarks[name]
        stdout_record = record.get("stdout", {})
        stderr_record = record.get("stderr", {})
        stdout = Path(stdout_record.get("path", ""))
        stderr = Path(stderr_record.get("path", ""))
        require(
            stdout.is_file()
            and sha256_file(stdout) == stdout_record.get("sha256")
            and stdout.stat().st_size == stdout_record.get("bytes")
            and stderr.is_file()
            and sha256_file(stderr) == stderr_record.get("sha256")
            and stderr.stat().st_size == stderr_record.get("bytes"),
            f"allocation {replicate}/{name} logs changed",
        )
        reparsed = monitor_base.validate_output(
            stdout, name, list(SIZES), NODES, RANKS, PPN, RUNS, WARMUP,
        )
        require(
            reparsed == record.get("benchmark")
            and reparsed.get("total_errors") == 0,
            f"allocation {replicate}/{name} runtime is invalid",
        )
        rows[name] = {
            key: float(number) for key, number in reparsed["results"].items()
        }
    resource = value.get("resource_set", {})
    nodelist = resource.get("nodelist")
    require(isinstance(nodelist, list) and len(nodelist) == NODES,
            f"allocation {replicate} lacks its exact N6 node set")
    summary = {
        "allocation": replicate,
        "job_id": value.get("job_id"),
        "nodelist": nodelist,
        "policy_order": order,
        "monitor": str(path.resolve()),
        "monitor_sha256": sha256_file(path),
        "artifact_set_id": bridge._fingerprint([
            {"path": str(item), "sha256": sha256_file(item)}
            for item in sorted(_required_artifacts(plan_dir, policies), key=str)
        ]),
    }
    return rows, summary


def weighted_cost(values: dict[str, float],
                  weights: dict[str, float]) -> float:
    expected = {f"message_bytes:{size}" for size in SIZES}
    require(set(weights) == expected, "runtime weights changed")
    require(set(values) == {str(size) for size in SIZES},
            "runtime cost size coverage changed")
    require(
        all(isinstance(weight, (int, float)) and weight > 0
            and math.isfinite(float(weight)) for weight in weights.values())
        and all(cost > 0 and math.isfinite(cost) for cost in values.values()),
        "runtime cost or weight is not finite positive",
    )
    total = sum(float(weight) for weight in weights.values())
    return math.exp(sum(
        float(weights[f"message_bytes:{size}"]) * math.log(values[str(size)])
        for size in SIZES
    ) / total)


def comparison(
    rows: dict[int, dict[str, dict[str, float]]],
    subject: str, comparator: str, weights: dict[str, float],
) -> dict[str, Any]:
    ratios = [
        weighted_cost(rows[index][comparator], weights)
        / weighted_cost(rows[index][subject], weights)
        for index in (1, 2, 3)
    ]
    bootstrap = n6_confirmation.base.paired_allocation_bootstrap(ratios)
    return {
        "subject_policy": subject,
        "comparator_policy": comparator,
        "subject_speedup": bootstrap,
        "point_estimate_at_least_1_03": bootstrap["estimate"] >= 1.03,
        "lower_95_strictly_above_1": bootstrap["lower_2_5_percent"] > 1.0,
        "passed": (
            bootstrap["estimate"] >= 1.03
            and bootstrap["lower_2_5_percent"] > 1.0
        ),
        "per_size_allocation_speedups": {
            str(size): [
                rows[index][comparator][str(size)]
                / rows[index][subject][str(size)]
                for index in (1, 2, 3)
            ]
            for size in SIZES
        },
    }


def oracle_regret(
    rows: dict[int, dict[str, dict[str, float]]],
    subject: str, oracle: str, weights: dict[str, float],
) -> dict[str, Any]:
    ratios = [
        weighted_cost(rows[index][subject], weights)
        / weighted_cost(rows[index][oracle], weights)
        for index in (1, 2, 3)
    ]
    return {
        "subject_policy": subject,
        "oracle_policy": oracle,
        "subject_cost_regret_to_oracle":
            n6_confirmation.base.paired_allocation_bootstrap(ratios),
        "per_size_allocation_cost_regret": {
            str(size): [
                rows[index][subject][str(size)]
                / rows[index][oracle][str(size)]
                for index in (1, 2, 3)
            ]
            for size in SIZES
        },
        "used_as_a_pass_fail_gate": False,
    }


def analyze_rows(
    plan: dict[str, Any], rows: dict[int, dict[str, dict[str, float]]],
    weights: dict[str, float],
) -> dict[str, Any]:
    policies = plan["policies"]
    names = {policy["name"] for policy in policies}
    require(set(rows) == {1, 2, 3}, "runtime allocations are incomplete")
    require(all(set(value) == names for value in rows.values()),
            "runtime allocations have different policy sets")
    weighted_costs = {
        name: {
            str(index): weighted_cost(rows[index][name], weights)
            for index in (1, 2, 3)
        }
        for name in sorted(names)
    }
    role_to_policy = {}
    for policy in policies:
        for role in policy["roles"]:
            require(role not in role_to_policy, f"duplicate runtime role: {role}")
            role_to_policy[role] = policy["name"]
    expected_roles = {f"control:{name}" for name in preparation.CONTROL_ROLES} | {
        f"{view}:{role}"
        for view in preparation.request_freezer.VIEWS
        for role in preparation.REPRESENTATIVE_ROLES
    }
    require(set(role_to_policy) == expected_roles,
            "runtime plan role coverage changed")
    anchor = role_to_policy["control:anchor"]
    deterministic = role_to_policy["control:deterministic"]
    oracle = role_to_policy["control:oracle"]

    comparisons = {}
    for role in sorted(expected_roles - {
            "control:anchor", "control:deterministic", "control:oracle"}):
        subject = role_to_policy[role]
        comparisons[role] = {
            "over_anchor": comparison(rows, subject, anchor, weights),
            "over_deterministic": comparison(
                rows, subject, deterministic, weights
            ),
            "distance_to_runtime_oracle": oracle_regret(
                rows, subject, oracle, weights
            ),
        }
    relational_primary = comparisons[
        "relational:primary_modal_representative"
    ]
    stable_gate = all(
        relational_primary[key]["passed"]
        for key in ("over_anchor", "over_deterministic")
    )

    context = {}
    relational_policy = role_to_policy[
        "relational:primary_modal_representative"
    ]
    for view in ("descriptors", "opaque"):
        other = role_to_policy[f"{view}:primary_modal_representative"]
        context[view] = comparison(rows, relational_policy, other, weights)
    context_gate = all(item["passed"] for item in context.values())
    ceiling_gate = any(
        both["over_anchor"]["passed"] and both["over_deterministic"]["passed"]
        for role, both in comparisons.items()
        if role.endswith(":posthoc_capability_upper_bound")
    )
    return {
        "runtime_unit_weights": {
            key: float(value) for key, value in sorted(weights.items())
        },
        "weighted_runtime_cost_us_by_policy_and_allocation": weighted_costs,
        "role_to_deduplicated_policy": role_to_policy,
        "representative_comparisons": comparisons,
        "relational_modal_over_other_modal": context,
        "claim_gates": {
            "stable_relational_modal_runtime_improvement_supported": stable_gate,
            "relational_context_runtime_effect_supported": context_gate,
            "posthoc_capability_ceiling_runtime_potential_observed": ceiling_gate,
            "portfolio_generalization_supported": False,
            "paper_mainline_complete": False,
        },
    }


def build_report(plan_dir: Path, monitor_paths: list[Path],
                 repo_root: Path) -> dict[str, Any]:
    manifest = preparation.verify_built(plan_dir, repo_root)
    plan = read_json(plan_dir / "plan.json")
    policies = plan["policies"]
    orders = plan.get("runtime_contract", {}).get("policy_order_by_allocation")
    require(
        isinstance(orders, dict)
        and set(orders) == {"1", "2", "3"},
        "runtime plan lacks frozen allocation orders",
    )
    require(len(monitor_paths) == 3,
            "runtime validation requires exactly three monitors")
    rows = {}
    summaries = []
    for replicate, path in zip((1, 2, 3), monitor_paths, strict=True):
        allocation, summary = validate_monitor(
            path.resolve(), replicate=replicate, plan_dir=plan_dir,
            policies=policies, order=orders[str(replicate)],
        )
        rows[replicate] = allocation
        summaries.append(summary)
    require(len({item["job_id"] for item in summaries}) == 3,
            "runtime validation did not use three independent jobs")
    require(len({item["artifact_set_id"] for item in summaries}) == 1,
            "runtime allocations used different compiler artifacts")
    screen_path = preparation.recorded_path(
        plan["evidence"]["policy_screen"], "policy screen"
    )
    screen = read_json(screen_path)
    require(screen.get("screen_id") == plan["policy_screen_id"],
            "runtime plan policy screen changed")
    analysis = analyze_rows(plan, rows, screen["unit_weights"])
    payload = {
        "schema_version": RESULT_SCHEMA,
        "status": "paired_runtime_complete",
        "plan_id": plan["plan_id"],
        "bundle_manifest_id": manifest["manifest_id"],
        "compiler_graph_id": plan["compiler_graph_id"],
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_modified": False,
            "model_invoked_by_runtime_validation": False,
            "provider_invoked_by_runtime_validation": False,
            "runtime_labels_visible_to_model": False,
            "all_policies_revalidated_and_materialized_by_compiler": True,
            "posthoc_ceiling_is_stable_policy_evidence": False,
        },
        "runtime_contract": plan["runtime_contract"],
        "allocation_monitors": summaries,
        **analysis,
        "evidence": {
            "plan": preparation.evidence(plan_dir / "plan.json"),
            "bundle": preparation.evidence(plan_dir / "manifest.json"),
            "policy_screen": preparation.evidence(screen_path),
            "analyzer": preparation.evidence(Path(__file__)),
            "runner": preparation.evidence(RUNNER),
            "controller": preparation.evidence(CONTROLLER),
        },
    }
    return {"result_id": bridge._fingerprint(payload), **payload}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-dir", type=Path, required=True)
    parser.add_argument("--monitor", type=Path, action="append", required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(
            args.plan_dir.resolve(),
            [path.resolve() for path in args.monitor],
            args.repo_root.resolve(),
        )
        require(not args.out.exists(), f"refusing to overwrite {args.out}")
        preparation.write_json_atomic(args.out.resolve(), report)
        print(json.dumps({
            "claim_gates": report["claim_gates"],
            "result_id": report["result_id"],
        }, sort_keys=True))
        return 0
    except (
        RuntimeAnalysisError, preparation.RuntimeValidationError,
        preparation.collective_eval.EvalError,
        policy_bridge.CompilerPolicyBridgeError, monitor_base.MonitorError,
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"collective-n6-llm-runtime-analysis: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
