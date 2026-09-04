#!/usr/bin/env python3
"""Rebuild N6 LLM evidence and derive conservative paper claim flags."""

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
ROOT = HERE.parents[2]
COLLECTIVE = HERE / "collective"
PASS_PYTHON = HERE.parent / "python"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(COLLECTIVE))
sys.path.insert(0, str(PASS_PYTHON))

import analyze_compiler_llm_capability_trials as capability  # noqa: E402
import analyze_collective_n6_llm_runtime_validation as runtime  # noqa: E402
import gicc_compiler_decision_suite as decision_suite  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


AUDIT_SCHEMA = "gicc-collective-n6-llm-paper-claim-audit-v1"


class PaperClaimAuditError(RuntimeError):
    """The N6 evidence cannot support a machine-derived claim audit."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PaperClaimAuditError(message)


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PaperClaimAuditError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise PaperClaimAuditError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def resolve_evidence(record: Any, role: str, repo_root: Path) -> Path:
    require(
        isinstance(record, dict)
        and set(record) == {"path", "sha256", "bytes"}
        and isinstance(record.get("path"), str),
        f"{role}: malformed evidence record",
    )
    path = Path(record["path"])
    if not path.is_absolute():
        path = repo_root / path
    path = path.resolve()
    require(
        path.is_file()
        and path.stat().st_size == record["bytes"]
        and sha256_file(path) == record["sha256"],
        f"{role}: evidence changed",
    )
    return path


def verified_capability(path: Path, repo_root: Path) -> tuple[
    dict[str, Any], dict[str, Path], dict[str, Any], dict[str, Any]
]:
    observed = read_json(path)
    require(
        isinstance(observed, dict)
        and observed.get("schema_version") == capability.ANALYSIS_SCHEMA,
        "wrong capability-analysis schema",
    )
    evidence = observed.get("evidence", {})
    request_json = resolve_evidence(
        evidence.get("request"), "capability request", repo_root
    )
    authorization = resolve_evidence(
        evidence.get("authorization"), "authorization", repo_root
    )
    run_index = resolve_evidence(
        evidence.get("run_index"), "run index", repo_root
    )
    policy_screen = resolve_evidence(
        evidence.get("policy_screen"), "policy screen", repo_root
    )
    request = read_json(request_json)
    request_evidence = request.get("evidence", {})
    sources = {
        role: resolve_evidence(
            request_evidence.get(role), f"request {role}", repo_root
        )
        for role in (
            "suite", "readiness", "input_separation", "sampling_null",
            "capability_protocol", "source_graph",
        )
    }
    require(
        sources["sampling_null"]
        == resolve_evidence(
            evidence.get("sampling_null"), "analysis sampling null", repo_root
        ),
        "request and analysis bind different sampling-null evidence",
    )
    prompt_dir = sources["suite"].parent / "prompts"
    expected = capability.build_report(
        suite_path=sources["suite"], prompt_dir=prompt_dir,
        readiness_path=sources["readiness"],
        separation_path=sources["input_separation"],
        sampling_null_path=sources["sampling_null"],
        protocol_path=sources["capability_protocol"],
        label="collective_n6", graph_path=sources["source_graph"],
        request_dir=request_json.parent, authorization_path=authorization,
        archive_dir=run_index.parent, screen_path=policy_screen,
        repo_root=repo_root,
    )
    require(observed == expected,
            "capability analysis does not rebuild from its raw archive")
    suite = decision_suite.verified_suite(
        read_json(sources["suite"]), prompt_dir
    )
    protocol = read_json(sources["capability_protocol"])
    return observed, sources, suite, protocol


def verified_runtime(path: Path, repo_root: Path) -> tuple[
    dict[str, Any], dict[str, Any]
]:
    observed = read_json(path)
    require(
        isinstance(observed, dict)
        and observed.get("schema_version") == runtime.RESULT_SCHEMA,
        "wrong N6 runtime-analysis schema",
    )
    plan_path = resolve_evidence(
        observed.get("evidence", {}).get("plan"), "runtime plan", repo_root
    )
    summaries = observed.get("allocation_monitors")
    require(isinstance(summaries, list) and len(summaries) == 3,
            "runtime analysis lacks three allocation monitors")
    monitors = []
    for index, summary in enumerate(summaries, start=1):
        require(
            isinstance(summary, dict)
            and isinstance(summary.get("monitor"), str)
            and isinstance(summary.get("monitor_sha256"), str),
            f"allocation {index}: malformed monitor evidence",
        )
        monitor = Path(summary["monitor"]).resolve()
        require(
            monitor.is_file()
            and sha256_file(monitor) == summary["monitor_sha256"],
            f"allocation {index}: monitor changed",
        )
        monitors.append(monitor)
    expected = runtime.build_report(plan_path.parent, monitors, repo_root)
    require(observed == expected,
            "runtime analysis does not rebuild from raw allocation evidence")
    return observed, read_json(plan_path)


def followup_entries(
    suite: dict[str, Any], protocol: dict[str, Any],
    current_family: str,
) -> list[dict[str, Any]]:
    entries = {entry["label"]: entry for entry in suite["entries"]}
    result = []
    for label, record in protocol.get("entries", {}).items():
        if (label == "collective_n6" or label not in entries
                or not isinstance(record, dict)
                or record.get("eligible_to_freeze_provider_request") is not True):
            continue
        entry = entries[label]
        if entry["decision_family"] == current_family:
            continue
        count = entry.get("decision_space", {}).get(
            "independent_policy_count"
        )
        require(
            isinstance(count, int) and not isinstance(count, bool) and count > 0,
            f"{label}: invalid policy count",
        )
        result.append({
            "label": label,
            "decision_family": entry["decision_family"],
            "independent_policy_count": count,
            "suite_entry_id": entry["entry_id"],
            "compiler_graph_id": entry["graph_id"],
        })
    return sorted(
        result,
        key=lambda entry: (-entry["independent_policy_count"], entry["label"]),
    )


def claim_summary(
    capability_report: dict[str, Any], runtime_report: dict[str, Any],
    plan: dict[str, Any], suite: dict[str, Any], protocol: dict[str, Any],
) -> dict[str, Any]:
    require(capability_report.get("label") == "collective_n6",
            "capability result is not collective_n6")
    require(
        plan.get("capability_analysis_id")
        == capability_report.get("analysis_id")
        and plan.get("policy_screen_id")
        == capability_report.get("screen_id")
        and runtime_report.get("plan_id") == plan.get("plan_id")
        and runtime_report.get("compiler_graph_id")
        == capability_report.get("compiler_graph_id"),
        "capability, runtime plan, and runtime result identities differ",
    )
    capability_boundary = capability_report.get("boundary", {})
    runtime_boundary = runtime_report.get("boundary", {})
    require(
        capability_boundary.get("compiler_lto_decisions_only") is True
        and capability_boundary.get("application_source_visible_or_modified")
        is False
        and runtime_boundary.get("compiler_lto_decisions_only") is True
        and runtime_boundary.get("application_source_modified") is False
        and runtime_boundary.get("runtime_labels_visible_to_model") is False,
        "LLM evidence crossed the compiler-only boundary",
    )
    gates = runtime_report.get("claim_gates", {})
    stable = gates.get(
        "stable_relational_modal_runtime_improvement_supported"
    ) is True
    context = gates.get(
        "relational_context_runtime_effect_supported"
    ) is True
    ceiling = gates.get(
        "posthoc_capability_ceiling_runtime_potential_observed"
    ) is True
    if stable and context:
        status = "single_family_relational_effect_supported"
    elif stable:
        status = "compiler_policy_gain_without_relational_attribution"
    elif ceiling:
        status = "posthoc_capability_ceiling_only"
    else:
        status = "no_upgraded_llm_performance_support"
    current_family = capability_report["decision_family"]
    followups = followup_entries(suite, protocol, current_family)
    relational = runtime_report.get("representative_comparisons", {}).get(
        "relational:primary_modal_representative", {}
    )
    return {
        "status": status,
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_visible_or_modified": False,
            "runtime_or_oracle_labels_visible_to_model": False,
            "model_output_authority": "existing graph-bound compiler IDs",
        },
        "claim_matrix": {
            "stable_relational_modal_runtime_improvement_supported": stable,
            "relational_context_runtime_effect_supported": context,
            "posthoc_capability_ceiling_runtime_potential_observed": ceiling,
            "single_family_capability_evidence_complete": True,
            "portfolio_generalization_supported": False,
            "paper_mainline_complete": False,
        },
        "relational_modal_runtime_distance_to_oracle": relational.get(
            "distance_to_runtime_oracle"
        ),
        "different_family_followup": {
            "currently_eligible": followups,
            "available": bool(followups),
            "next_action": (
                "freeze_one_separately_authorized_different-family_request"
                if followups else
                "establish_stable_compiler_oracle_in_another_family"
            ),
            "automatically_authorized": False,
        },
        "interpretation_constraints": {
            "modal_policy_is_primary": True,
            "best_of_20_is_posthoc_only": True,
            "one_family_cannot_establish_generalization": True,
            "offline_screen_is_not_runtime_speedup_evidence": True,
        },
    }


def build_audit(capability_path: Path, runtime_path: Path,
                repo_root: Path) -> dict[str, Any]:
    capability_report, sources, suite, protocol = verified_capability(
        capability_path, repo_root
    )
    runtime_report, plan = verified_runtime(runtime_path, repo_root)
    payload = {
        "schema_version": AUDIT_SCHEMA,
        **claim_summary(
            capability_report, runtime_report, plan, suite, protocol
        ),
        "evidence": {
            "capability_analysis": {
                "path": str(capability_path.resolve()),
                "sha256": sha256_file(capability_path),
                "bytes": capability_path.stat().st_size,
            },
            "runtime_analysis": {
                "path": str(runtime_path.resolve()),
                "sha256": sha256_file(runtime_path),
                "bytes": runtime_path.stat().st_size,
            },
            "suite": {
                "path": str(sources["suite"]),
                "sha256": sha256_file(sources["suite"]),
                "bytes": sources["suite"].stat().st_size,
            },
            "auditor": {
                "path": str(Path(__file__).resolve()),
                "sha256": sha256_file(Path(__file__)),
                "bytes": Path(__file__).stat().st_size,
            },
        },
    }
    return {"audit_id": bridge._fingerprint(payload), **payload}


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    for command in ("emit", "verify"):
        child = children.add_parser(command)
        child.add_argument("--capability-analysis", type=Path, required=True)
        child.add_argument("--runtime-analysis", type=Path, required=True)
        child.add_argument("--repo-root", type=Path, default=ROOT)
        child.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        audit = build_audit(
            args.capability_analysis.resolve(),
            args.runtime_analysis.resolve(), args.repo_root.resolve(),
        )
        if args.command == "emit":
            require(not args.out.exists(), f"refusing to overwrite {args.out}")
            write_json_atomic(args.out.resolve(), audit)
            action = "wrote"
        else:
            require(read_json(args.out.resolve()) == audit,
                    "paper claim audit does not match current evidence")
            action = "verified"
        print(json.dumps({
            "action": action, "status": audit["status"],
            "audit_id": audit["audit_id"],
            "paper_mainline_complete": False,
        }, sort_keys=True))
        return 0
    except (
        PaperClaimAuditError, capability.CapabilityAnalysisError,
        capability.trial_runner.CapabilityTrialError,
        capability.request_freezer.CapabilityRequestError,
        capability.policy_bridge.CompilerPolicyBridgeError,
        capability.metrics.CapabilityMetricsError,
        runtime.RuntimeAnalysisError, runtime.preparation.RuntimeValidationError,
        runtime.policy_bridge.CompilerPolicyBridgeError,
        runtime.monitor_base.MonitorError,
        runtime.n6_confirmation.base.ConfirmError,
        decision_suite.SuiteError, OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"collective-n6-llm-paper-audit: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
