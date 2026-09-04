#!/usr/bin/env python3
"""Score a verified compiler-only LLM archive against held-out controls.

This analyzer has no provider, compiler, scheduler, or source-edit path.  It
first re-verifies the exact eligibility-gated request, authorization, all 60
archived trials, compiler fallbacks, and private hints.  A separately
content-addressed family adapter must then supply held-out cost units for every
observed policy plus graph-bound oracle, anchor, and deterministic controls.
The generic metrics layer reports intention-to-treat behavior, stable modal
policies, and best-of-20 only as an explicitly post-hoc capability ceiling.
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
ROOT = HERE.parents[2]
PASS_PYTHON = HERE.parent / "python"
sys.path.insert(0, str(PASS_PYTHON))
sys.path.insert(0, str(HERE))

import gicc_compiler_policy_bridge as policy_bridge  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import gicc_llm_capability_metrics as metrics  # noqa: E402
import audit_llm_sampling_null as sampling_null  # noqa: E402
import prepare_compiler_llm_capability_request as request_freezer  # noqa: E402
import run_compiler_llm_capability_trials as trial_runner  # noqa: E402


SCREEN_SCHEMA = "gicc-compiler-llm-policy-screen-v1"
ANALYSIS_SCHEMA = "gicc-compiler-llm-capability-analysis-v1"
SCREEN_BOUNDARY = {
    "generated_after_archive_complete": True,
    "provider_visible": False,
    "application_source_visible": False,
    "application_source_modified": False,
    "runtime_labels_visible_to_model": False,
    "oracle_labels_visible_to_model": False,
    "compiler_graph_bound_policies_only": True,
    "offline_screen_is_runtime_speedup_evidence": False,
}


class CapabilityAnalysisError(RuntimeError):
    """The archive or held-out policy screen cannot support the analysis."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CapabilityAnalysisError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise CapabilityAnalysisError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def evidence(path: Path) -> dict[str, Any]:
    return {
        "path": display_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CapabilityAnalysisError(message)


def verify_evidence_record(record: Any, repo_root: Path,
                           label: str) -> dict[str, Any]:
    require(isinstance(record, dict) and set(record) == {
        "path", "sha256", "bytes",
    }, f"{label}: malformed evidence record")
    raw_path = record["path"]
    require(isinstance(raw_path, str) and raw_path,
            f"{label}: invalid evidence path")
    path = (repo_root / raw_path).resolve()
    try:
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise CapabilityAnalysisError(
            f"{label}: evidence escapes repository"
        ) from exc
    require(path.is_file(), f"{label}: evidence file is absent")
    require(
        sha256_file(path) == record["sha256"]
        and path.stat().st_size == record["bytes"],
        f"{label}: evidence bytes changed",
    )
    return record


def _screen_payload(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != SCREEN_SCHEMA:
        raise CapabilityAnalysisError(f"expected {SCREEN_SCHEMA}")
    payload = dict(value)
    screen_id = payload.pop("screen_id", None)
    require(screen_id == bridge._fingerprint(payload),
            "policy screen ID does not match content")
    return payload


def observed_policies(index: dict[str, Any],
                      graph: dict[str, Any]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for record in index.get("runs", []):
        selected = record.get("selected_ids_by_slot")
        verified = policy_bridge.verified_policy(graph, selected)
        require(record.get("policy_id") == verified["policy_id"],
                "run policy ID does not match selected graph IDs")
        previous = result.setdefault(
            verified["policy_id"], verified["selected_ids_by_slot"]
        )
        require(previous == verified["selected_ids_by_slot"],
                "one policy ID maps to inconsistent selections")
    require(result, "archive contains no policies")
    return result


def verified_screen(
    value: Any, *, graph: dict[str, Any], request: dict[str, Any],
    index: dict[str, Any], index_path: Path, repo_root: Path,
) -> dict[str, Any]:
    payload = _screen_payload(value)
    require(value.get("graph_id") == graph["graph_id"],
            "policy screen binds another compiler graph")
    require(value.get("request_id") == request["request_id"],
            "policy screen binds another provider request")
    require(value.get("run_index_sha256") == sha256_file(index_path),
            "policy screen binds another run archive")
    require(value.get("boundary") == SCREEN_BOUNDARY,
            "policy screen violates the held-out boundary")
    evidence_value = value.get("evidence")
    require(isinstance(evidence_value, dict)
            and set(evidence_value) == {"family_adapter", "runtime_controls"},
            "policy screen lacks exact provenance")
    verify_evidence_record(
        evidence_value["family_adapter"], repo_root, "family adapter"
    )
    runtime_controls = evidence_value["runtime_controls"]
    require(isinstance(runtime_controls, list) and runtime_controls,
            "policy screen lacks runtime-control evidence")
    for index_number, record in enumerate(runtime_controls, 1):
        verify_evidence_record(
            record, repo_root, f"runtime control {index_number}"
        )

    controls = value.get("controls")
    require(isinstance(controls, dict)
            and set(controls) == {"oracle", "anchor", "deterministic"},
            "policy screen lacks exact controls")
    for name, control in controls.items():
        require(isinstance(control, dict) and set(control) == {
            "selected_ids_by_slot", "cost_by_unit",
        }, f"{name} control has invalid fields")
        policy_bridge.verified_policy(graph, control["selected_ids_by_slot"])
    fallback = policy_bridge.decision_to_policy(graph, None)
    require(controls["anchor"]["selected_ids_by_slot"]
            == fallback["selected_ids_by_slot"],
            "screen anchor is not the atomic compiler fallback")

    policies = observed_policies(index, graph)
    costs = value.get("policy_cost_by_id")
    require(isinstance(costs, dict) and set(costs) == set(policies),
            "policy screen does not cover exactly every observed policy")
    require(isinstance(value.get("unit_weights"), dict),
            "policy screen lacks unit weights")
    # score_archive performs the exact positive-unit and control-shape checks
    # through the shared metrics implementation.
    return value


def score_archive(index: dict[str, Any], graph: dict[str, Any],
                  screen: dict[str, Any]) -> dict[str, Any]:
    require(index.get("status") == "complete", "archive is not complete")
    runs = index.get("runs")
    require(isinstance(runs, list), "archive has no run list")
    expected = [
        (view, trial)
        for trial in range(1, request_freezer.TRIALS_PER_VIEW + 1)
        for view in request_freezer.VIEWS
    ]
    require(
        sorted((record.get("view"), record.get("trial")) for record in runs)
        == sorted(expected),
        "archive does not contain exactly 20 trials for each view",
    )
    fallback = policy_bridge.decision_to_policy(graph, None)
    rows_by_view = {view: [] for view in request_freezer.VIEWS}
    record_by_key = {}
    for record in runs:
        view = record["view"]
        trial = record["trial"]
        accepted = record.get("bridge_accepted")
        require(isinstance(accepted, bool), "archive lacks bridge outcome")
        require(record.get("fallback_applied") is (not accepted),
                "archive fallback flag disagrees with bridge outcome")
        verified = policy_bridge.verified_policy(
            graph, record.get("selected_ids_by_slot")
        )
        require(record.get("policy_id") == verified["policy_id"],
                "archive policy ID changed")
        if not accepted:
            require(verified["selected_ids_by_slot"]
                    == fallback["selected_ids_by_slot"],
                    "invalid response did not use the compiler anchor")
        cost = screen["policy_cost_by_id"].get(verified["policy_id"])
        require(isinstance(cost, dict),
                "held-out screen lacks an observed policy")
        scored = metrics.score_trial({
            "view": view,
            "trial": trial,
            "bridge_accepted": accepted,
            "selected_ids_by_slot": verified["selected_ids_by_slot"],
            "cost_by_unit": cost,
        }, oracle=screen["controls"]["oracle"],
           anchor=screen["controls"]["anchor"],
           deterministic=screen["controls"]["deterministic"],
           unit_weights=screen["unit_weights"])
        rows_by_view[view].append(scored)
        record_by_key[(view, trial)] = record
    summary = metrics.summarize_views(rows_by_view)
    representatives = {}
    for view in request_freezer.VIEWS:
        view_summary = summary["views"][view]
        representatives[view] = {}
        for role, field in (
            ("primary_modal", "primary_modal_representative"),
            ("posthoc_best_of_20", "posthoc_capability_upper_bound"),
        ):
            policy_id = view_summary[field]["policy_id"]
            matching = [
                trial for trial in range(1, request_freezer.TRIALS_PER_VIEW + 1)
                if record_by_key[(view, trial)]["policy_id"] == policy_id
            ]
            representatives[view][role] = {
                "policy_id": policy_id,
                "archive_trials": matching,
                "representative_trial": min(matching),
                "posthoc": role == "posthoc_best_of_20",
            }
    return {
        "metrics": summary,
        "representative_provenance": representatives,
        "screened_trial_count": len(runs),
        "unique_policy_count": len({record["policy_id"] for record in runs}),
    }


def chance_calibration(metrics_summary: dict[str, Any],
                       null_entry: dict[str, Any]) -> dict[str, Any]:
    trial_count = null_entry.get("draw_count")
    threshold = null_entry.get("minimum_exact_hits_for_one_sided_alpha_0_05")
    policy_count = null_entry.get("legal_policy_count")
    require(trial_count == request_freezer.TRIALS_PER_VIEW,
            "sampling null has another trial count")
    require(isinstance(policy_count, int) and not isinstance(policy_count, bool)
            and policy_count > 0, "sampling null has invalid policy count")
    require(isinstance(threshold, int) and not isinstance(threshold, bool)
            and 1 <= threshold <= trial_count,
            "sampling null has invalid significance threshold")
    views = metrics_summary.get("views")
    require(isinstance(views, dict) and set(views) == set(request_freezer.VIEWS),
            "metrics summary lacks exact information views")
    calibrated = {}
    probability = sampling_null.Fraction(1, policy_count)
    for view in request_freezer.VIEWS:
        summary = views[view]
        observed_trials = summary.get("trial_count")
        rate = summary.get("intention_to_treat", {}).get(
            "exact_oracle_policy_rate"
        )
        require(observed_trials == trial_count
                and isinstance(rate, (int, float)) and not isinstance(rate, bool),
                f"{view}: cannot calibrate exact-oracle hits")
        hits = round(float(rate) * trial_count)
        require(abs(float(rate) - hits / trial_count) <= 1e-12,
                f"{view}: exact-oracle rate is not an integer hit count")
        tail = sampling_null.binomial_tail(trial_count, probability, hits)
        calibrated[view] = {
            "observed_exact_oracle_hits": hits,
            "uniform_null_tail_probability": sampling_null.fraction_record(tail),
            "meets_one_sided_alpha_0_05_hit_threshold": hits >= threshold,
        }
    return {
        "null_hypothesis": (
            "independent uniform legal-policy draws with exactly one "
            "predesignated oracle"
        ),
        "legal_policy_count": policy_count,
        "trials_per_view": trial_count,
        "uniform_single_draw_exact_oracle_probability": null_entry[
            "single_draw_exact_oracle_probability"
        ],
        "uniform_at_least_one_exact_oracle_hit_probability": null_entry[
            "at_least_one_exact_oracle_in_draws_probability"
        ],
        "minimum_exact_hits_for_one_sided_alpha_0_05": threshold,
        "views": calibrated,
        "null_is_model_distribution_or_performance_evidence": False,
    }


def build_report(
    *, suite_path: Path, prompt_dir: Path, readiness_path: Path,
    separation_path: Path, sampling_null_path: Path,
    protocol_path: Path, label: str,
    graph_path: Path, request_dir: Path, authorization_path: Path,
    archive_dir: Path, screen_path: Path, repo_root: Path,
) -> dict[str, Any]:
    inputs = request_freezer.verified_inputs(
        suite_path, prompt_dir, readiness_path, separation_path,
        sampling_null_path,
        protocol_path, label, graph_path,
    )
    request = request_freezer.verify_bundle(inputs, request_dir)
    authorization_value = read_json(authorization_path)
    graph = policy_bridge.verified_graph(
        read_json(request_dir / "private/compiler-graph.json")
    )
    index, authorization = trial_runner.verify_complete_archive(
        request=request, authorization_value=authorization_value,
        graph=graph, output_dir=archive_dir,
    )
    index_path = archive_dir / "run-index.json"
    screen = verified_screen(
        read_json(screen_path), graph=graph, request=request, index=index,
        index_path=index_path, repo_root=repo_root,
    )
    scored = score_archive(index, graph, screen)
    null_report = sampling_null.verify_report(read_json(sampling_null_path))
    null_entry = null_report.get("current_suite_uniform_null", {}).get(label)
    require(isinstance(null_entry, dict),
            f"{label}: sampling null lacks current suite entry")
    scored["chance_calibration"] = chance_calibration(
        scored["metrics"], null_entry,
    )
    payload = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": "offline_screen_complete_runtime_validation_required",
        "label": label,
        "decision_family": inputs["suite_entry"]["decision_family"],
        "compiler_graph_id": graph["graph_id"],
        "request_id": request["request_id"],
        "authorization_id": authorization_value["authorization_id"],
        "screen_id": screen["screen_id"],
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_visible_or_modified": False,
            "provider_invoked_by_analyzer": False,
            "compiler_invoked_by_analyzer": False,
            "scheduler_invoked_by_analyzer": False,
            "runtime_and_oracle_labels_were_provider_visible": False,
            "offline_screen_is_runtime_speedup_evidence": False,
            "representative_runtime_validation_still_required": True,
            "best_of_20_is_posthoc_capability_upper_bound": True,
            "best_of_20_is_action_space_chance_calibrated": True,
        },
        "provider": {
            "requested_model": authorization["provider"]["requested_model"],
            "provider_cli_version": authorization["provider"]["cli_version"],
            "fresh_session_per_trial": True,
            "total_archived_trials": scored["screened_trial_count"],
        },
        "results": scored,
        "implementation": {
            "analyzer": evidence(Path(__file__)),
            "metrics": evidence(PASS_PYTHON / "gicc_llm_capability_metrics.py"),
            "unified_bridge": evidence(
                PASS_PYTHON / "gicc_compiler_policy_bridge.py"
            ),
            "archive_verifier": evidence(
                HERE / "run_compiler_llm_capability_trials.py"
            ),
        },
        "evidence": {
            "request": evidence(request_dir / "request.json"),
            "authorization": evidence(authorization_path),
            "run_index": evidence(index_path),
            "policy_screen": evidence(screen_path),
            "sampling_null": evidence(sampling_null_path),
        },
    }
    result = dict(payload)
    result["analysis_id"] = bridge._fingerprint(payload)
    return result


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


def add_inputs(parser: argparse.ArgumentParser) -> None:
    request_freezer.add_inputs(parser)
    parser.add_argument("--request-dir", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--archive-dir", type=Path, required=True)
    parser.add_argument("--policy-screen", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    emit = subparsers.add_parser("emit")
    add_inputs(emit)
    emit.add_argument("--out", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    add_inputs(verify)
    verify.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_report(
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
        if args.command == "emit":
            write_json_atomic(args.out, report)
            action = "wrote"
        else:
            require(read_json(args.report) == report,
                    "analysis report does not match current evidence")
            action = "verified"
        print(
            f"compiler-llm-capability-analysis: {action}; "
            f"trials={report['results']['screened_trial_count']}; "
            f"policies={report['results']['unique_policy_count']}; "
            f"analysis_id={report['analysis_id']}"
        )
        return 0
    except (
        CapabilityAnalysisError, trial_runner.CapabilityTrialError,
        request_freezer.CapabilityRequestError,
        policy_bridge.CompilerPolicyBridgeError,
        metrics.CapabilityMetricsError, OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"compiler-llm-capability-analysis: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
