#!/usr/bin/env python3
"""Audit the historical source-free compiler/LTO LLM capability evidence.

This tool never invokes a provider, compiler, or scheduler.  It accepts only
already archived inputs and independently regenerated analyzer outputs.  Its
purpose is to preserve the useful compiler-only feasibility result while
preventing a post-hoc best-of-20 policy or an old-queue run from being
misreported as stable, current, relational-context evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
REPORT_SCHEMA = "gicc-historical-compiler-llm-ceiling-audit-v1"
PROTOCOL_SCHEMA = "gicc-compiler-lto-llm-protocol-v1"
ANALYSIS_SCHEMA = "gicc-compiler-lto-llm-trial-analysis-v1"
RUNTIME_SCHEMA = "gicc-compiler-lto-llm-runtime-v1"
MANIFEST_SCHEMA = "gicc-compiler-lto-eval-freeze-v1"
EXPECTED_TRIALS = 20
CURRENT_REQUIRED_QUEUE = "pdebug"


class HistoricalCeilingError(RuntimeError):
    """The archive cannot support the narrowly scoped historical claim."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HistoricalCeilingError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise HistoricalCeilingError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def fingerprint(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


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


def tree_evidence(path: Path) -> dict[str, Any]:
    files = sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    if not files:
        raise HistoricalCeilingError(f"evidence tree is empty: {path}")
    members = [{
        "path": candidate.relative_to(path).as_posix(),
        "sha256": sha256_file(candidate),
        "bytes": candidate.stat().st_size,
    } for candidate in files]
    return {
        "path": display_path(path),
        "file_count": len(members),
        "bytes": sum(row["bytes"] for row in members),
        "tree_id": fingerprint({"members": members}),
    }


def require(condition: bool, message: str) -> None:
    if not condition:
        raise HistoricalCeilingError(message)


def exact_regeneration(archived: Path, regenerated: Path, label: str) -> None:
    try:
        equal = archived.read_bytes() == regenerated.read_bytes()
    except OSError as exc:
        raise HistoricalCeilingError(f"cannot compare {label}: {exc}") from exc
    require(equal, f"{label} does not exactly reproduce archived bytes")


def parse_jobs(path: Path) -> dict[str, Any]:
    try:
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
    except OSError as exc:
        raise HistoricalCeilingError(f"cannot read job ledger {path}: {exc}") from exc
    require(rows, f"empty job ledger: {path}")
    require(
        set(rows[0]) == {"job_id", "rep", "order", "queue"},
        f"unexpected job-ledger columns: {path}",
    )
    require(all(row["job_id"] and row["rep"] and row["order"] and row["queue"]
                for row in rows), f"incomplete job ledger: {path}")
    return {
        "submitted_job_count": len(rows),
        "queues": sorted({row["queue"] for row in rows}),
        "submitted_replicates": sorted({int(row["rep"]) for row in rows}),
    }


def _finite_positive(value: Any, label: str) -> float:
    require(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value > 0,
        f"{label} is not positive and finite",
    )
    return float(value)


def summarize_policies(
    analysis: dict[str, Any],
    runtime_summaries: dict[str, dict[str, Any]],
    job_ledgers: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validate policy identities and separate stable from post-hoc results."""
    policies = analysis.get("policies")
    require(isinstance(policies, dict) and policies, "trial analysis has no policies")
    names = sorted(policies)
    require(set(runtime_summaries) == set(names), "runtime policy set mismatch")
    require(set(job_ledgers) == set(names), "job-ledger policy set mismatch")

    trial_owner: dict[int, str] = {}
    rows: list[dict[str, Any]] = []
    all_queues: set[str] = set()
    for name in names:
        policy = policies[name]
        summary = runtime_summaries[name]
        ledger = job_ledgers[name]
        frequency = policy.get("frequency")
        trials = policy.get("trials")
        require(
            isinstance(frequency, int) and not isinstance(frequency, bool)
            and frequency > 0 and isinstance(trials, list)
            and frequency == len(trials),
            f"{name}: invalid population frequency",
        )
        for trial in trials:
            require(
                isinstance(trial, int) and not isinstance(trial, bool)
                and 1 <= trial <= EXPECTED_TRIALS,
                f"{name}: invalid trial index",
            )
            require(trial not in trial_owner, f"trial {trial} belongs to two policies")
            trial_owner[trial] = name

        require(summary.get("schema_version") == RUNTIME_SCHEMA,
                f"{name}: wrong runtime schema")
        require(summary.get("dossier_id") == analysis.get("dossier_id"),
                f"{name}: dossier mismatch")
        require(summary.get("candidate_arm") == name,
                f"{name}: candidate-arm mismatch")
        require(summary.get("candidate_population_frequency") == frequency,
                f"{name}: runtime frequency mismatch")
        require(summary.get("candidate_population_trials") == trials,
                f"{name}: runtime trial partition mismatch")
        candidate_provenance = summary.get("response_provenance", {}).get(name, {})
        require(
            candidate_provenance.get("response_sha256")
            == policy.get("representative_response_sha256"),
            f"{name}: representative response mismatch",
        )
        binary_hashes = summary.get("binary_sha256_by_arm", {})
        require(name in binary_hashes and "default" in binary_hashes,
                f"{name}: missing materialized binary identity")
        require(binary_hashes[name] != binary_hashes["default"],
                f"{name}: candidate did not materialize a distinct compiler policy")

        paired = summary.get("paired_runtime", {}).get("default")
        require(isinstance(paired, dict), f"{name}: missing default comparison")
        replicate_count = paired.get("replicate_count")
        replicates = summary.get("replicates")
        require(
            isinstance(replicate_count, int) and replicate_count > 0
            and isinstance(replicates, list) and len(replicates) == replicate_count,
            f"{name}: runtime replicate mismatch",
        )
        by_rep = paired.get("speedup_reference_over_candidate_by_replicate")
        require(isinstance(by_rep, dict) and len(by_rep) == replicate_count,
                f"{name}: missing paired replicate ratios")
        speedup = _finite_positive(
            paired.get("geomean_speedup_reference_over_candidate"),
            f"{name} speedup",
        )
        ci = paired.get("paired_bootstrap_95_percentile_ci", {})
        lower = _finite_positive(ci.get("lower_2_5_percent"), f"{name} CI lower")
        upper = _finite_positive(ci.get("upper_97_5_percent"), f"{name} CI upper")
        require(lower <= upper, f"{name}: inverted confidence interval")
        sign = paired.get("two_sided_exact_sign_test", {})
        sign_p = _finite_positive(sign.get("p_value"), f"{name} sign p")
        require(sign_p <= 1.0, f"{name}: sign p exceeds one")
        all_queues.update(ledger["queues"])

        actions = policy.get("actions_by_site")
        require(isinstance(actions, dict) and actions,
                f"{name}: missing compiler site actions")
        require(set(actions.values()) <= {"default", "proxy", "trigger"},
                f"{name}: archive contains a non-route transformation")
        rows.append({
            "policy": name,
            "frequency": frequency,
            "trials": trials,
            "compiler_site_count": len(actions),
            "materialized_binary_sha256": binary_hashes[name],
            "measured_replicates": replicate_count,
            "submitted_jobs_in_ledger": ledger["submitted_job_count"],
            "job_ledger_queues": ledger["queues"],
            "default_over_candidate_speedup": speedup,
            "paired_bootstrap_95_percentile_ci": {
                "lower": lower,
                "upper": upper,
                "excludes_parity": lower > 1.0 or upper < 1.0,
                "archived_method": ci.get("method"),
            },
            "candidate_faster_replicates": paired.get("candidate_faster_replicates"),
            "two_sided_exact_sign_p": sign_p,
            "runtime_selection_interpretation": "not_individually_confirmatory",
        })

    require(sorted(trial_owner) == list(range(1, EXPECTED_TRIALS + 1)),
            "policies do not partition trials 1..20")
    modal = min(rows, key=lambda row: (-row["frequency"], row["policy"]))
    best = min(
        rows,
        key=lambda row: (-row["default_over_candidate_speedup"], row["policy"]),
    )
    weighted_geomean = math.exp(sum(
        row["frequency"] * math.log(row["default_over_candidate_speedup"])
        for row in rows
    ) / EXPECTED_TRIALS)
    historical_only = all_queues != {CURRENT_REQUIRED_QUEUE}
    modal_ci = modal["paired_bootstrap_95_percentile_ci"]
    modal_stable_speedup = (
        modal["default_over_candidate_speedup"] > 1.0
        and modal_ci["lower"] > 1.0
        and modal["two_sided_exact_sign_p"] < 0.05
        and not historical_only
    )
    return {
        "policies": rows,
        "modal_representative": {
            **modal,
            "selection_rule": "highest response frequency; policy ID breaks ties",
            "selected_without_runtime_labels": True,
        },
        "posthoc_best_of_20_capability_ceiling": {
            **best,
            "selection_rule": "largest archived default-over-candidate speedup",
            "posthoc": True,
            "multiple_comparison_adjusted": False,
            "statistically_confirmatory": False,
        },
        "response_population_descriptive": {
            "frequency_weighted_geomean_default_speedup": weighted_geomean,
            "paired_confidence_interval_permitted": False,
            "reason": (
                "policy campaigns used different allocations and one policy used "
                "fewer measured replicates"
            ),
        },
        "queue_scope": {
            "observed_queues": sorted(all_queues),
            "current_required_queue": CURRENT_REQUIRED_QUEUE,
            "historical_only": historical_only,
        },
        "claim_flags": {
            "compiler_only_feasibility_supported": True,
            "stable_modal_speedup_supported": modal_stable_speedup,
            "posthoc_historical_capability_ceiling_observed": (
                best["default_over_candidate_speedup"] > 1.0
            ),
            "posthoc_ceiling_statistically_confirmed": False,
            "current_pdebug_performance_claim_supported": False,
            "relational_context_value_supported": False,
            "cross_program_generalization_supported": False,
            "upgraded_suite_eligibility_affected": False,
        },
    }


def build_report(
    protocol_path: Path,
    analysis_path: Path,
    manifest_path: Path,
    source_path: Path,
    runtime_root: Path,
    regenerated_dir: Path,
    trial_analyzer: Path,
    runtime_analyzer: Path,
) -> dict[str, Any]:
    protocol = read_json(protocol_path)
    analysis = read_json(analysis_path)
    manifest = read_json(manifest_path)
    require(protocol.get("schema_version") == PROTOCOL_SCHEMA, "wrong protocol schema")
    require(analysis.get("schema_version") == ANALYSIS_SCHEMA, "wrong analysis schema")
    require(manifest.get("schema_version") == MANIFEST_SCHEMA, "wrong manifest schema")
    required_boundary = {
        "calibration_labels_visible": False,
        "candidate_may_generate_code": False,
        "evaluation_results_visible": False,
        "model_tools": [],
        "source_visible": False,
    }
    require(protocol.get("data_boundary") == required_boundary,
            "protocol violates source-free compiler-only boundary")
    require(protocol.get("trials") == EXPECTED_TRIALS, "protocol is not 20 trials")
    require(protocol.get("scoring", {}).get("distinct_accepted_policies_are_materialized")
            is True, "protocol did not require distinct-policy materialization")
    require(protocol.get("scoring", {}).get(
        "oracle_labels_are_withheld_until_all_responses_are_frozen") is True,
        "oracle labels were not held out")
    require(analysis.get("protocol_sha256") == sha256_file(protocol_path),
            "analysis binds different protocol bytes")
    require(analysis.get("dossier_id") == protocol.get("dossier_id"),
            "analysis/protocol dossier mismatch")
    require(analysis.get("trial_count") == EXPECTED_TRIALS,
            "analysis trial count mismatch")
    require(analysis.get("accepted_count") == EXPECTED_TRIALS,
            "not every historical response was accepted")
    require(analysis.get("fallback_count") == 0, "historical fallback was applied")
    require(analysis.get("retry_count") == 0, "historical semantic retry occurred")
    require(analysis.get("unique_response_count") == EXPECTED_TRIALS,
            "historical raw responses are not all distinct")
    require(analysis.get("unique_policy_count") == len(analysis.get("policies", {})),
            "unique-policy count mismatch")
    require(analysis.get("decision_models"), "decision model identity is absent")

    manifest_source = manifest.get("source", {})
    require(manifest.get("dossier_id") == protocol.get("dossier_id"),
            "manifest/protocol dossier mismatch")
    require(manifest.get("boundary", {}).get("source_in_prompt") is False,
            "manifest says source entered prompt")
    require(manifest_source.get("path") == display_path(source_path),
            "source path differs from frozen manifest")
    require(manifest_source.get("sha256") == sha256_file(source_path),
            "current source differs from frozen source bytes")
    for label, frozen in protocol.get("frozen_inputs", {}).items():
        frozen_path = ROOT / frozen.get("path", "")
        require(frozen_path.is_file(), f"missing frozen {label}")
        require(sha256_file(frozen_path) == frozen.get("sha256"),
                f"frozen {label} hash mismatch")

    exact_regeneration(
        analysis_path, regenerated_dir / "trial-analysis.json", "trial analysis",
    )
    runtime_summaries: dict[str, dict[str, Any]] = {}
    job_ledgers: dict[str, dict[str, Any]] = {}
    runtime_evidence: dict[str, Any] = {}
    policy_names = sorted(analysis["policies"])
    require(policy_names == [f"llm-policy{index:02d}" for index in range(1, 6)],
            "historical archive is not the expected five-policy population")
    analysis_sha = sha256_file(analysis_path)
    for name in policy_names:
        run_dir = runtime_root / f"compiler-lto-{name}-balanced-v1"
        summary_path = run_dir / "summary.json"
        jobs_path = run_dir / "jobs.tsv"
        regenerated = regenerated_dir / f"{name}-summary.json"
        exact_regeneration(summary_path, regenerated, f"{name} runtime summary")
        summary = read_json(summary_path)
        require(summary.get("trial_analysis_sha256") == analysis_sha,
                f"{name}: runtime binds different trial analysis")
        expected_raw = {
            f"rep{rep}-{arm}.log"
            for rep in summary.get("replicates", [])
            for arm in summary.get("binary_sha256_by_arm", {})
        }
        observed_raw = {path.name for path in (run_dir / "raw").glob("*.log")}
        require(observed_raw == expected_raw, f"{name}: raw log set mismatch")
        runtime_summaries[name] = summary
        job_ledgers[name] = parse_jobs(jobs_path)
        runtime_evidence[name] = {
            "summary": evidence(summary_path),
            "jobs": evidence(jobs_path),
            "raw_logs": tree_evidence(run_dir / "raw"),
            "regenerated_summary": evidence(regenerated),
        }

    policy_result = summarize_policies(analysis, runtime_summaries, job_ledgers)
    payload = {
        "schema_version": REPORT_SCHEMA,
        "status": "historical_exploratory_compiler_only_evidence",
        "boundary": {
            "compiler_lto_decisions_only": True,
            "application_source_visible_to_model": False,
            "application_source_modified": False,
            "candidate_generated_code": False,
            "all_distinct_policies_materialized_by_compiler_lto": True,
            "transformation_scope": "existing route actions only",
            "auditor_invokes_provider": False,
            "auditor_invokes_compiler": False,
            "auditor_invokes_scheduler": False,
        },
        "population": {
            "trial_count": analysis["trial_count"],
            "accepted_count": analysis["accepted_count"],
            "fallback_count": analysis["fallback_count"],
            "retry_count": analysis["retry_count"],
            "unique_raw_response_count": analysis["unique_response_count"],
            "unique_materialized_policy_count": analysis["unique_policy_count"],
            "decision_models": analysis["decision_models"],
            "all_provider_models_observed": analysis.get("all_provider_models_observed"),
            "provider_cost_usd_reported_total": analysis.get(
                "provider_cost_usd_reported_total"
            ),
        },
        "runtime": policy_result,
        "interpretation": {
            "supported": (
                "A source-free LLM produced legal compiler-only route decisions; "
                "the strict bridge accepted all 20 responses and LTO materialized "
                "all five distinct policies from unchanged application source."
            ),
            "exploratory_ceiling": (
                "The best runtime policy is a post-hoc best-of-20 observation, "
                "not typical model behavior or confirmatory performance evidence."
            ),
            "not_supported": [
                "stable modal-policy speedup",
                "current pdebug performance",
                "benefit from relational versus descriptor or opaque compiler input",
                "cross-program or cross-platform generalization",
                "readiness of the upgraded compiler decision suite",
            ],
        },
        "implementation": {
            "historical_auditor": evidence(Path(__file__)),
            "trial_analyzer": evidence(trial_analyzer),
            "runtime_analyzer": evidence(runtime_analyzer),
        },
        "evidence": {
            "protocol": evidence(protocol_path),
            "trial_analysis": evidence(analysis_path),
            "frozen_manifest": evidence(manifest_path),
            "unchanged_source": evidence(source_path),
            "regenerated_trial_analysis": evidence(
                regenerated_dir / "trial-analysis.json"
            ),
            "runtime_policies": runtime_evidence,
        },
    }
    result = dict(payload)
    result["audit_id"] = fingerprint(payload)
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
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--regenerated-dir", type=Path, required=True)
    parser.add_argument("--trial-analyzer", type=Path, required=True)
    parser.add_argument("--runtime-analyzer", type=Path, required=True)


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
            args.protocol, args.analysis, args.manifest, args.source,
            args.runtime_root, args.regenerated_dir, args.trial_analyzer,
            args.runtime_analyzer,
        )
        if args.command == "emit":
            write_json_atomic(args.out, report)
            action = "wrote"
        else:
            require(read_json(args.report) == report,
                    "report does not match current evidence")
            action = "verified"
        modal = report["runtime"]["modal_representative"]
        ceiling = report["runtime"]["posthoc_best_of_20_capability_ceiling"]
        print(
            f"historical-compiler-llm-ceiling: {action}; "
            f"modal={modal['policy']}:{modal['default_over_candidate_speedup']:.6f}x; "
            f"posthoc_best={ceiling['policy']}:"
            f"{ceiling['default_over_candidate_speedup']:.6f}x; "
            f"audit_id={report['audit_id']}"
        )
        return 0
    except (HistoricalCeilingError, OSError, KeyError, TypeError, ValueError) as exc:
        print(f"historical-compiler-llm-ceiling: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
