#!/usr/bin/env python3
"""Freeze analytic chance baselines for 20 compiler-policy samples.

The null assumes independent uniform draws over an entry's graph-bound legal
policies and exactly one predesignated oracle.  It is a calibration reference,
not a model-distribution assumption and not runtime evidence.  The tool reads
only frozen JSON artifacts; it invokes no compiler, scheduler, benchmark,
model, provider, or application-source path.
"""

from __future__ import annotations

import argparse
from fractions import Fraction
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
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE))

import audit_compiler_action_authority as action_authority  # noqa: E402
import audit_compiler_action_frontier as action_frontier  # noqa: E402
import gicc_compiler_decision_suite as decision_suite  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402


REPORT_SCHEMA = "gicc-llm-sampling-null-v1"
HISTORICAL_SCHEMA = "gicc-historical-compiler-llm-ceiling-audit-v1"
TRIALS_PER_VIEW = 20
ALPHA = Fraction(1, 20)
BOUNDARY = {
    "application_source_read": False,
    "application_source_modified": False,
    "compiler_invoked": False,
    "scheduler_invoked": False,
    "runtime_benchmark_invoked": False,
    "model_invoked": False,
    "provider_invoked": False,
    "provider_call_authorized": False,
    "null_is_runtime_evidence": False,
    "null_is_model_distribution_assumption": False,
}


class SamplingNullError(RuntimeError):
    """Frozen inputs do not support the requested chance calibration."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SamplingNullError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise SamplingNullError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def evidence(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise SamplingNullError(f"missing null evidence: {resolved}")
    return {
        "path": display_path(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def fraction_record(value: Fraction) -> dict[str, Any]:
    return {
        "exact": f"{value.numerator}/{value.denominator}",
        "value": float(value),
    }


def binomial_tail(draws: int, probability: Fraction, threshold: int) -> Fraction:
    if (draws <= 0 or probability <= 0 or probability > 1
            or threshold < 0 or threshold > draws + 1):
        raise SamplingNullError("invalid binomial-tail parameters")
    if threshold == draws + 1:
        return Fraction(0)
    return sum(
        Fraction(math.comb(draws, successes))
        * probability ** successes
        * (1 - probability) ** (draws - successes)
        for successes in range(threshold, draws + 1)
    )


def minimum_significant_hits(
    policy_count: int, draws: int, alpha: Fraction = ALPHA,
) -> tuple[int, Fraction]:
    probability = Fraction(1, policy_count)
    for threshold in range(1, draws + 2):
        tail = binomial_tail(draws, probability, threshold)
        if tail <= alpha:
            return threshold, tail
    raise SamplingNullError("cannot find a binomial significance threshold")


def uniform_oracle_null(policy_count: int, draws: int) -> dict[str, Any]:
    if policy_count <= 0 or draws <= 0:
        raise SamplingNullError("policy count and draw count must be positive")
    single = Fraction(1, policy_count)
    at_least_one = 1 - (1 - single) ** draws
    expected_hits = Fraction(draws, policy_count)
    expected_unique = policy_count * at_least_one
    threshold, threshold_tail = minimum_significant_hits(policy_count, draws)
    return {
        "legal_policy_count": policy_count,
        "draw_count": draws,
        "predesignated_unique_oracle_count": 1,
        "single_draw_exact_oracle_probability": fraction_record(single),
        "at_least_one_exact_oracle_in_draws_probability": fraction_record(
            at_least_one
        ),
        "expected_exact_oracle_hits": fraction_record(expected_hits),
        "expected_unique_policies": fraction_record(expected_unique),
        "minimum_exact_hits_for_one_sided_alpha_0_05": threshold,
        "tail_probability_at_threshold": fraction_record(threshold_tail),
    }


def all_categories_observed_probability(categories: int, draws: int) -> Fraction:
    if categories <= 0 or draws <= 0:
        raise SamplingNullError("categories and draws must be positive")
    if draws < categories:
        return Fraction(0)
    favorable = sum(
        (-1) ** omitted * math.comb(categories, omitted)
        * (categories - omitted) ** draws
        for omitted in range(categories + 1)
    )
    return Fraction(favorable, categories ** draws)


def maximum_occupancy_tail(
    categories: int, draws: int, threshold: int,
) -> Fraction:
    if (categories <= 0 or draws <= 0 or threshold <= 0
            or threshold > draws):
        raise SamplingNullError("invalid occupancy-tail parameters")
    # Coefficients of (sum_{j=0}^{threshold-1} x^j/j!)^categories.
    coefficients = [Fraction(1)] + [Fraction(0)] * draws
    base = [Fraction(1, math.factorial(count))
            for count in range(threshold)]
    for _ in range(categories):
        updated = [Fraction(0)] * (draws + 1)
        for used, coefficient in enumerate(coefficients):
            if coefficient == 0:
                continue
            for count, factor in enumerate(base):
                if used + count <= draws:
                    updated[used + count] += coefficient * factor
        coefficients = updated
    sequences_with_max_below = math.factorial(draws) * coefficients[draws]
    return 1 - sequences_with_max_below / categories ** draws


def verified_historical(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != HISTORICAL_SCHEMA:
        raise SamplingNullError(f"expected historical schema {HISTORICAL_SCHEMA}")
    payload = dict(value)
    audit_id = payload.pop("audit_id", None)
    if audit_id != bridge._fingerprint(payload):
        raise SamplingNullError("historical audit ID does not match content")
    population = value.get("population", {})
    modal = value.get("runtime", {}).get("modal_representative", {})
    best = value.get("runtime", {}).get(
        "posthoc_best_of_20_capability_ceiling", {}
    )
    if (population.get("trial_count") != TRIALS_PER_VIEW
            or population.get("accepted_count") != TRIALS_PER_VIEW
            or population.get("unique_materialized_policy_count") != 5
            or modal.get("frequency") != 10
            or best.get("posthoc") is not True
            or value.get("runtime", {}).get("claim_flags", {}).get(
                "posthoc_ceiling_statistically_confirmed") is not False):
        raise SamplingNullError("historical sampling population changed")
    return value


def resolved_frontier_sets(
    authority: dict[str, Any], frontier: dict[str, Any], suite_id: str,
) -> tuple[set[str], set[str]]:
    """Partition old conditional candidates against the current suite audit."""
    authority_frontier = authority.get(
        "conditional_compiler_policy_authority", {}
    )
    if authority_frontier.get("frontier_id") != frontier.get("frontier_id"):
        raise SamplingNullError("action authority binds another compiler frontier")
    frontier_labels = {
        item.get("label") for item in frontier.get("conditional_frontier", [])
    }
    unresolved = authority_frontier.get("unresolved_conditional_entries")
    realized = authority_frontier.get("realized_current_entries")
    if unresolved is None and realized is None:
        if frontier.get("suite_id") != suite_id:
            raise SamplingNullError("conditional frontier binds another suite")
        return set(frontier_labels), set()
    if (not isinstance(unresolved, list) or not isinstance(realized, list)
            or any(not isinstance(label, str)
                   for label in [*unresolved, *realized])):
        raise SamplingNullError("action authority has invalid frontier routing")
    unresolved_set = set(unresolved)
    realized_set = set(realized)
    if (unresolved_set & realized_set
            or unresolved_set | realized_set != frontier_labels
            or authority_frontier.get("frontier_suite_id")
            != frontier.get("suite_id")):
        raise SamplingNullError(
            "action authority does not partition the frozen frontier"
        )
    return unresolved_set, realized_set


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    suite_path = args.suite.resolve()
    suite = decision_suite.verified_suite(
        read_json(suite_path), args.prompt_dir.resolve(),
    )
    authority_path = args.action_authority.resolve()
    authority = action_authority.verify_report(read_json(authority_path))
    action_authority.verify_recorded_evidence(authority.get("evidence"))
    if authority.get("suite_id") != suite["suite_id"]:
        raise SamplingNullError("action authority binds another suite")

    frontier_path = args.conditional_frontier.resolve()
    frontier = action_frontier.verify_report(read_json(frontier_path))
    action_authority.verify_recorded_evidence(frontier.get("evidence"))
    unresolved_set, realized_set = resolved_frontier_sets(
        authority, frontier, suite["suite_id"],
    )

    historical_path = args.historical.resolve()
    historical = verified_historical(read_json(historical_path))

    current = {}
    for entry in sorted(suite["entries"], key=lambda item: item["label"]):
        count = entry.get("decision_space", {}).get("independent_policy_count")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise SamplingNullError(f"{entry.get('label')}: invalid policy count")
        current[entry["label"]] = uniform_oracle_null(
            count, TRIALS_PER_VIEW,
        )

    conditional = {}
    for item in frontier["conditional_frontier"]:
        if item["label"] in realized_set:
            continue
        count = item["conditional_independent_policy_count"]
        conditional[item["label"]] = {
            **uniform_oracle_null(count, TRIALS_PER_VIEW),
            "status": "conditional_on_positive_runtime_confirmation_and_refreeze",
            "model_visible_now": False,
        }

    empirical_support_size = historical["population"][
        "unique_materialized_policy_count"
    ]
    historical_null = uniform_oracle_null(
        empirical_support_size, TRIALS_PER_VIEW,
    )
    observed_modal_frequency = historical["runtime"]["modal_representative"][
        "frequency"
    ]
    historical_null.update({
        "scope": "conditional_uniform_null_over_the_five_observed_policies",
        "not_the_full_legal_policy_space": True,
        "observed_unique_policy_count": empirical_support_size,
        "observed_modal_frequency": observed_modal_frequency,
        "probability_all_five_policies_observed": fraction_record(
            all_categories_observed_probability(
                empirical_support_size, TRIALS_PER_VIEW,
            )
        ),
        "probability_max_frequency_at_least_observed": fraction_record(
            maximum_occupancy_tail(
                empirical_support_size, TRIALS_PER_VIEW,
                observed_modal_frequency,
            )
        ),
        "interpretation": (
            "Under a uniform null conditional on the five observed policies, "
            "covering all five and sampling a predesignated best policy at least "
            "once are expected, while a modal count of 10 is nonuniform. This "
            "supports response preference, not stable runtime benefit."
        ),
    })

    payload = {
        "schema_version": REPORT_SCHEMA,
        "boundary": dict(BOUNDARY),
        "suite_id": suite["suite_id"],
        "trials_per_view": TRIALS_PER_VIEW,
        "null_hypothesis": (
            "independent uniform legal-policy draws with exactly one "
            "predesignated oracle"
        ),
        "current_suite_uniform_null": current,
        "conditional_frontier_uniform_null": {
            label: conditional[label] for label in sorted(conditional)
        },
        "frontier_resolution": {
            "unresolved_conditional_entries": sorted(unresolved_set),
            "realized_current_entries": sorted(realized_set),
            "realized_entries_are_calibrated_in_current_suite": True,
        },
        "historical_empirical_support_null": historical_null,
        "claim_separation": {
            "best_of_20_requires_chance_calibration": True,
            "small_action_spaces_can_hit_oracle_by_chance": True,
            "uniform_null_is_a_model_distribution_claim": False,
            "uniform_null_is_performance_evidence": False,
            "modal_frequency_is_runtime_speedup": False,
            "llm_capability_claim_ready": False,
        },
        "evidence": {
            "auditor": evidence(Path(__file__)),
            "suite": evidence(suite_path),
            "action_authority": evidence(authority_path),
            "conditional_frontier": evidence(frontier_path),
            "historical_ceiling": evidence(historical_path),
        },
    }
    return {"null_id": bridge._fingerprint(payload), **payload}


def verify_report(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or value.get("schema_version") != REPORT_SCHEMA:
        raise SamplingNullError(f"expected report schema {REPORT_SCHEMA}")
    payload = dict(value)
    null_id = payload.pop("null_id", None)
    if null_id != bridge._fingerprint(payload):
        raise SamplingNullError("null_id does not match report content")
    if value.get("boundary") != BOUNDARY:
        raise SamplingNullError("sampling-null report crossed its offline boundary")
    claims = value.get("claim_separation", {})
    expected = {
        "best_of_20_requires_chance_calibration": True,
        "small_action_spaces_can_hit_oracle_by_chance": True,
        "uniform_null_is_a_model_distribution_claim": False,
        "uniform_null_is_performance_evidence": False,
        "modal_frequency_is_runtime_speedup": False,
        "llm_capability_claim_ready": False,
    }
    if claims != expected:
        raise SamplingNullError("sampling-null report overstates LLM evidence")
    return value


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--suite", type=Path, required=True)
    value.add_argument("--prompt-dir", type=Path, required=True)
    value.add_argument("--action-authority", type=Path, required=True)
    value.add_argument("--conditional-frontier", type=Path, required=True)
    value.add_argument("--historical", type=Path, required=True)
    value.add_argument("--out", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    report = verify_report(build_report(args))
    atomic_write(args.out.resolve(), json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
