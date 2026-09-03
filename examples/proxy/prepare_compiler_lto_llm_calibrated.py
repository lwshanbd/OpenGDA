#!/usr/bin/env python3
"""Freeze a source-free, calibration-enriched LTO LLM experiment package.

This tool never invokes a provider and never submits a scheduler job.  It gives
an LLM the same independent compiler-path calibration evidence used by the
calibrated GBT, plus compiler-derived relations among evaluation sites.  Test
runtime results and oracle labels remain excluded.
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
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/gicc-passes/python"))
sys.path.insert(0, str(HERE))

import compiler_lto_calibrated_gbt as calibrated  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import run_compiler_lto_llm_trials as zero_shot  # noqa: E402


REQUEST_SCHEMA = "gicc-compiler-lto-llm-calibrated-request-v1"
PACK_SCHEMA = "gicc-compiler-lto-in-context-calibration-v1"
TRIALS = 20
RUNNER = HERE / "run_compiler_lto_llm_calibrated_trials.py"
SYSTEM_PROMPT = (
    "You are a constrained decision component inside an LTO communication "
    "pipeline, not a coding agent. Use only the compiler-derived evaluation "
    "facts, relations, and independent measured calibration cases in the user "
    "prompt. Infer interactions rather than matching names. Do not request, "
    "infer, emit, or modify source code; do not assert new legality. Return "
    "only the JSON object required by the supplied schema.\n"
)


class PrepareError(RuntimeError):
    """The matched-evidence package is incomplete, stale, or unsafe."""


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
    ).encode()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PrepareError(f"cannot read JSON {path}: {exc}") from exc


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_json_atomic(path: Path, value: Any) -> None:
    write_text_atomic(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def source_record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": display_path(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def compiler_facts(facts: dict[str, Any]) -> dict[str, Any]:
    """Retain the exact GBT evidence but remove scenario/site name tokens."""
    fields = (
        "size_bytes", "batch_size", "grid_blocks", "site_count", "in_loop",
        "descriptor_reusable", "coalescable", "flops_to_first_use",
        "distance_exact",
    )
    result = {field: facts[field] for field in fields}
    result["total_bytes"] = facts["size_bytes"] * facts["batch_size"]
    return result


def calibration_pack(calibration_dossier_path: Path,
                     calibration_results_path: Path,
                     evaluation_dossier_path: Path) -> tuple[
                         dict[str, Any], dict[str, Any], dict[str, Any]
                     ]:
    calibration_dossier = bridge._verified_dossier(
        read_json(calibration_dossier_path)
    )
    evaluation_dossier = bridge._verified_dossier(
        read_json(evaluation_dossier_path)
    )
    results = read_json(calibration_results_path)
    calibration_sizes = {
        site["size_bytes"] for site in calibration_dossier["sites"]
    }
    evaluation_sizes = {
        site["size_bytes"] for site in evaluation_dossier["sites"]
    }
    overlap = calibration_sizes & evaluation_sizes
    if overlap:
        raise PrepareError(
            f"calibration/evaluation message sizes overlap: {sorted(overlap)}"
        )
    try:
        rows, _, _, excluded = calibrated.training_rows(
            calibration_dossier, results,
        )
    except calibrated.ModelError as exc:
        raise PrepareError(str(exc)) from exc
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["scenario"], []).append(row)
    cases = []
    for records in grouped.values():
        by_action = {record["action"]: record for record in records}
        if set(by_action) != {"proxy", "trigger"}:
            raise PrepareError("calibration case lacks both physical actions")
        facts = compiler_facts(by_action["proxy"]["facts"])
        if compiler_facts(by_action["trigger"]["facts"]) != facts:
            raise PrepareError("calibration action rows disagree on compiler facts")
        payload = {
            "compiler_facts": facts,
            "measured_end_to_end_us": {
                action: float(by_action[action]["cost_us"])
                for action in ("proxy", "trigger")
            },
        }
        cases.append({
            "case_id": "calibration:" + sha256_bytes(canonical(payload))[:24],
            **payload,
        })
    cases.sort(key=lambda row: row["case_id"])
    if len(cases) != 17 or len(rows) != 34:
        raise PrepareError(
            f"expected 17 stable cases/34 rows, got {len(cases)}/{len(rows)}"
        )
    payload = {
        "schema_version": PACK_SCHEMA,
        "scope": (
            "Independent real-LTO compiler-path calibration; evaluation "
            "runtime results, oracle labels, and source are absent."
        ),
        "evidence_parity": {
            "matched_comparator": "compiler-path calibrated GBT v2",
            "same_stable_scenarios": True,
            "same_action_cost_rows": True,
            "same_scalar_compiler_facts": True,
            "extra_llm_input": (
                "compiler-derived relations among evaluation sites"
            ),
        },
        "action_semantics": {
            "proxy": evaluation_dossier["action_semantics"]["proxy"],
            "trigger": evaluation_dossier["action_semantics"]["trigger"],
        },
        "excluded_unstable_case_count": len(excluded),
        "message_size_overlap_with_evaluation": [],
        "cases": cases,
    }
    pack = dict(payload)
    pack["pack_id"] = bridge._fingerprint(payload)
    return pack, calibration_dossier, evaluation_dossier


def evaluation_relations(dossier: dict[str, Any]) -> list[dict[str, Any]]:
    grouped = calibrated.group_dossier(dossier)
    try:
        proxy_worker_lanes = dossier["platform_profile"][
            "deployment_constraints"
        ]["proxy_worker_lanes"]
    except (KeyError, TypeError) as exc:
        raise PrepareError("evaluation dossier lacks proxy worker lanes") from exc
    if (not isinstance(proxy_worker_lanes, int)
            or proxy_worker_lanes <= 0):
        raise PrepareError("evaluation proxy worker lanes must be positive")
    relations = []
    for kernel, sites in sorted(grouped.items()):
        try:
            facts = calibrated.group_facts(sites)
        except calibrated.ModelError as exc:
            raise PrepareError(str(exc)) from exc
        relations.append({
            "kernel": kernel,
            "site_ids": facts["site_ids"],
            "compiler_facts": compiler_facts(facts),
            "shared_decision_context": len(sites) > 1,
            "legal_physical_actions_shared_by_all_sites": facts[
                "legal_actions"
            ],
            "resource_relations": {
                "proxy_worker_lanes": proxy_worker_lanes,
                "max_concurrent_proxy_producers": min(
                    facts["grid_blocks"], proxy_worker_lanes,
                ),
                "group_site_count": len(sites),
            },
            "relation": (
                "sites execute in one named scenario and share its measured "
                "end-to-end objective; concurrent proxy producers share the "
                "finite worker-lane budget"
            ),
        })
    return relations


def render_prompt(pack: dict[str, Any], evaluation: dict[str, Any]) -> str:
    relations = evaluation_relations(evaluation)
    preamble = (
        "You are the decision component of an LTO communication pass.\n"
        "The calibration cases were measured on the same real compiler/LTO "
        "path and platform family, with evaluation sizes and all evaluation "
        "results held out. Use their interactions as evidence; do not assume "
        "a one-dimensional size threshold. The evaluation relations are "
        "compiler-derived group structure, not source. Choose exactly one "
        "legal action for every site. A default action is an abstention and "
        "may be used only where it is listed. Do not invent actions, code, "
        "IR, legality, or site IDs.\n\n"
        "Return only the JSON object enforced by the supplied response schema."
    )
    body = {
        "calibration": pack,
        "evaluation_relations": relations,
        "evaluation_dossier": evaluation,
    }
    return preamble + "\n\nCOMPILER_EVIDENCE:\n" + json.dumps(
        body, indent=2, sort_keys=True,
    ) + "\n"


def delivery_record(path: Path, output_dir: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": path.resolve().relative_to(output_dir.resolve()).as_posix(),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def build_request(*, calibration_dossier_path: Path,
                  calibration_results_path: Path,
                  evaluation_dossier_path: Path,
                  output_dir: Path) -> dict[str, Any]:
    pack_path = output_dir / "calibration-pack.json"
    prompt_path = output_dir / "prompt.txt"
    system_path = output_dir / "system-prompt.txt"
    schema_path = output_dir / "response-schema.json"
    evaluation = bridge._verified_dossier(read_json(evaluation_dossier_path))
    payload = {
        "schema_version": REQUEST_SCHEMA,
        "status": "awaiting_explicit_provider_authorization",
        "question": (
            "Can an LLM use matched compiler-path calibration and relational "
            "compiler facts to choose compiler-owned LTO communication actions?"
        ),
        "compiler_only": True,
        "model_output_scope": "legal LTO action labels only",
        "evaluation_dossier_id": evaluation["dossier_id"],
        "source_inputs": [
            source_record(
                calibration_dossier_path, "calibration_compiler_dossier",
            ),
            source_record(
                calibration_results_path, "independent_calibration_labels",
            ),
            source_record(
                evaluation_dossier_path, "evaluation_compiler_dossier",
            ),
        ],
        "provider_delivery": {
            "files": [
                delivery_record(system_path, output_dir, "system_prompt"),
                delivery_record(prompt_path, output_dir, "user_prompt"),
                delivery_record(schema_path, output_dir, "response_schema"),
            ],
            "calibration_pack": delivery_record(
                pack_path, output_dir, "prompt_embedded_calibration",
            ),
            "independent_responses": TRIALS,
            "one_user_prompt_per_call": True,
            "calls_are_sequential": True,
        },
        "data_boundary": {
            "source_visible": False,
            "llvm_ir_visible": False,
            "evaluation_runtime_labels_visible": False,
            "evaluation_oracle_visible": False,
            "calibration_labels_visible": True,
            "calibration_and_evaluation_sizes_disjoint": True,
            "model_tools": [],
            "model_may_generate_code": False,
            "model_may_generate_ir": False,
            "compiler_revalidates_every_response": True,
        },
        "matched_evidence_control": {
            "comparator": "compiler-path calibrated GBT v2",
            "training_scenarios": 17,
            "action_cost_rows": 34,
            "calibration_cases_match_comparator": True,
            "calibration_action_cost_rows_match_comparator": True,
            "calibration_scalar_facts_match_comparator": True,
            "llm_additional_evidence": [
                "full source-free compiler evaluation dossier",
                "compiler-derived relational grouping of evaluation sites",
            ],
            "purpose": (
                "LLM compiler-stage capability upper bound, not a "
                "same-input model contest"
            ),
        },
        "preregistered_analysis": {
            "primary": "exact evaluation action-oracle agreement",
            "secondary": [
                "accepted_response_rate",
                "unique_policy_count",
                "modal_policy_frequency",
                "agreement_with_calibrated_gbt",
                "runtime only for distinct accepted policies",
            ],
            "invalid_response": (
                "one semantic failure and deterministic compiler fallback; "
                "no semantic retry"
            ),
            "runtime_protocol": (
                "reuse an existing byte-identical policy result when exact; "
                "otherwise compile through LTO and run at most one pdebug job "
                "at a time under a separately frozen protocol"
            ),
        },
        "authorization": {
            "required": True,
            "granted": False,
            "must_bind_exactly": [
                "request_id",
                "system_prompt_sha256",
                "prompt_sha256",
                "response_schema_sha256",
                "independent_responses",
                "provider.kind",
                "provider.executable",
                "provider.cli_version",
                "provider.requested_model",
                "provider.effort",
                "provider.fresh_session_per_trial",
                "provider.structured_output",
                "transport_retry",
            ],
            "note": (
                "Previous zero-shot or collective authorization does not bind "
                "this prompt hash and cannot be reused."
            ),
        },
        "implementation": {
            "preparer": source_record(
                HERE / "prepare_compiler_lto_llm_calibrated.py", "preparer",
            ),
            "provider_runner": source_record(RUNNER, "provider_runner"),
            "provider_runner_frozen": True,
        },
    }
    request = dict(payload)
    request["request_id"] = bridge._fingerprint(payload)
    return request


def prepare(*, calibration_dossier_path: Path,
            calibration_results_path: Path,
            evaluation_dossier_path: Path,
            output_dir: Path) -> dict[str, Any]:
    if output_dir.exists():
        raise PrepareError(f"refusing to overwrite output: {output_dir}")
    pack, _, evaluation = calibration_pack(
        calibration_dossier_path, calibration_results_path,
        evaluation_dossier_path,
    )
    output_dir.mkdir(parents=True)
    write_json_atomic(output_dir / "calibration-pack.json", pack)
    write_text_atomic(output_dir / "system-prompt.txt", SYSTEM_PROMPT)
    write_text_atomic(output_dir / "prompt.txt", render_prompt(pack, evaluation))
    write_json_atomic(
        output_dir / "response-schema.json",
        zero_shot.exact_response_schema(evaluation),
    )
    request = build_request(
        calibration_dossier_path=calibration_dossier_path,
        calibration_results_path=calibration_results_path,
        evaluation_dossier_path=evaluation_dossier_path,
        output_dir=output_dir,
    )
    write_json_atomic(output_dir / "request.json", request)
    return request


def verify(*, request_path: Path, calibration_dossier_path: Path,
           calibration_results_path: Path,
           evaluation_dossier_path: Path) -> dict[str, Any]:
    output_dir = request_path.resolve().parent
    request = read_json(request_path)
    if (not isinstance(request, dict)
            or request.get("schema_version") != REQUEST_SCHEMA):
        raise PrepareError(f"expected {REQUEST_SCHEMA}")
    pack, _, evaluation = calibration_pack(
        calibration_dossier_path, calibration_results_path,
        evaluation_dossier_path,
    )
    if read_json(output_dir / "calibration-pack.json") != pack:
        raise PrepareError("calibration pack changed")
    if (output_dir / "system-prompt.txt").read_text() != SYSTEM_PROMPT:
        raise PrepareError("system prompt changed")
    if (output_dir / "prompt.txt").read_text() != render_prompt(
        pack, evaluation,
    ):
        raise PrepareError("user prompt changed")
    if read_json(output_dir / "response-schema.json") != (
        zero_shot.exact_response_schema(evaluation)
    ):
        raise PrepareError("response schema changed")
    expected = build_request(
        calibration_dossier_path=calibration_dossier_path,
        calibration_results_path=calibration_results_path,
        evaluation_dossier_path=evaluation_dossier_path,
        output_dir=output_dir,
    )
    if request != expected:
        raise PrepareError("request does not regenerate exactly")
    return request


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "verify"):
        sub = subparsers.add_parser(name)
        sub.add_argument("--calibration-dossier", type=Path, required=True)
        sub.add_argument("--calibration-results", type=Path, required=True)
        sub.add_argument("--evaluation-dossier", type=Path, required=True)
        if name == "prepare":
            sub.add_argument("--output-dir", type=Path, required=True)
        else:
            sub.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    try:
        common = {
            "calibration_dossier_path": args.calibration_dossier.resolve(),
            "calibration_results_path": args.calibration_results.resolve(),
            "evaluation_dossier_path": args.evaluation_dossier.resolve(),
        }
        if args.command == "prepare":
            request = prepare(
                **common, output_dir=args.output_dir.resolve(),
            )
            print(
                "calibrated LTO LLM package frozen but NOT AUTHORIZED: "
                f"request_id={request['request_id']} trials={TRIALS}"
            )
        else:
            request = verify(**common, request_path=args.request.resolve())
            print(
                "calibrated LTO LLM package verified and NOT AUTHORIZED: "
                f"request_id={request['request_id']}"
            )
        return 0
    except (PrepareError, bridge.BridgeError, OSError, KeyError,
            TypeError, ValueError) as exc:
        print(f"prepare-compiler-lto-llm-calibrated: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
