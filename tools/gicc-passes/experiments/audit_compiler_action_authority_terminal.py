#!/usr/bin/env python3
"""Close terminal-negative transforms in the compiler authority audit.

The base authority audit measures interface width independently of model
identity.  This adapter retains that result while partitioning the historical
conditional transform frontier into unresolved, realized, and terminally
closed entries.  It invokes no compiler, scheduler, benchmark, model, provider,
or source reader/editor.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import audit_compiler_action_authority as base  # noqa: E402
import audit_compiler_terminal_negatives as terminal  # noqa: E402


PROGRAM_NAME = "compiler-action-authority-terminal"
TERMINAL_LABELS = {"jacobi", "loop_lto", "mm_minimal"}


def apply_terminal_negatives(
    authority: dict[str, Any], terminal_report: dict[str, Any],
    terminal_path: Path, prompt_dir: Path,
) -> dict[str, Any]:
    result = copy.deepcopy(authority)
    candidates = terminal_report.get("candidates")
    if not isinstance(candidates, dict) or set(candidates) != TERMINAL_LABELS:
        raise base.AuthorityError(
            "terminal-negative report does not bind the conditional frontier"
        )
    if any(
        not isinstance(item, dict)
        or item.get("status") != "closed_negative"
        or item.get("model_visible") is not False
        or item.get("confirmation_eligible") is not False
        or item.get("paper_performance_claim") is not False
        for item in candidates.values()
    ):
        raise base.AuthorityError(
            "terminal-negative authority disposition is not fail-closed"
        )
    conditional = result.get("conditional_compiler_policy_authority")
    if not isinstance(conditional, dict):
        raise base.AuthorityError("authority report lacks conditional frontier")
    if (
        set(conditional.get("unresolved_conditional_entries", []))
        != TERMINAL_LABELS
        or conditional.get("realized_current_entries") != []
        or conditional.get("model_visible_transform_count") != 0
        or conditional.get("runtime_unconfirmed_transform_count") != 3
    ):
        raise base.AuthorityError(
            "terminal overlay would replace a non-pending frontier"
        )
    conditional.update({
        "unresolved_conditional_entries": [],
        "realized_current_entries": [],
        "closed_terminal_entries": sorted(TERMINAL_LABELS),
        "runtime_unconfirmed_transform_count": 0,
        "terminal_negative_transform_count": len(TERMINAL_LABELS),
        "terminal_negative_reason_codes": {
            label: candidates[label]["reason_code"]
            for label in sorted(TERMINAL_LABELS)
        },
        "model_visible_transform_count": 0,
        "performance_claim_supported": False,
    })
    claims = result.get("claim_separation")
    if not isinstance(claims, dict):
        raise base.AuthorityError("authority report lacks claim separation")
    claims.update({
        "conditional_runtime_unconfirmed_nonroute_transforms": 0,
        "terminal_negative_nonroute_transforms": len(TERMINAL_LABELS),
        "llm_performance_superiority_claimed": False,
    })
    old_id = result.pop("authority_id", None)
    if not isinstance(old_id, str):
        raise base.AuthorityError("base authority report lacks an identity")
    result["pre_terminal_overlay_authority_id"] = old_id
    result["terminal_negatives_id"] = terminal_report["terminal_negatives_id"]
    result["prompt_dir"] = base.display_path(prompt_dir)
    evidence = result.get("evidence")
    if not isinstance(evidence, dict):
        raise base.AuthorityError("base authority report lacks evidence")
    evidence["terminal_negatives"] = base.evidence(terminal_path)
    payload = dict(result)
    return {"authority_id": base.bridge._fingerprint(payload), **payload}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    authority = base.verify_report(base.build_report(args))
    terminal_report = terminal.verify_contained(args.terminal_negatives)
    result = apply_terminal_negatives(
        authority, terminal_report, args.terminal_negatives,
        args.prompt_dir.resolve(),
    )
    return base.verify_report(result)


def add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--gbt-report", type=Path, required=True)
    parser.add_argument("--input-separation", type=Path, required=True)
    parser.add_argument(
        "--communication", type=base.parse_spec, action="append", required=True
    )
    parser.add_argument("--structural-graph", type=Path, required=True)
    parser.add_argument("--collective-graph", type=Path, required=True)
    parser.add_argument(
        "--collective-label",
        choices=("collective_n8", "collective_n6"),
        default=base.DEFAULT_COLLECTIVE_LABEL,
    )
    parser.add_argument("--conditional-frontier", type=Path, required=True)
    parser.add_argument("--terminal-negatives", type=Path, required=True)


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
        result = build_report(args)
        if args.command == "emit":
            base.atomic_write(
                args.out.resolve(),
                json.dumps(result, indent=2, sort_keys=True) + "\n",
            )
            action = "wrote"
        else:
            if base.read_json(args.report) != result:
                raise base.AuthorityError(
                    "terminal authority report does not match current evidence"
                )
            action = "verified"
        conditional = result["conditional_compiler_policy_authority"]
        print(
            f"{PROGRAM_NAME}: {action}; current_nonroute_ids="
            f"{result['claim_separation']['current_nonroute_compiler_candidate_or_option_ids']}; "
            f"terminal_negative="
            f"{conditional['terminal_negative_transform_count']}; "
            f"provider_calls_authorized=0; authority_id="
            f"{result['authority_id']}"
        )
        return 0
    except Exception as exc:
        print(f"{PROGRAM_NAME}: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
