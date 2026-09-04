#!/usr/bin/env python3
"""Freeze at most one compiler-only request for the widest eligible graph.

This is an offline selector.  It regenerates the final capability protocol,
ranks only runtime-eligible suite entries by legal compiler-policy count, and
freezes one source-free request bundle.  It never authorizes or invokes a
model, compiler, scheduler, runtime benchmark, or application-source path.
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
sys.path.insert(0, str(HERE.parent / "python"))
sys.path.insert(0, str(HERE))

import audit_compiler_llm_capability_protocol as capability  # noqa: E402
import gicc_compiler_decision_suite as decision_suite  # noqa: E402
import gicc_llm_bridge as bridge  # noqa: E402
import prepare_compiler_llm_capability_request as request_freezer  # noqa: E402


SELECTION_SCHEMA = "gicc-compiler-llm-priority-request-selection-v1"
PREFERRED_LABEL_ORDER = (
    "collective_n6",
    "coalescing_placement",
    "jacobi",
    "minimod",
    "mixed_lto",
    "mm_minimal",
    "loop_lto",
    "collective_n8",
)
BOUNDARY = {
    "compiler_lto_decisions_only": True,
    "application_source_read": False,
    "application_source_modified": False,
    "model_invoked": False,
    "provider_invoked": False,
    "provider_call_authorized": False,
    "scheduler_invoked": False,
    "runtime_benchmark_invoked": False,
    "maximum_requests_frozen": 1,
}


class PriorityRequestError(RuntimeError):
    """The final audit cannot support one exact request selection."""


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PriorityRequestError(f"cannot read JSON {path}: {exc}") from exc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
    except OSError as exc:
        raise PriorityRequestError(f"cannot hash {path}: {exc}") from exc
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
        raise PriorityRequestError(f"missing selection evidence: {resolved}")
    return {
        "path": display_path(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def parse_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("expected LABEL=PATH")
    label, raw = value.split("=", 1)
    if not label or not raw:
        raise argparse.ArgumentTypeError("expected nonempty LABEL=PATH")
    return label, Path(raw)


def policy_count(entry: dict[str, Any]) -> int:
    value = entry.get("decision_space", {}).get("independent_policy_count")
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PriorityRequestError(
            f"{entry.get('label')}: invalid independent policy count"
        )
    return value


def prioritized_entries(
    suite: dict[str, Any], protocol: dict[str, Any],
) -> list[dict[str, Any]]:
    entries = {entry["label"]: entry for entry in suite["entries"]}
    protocol_entries = protocol.get("entries")
    if not isinstance(protocol_entries, dict):
        raise PriorityRequestError("capability protocol lacks entries")
    eligible_summary = protocol.get("summary", {}).get(
        "runtime_eligible_entries"
    )
    eligible = sorted(
        label for label, record in protocol_entries.items()
        if isinstance(record, dict)
        and record.get("eligible_to_freeze_provider_request") is True
    )
    if eligible != sorted(eligible_summary or []):
        raise PriorityRequestError("capability eligibility summary changed")
    if any(label not in entries for label in eligible):
        raise PriorityRequestError("eligible entry is absent from suite")
    preference = {
        label: index for index, label in enumerate(PREFERRED_LABEL_ORDER)
    }
    return sorted(
        ({
            "label": label,
            "decision_family": entries[label]["decision_family"],
            "independent_policy_count": policy_count(entries[label]),
            "suite_entry_id": entries[label]["entry_id"],
            "compiler_graph_id": entries[label]["graph_id"],
        } for label in eligible),
        key=lambda item: (
            -item["independent_policy_count"],
            preference.get(item["label"], len(preference)),
            item["label"],
        ),
    )


def selection_payload(
    *, suite: dict[str, Any], protocol: dict[str, Any],
    ranked: list[dict[str, Any]], request_id: str | None,
    evidence_paths: dict[str, Path],
) -> dict[str, Any]:
    selected = ranked[0] if ranked else None
    return {
        "schema_version": SELECTION_SCHEMA,
        "status": (
            "one_request_frozen_awaiting_exact_authorization"
            if selected else "no_runtime_eligible_entry"
        ),
        "suite_id": suite["suite_id"],
        "capability_protocol_id": protocol["protocol_id"],
        "boundary": dict(BOUNDARY),
        "selection_rule": {
            "primary": "maximum independent compiler policy count",
            "tie_break": list(PREFERRED_LABEL_ORDER),
            "rationale": (
                "A wider graph is the stricter capability-ceiling test and "
                "has a lower uniform exact-oracle chance; this ranks compiler "
                "interfaces, not model brands."
            ),
            "freeze_only_one_request": True,
        },
        "eligible_entries_ranked": ranked,
        "selected_entry": selected,
        "request_id": request_id,
        "authorization_granted": False,
        "currently_permitted_provider_calls": 0,
        "evidence": {
            label: evidence(path)
            for label, path in sorted(evidence_paths.items())
        },
    }


def with_id(payload: dict[str, Any]) -> dict[str, Any]:
    return {"selection_id": bridge._fingerprint(payload), **payload}


def verify_selection(value: Any) -> dict[str, Any]:
    if (not isinstance(value, dict)
            or value.get("schema_version") != SELECTION_SCHEMA):
        raise PriorityRequestError(f"expected {SELECTION_SCHEMA}")
    payload = dict(value)
    observed = payload.pop("selection_id", None)
    if observed != bridge._fingerprint(payload):
        raise PriorityRequestError("selection ID does not match content")
    if value.get("boundary") != BOUNDARY:
        raise PriorityRequestError("selection crossed its offline boundary")
    if (value.get("authorization_granted") is not False
            or value.get("currently_permitted_provider_calls") != 0):
        raise PriorityRequestError("selection unexpectedly authorizes calls")
    ranked = value.get("eligible_entries_ranked")
    if not isinstance(ranked, list):
        raise PriorityRequestError("selection lacks ranked entries")
    if value.get("selected_entry") != (ranked[0] if ranked else None):
        raise PriorityRequestError("selection does not choose its first entry")
    return value


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


def derive(args: argparse.Namespace, output_dir: Path,
           freeze: bool) -> dict[str, Any]:
    suite_path = args.suite.resolve()
    prompt_dir = args.prompt_dir.resolve()
    suite = decision_suite.verified_suite(read_json(suite_path), prompt_dir)
    expected_protocol = capability.build_report(
        suite_path, prompt_dir, args.readiness.resolve(),
        args.input_separation.resolve(), args.sampling_null.resolve(),
    )
    protocol_path = args.capability_protocol.resolve()
    protocol = read_json(protocol_path)
    if protocol != expected_protocol:
        raise PriorityRequestError(
            "capability protocol does not regenerate from final audit"
        )
    ranked = prioritized_entries(suite, protocol)
    graph_specs = dict(args.graph)
    if len(graph_specs) != len(args.graph):
        raise PriorityRequestError("duplicate graph label")

    request_id = None
    if ranked:
        label = ranked[0]["label"]
        if label not in graph_specs:
            raise PriorityRequestError(f"selected graph is absent: {label}")
        inputs = request_freezer.verified_inputs(
            suite_path, prompt_dir, args.readiness.resolve(),
            args.input_separation.resolve(), args.sampling_null.resolve(),
            protocol_path, label, graph_specs[label].resolve(),
        )
        if freeze:
            request = request_freezer.prepare(inputs, output_dir / "request")
        else:
            request = request_freezer.verify_bundle(
                inputs, output_dir / "request"
            )
        request_id = request["request_id"]

    payload = selection_payload(
        suite=suite, protocol=protocol, ranked=ranked, request_id=request_id,
        evidence_paths={
            "selector": Path(__file__),
            "suite": suite_path,
            "capability_protocol": protocol_path,
            "request_freezer": Path(request_freezer.__file__),
        },
    )
    return verify_selection(with_id(payload))


def add_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--prompt-dir", type=Path, required=True)
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--input-separation", type=Path, required=True)
    parser.add_argument("--sampling-null", type=Path, required=True)
    parser.add_argument("--capability-protocol", type=Path, required=True)
    parser.add_argument("--graph", type=parse_spec, action="append", default=[])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    children = parser.add_subparsers(dest="command", required=True)
    freeze_parser = children.add_parser("freeze")
    add_inputs(freeze_parser)
    freeze_parser.add_argument("--output-dir", type=Path, required=True)
    verify_parser = children.add_parser("verify")
    add_inputs(verify_parser)
    verify_parser.add_argument("--request-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        output_dir = (
            args.output_dir.resolve() if args.command == "freeze"
            else args.request_root.resolve()
        )
        if args.command == "freeze":
            if output_dir.exists():
                raise PriorityRequestError(
                    f"refusing to overwrite request root: {output_dir}"
                )
            output_dir.mkdir(parents=True)
            report = derive(args, output_dir, True)
            write_json_atomic(output_dir / "selection.json", report)
            action = "frozen"
        else:
            report = derive(args, output_dir, False)
            if read_json(output_dir / "selection.json") != report:
                raise PriorityRequestError(
                    "selection report does not regenerate"
                )
            action = "verified"
        selected = report["selected_entry"]
        print(
            f"compiler-llm-priority-request: {action}; "
            f"selected={selected['label'] if selected else 'none'}; "
            f"calls_permitted=0; selection_id={report['selection_id']}"
        )
        return 0
    except (
        PriorityRequestError, capability.CapabilityProtocolError,
        decision_suite.SuiteError, request_freezer.CapabilityRequestError,
        OSError, KeyError, TypeError, ValueError,
    ) as exc:
        print(f"compiler-llm-priority-request: ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
