#!/usr/bin/env python3
"""Emit and validate the legality-masked O5 compiler hint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    features = json.loads(args.features.read_text())
    puts = [row for row in features if row.get("op_kind") == "put_no_db"]
    mirrored = [row for row in puts if row.get("kernel") == "halo_kernel_mirrored"]
    unmirrored = [row for row in puts if row.get("kernel") == "halo_kernel_unmirrored"]
    if len(mirrored) != 1 or len(unmirrored) != 1:
        raise SystemExit(
            f"expected one looped put per O5 kernel, got "
            f"mirrored={len(mirrored)} unmirrored={len(unmirrored)}")

    good = mirrored[0]
    bad = unmirrored[0]
    if not good.get("hk_capable") or "trigger" not in good.get("legal_paths", []):
        raise SystemExit("host-mirrored kernel did not unlock trigger legality")
    if bad.get("hk_capable") or bad.get("legal_paths") != ["proxy"]:
        raise SystemExit("unmirrored kernel was not restricted to proxy-only")

    hint = {
        "version": 1,
        "schema_version": "gicc-hint-v1",
        "default_dispatch": "CPU_PROXY_ENQUEUE",
        "sites": {
            good["site_id"]: {
                # A looped host-mirror trace is emitted in batched form.  The
                # current batched lowering supports the DWQ path directly;
                # IPC_OR_DWQ is legal in the abstract feature set but not yet
                # materializable for this trace shape.
                "dispatch": "DWQ_TRIGGER",
                "reason": "O5 host mirror makes pre-launch staging legal",
            },
            bad["site_id"]: {
                "dispatch": "CPU_PROXY_ENQUEUE",
                "reason": "O5 no mirror: descriptor fields are device-only",
            },
        },
        "o5_gate": {
            "mirrored_kernel": good["kernel"],
            "mirrored_legal_paths": good["legal_paths"],
            "unmirrored_kernel": bad["kernel"],
            "unmirrored_legal_paths": bad["legal_paths"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(hint, indent=2) + "\n")
    print(json.dumps(hint["o5_gate"], indent=2))


if __name__ == "__main__":
    main()
