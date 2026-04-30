#!/usr/bin/env python3
"""GICC dispatch decider stub.

Reads gicc-features.json (output of GICCFeatureExtraction) and emits
gicc-hint.json (input to GICCDispatchLowering).

v1: rule-based.
v2: replace decide_one() with an ML model invoked from a Python
    inference framework (sklearn / onnxruntime / torch).

Environment:
  GICC_FEATURES_FILE  path to features.json (input)   [required]
  GICC_HINT_FILE      path to hint.json (output)      [required]
"""

import json
import os
import sys
from pathlib import Path
from typing import Any


def decide_one(feat: dict[str, Any]) -> dict[str, Any] | None:
    """Pick a dispatch for a single (launch site x op) record.

    Returns a hint entry suitable for hint.json's "sites" map, or None
    if the op is not eligible for routing (flush / quiet / unknown
    op_kind always inherit the default).
    """
    op_kind = feat.get("op_kind")
    if op_kind not in ("put_no_db", "get_no_db"):
        return None

    # Same-node peers go through IPC (hipMemcpyAsync). Anything else
    # falls through to the safe default (DWQ_TRIGGER, set globally).
    if (feat.get("peer_kind") == "const"
            and feat.get("peer_locality") == "same_node"):
        return {"dispatch": "IPC_PUSH"}

    return None


def main() -> int:
    try:
        feat_path = Path(os.environ["GICC_FEATURES_FILE"])
        hint_path = Path(os.environ["GICC_HINT_FILE"])
    except KeyError as e:
        print(
            f"gicc-decider: required env var {e} not set", file=sys.stderr)
        return 1

    if not feat_path.exists():
        print(f"gicc-decider: {feat_path} does not exist", file=sys.stderr)
        return 1

    features = json.loads(feat_path.read_text())
    hint: dict[str, Any] = {
        "version": 1,
        "schema_version": "gicc-hint-v1",
        "default_dispatch": "DWQ_TRIGGER",
        "sites": {},
    }

    for feat in features:
        site_id = feat.get("site_id")
        if not site_id:
            continue
        decision = decide_one(feat)
        if decision is not None:
            hint["sites"][site_id] = decision

    hint_path.parent.mkdir(parents=True, exist_ok=True)
    hint_path.write_text(json.dumps(hint, indent=2) + "\n")
    print(
        f"gicc-decider: wrote {len(hint['sites'])} site hint(s) "
        f"to {hint_path}",
        file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
