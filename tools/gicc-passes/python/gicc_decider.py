#!/usr/bin/env python3
"""GICC dispatch decider.

Reads gicc-features.json (output of GICCFeatureExtraction) and emits
gicc-hint.json (input to GICCDispatchLowering).

The decision is a calibrated analytic rule, not a learned model. Both
dispatch paths pay a fixed per-phase cost plus a per-op issue cost, and
both can hide the wire time behind whatever compute separates the issue
from the first use of the result:

    exposed_trigger = C_T + K * T_stage + max(0, wire - D)
    exposed_proxy   = C_P + max(K * T_push, wire - D)
    wire            = bytes * K / BW

The trigger path stages descriptors on the host, so its per-op cost is
serial and cannot overlap the transfer; the proxy path pushes from
inside the kernel, so its per-op cost pipelines against the wire but is
several times larger. That asymmetry is why the winner flips with
issue-to-first-use distance at fixed message size, peer, and trip count
-- a flip no runtime-visible feature can predict.

The four constants are platform properties, measured once with
examples/proxy/ctx_bench --exp=distance and overridable from the
environment. On Tioga (MI250X + Slingshot 11) they are the defaults
below; on another machine, recalibrate rather than retrain.

Environment:
  GICC_FEATURES_FILE  path to features.json (input)   [required]
  GICC_HINT_FILE      path to hint.json (output)      [required]
  GICC_CAL_*          override a calibration constant [optional]
  GICC_FLOPS_PER_US   device compute rate, converts the static
                      flops_to_first_use count into a distance in us
"""

import json
import os
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Platform calibration. Measured on Tioga with
#   ctx_bench --path={proxy,trigger} --exp=distance --ops=1,4,16,64
# by fitting the fully-hidden regime (D >> wire), where exposed cost is
# linear in K and the slope is the per-op issue cost.
# ---------------------------------------------------------------------------
def _cal(name: str, default: float) -> float:
    return float(os.environ.get("GICC_CAL_" + name, default))


C_T = _cal("C_T", 7.88)        # us, fixed per-phase cost, trigger path
T_STAGE = _cal("T_STAGE", 1.033)   # us per op, host descriptor staging
C_P = _cal("C_P", 5.37)        # us, fixed per-phase cost, proxy path
T_PUSH = _cal("T_PUSH", 2.872)     # us per op, in-kernel ring push
BW_GBPS = _cal("BW", 24.0)     # GB/s, one NIC's ceiling

# Static flop counts are a proxy for time. This converts them; it is the
# weakest link in the chain, so the rule only trusts the distance when it
# changes the answer (see decide_one).
FLOPS_PER_US = float(os.environ.get("GICC_FLOPS_PER_US", 2000.0))


def exposed_us(bytes_: float, k: int, dist_us: float) -> tuple[float, float]:
    """Exposed communication cost of each path, in microseconds."""
    wire = bytes_ * k / (BW_GBPS * 1e9) * 1e6
    trig = C_T + k * T_STAGE + max(0.0, wire - dist_us)
    prox = C_P + max(k * T_PUSH, wire - dist_us)
    return trig, prox


def decide_one(feat: dict[str, Any]) -> dict[str, Any] | None:
    """Pick a dispatch for a single (launch site x op) record."""
    op_kind = feat.get("op_kind")
    if op_kind not in ("put_no_db", "get_no_db"):
        return None

    # Legality first. A site whose descriptor is not host-knowable cannot
    # be staged on the host at all, so the trigger path is not merely
    # slower, it is unavailable. GICCDispatchLowering enforces this too.
    if feat.get("hk_capable") is False:
        return {"dispatch": "CPU_PROXY_ENQUEUE",
                "reason": "not host-knowable"}

    # Same-node peers are a different question entirely (xGMI, not the
    # NIC); leave them on the hybrid default.
    if (feat.get("peer_kind") == "const"
            and feat.get("peer_locality") == "same_node"):
        return {"dispatch": "IPC_PUSH", "reason": "same-node peer"}

    # Performance. Needs a message size and a trip count; without either
    # the rule has nothing to weigh, so inherit the default rather than
    # guess.
    size_log2 = feat.get("size_log2")
    if size_log2 is None:
        return None
    bytes_ = float(1 << int(size_log2))

    k = feat.get("trip_count")
    if k is None:
        # In a loop whose bound is a runtime value we cannot know K. The
        # measured surface says K=1 favours the proxy only marginally
        # while large K favours the trigger strongly, so guessing K=1 is
        # the conservative reading of an unknown loop.
        k = 1 if not feat.get("in_loop") else None
        if k is None:
            return None
    k = max(1, int(k))

    flops = feat.get("flops_to_first_use")
    dist_us = 0.0 if flops is None else float(flops) / FLOPS_PER_US

    trig, prox = exposed_us(bytes_, k, dist_us)
    margin = abs(trig - prox) / max(min(trig, prox), 1e-9)
    if margin < 0.05:
        # Within noise of each other. Measurements show the model's
        # mispredictions are concentrated exactly here, so decline to
        # pin the site and let the default stand.
        return None

    if trig < prox:
        return {"dispatch": "DWQ_TRIGGER",
                "reason": f"K={k} bytes={int(bytes_)} D={dist_us:.1f}us "
                          f"trig={trig:.1f} prox={prox:.1f}"}
    return {"dispatch": "CPU_PROXY_ENQUEUE",
            "reason": f"K={k} bytes={int(bytes_)} D={dist_us:.1f}us "
                      f"trig={trig:.1f} prox={prox:.1f}"}


def main() -> int:
    try:
        feat_path = Path(os.environ["GICC_FEATURES_FILE"])
        hint_path = Path(os.environ["GICC_HINT_FILE"])
    except KeyError as e:
        print(f"gicc-decider: required env var {e} not set", file=sys.stderr)
        return 1

    if not feat_path.exists():
        print(f"gicc-decider: {feat_path} does not exist", file=sys.stderr)
        return 1

    features = json.loads(feat_path.read_text())
    hint: dict[str, Any] = {
        "version": 1,
        "schema_version": "gicc-hint-v1",
        "default_dispatch": "IPC_OR_DWQ",
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
    for sid, d in hint["sites"].items():
        print(f"gicc-decider: {sid} -> {d['dispatch']}"
              f"  ({d.get('reason', '')})", file=sys.stderr)
    print(f"gicc-decider: wrote {len(hint['sites'])} site hint(s) "
          f"to {hint_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
