# rule_fitter — Offline rule-list fitter

Offline CN2-style greedy inducer for GICC-Pilot `policy_<platform>.json`
files. Input is M3 calibration data (R021 / R022); output is a file that
`libgicc_policy::load_policy` accepts directly.

## Usage

```
fit.py --calibration r022_tioga.json --out policies/policy_tioga.json
```

Optional flags:
- `--slack 0.05` — how close a config must be to oracle median to count as
  "equivalent" (for the least-NIC-budget tiebreaker).
- `--max-rules 20` — upper bound on induced rules (excluding the catch-all).
- `--min-coverage 0.95` — warn-and-nonzero-exit threshold.

## What it does

1. For every calibration cell, pick the **target label**: among configs
   within `--slack` of the oracle (min-median T2S), choose the one with the
   smallest `(pool_size * R(P), slot_depth, prefer-proxy)` tuple. This
   matches the bias described in `refine-logs/FINAL_PROPOSAL.md`
   §Calibration — scale-survival over marginal speed.
2. Greedy depth-1 induction: repeatedly pick the feature-equality guard
   that covers the most uncovered cells with a single, NIC-cap-feasible
   decision. Emit, remove covered cells, loop.
3. Sort induced rules specific-first; append a conservative catch-all
   (empty guard).

## Testing

`test_fit.py` synthesizes a 3-cell calibration set with known-best
configs and asserts the fitter produces rules that cover them. Run:

```
python3 test_fit.py
```

## Schema of expected calibration input

```json
{
  "platform": "tioga",
  "platform_desc": {
    "fabric": "ofi_cxi",
    "gpu_family": "mi250x",
    "nic_caps": {"counter_max": 2047, "dwq_max": 256, "ctrs_per_op": 2, "qp_max_per_peer": 0}
  },
  "cells": [
    {
      "workload": "jacobi",
      "scale": 32,
      "features": {"peer_class": "k_from_topology_hint",
                   "size_class": "large_const",
                   "freq_class": "hot_loop"},
      "configs": [
        {"path": "ofi_triggered", "slot_depth": 4, "pool_size": 16,
         "t2s_ms": 276.2, "seeds": 5, "feasible": true},
        ...
      ]
    }, ...
  ]
}
```
