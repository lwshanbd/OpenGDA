# GICC LTO pass interchange schemas

Two JSON files cross the boundary between the LLVM passes and the
external decider:

```
                features.json
   passes  ──────────────►  decider  ──────────►  passes
                                       hint.json
```

`gicc-features.json` is produced by `GICCFeatureExtraction` (host-side,
once per build). `gicc-hint.json` is consumed by `GICCDispatchLowering`
to pick the dispatch for each `put_no_db.placeholder` call.

The schemas are forward-compatible: producers MAY emit additional
fields and consumers MUST tolerate fields they don't recognize.


## features.json

A JSON array of per-(launch site × op) records.

```json
[
  {
    "schema_version": 1,
    "site_id":            "<TU>:<line>:<kernel>::<idx>",
    "kernel":             "halo_kernel",
    "op_kind":            "put_no_db",       // | get_no_db | flush | quiet
    "size_kind":          "const",           // | param | binop | cast | derived
    "size_log2":          12,                // null when size is non-const
    "peer_kind":          "param",           // | const | binop | cast | derived
    "peer_locality":      null,              // | same_node | cross_node
    "in_loop":            false,
    "guard_density":      0.5,               // 1.0 if Always else 0.5
    "fan_out":            2,                 // distinct param-keyed peers
    "compute_before_flops": 0,
    "iter_estimate":      null
  }
]
```

`site_id` matches the `gicc.site_id` metadata operand on every
placeholder call and is the join key between features and hint.

`peer_locality` is reserved for v2 — the rule-based decider in v1
infers it lazily from `peer_kind` + runtime context.


## hint.json

```json
{
  "version": 1,
  "schema_version": "gicc-hint-v1",
  "default_dispatch": "DWQ_TRIGGER",         // | IPC_PUSH | DWQ_BATCHED
  "sites": {
    "<site_id>": {
      "dispatch":     "IPC_PUSH",            // required
      "stream_hint":  "ipc",                 // optional, v2
      "batch_group":  "halo_left"            // optional, v2 (DWQ_BATCHED)
    }
  },
  "global": {                                 // optional, v2
    "max_batch_size":   16,
    "ipc_stream_count": 1
  }
}
```

`default_dispatch` applies to any site_id not present in `sites`. Sites
referencing a kind the lowering pass doesn't yet know about (e.g.
`DWQ_BATCHED` in v1) silently fall back to the default — no build
failure.


## Decider invocation

```
GICC_FEATURES_FILE=<features.json> \
GICC_HINT_FILE=<hint.json> \
    python3 tools/gicc-passes/python/gicc_decider.py
```

The v1 decider is a deterministic rule:

```
op_kind in {put_no_db, get_no_db} and peer_kind == "const"
                                  and peer_locality == "same_node"
   →  IPC_PUSH
otherwise
   →  inherit hint.default_dispatch (DWQ_TRIGGER)
```

v2 will swap `decide_one()` for an ML model trained on per-rank
profiling data. The pass-side schema will not change.
