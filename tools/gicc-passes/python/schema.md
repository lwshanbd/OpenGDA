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
    "hk_capable":         true,              // false → MUST route to CPU_PROXY_ENQUEUE
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

`hk_capable` is computed by `GICCHKAnalysis`. It is `true` when every
non-ctx argument of this call site is host-knowable per HK Analysis (a
constant, a kernel formal, or a pure composition over those). When
`false`, at least one argument depends on per-thread state
(`threadIdx`, device load, non-canonical PHI, ...) and the call cannot
be hoisted into a host trace function. Such sites MUST be routed to
`CPU_PROXY_ENQUEUE` by the decider; routing them to `IPC_PUSH`,
`DWQ_TRIGGER`, `IPC_OR_DWQ`, or `DWQ_BATCHED` is a compile-time error
(enforced by `GICCDispatchLowering`).


## hint.json

```json
{
  "version": 1,
  "schema_version": "gicc-hint-v1",
  "default_dispatch": "IPC_OR_DWQ",           // | IPC_PUSH | DWQ_TRIGGER | DWQ_BATCHED | CPU_PROXY_ENQUEUE
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

### Dispatch values

| Value | Semantics |
|---|---|
| `IPC_PUSH` | Force IPC `hipMemcpyAsync` on the IPC stream. Caller asserts the peer is mapped. |
| `DWQ_TRIGGER` | Force `gicc_runtime_dwq_enqueue` (libfabric Deferred Work Queue). |
| `DWQ_BATCHED` | Collapse N consecutive same-BB sites into a single `gicc_runtime_dwq_enqueue_batched`. |
| `IPC_OR_DWQ` | Hybrid runtime branch: IPC if peer base is mapped, DWQ otherwise. Default. |
| `CPU_PROXY_ENQUEUE` | Device-side enqueue to the CPU proxy ring; the host trace function emits **nothing** for the site. The actual RDMA work is performed by a CPU proxy thread that drains the ring (see Task 8 for the device-side body that is preserved when this dispatch is selected). Required for sites with `hk_capable=false`. The pass `report_fatal_error`s if `CPU_PROXY_ENQUEUE` is requested without the `GICC_PROXY_ENABLED` env var set. |

### HK / dispatch cross-check

`GICCDispatchLowering` enforces two hard errors:

1. A site with `hk_capable=false` (per the per-kernel JSON) MUST be routed to `CPU_PROXY_ENQUEUE`. Routing it to `IPC_PUSH`, `DWQ_TRIGGER`, `IPC_OR_DWQ`, or `DWQ_BATCHED` is a build error.
2. `CPU_PROXY_ENQUEUE` MUST be paired with `GICC_PROXY_ENABLED=1` at compile time. Otherwise the build fails (the runtime helpers it would require are not linked in).


## kernel JSON (per-kernel template)

Written by `GICCDeviceDiscovery` to `${GICC_META_DIR}/<mangled>.json`.
Read by every host-side pass and by `GICCDispatchLowering` (which
read-modify-writes `proxy_aware` after lowering). Schema (relevant
top-level fields):

```json
{
  "version": 1,
  "kernel_mangled": "_Z11halo_kernel...",
  "kernel_simple":  "halo_kernel",
  "proxy_aware":    false,            // see below
  "params": [ ... ],
  "ops":    [ { "site_id": "...", "hk_capable": true, ... } ]
}
```

`proxy_aware` is set to `true` by `GICCDispatchLowering` when at least
one of the kernel's call sites was lowered to `CPU_PROXY_ENQUEUE`. The
device-side lowering pass (Task 8) reads this bit to decide whether to
preserve the device-side `put_no_db` body so the proxy ring enqueue
stays in the kernel. Defaults to `false`; absent in JSON files written
before Task 2.


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
   →  inherit hint.default_dispatch (IPC_OR_DWQ)
```

The default is `IPC_OR_DWQ`: a hybrid runtime branch in the lowered
trace function. If the peer's IPC base pointer is non-null (i.e. the
peer is on the same node and the buffer is mapped via
`hipIpcOpenMemHandle`) the trace issues a `hipMemcpyAsync`; otherwise
it falls through to `gicc_runtime_dwq_enqueue`. This matches what
the no-hint default in `GICCDispatchLowering` already does, so 3-pass
builds inherit the same perf as single-pass builds.

v2 will swap `decide_one()` for an ML model trained on per-rank
profiling data. The pass-side schema will not change.
