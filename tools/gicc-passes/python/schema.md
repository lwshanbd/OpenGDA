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

A JSON array with one record per device communication op. If the kernel has
multiple host launch sites, their contexts are aggregated because the current
lowering hint names the shared device op and therefore selects one lowering
for all of them.

```json
[
  {
    "schema_version": 6,
    "site_id":            "<TU>:<line>:<kernel>::<idx>",
    "kernel":             "halo_kernel",
    "op_kind":            "put_no_db",       // | get_no_db | flush | quiet
    "hk_capable":         true,              // false → MUST route to CPU_PROXY_ENQUEUE
    "size_kind":          "const",           // | param | binop | cast | derived
    "size_bytes":         4096,              // literal or host-LTO-resolved;
                                             //   null when contexts differ/dynamic
    "size_log2":          12,                // null without one common,
                                             //   positive LTO-resolved size
    "transfer_interval": {                   // source-free descriptor expression
      "semantics": "source_buffer_half_open_byte_interval",
      "source_buffer": {"kind": "param", "param": 12},
      "byte_offset":   {"kind": "param", "param": 13},
      "byte_size":     {"kind": "param", "param": 17},
      "symbolically_exact": true,            // no opaque expression leaf
      "affine": true,                        // conservative compiler classification
      "host_knowable": true                  // exact and HK-reconstructible
    },
    "peer_kind":          "param",           // | const | binop | cast | derived
    "peer_locality":      null,              // | same_node | cross_node
    "in_loop":            true,              // real value from device-side LoopInfo
    "loop": {                                // present iff in_loop=true
      "iv_start":         0,
      "iv_step":          1,
      "bound_known":      true,
      "bound_param_idx":  3,
      "degraded":         false              // omitted unless true
    },
    "guard_density":      0.5,               // 1.0 if Always else 0.5
    "fan_out":            2,                 // distinct param-keyed peers
    "static_launch_sites": 1,                // host callsites sharing this device op
    "launch_contexts": [{                    // distinct host-LTO contexts
      "static_callsite_count": 1,
      "size_bytes":       4096,
      "trip_count":       64,
      "grid_blocks":      8,
      "threads_per_block": 1,
      "phase_launch_supported": true,
      "phase_launch_stream": "explicit",
      "phase_launch_materialization": "device_stub",
      "phase_launch_reason": "one original kernel launch with reusable parameters and unchanged stream",
      "kernel_argument_slots_exact": true,
      "kernel_argument_slot_count": 18,
      "kernel_argument_slot_reason": "every HIP parameter slot has distinct launch-local, metadata-typed storage",
      "launch_grid":      {"x": 8, "y": 1, "z": 1},
      "launch_block":     {"x": 1, "y": 1, "z": 1}
    }],
    "launch_grid": {                         // host LTO callsite constants;
      "x":                 8,                 //   individual values null when
      "y":                 1,                 //   not statically known
      "z":                 1
    },
    "launch_block": {"x": 1, "y": 1, "z": 1},
    "grid_blocks":        8,                 // x*y*z; null if any dimension
                                             //   is dynamic or product overflows
    "threads_per_block":  1,                 // same rule for launch_block
    "phase_launch_supported": true,           // host LTO proved that the
    "phase_launch_stream": "explicit",        //   wrapper launch is cloneable
    "phase_launch_materialization": "device_stub", // exact rewrite owner
    "phase_launch_reason": "one original ...",//   on one unchanged stream
    "kernel_argument_slots_exact": true,       // every void** slot is distinct,
    "kernel_argument_slot_count": 18,          //   launch-local and metadata-typed
    "kernel_argument_slot_reason": "every HIP ...",
    "compute_before_flops": 17,              // arith ops in BBs dominating the call;
                                             //   null if not measured
    "flops_to_first_use": 205,               // arith ops between the call and the
                                             //   kernel's completion point: the static
                                             //   issue-to-first-use distance. null if
                                             //   not measured.
    "distance_exact":     false,             // false => flops_to_first_use skipped a
                                             //   loop with a runtime bound, so it is a
                                             //   LOWER BOUND on hideable work
    "trip_count":         64,                // ops per phase, from ScalarEvolution
                                             //   or a constant host launch binding
    "iter_estimate":      64,                // set when host LTO resolved the bound

    "descriptor_reusable": true,             // every descriptor field repeats
                                             //   across the enclosing loop, so the
                                             //   host can stage ONCE and trigger
                                             //   trip_count times. false outside a
                                             //   loop: nothing to amortise.
    "buffer_reusable":     true,             // the BUFFERS do not vary even if the
                                             //   offsets do -- the ordinary stencil
                                             //   shape, where registration hoists
                                             //   but the descriptor does not.
    "batch_size":          8,                // transfers released by the same
                                             //   completion point, a looped one
                                             //   counted trip_count times. null when
                                             //   not measured.
    "producer_frontier": {                   // optional source-free write facts;
      "analyzed": true,                      //   NOT a legality assertion
      "write_footprint_known": true,
      "completion_site_id": "<site_id>",
      "ordinary_store_params": [1],          // kernel formal indices only
      "atomic_write_params": [3],
      "buffer_identity_guardable": true,     // compiler fixed both argument slots
      "producer_pointer_param": 1,
      "source_buffer_index_param": 12,
      "buffer_identity_guard_reason": "one producer pointer and one shared i32 source-buffer formal can be guarded at launch",
      "source_identity_guardable": false,
      "source_pointer_candidates": [],       // readonly noalias candidates
      "source_identity_buffer_index_param": null,
      "source_identity_guard_reason": "...",
      "producer_domains_known": true,         // every ordinary store below is exact
      "producer_store_domains": [{
        "pointer_param": 1,                   // destination pointer formal
        "byte_offset": {                      // source-free device expression
          "kind": "binop", "op": "mul", "type": "i64",
          "children": [
            {"kind": "param", "param": 6, "type": "i64"},
            {"kind": "const", "value": 4, "type": "i64"}
          ]
        },
        "byte_size": 4,
        "address_exact": true,
        "predicates": [{                      // conjunction; all entries required
          "condition": {
            "kind": "compare", "op": "ult", "type": "i1",
            "children": [
              {"kind": "param", "param": 6, "type": "i64"},
              {"kind": "const", "value": 1024, "type": "i64"}
            ]
          },
          "required_value": true
        }],
        "predicates_exact": true,
        "domain_exact": true,                 // local IR recovery only
        "partition_region_exact": true,       // one single-entry CFG region
        "partition_predicate_index": 0,       // entry in predicates above
        "partition_region_reason": "one exact controlling edge gates a single-entry producer region",
        "reason": "exact formal-rooted byte interval and controlling predicates recovered"
      }],
      "atomic_domains_known": true,
      "producer_atomic_domains": [{
        "pointer_param": 3,
        "operation": "atomic_add",
        "result_unused": true,
        "predicates": [{
          "condition": {"kind": "param", "param": 7, "type": "i1"},
          "required_value": true
        }],
        "predicates_exact": true,
        "domain_exact": true,
        "reason": "exact formal-rooted atomic and controlling predicates recovered"
      }],
      "phase_sensitive_domains_known": true,
      "producer_phase_sensitive_domains": [{
        "operation": "_Z11__shfl_downfji",
        "guard_predicates": [{
          "condition": {"kind": "param", "param": 7, "type": "i1"},
          "required_value": true
        }],
        "guard_predicates_exact": true,
        "domain_exact": true,
        "reason": "exact dominating direct i1 kernel-formal guards recovered"
      }],
      "overlap_partition": {
        "analyzed": true,
        "exact": true,
        "mode": "checked_store_interval_overlap",
        "formal_binding": "same_kernel_formal_indices",
        "producer_pointer_param": 1,
        "source_buffer_index_param": 12,
        "producer_store": {"...": "same exact store domain"},
        "transfer_intervals": [
          {"site_id": "<put0>", "source_buffer": {"kind": "param", "param": 12},
           "byte_offset": {"kind": "param", "param": 13},
           "byte_size": {"kind": "param", "param": 17}},
          {"site_id": "<put1>", "source_buffer": {"kind": "param", "param": 12},
           "byte_offset": {"kind": "param", "param": 14},
           "byte_size": {"kind": "param", "param": 17}}
        ],
        "boundary_predicate": "store_interval_overlaps_any_transfer_interval",
        "remainder_predicate": "logical_complement_of_boundary",
        "checked_interval_ends_required": true,
        "buffer_identity_guard_required": true,
        "store_instance_partition_disjoint": true,
        "store_instance_partition_complete": true,
        "proof_scope": "ordinary_producer_store_instances",
        "full_compute_region_partition_proved": true,
        "side_effect_safety_exact": true,
        "side_effect_safety_mode": "all_nonduplicable_operations_disabled_by_formal_guard",
        "side_effect_free_guard": {
          "kind": "param_eq", "param": 7, "value": false
        },
        "side_effects_excluded_on_optimized_path": true,
        "side_effect_partition_proved": false
      },
      "ordinary_store_sites": 1,
      "atomic_write_sites": 1,
      "phase_sensitive_sites": 2,
      "unknown_write_sites": 0,
      "reason": "formal-rooted writes recovered; ...",
      "remaining_proofs": [
        "buffer_identity_guarded_fallback_materialization",
        "checked_interval_guard_materialization",
        "device_phase_partition_materialization",
        "side_effect_guarded_fallback_materialization",
        "launch_phase_materialization"
      ]
    },
    "legal_paths":        ["proxy", "trigger", "ipc"]
                                             // LEGALITY, not preference. See below.
  }
]
```

`legal_paths` is what the decider must not step outside of. The proxy path
is always present: the device pushes a command and the worker reads the
descriptor at submit time, so nothing has to be knowable in advance. The
trigger and IPC paths need the host to reconstruct the descriptor before
the kernel launches, which is what `hk_capable` proves -- and a loop the
pass recognised but could not model (`loop.degraded`) disqualifies them
too, because trace synthesis would drop the transfer rather than emit
wrong code.

`descriptor_reusable` and `buffer_reusable` are DERIVED from the `args`
expressions on read, not stored in the per-kernel JSON, so there is one
definition of them rather than two that can drift. `batch_size` is stored,
because grouping transfers by completion point needs the CFG: it is the
first completion point REACHABLE from the transfer, not the next one in
any linear block order -- a depth-first numbering ranks a loop's exit
block ahead of its body, which would group the loop's own transfers with
whatever follows the flush that actually releases them.

Schema v5 → v6 changes:
- One record is emitted per device op rather than per `(launch site × op)`.
  `launch_contexts` retains the distinct host contexts and
  `static_callsite_count`/`static_launch_sites` record their multiplicity.
  This matches the current hint granularity: without kernel cloning, one
  device op cannot legally receive different lowerings at two launches.
- `size_bytes`, per-context `size_bytes`, and per-context `trip_count` bind
  constant host wrapper operands back to device kernel formals. Top-level
  values are present only when every launch context agrees; disagreement or
  dynamic operands remain `null`.

Optional `producer_frontier` facts are a forward-compatible v6 extension.
They summarize only writes between a compiler-discovered transfer group and
its mandatory flush. `write_footprint_known` means those writes are rooted in
named kernel pointer formals; it does **not** authorize fission. The listed
remaining proofs must all be discharged by later host/device LTO before a
candidate may enter the model-visible legal set.

`buffer_identity_guardable` does not claim that the pointer and registered
buffer are equal. It says the compiler found exactly one producer pointer
formal and one shared i32 `src_buf` formal, so it can call the fixed runtime
identity helper using those two constant slot indices. A future materializer
must keep one untouched fused launch on the guard's false edge. Until that
branch exists, `buffer_identity_guarded_fallback_materialization` remains in
`remaining_proofs` and no fission action is legal.

`source_identity_guardable` is only an identity-candidate relation, not a
disjointness proof or an early-trigger legality result. Device discovery
retains `noalias` and compiler-proved `readonly` bits on pointer formals, roots
every intervening write in a formal, and lists read-only noalias pointer slots
that host LTO could compare with the shared registered source-buffer slot.
LLVM `noalias` constrains memory locations accessed during the invocation; it
does not prove that the entire registered source allocation misses every
write-root allocation. A later materializer must separately prove or check
that whole-allocation relation. If an intervening write is atomic or another
non-duplicable side effect is present, it must also prove that moving
communication across the effect preserves its observable ordering; memory
disjointness alone is insufficient. The materializer must retain the untouched
original schedule on every false or unknown edge. Until then, the fact neither
moves a trigger nor adds a legal model action.

`transfer_interval` is the compiler-recovered half-open source byte interval
`[byte_offset, byte_offset + byte_size)` within `source_buffer`. Its nested
expressions contain only formal indices, literals, operations, and modeled
compiler leaves; they contain no source text. `symbolically_exact` means the
expression has no opaque leaf, while `affine` is a stricter, conservative
classification for the first producer-frontier candidate. Neither flag proves
buffer identity, binds runtime formal values, proves a producer domain, or
adds a fission action to `legal_paths`.

`producer_store_domains` is a device-IR slice, not source reconstruction. A
store is represented by its pointer-formal index, byte offset/size, and the
conjunction of controlling predicates. Expressions may contain kernel
formals, typed constants, typed arithmetic/casts/comparisons, selects, and
target builtins such as `block_id_y`, `block_size_y`, and `thread_id_y`. Any
unmodeled leaf,
loop-carried store, or control terminator makes the corresponding exact flag
false. `producer_domains_known=true` means every ordinary store was recovered
in this local form; it still does not prove that a registered transfer buffer
is the pointer, that its interval equals a producer subset, or that the
boundary/remainder partition is complete.

`partition_region_exact` is a separate CFG proof. The named controlling edge
enters a single-entry region with a post-dominating merge and contains the
producer store. It identifies where device LTO can materialize a whole-region
phase predicate; it does not claim that the rewrite already exists.

`producer_atomic_domains` records exact formal-rooted atomics and their full
control domains. `producer_phase_sensitive_domains` records convergent or
`noduplicate` calls that are not compiler-recognized, freely replicable GPU
identity queries. The latter retains exact dominating direct-`i1`-formal
guards even when the call is inside a loop. The first guarded candidate
requires one formal condition
shared by every atomic and every phase-sensitive call. On unchanged Jacobi,
both `atomicAdd` and `__shfl_down` require formal 7 to be true, so only formal
7 equal to false may enter the optimized path; the true edge must retain the
original fused launch. Opposite or unmatched guards fail closed.

`kernel_argument_slots_exact` closes the host/device formal-namespace gap
without reverse-engineering source expressions or lambda captures. It is true
only when every metadata formal has a distinct launch-owner-local value cell
at the same `hipLaunchKernel` parameter-array index and its storage type
matches device metadata (including the audited i1-to-i8 host ABI case). The
phase launches reuse this exact array, so a device expression referring to
formal `i` and a transfer expression referring to formal `i` consume the same
runtime value even when that value is dynamic.

`overlap_partition` is emitted only for the first narrow case: one exact
ordinary producer store and one all-PUT completion group whose source-buffer
and producer facts agree. Rather than requiring a source-derived algebraic
identity such as `offset = row * stride`, device LTO can classify a store
instance with checked half-open byte-interval overlap against the exact
transfer intervals. The boundary predicate and its logical complement are
therefore disjoint and complete for ordinary producer-store instances by
construction. A separate single-entry-region proof now establishes where that
predicate can gate the full Jacobi stencil region. Atomic and subgroup
behavior is not reassociated: the first candidate is restricted by a shared
compiler-proved formal guard that makes all atomics and non-replicable
convergent calls unreachable. Checked arithmetic, device-phase rewriting,
both runtime fallback branches, and launch cloning remain explicit
materialization obligations; until those exist, no fission candidate enters
`legal_paths`.

`phase_launch_supported` is also a compiler proof, not a model assertion. An
application-side call or exception-aware `invoke` may reach the annotated
wrapper because the materialization point is inside that wrapper and its
caller successors remain untouched. The compiler still requires exactly one
matching `hipLaunchKernel`, an unused return value, and a launch-owner-local parameter
array. At the early-simplification pass point the launch may still live in an
exclusive HIP device stub reached through the kernel's constant global; the
compiler then additionally proves the unique push/pop configuration chain and
reports `phase_launch_materialization: "device_stub"`. After inlining, the
equivalent location is `"wrapper"`. The materializer must recompute the same
shape on final host IR before cloning anything. The proof now also rejects an
undersized, missing, aliased, non-local, or metadata-type-mismatched kernel
parameter slot.

Schema v4 → v5 changes:
- `launch_grid`, `launch_block`, `grid_blocks`, and `threads_per_block`
  added. They are decoded conservatively from constant `dim3` operands on the
  host-side `gicc::launch` call in LTO IR. Dynamic values remain `null`.
  This gives a compiler-level decider the launch concurrency context without
  exposing or modifying application source.

Schema v3 → v4 changes:
- NOTE: no producer ever emitted `schema_version: 3`. The emitter was left
  at 2 while this document already described v3, so a file claiming 2 may
  or may not carry the v3 fields; check for their presence rather than
  trusting the number. v4 onwards the two agree.
- `descriptor_reusable`, `buffer_reusable`, `batch_size`, `legal_paths`
  added. These are the reuse and batching properties a runtime cannot
  establish at the moment of the call: it sees one transfer, not the loop
  it sits in nor the group it belongs to.

Schema v2 → v3 changes:
- `flops_to_first_use`, `distance_exact`, `trip_count` added. Together with
  `size_log2` these are what the shipped decider weighs: both dispatch
  paths pay a fixed per-phase cost plus a per-op issue cost, and only the
  work between the issue and the first use can hide the wire time.
- `trip_count` is the number of times the CALL runs, i.e. the loop's
  backedge-taken count, not LLVM's "trip count" (which counts header
  executions and is one larger for a loop seen before rotation).

Schema v1 → v2 changes:
- `in_loop` now reflects the real LoopInfo result (was hardcoded false).
- `compute_before_flops` is the DominatorTree-derived count; was hardcoded 0.
  `null` means the device pass ran without DT available.
- New optional `loop` sub-object present iff `in_loop=true`, exposing the
  canonical loop descriptor (`iv_start`, `iv_step`, `bound_param_idx`).
- `peer_locality` still null — requires a runtime topology side-band that
  is not yet wired up (filled by a future v2.x decider hook).

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

For each transfer assigned to a compiler-discovered completion group, newer
kernel templates may also carry:

```json
{
  "batch_size": 2,
  "completion_site_id": "<flush site_id>",
  "group_early_trigger_legal": false,
  "group_early_trigger_reason":
    "intervening instruction may write a registered source buffer"
}
```

`group_early_trigger_legal` is a compiler proof, not a model feature that can
be overridden. A registered source buffer is named by an integer handle at the
device API, so LLVM alias analysis cannot generally relate it to pointer
stores. The proof therefore rejects `TRIGGER_GROUP_EARLY` if moving the shared
flush would cross *any* instruction that may write memory. The final device
lowering repeats dominance, post-dominance, group-completeness, operand
dominance, and intervening-write checks before moving the MMIO trigger.

`PRODUCER_FRONTIER_TWO_PHASE` is a separate compiler-owned transform for a
completion group whose `producer_frontier.overlap_partition.exact` proof is
complete. The group bridge exposes it only when every member is an
unconditional, non-loop, host-knowable PUT; the host launch shape is reusable;
the checked producer/remainder partition excludes non-duplicable effects; and
the platform profile explicitly enables `producer_frontier_fission` after a
runtime oracle gate. Selecting its opaque candidate ID emits the transform on
every group member with `DWQ_TRIGGER`. Both final LTO halves rebuild the facts,
successful device LTO records `producer_fission_device_materialized: true` in
the freshly regenerated kernel template, and explicit-transform host LTO
requires that attestation before cloning a launch. Host runtime
identity/interval checks then retain the original fused launch on every false
edge. A route-only hint cannot activate this schedule.


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


## LLM decision bridge

The LLM path uses the same pass-side `features.json -> hint.json` contract.
It does not give source text to the model and it does not put a provider call
inside the linker.  The two LTO invocations are separated by a replayable,
content-addressed decision step:

```text
LTO feature extraction
        |
        v
features.json + measured platform profile
        |
        v
gicc-llm-dossier-v1 --external model--> gicc-llm-decision-v1
        |
        v  schema, dossier hash, site completeness, legal_actions
gicc_llm_bridge.py accept
        |
        v
gicc-hint-v1 --second LTO--> validated lowering
```

Create the provider-neutral artifacts with:

```bash
python3 tools/gicc-passes/python/gicc_llm_bridge.py emit \
  --features build/features.json \
  --platform tools/gicc-passes/python/profiles/tioga-mi250x-slingshot11.json \
  --dossier build/llm-dossier.json \
  --prompt build/llm-prompt.txt
```

The response must bind itself to the dossier hash and cover every decision
site exactly once:

```json
{
  "schema_version": "gicc-llm-decision-v1",
  "dossier_id": "sha256:<digest>",
  "decisions": {
    "<site_id>": {
      "action": "proxy",
      "confidence": 0.93,
      "rationale": "short compiler/platform-fact explanation"
    }
  }
}
```

`action` must be one of that site's `legal_actions`.  `default` is an explicit
abstention and leaves the pass's `IPC_OR_DWQ` baseline in place, so it is
offered only when the compiler says both IPC and trigger are materializable.
Forced `ipc` is withheld unless topology proves `peer_locality=same_node`.
A proxy-only site cannot abstain because the hybrid host path is not legal
there.  A modeled loop is represented by one batched host placeholder; the
current pass exposes only `proxy` and `trigger` for that shape, not `ipc` or
`default`.

Accept and translate with:

```bash
python3 tools/gicc-passes/python/gicc_llm_bridge.py accept \
  --dossier build/llm-dossier.json \
  --response build/llm-response.json \
  --hint build/llm-hint.json
```

By default any malformed, stale, incomplete, or illegal response produces a
deterministic, materializable fallback hint: ordinary host-capable sites
inherit `IPC_OR_DWQ`, batched-loop sites are pinned to `DWQ_TRIGGER`, and
compiler-proven proxy-only sites are pinned to `CPU_PROXY_ENQUEUE`.  No model
choice is partially applied.  `--strict` instead rejects the sample without
writing a hint.

### Relational communication-group bridge

`gicc_comm_group_plan_bridge.py` augments the flat per-site dossier with the
compiler's source-free kernel template. It groups multiple communication sites
by `completion_site_id`, and represents an otherwise ungrouped transfer as a
singleton opportunity whenever it has at least two materializable routes. The
planner receives formal argument-expression relations, transfer intervals,
producer-frontier facts, launch contexts, compute distance, and dependence
legality. This is intentionally richer than the flat scalar input used by the
original cost model.

The bridge enumerates complete compiler-materializable route combinations and
adds `group_trigger_early` only when both the platform profile enables it and
the compiler proof above succeeds. A rejected transformation remains visible
as a masked candidate and reason, but has no candidate ID and therefore cannot
be selected. The private content-addressed graph retains site-to-materializer
bindings for `accept`; the prompt contains a separately constructed model view
that expresses transfers by ordinal, removes kernel/site/path identities and
materializer strings, and omits profile provenance paths.

```bash
python3 tools/gicc-passes/python/gicc_comm_group_plan_bridge.py emit \
  --dossier build/dossier.json --meta-dir build/meta \
  --graph build/group-graph.json --prompt build/group-prompt.txt
```

The response contains only one existing candidate ID per opportunity. `accept`
content-validates the graph and candidate, then emits `gicc-hint-v1`; source,
model-authored code/IR, dispatch strings, site IDs, or new legality assertions
are rejected.

### Relational collective algorithm and size-policy bridge

`GICCCollectivePlanningPass` recognizes a trusted, fixed-ABI compiler catalog
at the early LTO extension point. A semantic anchor declares the collective
family, contract, element width, argument roles, and the only message-size
thresholds the compiler may materialize. Catalog entries declare algorithms
and structural facts such as communication graph, topology, step complexity,
cross-node pattern, synchronization, pipeline depth, and resource model.

With `GICC_COLLECTIVE_OUT=<path>`, the pass emits
`gicc-collective-inventory-v1`. It contains no application source, source
location, or function name. Every catalog target is content-addressed and is
included only when LLVM proves exact function-type, family, semantic-contract,
and void-call materializer equality. Call facts include constant/dynamic
message shape, element width, ranks-per-node shape, loop depth, and structural
arithmetic counts.

The bridge joins that inventory with a platform topology/resource profile:

```bash
python3 tools/gicc-passes/python/gicc_collective_plan_bridge.py emit \
  --inventory build/inventory.json \
  --platform tools/gicc-passes/python/profiles/tioga-mi250x-cxi-collective-capacity.json \
  --graph build/collective-graph.json \
  --prompt build/collective-prompt.txt
```

For each compiler-owned message interval, the model sees semantic algorithm
descriptors and opaque `option_id` values. The model-facing v2 view normalizes
each compiler candidate into one source-free `candidate_class` entity instead
of repeating its descriptor in every interval. Explicit relations preserve
message-size order and identify candidates that share a communication graph,
topology strategy, cross-node pattern, resource model, or synchronization
scheme. Each interval maps its opaque options to those entities, so the model
can reason jointly about structural variants while still returning only
compiler-generated option IDs. Its complete response consists of one existing
option ID per interval plus bounded confidence/rationale fields.

`emit --prompt-view relational` is the primary rich view. Two preregistered
input ablations preserve the identical compiler graph, action space, response
schema, validator, and materializer: `descriptors` removes explicit relation
edges, while `opaque` removes candidate descriptors and option-to-candidate
mappings. These are input-information ablations, not different compiler
permissions; none can name or create a materializer target.

A platform profile may carry a strictly validated
`gicc-collective-primitive-calibration-v1` block. It must identify immutable
artifact hashes, declare that no collective action labels are visible, bound
its applicability, and contain only positive finite primitive measurements.
Such measurements are compiler cost-model priors (for example link bandwidth
or proxy issue cost), never results from the collective controls being scored.
When a profile contains this block, `emit` requires one
`--calibration-artifact <path>` per declared hash and verifies exact content
before constructing the graph. Artifact paths are never serialized into the
model view.
It cannot output a target symbol, algorithm string, threshold, source, code,
IR, or legality. `accept` converts valid option IDs to a narrow
`gicc-collective-hint-v1`; malformed or invented content falls back atomically
to the semantic anchor unless `--strict` is requested.

During the second LTO build, set
`GICC_COLLECTIVE_HINT_IN=<collective-hint.json>` and
`GICC_COLLECTIVE_ONLY=1`. The latter restricts the automatically attached
pipeline to collective planning: a collective-only experiment has no ordinary
`GICC_HINT_IN` or synthesized per-transfer host trace, so running transfer
device lowering would otherwise erase proxy operations with no replacement.
The pass independently
recomputes opportunity, target, and composite-plan IDs; checks catalog
membership, exact ABI, family, contract, and the complete ordered threshold
list; and only then retargets the call or creates the size-policy CFG. Every
materialized call receives `gicc.collective.candidate_id` and
`gicc.collective.target_id` IR metadata for pre-runtime audit. Any mismatch
preserves the original semantic anchor call.

The fixed-source capacity experiment and its control generator live under
`tools/gicc-passes/experiments/collective/`. Uniform catalog arms establish the
measurable headroom before any model is evaluated; their source and catalog
hashes, option IDs, hints, binaries, and materialized IR are auditable without
changing the benchmark source.
