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
      "ordinary_store_sites": 1,
      "atomic_write_sites": 1,
      "unknown_write_sites": 0,
      "reason": "formal-rooted writes recovered; ...",
      "remaining_proofs": [
        "buffer_identity_guarded_fallback_materialization",
        "exact_transfer_intervals",
        "exact_producer_domains",
        "complete_disjoint_partition",
        "side_effect_partition",
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

`transfer_interval` is the compiler-recovered half-open source byte interval
`[byte_offset, byte_offset + byte_size)` within `source_buffer`. Its nested
expressions contain only formal indices, literals, operations, and modeled
compiler leaves; they contain no source text. `symbolically_exact` means the
expression has no opaque leaf, while `affine` is a stricter, conservative
classification for the first producer-frontier candidate. Neither flag proves
buffer identity, binds runtime formal values, proves a producer domain, or
adds a fission action to `legal_paths`.

`phase_launch_supported` is also a compiler proof, not a model assertion. It
is true only when every aggregated host call is a non-throwing direct call to
an annotated wrapper and the compiler finds exactly one matching
`hipLaunchKernel`, an unused return value, and a launch-owner-local parameter
array. At the early-simplification pass point the launch may still live in an
exclusive HIP device stub reached through the kernel's constant global; the
compiler then additionally proves the unique push/pop configuration chain and
reports `phase_launch_materialization: "device_stub"`. After inlining, the
equivalent location is `"wrapper"`. The materializer must recompute the same
shape on final host IR before cloning anything.

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
by `completion_site_id` and supplies the planner with operation order, shared
completion, formal argument-expression relations, launch contexts, compute
distance, and dependence legality. This is intentionally richer than the
flat scalar input used by the original cost model.

The bridge enumerates complete compiler-materializable route combinations and
adds `group_trigger_early` only when both the platform profile enables it and
the compiler proof above succeeds. A rejected transformation remains visible
as a masked candidate and reason, but has no candidate ID and therefore cannot
be selected.

```bash
python3 tools/gicc-passes/python/gicc_comm_group_plan_bridge.py emit \
  --dossier build/dossier.json --meta-dir build/meta \
  --graph build/group-graph.json --prompt build/group-prompt.txt
```

The response contains only one existing candidate ID per group. `accept`
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
