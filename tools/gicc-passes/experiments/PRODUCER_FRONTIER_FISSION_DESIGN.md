# Compiler-owned producer-frontier fission

Status: a dormant end-to-end compiler oracle is implemented; runtime
correctness and performance remain unmeasured, so this is not yet a
performance claim.

## Research question

Can an LTO planner improve communication overlap by selecting among
compiler-generated schedules, while its entire output remains an opaque
candidate ID and the application source remains unchanged?

The motivating real case is `examples/ofi/jacobi.cpp`. Device discovery sees
two halo PUT descriptors, a stencil store region, one shared flush, and a
quiet. The descriptors are issued before the stencil, but the flush occurs
after it because the stencil produces the halo source rows. Moving the flush
to the current post-PUT frontier sends stale halo data. The existing compiler
therefore correctly masks `TRIGGER_GROUP_EARLY`.

The missing schedule is:

1. compute the two halo-producing boundary rows;
2. stage and trigger the two halo PUTs;
3. compute the disjoint interior rows while communication progresses;
4. quiet before the iteration boundary.

This changes the compiler schedule, not application source or communication
semantics.

## Model boundary

The model may receive only a content-addressed, source-free compiler graph:

- communication-group membership and operation order;
- registered-buffer/pointer formal relations derived by host LTO;
- symbolic source intervals for every transfer;
- producer write regions and iteration-domain partitions;
- atomic domains and non-replicable convergent-operation guards;
- dependence, dominance, post-dominance, and stream-order proofs;
- launch geometry, topology, resource pressure, and calibrated costs;
- existing compiler candidate IDs and their predicted effects.

The response contains exactly one existing candidate ID. It cannot contain
source, IR, a function/site name, a predicate, a legality assertion, code, or
a new transformation. The bridge rejects the whole response on any mismatch,
and final host and device LTO independently re-prove the selected candidate.

## ABI-preserving execution mechanism

Creating new device kernels during device LTO would not automatically create
the corresponding HIP host stubs. Instead, preserve the original kernel
symbol and ABI and launch it twice in two compiler phases.

Place a schedule-phase word at fixed byte offset 16 in the
compiler/runtime-owned prefix of `DeviceCtx`. A compile-time layout assertion
keeps that offset independent of the optional proxy fields. A small pre-authored
GICC setter kernel writes the word. Host LTO rewrites the body of
the existing annotated `gicc::launch<Kernel>` instantiation, or its exact call
site, into the following same-stream sequence:

```text
set_schedule_phase(ctx, boundary_producer)
launch original Kernel(ctx, original args...)
set_schedule_phase(ctx, interior_and_communication)
launch original Kernel(ctx, original args...)
set_schedule_phase(ctx, original)
```

The setter is a device command on the same stream, not a racing host store.
HIP stream order therefore makes each launch observe its own phase. The
original kernel stub, symbol, formal arguments, and application call remain
unchanged.

The runtime-owned mechanism is now present but dormant: `DeviceCtx` contains a
zero-defaulted phase word in its fixed compiler/runtime prefix, every
`Runtime::prepare*` path restores
`original`, and the C ABI helper
`gicc_runtime_set_schedule_phase_from_kernel_args` launches a one-thread
setter on the supplied stream. Host/device IR audit confirms that the helper
dereferences kernel-argument slot zero, forwards the stream to HIP, and that
the setter performs one 32-bit store to the appended context field. No pass
calls the helper and no kernel reads the phase yet, so this milestone cannot
change application behavior by itself.

Static whole-program pointer/registration recovery is not required for the
first candidate. A second compiler-only C ABI helper now compares a pointer
kernel-argument slot with the base of the local registered-buffer index held
in another slot. The eventual host materializer can branch on that check:
the true edge executes the two phase launches, while the false edge executes
the untouched original launch once. Parameter indices and types remain
compiler-proved constants; neither the model nor the runtime invents them.
The helper is also dormant until the materializer and the remaining static
domain proofs exist.

A second dormant guard helper validates each compiler-recovered half-open
transfer interval against the registered local buffer size. It uses
`offset <= registered_size && size <= registered_size - offset`, so the check
cannot itself wrap. The first materializer will require both buffer identity
and every interval-bounds check before entering the two-phase branch; any
failure executes the untouched fused launch.

Discovery also records source-identity candidates for kernels whose transfer
source may correspond to a read-only pointer formal. It carries frontend
`noalias` and a conservative IR use-chain proof of `readonly`, roots every
intervening write in other formals, and lists the pointer slots that host LTO
could compare with the shared registered source-buffer slot. This is useful
for matrix multiply, where `As` or `Bs` may name the registered source while
atomics are rooted in `Cs`. The relation is not yet an executable candidate:
LLVM `noalias` does not prove that the complete registered source allocation
misses every write-root allocation. A future materializer must separately
prove or check that whole-allocation relation, keep the original fused
schedule on false or unknown edges, and final device LTO must re-prove the
same identity-candidate and write-root facts before relocating the trigger.

Device discovery now emits this guard shape only for the narrow case of one
ordinary producer pointer and one shared i32 source-buffer formal across the
whole PUT group. The unchanged Jacobi kernel satisfies that shape with fixed
kernel slots `(pointer=1, buffer-index=12)`. A synthetic group whose members
use different source-buffer formals is rejected. These are compiler facts;
the fission action remains absent from the legal action set.

Feature extraction now also preserves every transfer's symbolic source
interval as a half-open byte range over compiler formal indices. It reports
separate `symbolically_exact`, `affine`, and `host_knowable` facts, and the
source-free bridge forwards the nested expression unchanged. This closes the
previous information-loss gap between device metadata and the model dossier.

Device discovery now preserves the matching local store semantics as well.
For each ordinary producer store it records a formal-rooted byte address and
the conjunction of dominating control predicates, using target-independent
expression nodes plus explicit GPU coordinate builtins. On unchanged Jacobi,
the real O3 HIP pipeline recovers the one `a_new` store as
`4 * ((global_y + iy_start) * nx + global_x + 1)` under
`global_y + iy_start < iy_end` and `global_x + 1 < nx - 1`. This is marked
`domain_exact` only as a local device-IR fact; the host ABI binding and
cross-operation partition are proved separately below.

Host LTO now audits every entry in the `hipLaunchKernel` parameter array. On
unchanged Jacobi it proves all 18 slots are distinct launch-local cells with
the storage types recorded by device metadata. This supplies a stronger
binding than attempting to reverse a source lambda: transfer expressions and
producer expressions name the same runtime kernel-formal slots, and both
phase launches reuse that exact array.

That shared formal namespace also removes the need to guess static scalar
identities such as `top_offset = 4 * nx`. For the first one-store PUT group,
the compiler constructs the boundary predicate directly as checked byte
overlap between the exact producer-store interval and either exact transfer
interval; the interior predicate is its logical complement. This proves a
disjoint and complete partition of ordinary store instances for unchanged
Jacobi, including the important fact that the transferred boundary columns
need not be producer writes.

Device discovery now separately proves that one exact controlling edge gates
a single-entry stencil region with a post-dominating merge. On unchanged
Jacobi this is the `global_x + 1 < nx - 1` edge, so the checked-overlap
predicate has a compiler-proved whole-region insertion point rather than only
a store-instance interpretation.

Fission must also preserve operations whose meaning depends on participation
or execution count. Discovery inventories exact atomic domains and every
convergent or `noduplicate` call other than compiler-recognized, replicable
GPU identity query. It retains dominating direct-`i1` kernel-formal guards
even for a call inside a loop. Real Jacobi contains one `atomicAdd` site and one
`__shfl_down` site (the latter executes repeatedly); both require formal 7,
`calculate_norm`, to be true. The compiler therefore permits the optimized
path only under formal 7 equal to false and retains the untouched fused launch
when it is true. It performs no floating-point reassociation and assumes no
subgroup law. A negative compiler test with opposite guards is rejected.

These are still not a default-enabled schedule. A dormant final host-IR pass
re-proves the launch and parameter-array shape, checks buffer identity plus
every interval bound at runtime, gates the optimized edge on the shared
side-effect-disabling formal, and emits two same-stream launches followed by a
phase reset, with an untouched fused-launch fallback. It rejects
dynamic/cross-block parameter-array writes and interval casts whose bit-width
semantics cannot be replayed from the v1 metadata.

A matching device-IR pass now rebuilds the final producer, transfer, completion,
and side-effect facts and exact-compares them with persisted discovery metadata.
It materializes the overlap/complement compute partition, gates the original
PUTs, and clones one synthetic early flush after the final PUT. The synthetic
flush carries compiler metadata so site-ID assignment ignores it while ordinary
device lowering still converts it to the same DWQ trigger operation. Both passes
are available as explicitly named test pipelines. Automatic host/device LTO
attachment requires the opt-in `GICC_PRODUCER_FISSION_ORACLE=1`; its default is
false, and this candidate remains absent from the model-visible legal action
set.

Device LTO rewrites the original kernel body:

- boundary phase skips PUT/flush/quiet and executes only iterations whose
  stores produce the exact transfer source intervals;
- interior/communication phase stages the original PUTs, relocates the flush
  immediately after them, executes only the proven-disjoint interior domain,
  and retains quiet at the original completion frontier;
- atomics, reductions, and non-replicable convergent operations are either
  disabled by the shared compiler-proved launch guard or separately proven
  safe; otherwise the candidate is masked.

### Audited host-IR materialization shape

At the early-simplification extension point used for compiler facts, the real
Jacobi host IR retains one annotated wrapper that pushes launch configuration
and dispatches through a constant kernel global. That global has one HIP
device-stub initializer; the stub pops the same configuration and owns the
single `hipLaunchKernel` call and its parameter array. The normal optimizer
later folds and inlines this chain, at which point the same launch lives
directly in the wrapper.

The feature pass now recognizes both forms and fails closed unless the stub
dispatch (when present), push/pop chain, launch target, return-value use, and
parameter-array lifetime are unique. It additionally proves every parameter
slot is distinct, launch-local, and metadata-typed. It reports the exact recomputation point
as `phase_launch_materialization: device_stub|wrapper`. On the unchanged
Jacobi source, the real early pass proves `device_stub`; it also rejects a
synthetic stub shared by another launch. Host LTO can insert a pre-authored
runtime helper around the proven call and clone it while it is still inside
the parameter-array lifetime:

```text
set_phase_from_kernel_args(kernel_params, boundary, stream)
hipLaunchKernel(original symbol, original geometry, kernel_params, shmem, stream)
set_phase_from_kernel_args(kernel_params, interior, stream)
hipLaunchKernel(original symbol, original geometry, kernel_params, shmem, stream)
set_phase_from_kernel_args(kernel_params, original, stream)
```

The helper reads the already-materialized `DeviceCtx*` from kernel argument
zero and launches the GICC-owned setter kernel on the supplied stream. This
avoids reverse-engineering `Runtime::prepare()` internals in the pass and
does not require a new application-visible kernel stub. Final host LTO must
rerun the shape proof because the optimizer may have moved ownership from the
stub to the wrapper.

Trace synthesis currently executes before the annotated wrapper. An IPC route
may therefore enqueue a copy before boundary producers run. The first fission
candidate consequently requires a valid final hint file in which every group
member resolves to exactly one untransformed `DWQ_TRIGGER`. `IPC_PUSH`,
`IPC_OR_DWQ`, batching, proxy transforms, missing hints, and malformed hints
all fail closed until trace synthesis itself can be phase-split.

## Required compiler proofs

Candidate generation is fail-closed and requires all of the following:

1. **One launch, one stream.** Host LTO recovers the exact launch wrapper,
   stream operand, kernel template, and device metadata ID.
2. **Buffer identity.** Host LTO either proves that a kernel pointer formal
   and a registered-buffer handle formal refer to the same allocation, or
   emits the compiler-owned runtime identity check with an untouched fused
   launch on its false edge.
3. **Exact transfer intervals.** Source offset and size are host-knowable
   affine expressions over launch operands.
4. **Exact producer footprint.** Device LTO maps stores through the related
   pointer formal to an affine iteration domain that produces each interval.
5. **Disjoint remainder.** Interior-phase stores cannot overlap a triggered
   source interval; loads and stores retain their original dependence order.
6. **Complete partition.** Boundary and interior predicates are disjoint and
   cover every originally active compute iteration exactly once.
7. **Non-duplicable-operation safety.** Atomics/reductions and subgroup or
   other non-replicable convergent operations receive every original effect
   exactly once, or one compiler-proved formal guard makes all of them
   unreachable on the optimized path. No candidate is emitted otherwise.
8. **Communication completeness.** Every original group member is staged and
   exactly one trigger and required quiet remain on every path.
9. **Launch safety.** Both launches use the original grid, block, shared
   memory, stream, and arguments. An exception-aware `invoke` of the annotated
   wrapper is accepted because materialization changes only the separately
   audited `hipLaunchKernel` call inside that wrapper and leaves the caller's
   normal and unwind successors untouched. Invoke-form HIP launches and
   multi-stream aliases remain rejected in the first implementation.
10. **Final replay.** The lowering pass recomputes all relations from final IR
    and checks the candidate content ID before changing either host or device
    code.

Unknown aliasing, non-affine offsets, irreducible CFGs, non-composable side
effects, unmatched convergent-operation guards, unknown stream order, or
inconsistent host/device metadata masks the candidate. A model cannot
override a mask.

## Candidate space and evaluation

The initial compiler catalog contains only:

- `original_fused` — semantic anchor;
- `producer_frontier_two_phase` — the exact schedule above.

Additional tilings, phase counts, or runtime predicates are not introduced
until the two-phase materializer is correct. Richness comes first from applying
the same legal schedule decision across multiple compiler-discovered groups,
problem sizes, topologies, compute distances, and producer footprints—not from
inventing many unimplemented labels.

Evaluation proceeds in gates:

1. lit tests for host/device discovery, content IDs, materialization, and every
   fail-closed proof above;
2. offline host/device IR equivalence audits on Jacobi, without a model;
3. one `pdebug` correctness smoke at a time;
4. paired compiler controls for end-to-end time and overlap attribution;
5. freeze a source-free relational prompt only if the compiler oracle has
   stable headroom over `original_fused`;
6. request separate, exact provider authorization; only then evaluate model
   candidate-ID selections and compile them through the same verifier.

Hand-edited Jacobi variants may be used only as clearly labeled engineering
diagnostics. They cannot serve as the claimed model or compiler result.
