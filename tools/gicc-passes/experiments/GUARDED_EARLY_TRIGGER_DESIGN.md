# Compiler-owned guarded early trigger

Status: frozen implementation contract. The runtime allocation guard exists,
but no compiler candidate is selectable and no application binary has yet
been materialized or measured.

## Research question

Can LTO start an already-staged DWQ transfer before an independent compute
region, while preserving the application source, kernel ABI, and fallback
behavior? The first real target is the singleton PUT in `mm_minimal`: the
logical PUT precedes matrix multiplication, but its hardware trigger remains
at the later source-level flush. The candidate moves only that compiler-owned
trigger; it does not move the logical communication operation or any
application instruction.

This is an action-space experiment, not an LLM experiment. The compiler oracle
must first demonstrate stable end-to-end headroom. A later model may select
only an existing content-addressed candidate ID and cannot supply source, IR,
pointer identities, ranges, predicates, or legality assertions.

## Frozen execution shape

The original kernel symbol, arguments, grid, block size, shared-memory size,
and stream are unchanged. Host LTO emits the guard in the annotated launch
wrapper before general inlining. At that extension point HIP may still
represent the application launch as a compiler-owned `push configuration ->
device stub -> hipLaunchKernel` chain. The pass audits that whole chain,
builds a temporary kernel-parameter view from the exact stub operands, and
then may emit exactly:

```text
guard = source-handle matches one compiler-proved readonly pointer formal
     && every transfer interval is within the registered source buffer
     && the registered source interval is disjoint from every compiler-proved
        write-pointer GPU allocation

if guard:
    set_schedule_phase(EARLY_TRIGGER) on the original stream
    launch original kernel once
    set_schedule_phase(ORIGINAL) on the original stream
else:
    launch original kernel once
```

Final device LTO may change the kernel only as follows:

- in `EARLY_TRIGGER`, issue one compiler-owned MMIO trigger immediately after
  the unique post-PUT frontier and skip the original flush;
- in every other phase, skip the synthetic trigger and retain the original
  flush;
- retain all compute, loads, stores, atomics, quiet operations, and control
  flow exactly once.

The existing same-stream setter kernel orders the phase write before the
application kernel and resets it afterward. A false or failed guard runs the
untouched schedule once. The new phase value is compiler/runtime-owned and is
not application-visible. Transforming before inlining is required: otherwise
O3 can distribute unguarded `hipLaunchKernel` calls into application callers
before a late wrapper-only pass sees them. Final optimized IR must show that
every source launch still reaches the guarded wrapper (or contains an inlined
copy of its guard), with no reachable direct-launch bypass.

## Required compiler proof

Candidate generation, device materialization, and final host materialization
must each fail closed unless all applicable facts below agree:

1. There is one completion group of unconditional, non-loop, host-knowable
   PUTs and exactly one mandatory flush after a unique post-issue frontier.
2. Every PUT uses one shared direct `i32` registered-source-buffer formal and
   has an exact host-materializable half-open source interval.
3. At least one pointer formal is `readonly noalias` and may be compared with
   that registered source at runtime; no candidate pointer is a write root.
4. Every memory write crossed by the early trigger is rooted in a known kernel
   pointer formal. Unknown writes, inline assembly, volatile reads, fences,
   barriers, and unclassified convergent or `noduplicate` calls reject the
   candidate.
5. Every crossed write-root formal is `noalias`, is distinct from every source
   identity candidate, and is checked at runtime against the complete
   registered source interval using the GPU allocator's allocation range.
6. The helper returns false for invalid indices, null pointers, zero sizes,
   allocation-query failures, or address overflow. Platforms without the
   audited HIP allocation query retain the original schedule.
7. The explicit hint selects `DWQ_TRIGGER` and the pre-authored guarded
   transform for every group member. A route-only hint cannot change trigger
   placement.
8. Final device LTO rebuilds the write-root and communication facts, exact
   compares them with discovery metadata, materializes the phase CFG, and
   persists a fresh attestation. Host LTO requires that attestation and
   independently audits either the final direct-launch ABI or the pre-inliner
   compiler HIP-stub chain before it adds the guard.
9. The original completion semantics remain: hardware may begin the logical
   PUT any time after its source-level call, and quiet/reset still guarantee
   completion at the original boundary. No ordering is inferred for a GET.

## Deliberate exclusions

- No application-source rewrite or generated source variant.
- No early trigger for GET, loop-carried communication, conditional PUTs,
  multiple completion groups, unknown aliases, or non-HIP targets.
- No proof based only on pointer inequality. Runtime checks compare the full
  registered source interval with the full allocator range of every write
  pointer.
- No claim that `noalias` alone covers a registered-buffer handle.
- No model/provider call, and no model-visible candidate, before the compiler
  oracle passes a separately frozen runtime gate.

## Evaluation gate

After lit tests, a source-unchanged disjoint build must show the explicit phase
CFG in final host/device IR, distinct binary hashes for baseline and guarded
oracle, identical correctness outputs, and the expected route/trigger audit.
Runtime evaluation will use `pdebug`, one scheduler job at a time, paired
order-balanced runs, and end-to-end wall time. A scout may advance only to an
independently frozen confirmation; neither stage authorizes a model call.
