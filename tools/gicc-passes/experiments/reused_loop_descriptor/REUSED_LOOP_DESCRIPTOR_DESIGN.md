# Reused loop descriptor: frozen compiler-only design

## Question

Can LTO use a compiler-proved loop-invariant communication descriptor to
remove redundant host-side descriptor construction without changing the
application source or collapsing observable network operations?

The motivating `loop_lto` kernel has one `put_no_db` in a runtime-bounded
device loop.  Device discovery already proves that all six descriptor
arguments are loop invariant and reports `descriptor_reusable=true`.  The
current host trace nevertheless allocates six runtime-sized arrays, executes
six stores per iteration, and reloads those arrays in
`gicc_runtime_dwq_enqueue_batched`.

## Frozen transform

The new, default-off hint transform is `REUSE_LOOP_DESCRIPTOR`.

For a legal selected PUT loop, trace synthesis will emit one compiler-owned
placeholder with this logical shape:

```text
repeat(rt, n_ops, peer, dst_buf, dst_off, src_buf, src_off, size)
```

Dispatch lowering maps the placeholder one-to-one to
`gicc_runtime_dwq_enqueue_repeated`.  The runtime helper resolves the local
registration and remote address once, then queues exactly `n_ops` deferred
RMA writes with the same monotonically increasing thresholds and accounting
as the existing batched helper.

This is descriptor-template reuse, not transfer coalescing:

- network operation count remains `n_ops`;
- operation order and per-operation trigger thresholds remain unchanged;
- the original device kernel and its completion point remain intact;
- no source operation, computation, or memory access is moved;
- application source is neither rewritten nor exposed to a model.

## Compiler legality contract

Both the graph generator and trace materializer must fail closed unless all
of the following are independently present in compiler metadata:

1. the operation is `put_no_db` and the host descriptor is knowable;
2. the operation is in a modeled, non-degraded natural loop;
3. the loop starts at zero, steps by one, and has a known bound;
4. a dynamic bound is an in-range `i32` kernel formal; a constant bound fits
   signed `i32`;
5. every descriptor expression (`target_rank`, both buffers, both offsets,
   and size) is recursively loop invariant;
6. there is no per-iteration `field_not_null` guard;
7. the selected route is `DWQ_TRIGGER`.

For the canonical signed loop `for (i = 0; i < bound; ++i)`, the generated
count is `max(bound, 0)`.  No model-provided legality statement, count,
argument expression, site identity, code, or IR is trusted.

## Model boundary and rollout gate

The compiler bridge may understand this materializer, but it must remain
absent from production/model-visible opportunity graphs unless a platform
profile explicitly enables `compiler_transforms.reused_loop_descriptor`.
The Tioga profile stays unchanged until a content-addressed, order-balanced
runtime scout passes correctness and launch/enqueue-count audits.  Enabling
the candidate after a scout permits a new graph; it does not authorize a
provider call.

The eventual model output remains one opaque compiler-generated candidate ID.
The compiler revalidates the frozen legality contract before lowering.

## Validation before runtime

- LLVM tests must prove the positive scalar-placeholder shape and fail-closed
  cases for loop-variant arguments and unsupported bounds.
- Dispatch-lowering tests must prove the exact helper ABI and absence of the
  placeholder after lowering.
- Python tests must prove default invisibility, profile-gated visibility,
  content-addressed candidate identity, strict acceptance, and safe fallback.
- The complete Python and LLVM test suites must remain green.

No scheduler job is authorized by this design document.  A separate frozen
oracle protocol is required before submitting any runtime experiment.
