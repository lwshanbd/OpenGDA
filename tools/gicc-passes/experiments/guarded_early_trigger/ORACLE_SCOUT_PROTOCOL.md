# Guarded early-trigger compiler-oracle scout

Status: preregistered before runtime execution. This protocol evaluates one
compiler/LTO materialization over unchanged `examples/ofi/mm_minimal.cpp`. It
does not expose a candidate to an LLM, invoke a provider, or authorize any
application-source change.

## Frozen arms and correctness

Both arms select `DWQ_TRIGGER` for the same discovered PUT. `baseline` keeps
the source-level flush position. `guarded` additionally selects the existing
`GUARDED_EARLY_TRIGGER` transform. The build must prove identical feature
facts, kernel metadata identical after removing the one-way device
attestation, distinct executable hashes, and the frozen final-IR host/device
shape.

Because `mm_minimal` does not print its result matrix, both arms link the same
`--wrap=hipMemcpy` validation harness. After the timed region, it hashes the
final D2H matrix copy on every rank. A pair is valid only when all 16 rank
hashes match exactly. This wrapper is outside the measured interval and does
not alter the application source or select a schedule.

## Runtime design

- Queue: `pdebug` only.
- One two-node allocation, 16 ranks, eight ranks/GPUs per node, eight CPU
  cores per rank.
- Sizes: `N=4096` and `N=8192`, retained regardless of outcome.
- Four paired replicates per size in order blocks `AB`, `BA`, `BA`, `AB`.
- Each invocation uses the application's fixed ten runs and excludes its two
  declared warmups. The estimand is baseline mean microseconds divided by
  guarded mean microseconds.
- The controller waits for the current compiler-headroom campaign to become
  terminal before submitting exactly one Flux job. It never uses `pci`.

The scout gate passes when either size has at least three of four paired wins
and median speedup of at least `1.02`, after every checksum pair and scheduler
audit passes. This is only evidence that the compiler action space has runtime
headroom. A positive result permits a separately frozen confirmation; a
negative result keeps the transform absent from model-visible candidates.
Neither outcome is a paper-level LLM claim.
