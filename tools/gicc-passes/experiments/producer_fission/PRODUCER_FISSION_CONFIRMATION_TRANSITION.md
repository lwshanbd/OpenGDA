# Producer-fission scout-to-confirmation transition

Status: preregistered before the pending Jacobi producer-fission scout was
submitted. This file is separate from the immutable scout artifact set.

The transition applies only to compiler/LTO-generated schedules over unchanged
Jacobi source. It has no provider, compiler, scheduler, or source-edit path.
The dormant candidate is
`sha256:f0b4cb2a9ac13094c46d88ce7d706094e25ac95a33657b8e58031df6bfcac5e2`,
a compiler-proved guarded host/device fission with two phases. It remains
absent from the model-visible candidate list until independent runtime
confirmation passes.

## No post-hoc size selection

The exploratory gate passes when either 1 KiB or 4 KiB has at least three of
four paired wins and median speedup of at least `1.03`. That gate decides only
whether confirmation is worth running. It does not choose the confirmation
estimand.

The compiler schedule graph has no runtime-size decision slot, so confirmation
always retains both preregistered sizes. A favorable size cannot be kept while
an unfavorable one is discarded. The primary metric is the geometric mean of
paired baseline/fission speedups across both sizes and both order-balanced
blocks, clustered by independent allocation.

## Confirmation contract

Only a passed, regenerated scout may produce a transition manifest. The future
confirmation consists of three independent two-node `pdebug` allocations,
submitted and completed one at a time. Each uses 16 ranks, eight ranks/GPUs per
node, eight CPU cores per rank, `-niter 200 -nccheck 10`, both sizes, and both
`AB` and `BA` blocks.

The compiler oracle is confirmed only if:

- all iteration-count and final-norm correctness pairs pass;
- the aggregate fission speedup point estimate is at least `1.03`;
- the exact allocation-cluster paired-bootstrap 95% lower bound is above one;
- at least two of three allocation-level geometric-mean speedups exceed one.

Scout and confirmation labels are excluded from every later model prompt.
Failure keeps the candidate dormant. Success permits creation of a *new*
content-addressed compiler graph that exposes the existing candidate ID; it
does not mutate the frozen graph or authorize a provider call. One positive
Jacobi case establishes compiler schedule headroom, not an LLM-selection
claim; that claim still requires multiple held-out decision cases.

## Deferred use

Only after the existing successor controller completes with a positive scout:

```sh
python3 tools/gicc-passes/experiments/producer_fission/prepare_producer_fission_confirmation.py prepare \
  --monitor build_ofi/producer_fission_oracle_scout_90b9123_20260904/monitor.json \
  --analysis build_ofi/producer_fission_oracle_scout_90b9123_20260904/analysis.json \
  --coverage-report build_ofi/compiler_fact_coverage_20260904/coverage-report.json \
  --binary-dir build_ofi/producer_fission_oracle_90b9123 \
  --out build_ofi/producer_fission_confirmation_transition_20260904.json
```

The preparer rehashes the complete scout artifact set, regenerates every
reported speedup from baseline/fission seconds, verifies the dormant candidate
identity and source-free coverage graph, and rejects missing, failed, or
tampered evidence. It emits a plan only and never submits confirmation work.

The confirmation implementation is also frozen before the scout result. After
the transition exists, and only when it verifies as
`confirmation_plan_ready`, the serial controller can be started with:

```sh
bash tools/gicc-passes/experiments/producer_fission/continue_producer_fission_confirmation.sh \
  build_ofi/producer_fission_confirmation_transition_20260904.json \
  build_ofi/producer_fission_oracle_90b9123 \
  build_ofi/producer_fission_confirmation_20260904
```

The controller reuses the exact scout-bound baseline and compiler-fission
binaries. It submits one `pdebug` allocation, waits for a clean audited result,
and only then submits the next. Each allocation contains an `AB` block and a
`BA` block over both sizes. The analyzer rejects shared job IDs, altered
artifacts or execution order, queue/resource drift, and any iteration-count or
final-norm mismatch. This command is documented here but is not launched by
the preparer or by the pending scout controller.
