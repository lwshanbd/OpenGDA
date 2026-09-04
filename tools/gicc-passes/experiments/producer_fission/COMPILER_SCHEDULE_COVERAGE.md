# Compiler schedule coverage snapshot

Status: source-free compiler-fact coverage audit recorded before any provider
call.  This is structural evidence about the optimization search space, not a
runtime or LLM-quality result.

## Boundary and integrity

- Inputs are schema-6 compiler facts and sibling schema-1 kernel templates.
- Application source text is never read by the analyzer or placed in the
  `model_graph`.
- The model-visible graph contains no case label, filesystem path, kernel name,
  source filename, or operation site ID.  The analyzer fails closed if one of
  those identities escapes.
- Every schedule candidate commits to its complete payload with a SHA-256 ID.
  Case and graph IDs are also recomputed before the report is written.
- The model can select only IDs in `model_visible_candidate_ids`.  Compiler
  oracles remain dormant until independent runtime evidence establishes
  headroom and a later request explicitly authorizes a provider call.

The stable source-free graph ID for this snapshot is
`sha256:6264bb8d57dbb563c2af9f8dea30024cc4eb1102d86c013c2458cb87074b8f02`.
The only materializable two-phase oracle is
`sha256:dae68f7745980e8aa4f6a1fdc338898c36c66b611e97e5301636aeaf54269baf`;
it is deliberately absent from the model-visible candidate lists.

## Coverage result

| Case | Feature SHA-256 | Compiler classification | Missing compiler proof or materialization |
|---|---|---|---|
| Jacobi | `0f94ca074a4d9c3efdf3935051ae55afaf61d1d94ce52077ea362f58ae29f332` | `intra_kernel_exact_partition` | none for the guarded two-phase oracle |
| Minimod | `67950e39e52de4307072a3032644e23feda5beabbb14f0e997f6d8c052f34a33` | `conditional_communication_only` | guarded completion region; cross-launch producer pipeline |
| minimal matrix multiply | `42edd118e147a7af658cbdd7366bf60a2fd8d5114ce19079a9bcd98c6425fda3` | `guarded_source_identity_candidate` | whole-write-allocation disjointness; guarded early-trigger materialization |
| mixed-side-effect LTO | `b45161addf18a307cc78e67e35202b0cda7f015cff2e39b653bde35267c6d879` | `unknown_side_effect_frontier` | side-effect alias partition |
| loop-carried LTO | `c139fa05703d764d5e691611ab6ed2649b71cb973ca1025e126b3043cd43caf4` | `loop_carried_communication` | guarded loop phase schedule |

This small, deliberately heterogeneous set does not estimate a population
rate.  It does show that the compiler-level search space is larger than a
single transfer-policy classifier: multiple distinct proof families block
otherwise plausible scheduling transformations, while one case already has a
fully materializable compiler-only oracle.  All five cases now pass the host
phase-materialization shape audit.  Three had previously been rejected solely
because the application-side wrapper call used LLVM `invoke`; the compiler
now correctly audits the independent HIP launch inside the wrapper without
changing the caller's normal or unwind edge.

The matrix-multiply atomic is not assumed to produce the communicated
buffer. Device IR identifies its root as the `Cs` pointer formal, whereas the
transfer names a registered-buffer index. A conservative use-chain proof
finds two `readonly noalias` pointer formals (`As` and `Bs`) that may name that
registered source. The source-free relation records those identity candidates,
the `Cs` write root, and the source-buffer formal. It deliberately makes no
whole-allocation disjointness claim: LLVM `noalias` alone is insufficient for
transfer bytes not otherwise accessed through the candidate pointer. A future
host materializer must both match source identity and prove or check that the
full transfer interval misses every write-root allocation; otherwise it must
retain the fused schedule. No executable candidate is exposed yet.

## Reproduction

From the repository root, after producing the five fact directories:

```sh
python3 tools/gicc-passes/experiments/producer_fission/analyze_compiler_schedule_coverage.py \
  --case jacobi=build_ofi/compiler_fact_coverage_20260904/jacobi_disjoint/meta/features.json \
  --case minimod=build_ofi/compiler_fact_coverage_20260904/minimod_disjoint/meta/features.json \
  --case mm_minimal=build_ofi/compiler_fact_coverage_20260904/mm_minimal_disjoint/meta/features.json \
  --case mixed_lto=build_ofi/compiler_fact_coverage_20260904/bench_mixed_disjoint/meta/features.json \
  --case loop_lto=build_ofi/compiler_fact_coverage_20260904/bench_pingpong_disjoint/meta/features.json \
  --out build_ofi/compiler_fact_coverage_20260904/coverage-report.json \
  --markdown build_ofi/compiler_fact_coverage_20260904/coverage-report.md
```

The JSON report retains paths and human labels only in its internal audit
section.  Only its separately content-addressed `model_graph` is eligible for a
future model prompt.

## Consequence for the paper direction

The immediate scientific question is the compiler-oracle upper bound: does a
legal communication schedule generated and materialized entirely in LTO have
measurable runtime headroom?  The frozen Jacobi A/B scout answers the first
instance of that question.  Only if it passes the preregistered gate should the
two-phase candidate become model-visible.

LLM utility is a later, separate question.  It requires multiple compiler-
generated legal candidates, held-out decisions from the source-free graph, and
runtime regret against compiler oracles.  Until those conditions exist, this
snapshot supports investment in richer compiler proofs and candidate
generation, but it makes no claim that an LLM improves performance.
