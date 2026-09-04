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
`sha256:4ec6b7d42622988f57c4b740b38fd872fd4d8a8e18cf49b209a993c1ea0c8f4f`.
The only materializable two-phase oracle is
`sha256:be6d9caef2578ca61bd623eca44e590533cf0fd347d739b8ad35c5b2cb33aa53`;
it is deliberately absent from the model-visible candidate lists.

## Coverage result

| Case | Feature SHA-256 | Compiler classification | Missing compiler proof or materialization |
|---|---|---|---|
| Jacobi | `9ef34ae5ad60516a97c8736dfc4d263bfc4c0b958410450f0bea6e399b184539` | `intra_kernel_exact_partition` | none for the guarded two-phase oracle |
| Minimod | `67950e39e52de4307072a3032644e23feda5beabbb14f0e997f6d8c052f34a33` | `conditional_communication_only` | guarded completion region; cross-launch producer pipeline |
| minimal matrix multiply | `bc83a7afe5b6b69d56c954c8c66016932c882cc18d245cba714447867124f5b4` | `atomic_producer_no_store_remainder` | atomic producer partition |
| mixed-side-effect LTO | `934dd3c7084088ad25f357b984c9c3c336561eca718f90a7089baa0f622448e4` | `unknown_side_effect_frontier` | side-effect alias partition |
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

## Reproduction

From the repository root, after producing the five fact directories:

```sh
python3 tools/gicc-passes/experiments/producer_fission/analyze_compiler_schedule_coverage.py \
  --case jacobi=build_ofi/compiler_fact_coverage_20260904/jacobi_current/meta/features.json \
  --case minimod=build_ofi/compiler_fact_coverage_20260904/minimod_current/meta/features.json \
  --case mm_minimal=build_ofi/compiler_fact_coverage_20260904/mm_minimal_invoke/meta/features.json \
  --case mixed_lto=build_ofi/compiler_fact_coverage_20260904/bench_mixed_invoke/meta/features.json \
  --case loop_lto=build_ofi/compiler_fact_coverage_20260904/bench_pingpong_invoke/meta/features.json \
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
