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
`sha256:4ebc2d36ee251950bc6af7a3360c9a75516989fb46c863f3d54cb02074b9e712`.
The only materializable two-phase oracle is
`sha256:be6d9caef2578ca61bd623eca44e590533cf0fd347d739b8ad35c5b2cb33aa53`;
it is deliberately absent from the model-visible candidate lists.

## Coverage result

| Case | Feature SHA-256 | Compiler classification | Missing compiler proof or materialization |
|---|---|---|---|
| Jacobi | `9ef34ae5ad60516a97c8736dfc4d263bfc4c0b958410450f0bea6e399b184539` | `intra_kernel_exact_partition` | none for the guarded two-phase oracle |
| Minimod | `67950e39e52de4307072a3032644e23feda5beabbb14f0e997f6d8c052f34a33` | `conditional_communication_only` | guarded completion region; cross-launch producer pipeline |
| minimal matrix multiply | `3d5b0a86eb3023c1186ef51455d83598e5c9295c2218d82eb154983700d6f5e5` | `atomic_producer_no_store_remainder` | atomic producer partition; invoke-preserving host materialization |
| mixed-side-effect LTO | `02c67cb5768c71a7461afecbd9e4831fefc2f36fcc7532e8132630b85ad82105` | `unknown_side_effect_frontier` | side-effect alias partition; invoke-preserving host materialization |
| loop-carried LTO | `f2b377ac1adce656897dca7c6a6841a3aa1c71d08df7347e20fb2d1139fb61a1` | `loop_carried_communication` | guarded loop phase schedule; invoke-preserving host materialization |

This small, deliberately heterogeneous set does not estimate a population
rate.  It does show that the compiler-level search space is larger than a
single transfer-policy classifier: multiple distinct proof families block
otherwise plausible scheduling transformations, while one case already has a
fully materializable compiler-only oracle.

## Reproduction

From the repository root, after producing the five fact directories:

```sh
python3 tools/gicc-passes/experiments/producer_fission/analyze_compiler_schedule_coverage.py \
  --case jacobi=build_ofi/compiler_fact_coverage_20260904/jacobi_current/meta/features.json \
  --case minimod=build_ofi/compiler_fact_coverage_20260904/minimod_current/meta/features.json \
  --case mm_minimal=build_ofi/compiler_fact_coverage_20260904/mm_minimal_singlefrontier/meta/features.json \
  --case mixed_lto=build_ofi/compiler_fact_coverage_20260904/bench_mixed_lto/meta/features.json \
  --case loop_lto=build_ofi/compiler_fact_coverage_20260904/bench_pingpong_lto/meta/features.json \
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
