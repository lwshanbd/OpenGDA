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
`sha256:905d2b063a7b82a9afdca5ac0c5674cc4fc85dba96e1f5b7dda6b8cd8bfadd82`.
The only materializable two-phase oracle is
`sha256:f0b4cb2a9ac13094c46d88ce7d706094e25ac95a33657b8e58031df6bfcac5e2`;
it is deliberately absent from the model-visible candidate lists.

## Coverage result

| Case | Feature SHA-256 | Compiler classification | Missing compiler proof or materialization |
|---|---|---|---|
| Jacobi | `0f94ca074a4d9c3efdf3935051ae55afaf61d1d94ce52077ea362f58ae29f332` | `intra_kernel_exact_partition` | none for the guarded two-phase oracle |
| Minimod | `67950e39e52de4307072a3032644e23feda5beabbb14f0e997f6d8c052f34a33` | `conditional_communication_only` | guarded completion region; cross-launch producer pipeline |
| minimal matrix multiply | `42edd118e147a7af658cbdd7366bf60a2fd8d5114ce19079a9bcd98c6425fda3` | `guarded_source_identity_candidate` | whole-write-allocation disjointness; communication-side-effect ordering; guarded early-trigger materialization |
| mixed-side-effect LTO | `b45161addf18a307cc78e67e35202b0cda7f015cff2e39b653bde35267c6d879` | `unknown_side_effect_frontier` | side-effect alias partition |
| loop-carried LTO | `c139fa05703d764d5e691611ab6ed2649b71cb973ca1025e126b3043cd43caf4` | `loop_carried_communication` | guarded loop phase schedule |

The table above counts structural schedule shapes, not all compiler-level
choices. The existing route materializer exposes the following independent
per-workload portfolios under platform profile SHA-256
`9c40639b6a14f439905fbd46dda682ad17ff017e79d8a15dc61de4f73e437fe3`:

| Case | Opportunity shape | Materializable route plans | Private graph ID | Model-prompt SHA-256 |
|---|---|---:|---|---|
| Jacobi | two-transfer completion group | 9 | `sha256:14cde78ee49478bae91ef42f9dcf711064f9e4a9187d1f7dca0aa9d390c78c6d` | `c3ab1b8a8f911e75d14a7a685a0f95c5f872c4589973cb1589b1ea388ada4b4c` |
| Minimod | two-transfer completion group | 9 | `sha256:5766ac7cf068d3a0878054ce87d8655cb2967d352a2053f4d75e7f565406c864` | `b4708dd6c1a4c3f745d4185a36608f8c9ecd199e9409fb26adeebeca43bf5f86` |
| minimal matrix multiply | singleton transfer | 3 | `sha256:e8ba0ec787b531a9f900e09334dd938a5c805c3b2be79e29876e9fcb2d700987` | `f8d51838bd326aadab6efac8a0589b3ac7a63d5f20c005e3ed4d99ce8c9175a3` |
| mixed-side-effect LTO | two-transfer completion group | 9 | `sha256:4c9e9ea773e2ae2f560bf89434e19a762cd27277fcb98b0597a9a84bf498e731` | `83f543591047a25f1ea36c19dbacdec7cdf651fa970b889e7b43d8736e1a7099` |
| loop-carried LTO | singleton transfer | 2 | `sha256:0e53a7261a7be87fc9ba363c56d203db661cf99400c3f781efbc10510bb431e9` | `788c7aaaac36373f0b9fb43ec06b13482c48ad356206a2e3ccd3f809b37454a9` |

These are 32 catalog entries across five separate workload decisions, not one
32-way joint decision and not 32 distinct program transformations. They cover
compiler-legal transport routing (`default`, proxy, or trigger, with the loop
shape restricted to proxy/trigger). The structural axis remains narrower:
the original phase layout is visible in every case, while the Jacobi
producer-fission layout remains dormant pending its preregistered runtime
gate. A future composed portfolio must enumerate and revalidate only route ×
schedule combinations that the corresponding LTO materializers jointly
support; it must not assume a Cartesian product.

The model prompts above contain the richer compiler relations needed for that
study—formal argument expressions, transfer intervals, producer-frontier
facts, launch contexts, compute distance, dependence legality, and measured
platform data. The private graphs retain site/materializer bindings, while the
prompt view removes kernel names, site IDs, paths, profile provenance paths,
and materializer strings. No provider call is authorized by generating these
local prompts.

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
retain the fused schedule. Because the write is atomic, the compiler must also
establish that relocating communication across it preserves observable
side-effect order. No executable candidate is exposed yet.

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

For each case, generate the private route graph and identity-free model prompt
locally. For example, for Jacobi:

```sh
python3 tools/gicc-passes/python/gicc_llm_bridge.py emit \
  --features build_ofi/compiler_fact_coverage_20260904/jacobi_disjoint/meta/features.json \
  --platform tools/gicc-passes/python/profiles/tioga-mi250x-slingshot11.json \
  --dossier build_ofi/compiler_fact_coverage_20260904/portfolio/jacobi/dossier.json \
  --prompt build_ofi/compiler_fact_coverage_20260904/portfolio/jacobi/site-prompt.txt
python3 tools/gicc-passes/python/gicc_comm_group_plan_bridge.py emit \
  --dossier build_ofi/compiler_fact_coverage_20260904/portfolio/jacobi/dossier.json \
  --meta-dir build_ofi/compiler_fact_coverage_20260904/jacobi_disjoint/meta \
  --graph build_ofi/compiler_fact_coverage_20260904/portfolio/jacobi/group-graph.json \
  --prompt build_ofi/compiler_fact_coverage_20260904/portfolio/jacobi/group-prompt.txt
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
