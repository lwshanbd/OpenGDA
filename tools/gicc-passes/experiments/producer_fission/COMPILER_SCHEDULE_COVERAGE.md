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

| Case | Opportunity shape | Plans | Private graph ID | Relational prompt SHA-256 | Descriptors prompt SHA-256 | Opaque prompt SHA-256 |
|---|---|---:|---|---|---|---|
| Jacobi | two-transfer completion group | 9 | `sha256:1f7a4f9adeddc978846484d938cbd3d545bb6d3a758188e53a6b6528bb9a72dd` | `0af23ab3d79dbbda6a6f7345841e3e9e0ac6a96a7b4a2266893c804e999951a2` | `d97e51daeef2baec7f589232b1b12391ecc7467757c286fb78325c892dc5c049` | `1ee5457271d348149fdfc702e07a9e98a74946ac6da8e712af00560b262fcca7` |
| Minimod | two-transfer completion group | 9 | `sha256:635f115d1fdcc65e48b49ef8de94d29734edfaf22f6a66b035e3e57c4396889d` | `f70a00222199f8938cdea2c7c194073e985cb7e1ea1fa3c8ba6a31fb4a5c1568` | `9299ac535ea9913c4de92f9008f3df91be26a1c8818d1ee067cd677dfd8cc0da` | `e58a0dbfafccdb3275477583f9119565cd1a4f45057a57b867b10c15bc6673a4` |
| minimal matrix multiply | singleton transfer | 3 | `sha256:e8ba0ec787b531a9f900e09334dd938a5c805c3b2be79e29876e9fcb2d700987` | `431596fcc224991069f5c84094bec650c896e07d78d45477aa9dde97485db74c` | `d7a573699a03ef159f3b60833d052a21a290687183cabe0f421d870b4ceb6b31` | `c5d3e28f79efb1bf867a4ebd269ec9bffa7caadf2b50797d811080053c1349f4` |
| mixed-side-effect LTO | two-transfer completion group | 9 | `sha256:b3e137e2fe3f2cad2807eb534383066f285182c7588e8a895b73675087c0f5a5` | `194948387a5123f962b2ba6bfc1321d4acac48def0e62e1c1244005882aea671` | `900eed1308b6b1516c401ed330524c357a0171dfcb0f2e93da5fdd3860b2dc3c` | `c24cf03fdda58094ec7a13b1cd743c9cd38492e9d99d4be97bb03c0be0a4fa2c` |
| loop-carried LTO | singleton transfer | 2 | `sha256:0e53a7261a7be87fc9ba363c56d203db661cf99400c3f781efbc10510bb431e9` | `b90b76b71e73e12dc0e4e0bde7e9948ea78b0d775869a4765ea8f25782956607` | `92bbd78f83009ce5473d8f43464f18ee08902384b7177525a8d6d516ba5f6d5c` | `acb4dcf91dc949869e216eb66c5474bf88dd49580ba687366535f5ea55b6bc3f` |

These are 32 catalog entries across five separate workload decisions, not one
32-way joint decision and not 32 distinct program transformations. They cover
compiler-legal transport routing (`default`, proxy, or trigger, with the loop
shape restricted to proxy/trigger). The structural axis remains narrower:
the original phase layout is visible in every case, while the Jacobi
producer-fission layout is now represented by one compiler-generated masked
candidate. Its opaque selection maps only to
`PRODUCER_FRONTIER_TWO_PHASE`; the host and device LTO passes independently
re-prove it, and host LTO additionally requires the successful final-device
materialization attestation. The base platform profile keeps it non-selectable
pending its preregistered runtime gate. Enabling that gate would make the
Jacobi portfolio 10 candidates and the five-workload total 33, without
inventing another transformation. A future composed portfolio must enumerate
and revalidate only route × schedule combinations that the corresponding LTO
materializers jointly support; it must not assume a Cartesian product.

The primary relational prompts above contain the richer compiler relations
needed for that study—formal argument expressions, transfer intervals,
producer-frontier facts, launch contexts, compute distance, dependence
legality, and measured platform data. Their explicit relation edges name
transfer order and shared/distinct argument structure. The `descriptors`
ablation removes those edges while retaining the same compiler and candidate
descriptors. The `opaque` ablation additionally removes selectable-candidate
semantics while preserving exactly the same candidate IDs and response
schema. The private graphs retain site/materializer bindings, while every
prompt view removes kernel names, site IDs, paths, profile provenance paths,
and materializer strings. No provider call is authorized by generating these
local prompts.

Shared transfer arguments, launch facts, and producer-frontier proofs are
factored into one compiler entity instead of repeated for every transfer; only
per-transfer differences remain in the ordinal records. This preserves all
semantic content while reducing the five portfolios' three-view prompt bytes
from 483,350 to 302,690 (37.4%). In particular, the Jacobi relational prompt
falls from 93,813 to 50,970 bytes without changing its graph or candidate IDs.

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
  --prompt build_ofi/compiler_fact_coverage_20260904/portfolio/jacobi/group-prompt.txt \
  --prompt-view relational
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
