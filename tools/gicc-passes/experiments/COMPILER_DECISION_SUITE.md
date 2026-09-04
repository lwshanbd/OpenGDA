# Source-free compiler decision suite

Status: locally frozen capability inventory; no model or provider call and no
runtime-performance claim.

This suite answers a narrower question than model accuracy: do we have a
replayable set of non-source compiler decisions whose inputs are materially
richer than the scalar rows used by the original path GBT?  The current answer
is yes.  The frozen index contains seven **independent** compiler/LTO decision
tasks from three decision families:

| Entry | Compiler decision | Independent policy count | Runtime status |
|---|---|---:|---|
| `coalescing_placement` | proxy/trigger route, loop coalescing, and early/late trigger placement over six compiler opportunities | 4096 | four-replicate runtime capacity measured; preregistered LLM gate failed |
| `collective_n8` | collective algorithm jointly across four message regions | 4096 | topology-matched hierarchy-pipeline scout pending |
| `jacobi` | two-transfer communication-group route | 9 | producer-frontier schedule exists but remains masked pending its oracle A/B |
| `minimod` | two-transfer communication-group route | 9 | route capacity only |
| `mixed_lto` | two-transfer communication-group route | 9 | route capacity only |
| `mm_minimal` | singleton communication route | 3 | early trigger remains illegal without allocation and side-effect-order proofs |
| `loop_lto` | singleton communication route | 2 | loop phase scheduling remains unmaterialized |

The counts above are reported per entry.  They must not be summed or
multiplied into a fictitious joint action space: the programs execute
independently, and no compiler materializer composes their decisions.

## Frozen identity

- suite ID:
  `sha256:0089efb08a37fde688717be90d2004460a3f6ee82ccbda75c9234e0214bc3483`
- serialized suite SHA-256:
  `748f816f48a379b27d5c8f4d4157357d009047202eabfa2952f11082be9f7bba`
- decision-family counts: one collective size-policy graph, five
  communication route/schedule graphs, and one communication coalescing and
  trigger-placement graph;
- selectable compiler IDs: 32 collective option IDs over four slots and 32
  communication route/schedule candidate IDs over five separate tasks, plus
  24 structural candidate IDs over six opportunities;
- prompt views: `relational`, `descriptors`, and `opaque` for every entry.
- one graph-derived exact response schema per entry; every selectable field is
  an enum of existing option/candidate IDs and every unknown field is rejected.

The suite manifest content-addresses every private graph, selectable-ID set,
model view, rendered prompt, and exact response schema. Its verifier checks the
suite and entry IDs, all prompt/schema hashes and byte counts, and the
independence declaration. The tool has no provider, scheduler, compiler, or
source-edit code path.

The communication views normalize identical per-transfer facts into shared
compiler entities. This retains every semantic field and all ordinal-specific
differences while avoiding repeated producer and launch subgraphs. Across the
five communication entries, the 15 prompt files shrink by 37.4%; the Jacobi
relational view shrinks from 93,813 to 50,970 bytes with the private graph and
all selectable IDs unchanged.

## Information ablation

All three views of an entry bind to the exact same private compiler graph,
candidate IDs, response schema, strict validator, and LTO materializer.

- `relational` is primary.  It exposes compiler-derived topology, operation
  order, shared/distinct argument relations, symbolic transfer intervals,
  producer/dependence facts, launch shape, resource constraints, and
  source-free calibration where available.
- `descriptors` removes explicit relation edges but retains the underlying
  compiler and candidate descriptors.
- `opaque` removes selectable-candidate semantics while retaining the same
  selectable IDs and response shape.

This makes future differences among the three views evidence about use of
compiler semantic context, rather than differences in transformation
authority.  It also states precisely how the LLM input differs from the
existing GBT: the GBT consumes declared scalar features for one route label;
the primary LLM view receives graph structure and compiler-proved relations
needed for joint policies.  This does not imply that the LLM will perform
better; only a held-out, runtime-validated regret experiment can show that.

An identity-leak audit over all 21 rendered prompts found no application or
kernel name, source suffix, site ID, materializer binding, or provenance path.
No prompt is authorized for external transmission by this local freeze.

## Structural action-space entry

The added `coalescing_placement` entry is the largest already materialized
compiler transformation space in the suite.  Its six opportunities each offer
four compiler-generated candidates: device proxy, trigger descriptor batch,
late loop coalescing, and early coalescing/trigger placement.  This is a
factorized `4^6 = 4096` space with 20 additional fixed sites.  The model sees
compiler facts and compiler-proved effects but not source, kernel names, site
IDs, provenance paths, or materializer bindings; an exact response schema
allows only the 24 existing candidate IDs.  All three information views bind
to that same action set and strict compiler validator.

The official suite binds the current-plugin graph:

- graph ID:
  `sha256:77a0f61a8e569b76f43391f1c4fbcae04d606db78b20a69ab65460aba7371c24`;
- graph file SHA-256:
  `80799c6e8acb056f2f213fa12463d57ad4f8863a75a2200d7c3bd86ea78242ef`;
- entry ID:
  `sha256:37e902c3c1eaf75bfa9cecc0259b3297436066a8df1be36e334e6e9c2912930d`;
- selectable-ID-set SHA-256:
  `f0d9c5e04b14890571625cfd20e4dfb6176b6b72f0c60d210dac4535b73e447a`;
- relational/descriptors/opaque prompt SHA-256 values:
  `324be5d0e8f2ef5ecc2a2480f1ef3bfd387f06a123afb6aab5fd3b73d6982974`,
  `ed4c9bd02b2115c88e818ae6745b87c78cf6bb38229651729c99b43213bbb1f7`,
  and `e7e5d0f08e2f92a3d38ab0ba3e2158945728298cf7e95d3492b9fd0718fe930b`;
- exact response-schema ID:
  `sha256:174375e342426dfa006b3ab832d1b5c0146e1febc42c6aa91c5f03ce3adab4b8`.

The historical four-replicate runtime bundle is intentionally not rewritten.
It was frozen against plugin SHA-256
`7f11ba4ccbbee6d2d79d0c02059476cefc87d4339a73af471c1639629bab8a73`
and graph ID
`sha256:544a66f3b4c217fe5aea56c1c0ed7a87f685e47f460659110764ed26f9756d7b`.
The available plugin is now
`e8ef1675c02b580efb868a70211bd896891709b4a87aff5eea6e635b7c09e33d`,
so the historical bundle verifier reports that plugin mismatch rather than
pretending bitwise provenance still holds.  A disjoint rebuild with the
current plugin produced the graph above and passed the compiler's manifest,
host-IR, and device-IR semantic checks for all four uniform arms.  After
removing only `compiler_dossier_id` and `graph_id`, the historical and current
private graphs are byte-for-byte canonical-equal, with normalized SHA-256
`413346b17a40fef5fc3d96ecbe32bd944c3605c01db59c69557280e9be0821fa`;
all opportunity and candidate IDs are identical.  Thus the historical timings
support the unchanged compiler action semantics, while any future response is
still bound to and revalidated against the current graph.

The historical runtime result is a useful positive and negative control at
once.  Trigger descriptor batch is `5.485628x` slower than the exact factorized
oracle, demonstrating large compiler-owned optimization headroom from the
available transformations.  But the best uniform arm is only `1.017982x`
slower than that oracle, with paired-bootstrap 95% CI
`[0.989739, 1.054439]`; there are two stable early winners and no stable late
winner.  The preregistered placement LLM gate therefore failed.  This supports
the paper's compiler transformation/action-space claim, not an LLM performance
claim or a stable per-site placement claim.

## Frozen deterministic compiler control

The eight-node collective graph also has a preregistered, source-free
structural heuristic.  This is a compiler baseline, not a model result.  It
reads only the verified graph and topology descriptors; it cannot read source,
runtime measurements, or provider output.  It chooses the guarded
`node_double_tree` cohort at four or more nodes with more than one rank per
node, using pipeline depths `(1, 1, 4, 8)` over the four ordered message bins.
If the topology or the complete compiler candidate cohort is unavailable, it
falls back atomically to the semantic anchor in every bin.

The control was frozen before the pending N8 runtime result:

- graph ID:
  `sha256:ce569e2575cfdc924004631dcf96a2d81077520c3198c8f78edde3a4405cf806`;
- control ID:
  `sha256:51363db826b95661fa1aaafd5617737eba3e3513b7e2b0a9d45c10b6fabe318a`;
- decision, hint, and control file SHA-256 values:
  `221a3692960539b8ed2a84aee7efa09ed6d1c04d6354a1857c28204f24f27c53`,
  `2b2577815e124c263a66b3f3a33e5375671087b08b51dd544376747cddc0260c`,
  and `5e630d76bf62c54622bf5062c60f83f25ea96f86187405d0c829fd602b0d66cb`;
- materialized binary SHA-256:
  `b71a98f2627a78d0d4869225fc1b2ac7034be06f44f6e18713a8437c19f32d5d`;
- materialized host/device IR SHA-256 values:
  `c341e2d2130642f6881d74b6903ba18281bb1f70f653ad63a5d56989b6060e65`
  and `5d0e1e64bd3db53307831b34735cdd7a5c8be460e3ecdf010d8e63abe0a7d1e6`.

The compiler independently revalidated the opaque option IDs and lowered the
four-bin decision to an explicit size-dispatch CFG: tree through 4 KiB, tree
through 256 KiB, four-way pipelined tree through 8 MiB, and eight-way
pipelined tree above 8 MiB.  Build provenance, dependency closure, command
records, host IR, and device proxy-ring IR all verify.  This comparator remains
valid whether the N8 runtime gate is positive or negative; it prevents a later
model result from being credited for a policy already expressible by a simple
compiler rule.

The five communication graphs now have the analogous, deliberately narrower
source-free control documented in
`COMMUNICATION_STRUCTURAL_CONTROLS.md`.  It uses only independent platform
calibration and compiler-derived group/launch facts to choose between uniform
proxy and uniform trigger candidates.  It cannot select mixed routes or
schedule transforms, and it falls back to the semantic anchor when a required
compiler fact is dynamic.  This is the appropriate falsifier for the richer
LLM input: relational reasoning is useful only if its validated plan improves
on both this simple rule and the unchanged compiler anchor.

## Current scientific support

Existing measurements establish that compiler-level communication decisions
can have large performance effects, but not yet that an LLM captures them:

- the compiler coalescing/placement graph has a `5.485628x` baseline-to-oracle
  gap, but only a noisy `1.017982x` best-uniform-to-oracle gap, so its
  preregistered LLM gate failed;
- the two-node collective capacity screen was negative;
- a four-node topology crossover favored the semantic baseline at two nodes
  and the hierarchical tree at four nodes, with `1.274x` equal-weight
  topology-selection headroom over the best single algorithm, but a trivial
  two-case rule fit both observations;
- the four-node hierarchy-pipeline confirmation was negative after its scout
  gain proved to be a 1 KiB outlier;
- the topology-matched eight-node hierarchy-pipeline scout and the guarded
  Jacobi producer-frontier A/B are the remaining preregistered headroom tests.

Thus the suite supports the paper's **method and action-space** claim now.  A
performance claim about LLM-guided optimization remains gated on stable oracle
headroom, held-out decisions, explicit content-addressed provider
authorization, and runtime measurement of each selected compiler plan.
The machine-derived per-entry state and fail-closed transitions are documented
in `LLM_CAPABILITY_READINESS.md`; prompt availability is never treated as
runtime or model evidence.

## Reproduction

From the repository root, after producing the already documented private
graphs:

```sh
python3 tools/gicc-passes/python/gicc_compiler_decision_suite.py emit \
  --communication jacobi=build_ofi/compiler_fact_coverage_20260904/portfolio/jacobi/group-graph.json \
  --communication minimod=build_ofi/compiler_fact_coverage_20260904/portfolio/minimod/group-graph.json \
  --communication mm_minimal=build_ofi/compiler_fact_coverage_20260904/portfolio/mm_minimal/group-graph.json \
  --communication mixed_lto=build_ofi/compiler_fact_coverage_20260904/portfolio/mixed_lto/group-graph.json \
  --communication loop_lto=build_ofi/compiler_fact_coverage_20260904/portfolio/loop_lto/group-graph.json \
  --collective collective_n8=build_ofi/compiler_collective_capacity_n8_hierpipe_v1_20260903/discovery/graph.json \
  --structural coalescing_placement=build_ofi/compiler_comm_plan_placement_current_20260904/generated/opportunity-graph.json \
  --prompt-dir build_ofi/compiler_decision_suite_20260904/prompts \
  --out build_ofi/compiler_decision_suite_20260904/suite.json
python3 tools/gicc-passes/python/gicc_compiler_decision_suite.py verify \
  --suite build_ofi/compiler_decision_suite_20260904/suite.json \
  --prompt-dir build_ofi/compiler_decision_suite_20260904/prompts
```

The current structural graph and four uniform materializations can be rebuilt
without overwriting the historical runtime bundle:

```sh
GICC_COMM_PLAN_OUT="$PWD/build_ofi/compiler_comm_plan_placement_current_20260904" \
  bash examples/proxy/build_compiler_comm_plan_placement.sh controls
```
