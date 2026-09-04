# Source-free compiler decision suite

Status: locally frozen capability inventory; no model or provider call and no
runtime-performance claim.

This suite answers a narrower question than model accuracy: do we have a
replayable set of non-source compiler decisions whose inputs are materially
richer than the scalar rows used by the original path GBT?  The current answer
is yes.  The frozen index contains six **independent** compiler/LTO decision
tasks from two decision families:

| Entry | Compiler decision | Independent policy count | Runtime status |
|---|---|---:|---|
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
  `sha256:0c6392c716646f0de444ca4916b115ab1e81aa118b45641ca4c8c85a01e214c2`
- serialized suite SHA-256:
  `59afd8a18469afceef4dc9def741a0efde40d1e4adead42a8a3d21a719a50602`
- decision-family counts: one collective size-policy graph and five
  communication route/schedule graphs;
- selectable compiler IDs: 32 collective option IDs over four slots and 32
  communication candidate IDs over five separate tasks;
- prompt views: `relational`, `descriptors`, and `opaque` for every entry.

The suite manifest content-addresses every private graph, selectable-ID set,
model view, and rendered prompt.  Its verifier checks the suite and entry IDs,
all prompt hashes and byte counts, and the independence declaration.  The tool
has no provider, scheduler, compiler, or source-edit code path.

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

An identity-leak audit over all 18 rendered prompts found no application or
kernel name, source suffix, site ID, materializer binding, or provenance path.
No prompt is authorized for external transmission by this local freeze.

## Current scientific support

Existing measurements establish that compiler-level communication decisions
can have large performance effects, but not yet that an LLM captures them:

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
  --prompt-dir build_ofi/compiler_decision_suite_20260904/prompts \
  --out build_ofi/compiler_decision_suite_20260904/suite.json
python3 tools/gicc-passes/python/gicc_compiler_decision_suite.py verify \
  --suite build_ofi/compiler_decision_suite_20260904/suite.json \
  --prompt-dir build_ofi/compiler_decision_suite_20260904/prompts
```
