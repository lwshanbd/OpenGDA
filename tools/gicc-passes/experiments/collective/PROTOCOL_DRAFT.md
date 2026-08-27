# Compiler-only collective policy capacity protocol (draft)

Status: **draft until the pdebug safety qualification completes**. This file
does not authorize a provider call or a scheduler job by itself.

## Scientific question

Can an LLM use a source-free, relational compiler description of collective
semantics, topology, communication graphs, synchronization, pipeline depth,
and resource constraints to choose a non-local compiler policy that approaches
the measurable oracle over a materially larger action space than per-site
route selection?

The experiment is a capability upper-bound study, not a claim that an LLM is
universally a better latency regressor. The claim is supported only if the
compiler action space has non-trivial headroom and different algorithms win in
different compiler-owned message regions.

## Immutable boundary

- The application source calls one neutral semantic collective anchor.
- The model receives no source, source location, function name, IR, or
  materializer target.
- The model-facing compiler graph normalizes catalog candidates into entities
  and exposes explicit message-order and shared-structure relations; it is not
  a prose rendering of a per-size scalar feature row.
- The model response contains existing compiler-generated `option_id` values,
  bounded confidence, and a short rationale only.
- The model cannot name or create an algorithm, threshold, function, dispatch,
  code fragment, IR fragment, or legality assertion.
- The bridge rejects the complete response atomically on any stale, unknown,
  incomplete, or extra field and falls back to the semantic anchor.
- LLVM independently recomputes opportunity, target, option, and composite
  plan IDs; checks catalog membership, family, semantic contract, exact ABI,
  and the complete threshold list; and only then rewrites the call.
- Every materialized call is audited through
  `gicc.collective.candidate_id` and `gicc.collective.target_id` IR metadata.
- No LLM output edits application or catalog source.

## Frozen source inputs for the current draft

- `compiler_collective_eval.cpp`:
  `391b4888f88039975f3d990802763b3694b185a20d2b23c2419d23d4b0ed19c8`
- `compiler_collective_catalog.hpp`:
  `8958731e3f5b2f6fb6b2380437f62363c52dddcc8d8a32646ff6c9133ab5f77f`
- implementation commit: `4b98393`
- target: Tioga MI250X GCD + Slingshot-11/CXI, CPU-proxy collective catalog
- scheduler queue: `pdebug` only
- submission rule: exactly one scheduler job active or queued at a time
- Gate-B two-node compiler profile:
  `tioga-mi250x-cxi-collective-capacity-n2.json`

The source and catalog hashes must be rechecked before every build or runtime
phase. A mismatch invalidates, rather than silently updates, this draft.
Every graph used for model reasoning must describe the exact runtime topology.
A graph or response generated for two nodes is invalid evidence at eight nodes,
and vice versa. Safety results may motivate a later topology, but are never
silently extrapolated across topology profiles.
The two-node profile may expose only hash-pinned, pre-existing primitive link
and proxy cost measurements whose scope is declared and whose data contains no
collective action labels. Gate-B timings, winners, and oracle choices are
evaluation-only and must never enter a model prompt.
Graph generation must content-verify every declared primitive-calibration
artifact; a hash string in the profile without the matching local artifact is
not sufficient provenance.

## Compiler action space

The current source-free graph has one semantic opportunity, four fixed message
regions (`<=4 KiB`, `4 KiB..256 KiB`, `256 KiB..8 MiB`, `>8 MiB`), and eight
compiler targets:

1. semantic `baseline_auto` anchor;
2. flat double tree;
3. flat double tree with four pipeline chunks;
4. flat double tree with eight pipeline chunks;
5. locality-aware ring;
6. hierarchical ring;
7. hierarchical direct reduce-scatter/exchange/all-gather;
8. hierarchical double tree.

This gives `8^4 = 4096` joint size policies without letting the model author a
single transformation. Every wrapper has the same compiler-verified ABI and a
runtime-guarded fallback for dynamic preconditions.

## Sequential gates

No later gate runs unless the previous gate passes. A gate submission is one
`pdebug` job and must finish before another is submitted.

### Gate A: diagnostic runtime smoke

- topology: 2 nodes, 8 ranks/node, one rank/GCD;
- sizes: 1 KiB and 4 KiB;
- arm: `baseline_auto` only;
- one correctness call, no warmup, one timed call;
- pass: exit 0, `COLLECTIVE_DONE total_errors=0`, two unique result rows, no
  timeout/hang.

This is diagnostic and is not paper evidence.
The already submitted baseline smoke predates topology-specific graph freezing;
because it exercises only the semantic anchor and invokes no model, it may
qualify runtime safety but cannot qualify a graph or model prompt.

### Gate B: catalog safety qualification

- same 2-node topology;
- regenerate the graph, prompt, controls, manifest, and materialized IR from
  the frozen two-node profile before the first Gate-B submission;
- all eight uniform compiler arms, one job at a time;
- sizes: 1 KiB, 4 KiB, 8 KiB, 64 KiB, 256 KiB, 1 MiB, 4 MiB, 8 MiB, 16 MiB;
- one correctness call per size, one warmup, three timed calls;
- each arm must exit 0 with zero errors at every size and no timeout.

An arm that fails is masked in a new content-addressed platform profile with
the observed reason. It is never silently retained as a selectable option.
After masking, discovery, graph generation, controls, builds, and IR audit are
rerun from the unchanged source.

### Gate C: headroom screen

Use the Gate-B timings only as a screen, not a final estimate. Continue only if
all of the following hold:

- at least two distinct compiler targets win different measured sizes;
- the baseline-to-pointwise-oracle geometric-mean ratio is at least `1.05`;
- at least one measured size has at least `1.10x` baseline headroom;
- the best compiler-bin policy is not identical to `baseline_auto` in all four
  regions.

Failure narrows the paper claim to a negative capacity result; it does not
trigger more model calls or a larger scheduler sweep.
The analysis artifact must record every observed value, threshold, and Boolean
result for these four criteria; a narrative judgment is not sufficient.

### Gate D: confirmatory compiler controls

Only after Gate C passes:

- three paired replicates per surviving uniform arm;
- one arm per pdebug job, sequentially submitted;
- rotate arm order between replicates before submission;
- warmup 2, timed calls 7 per size;
- report rank-max median latency per job;
- analyze per-size winners, best uniform target, pointwise oracle, and the
  compiler-bin oracle restricted to the four frozen regions;
- paired bootstrap confidence intervals over replicate-level log ratios;
- freeze graph, manifest, hints, source/catalog hashes, materialized-IR hashes,
  binary hashes, job IDs, node lists, and raw logs.

The paper's action-space headroom is the compiler-bin oracle relative to the
semantic baseline and best uniform target, not an unconstrained source oracle.

### Gate E: model capability evaluation

Requires separate explicit authorization before sending the new collective
graph to any external provider. The previous authorization for cleaned
Minimod/Jacobi prompts does not cover this graph.

The prompt profile and runtime topology must match exactly. A later eight-node
evaluation therefore requires its own eight-node qualification and frozen
graph; the two-node screen cannot serve as its performance control.

If authorized, use one frozen graph and prompt per evaluated topology. Compare:

- semantic compiler baseline;
- best uniform compiler target;
- measured compiler-bin oracle;
- deterministic compiler heuristic;
- GBT over its declared scalar compiler features;
- LLM over the richer source-free relational compiler graph.

The goal is not to rank model families in the abstract. The capability claim
requires the LLM policy to exploit the richer compiler input: report distance
to the restricted oracle, baseline speedup, bin-choice accuracy, invalid-output
rate, and stability over 20 independent responses. Every accepted response is
compiled through the same LTO verifier; rejected responses execute the anchor.

Subject to explicit authorization for every exact prompt hash, use the same
model and decoding settings for three compiler-input views: primary
`relational`, `descriptors` without explicit relation edges, and `opaque`
without candidate semantics. All three expose the same option IDs and response
schema. This ablation tests the value of compiler semantic/relational context;
it does not grant any view additional transformation authority and is not a
general model-family ranking.

Before scheduling any model-selected plan, score all raw responses through the
strict bridge against the frozen compiler controls. Record response/prompt
hashes, invalid-output rate, exact-oracle rate, mean bin accuracy, policy
stability, and distance to the compiler-bin oracle. This is explicitly a
counterfactual screen used to deduplicate policies; it is never reported as
runtime speedup. Every unique policy used for a performance claim must still
be compiled, IR-audited, and measured on the matching pdebug topology.

## Stop conditions

- Any source/catalog hash mismatch.
- Any non-`pdebug` queue request.
- More than one active or queued experiment job.
- Any correctness error, hang, stale graph, missing compiler metadata, or
  materializer mismatch.
- Any prompt containing a source fragment, source location, function name,
  target ID, or IR.
- Any provider call without new explicit authorization for the exact frozen
  prompt hash.
