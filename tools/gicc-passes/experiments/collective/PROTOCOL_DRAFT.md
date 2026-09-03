# Compiler-only collective policy capacity protocol (draft)

Status: **draft until the pdebug safety qualification completes**. This file
does not authorize a provider call or a scheduler job by itself.

The content-addressed v2 offline bundle remains recorded, unchanged, in
`FROZEN_V2_MANIFEST.json` as a historical diagnostic. Its lower builds ran the
independent per-transfer lowering pipeline without a matching transfer hint;
that pipeline erased the catalog kernels' proxy operations. It also borrowed
three runtime LTO objects from a different CMake target. Consequently v2 is
superseded and is ineligible for runtime or model evidence even if its stored
top-level hashes still verify. A v3 freeze requires the collective-only scope
and same-build provenance described below.

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
- transitive catalog body `examples/proxy/coll_common.hpp`:
  `98aff956bbc63d36b74935b21b0ac825ca78c0ae3695888a9a6d5583dc468429`
- collective-only pipeline implementation commit: `5a47357`
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

## Same-build provenance required for v3

Every discover and lower build must emit and pass verification of a
`gicc-collective-build-provenance-v1` manifest. The manifest freezes:

- the benchmark, catalog, compiler, compiler configuration, pass plugin,
  build/evaluation tools, and (for lower builds) the exact compiler hint;
- compiler-generated dependency files and every repository translation-unit
  dependency they name, including the catalog's transitive `coll_common.hpp`;
- exact compiler/linker argument vectors and the lowering-scope environment;
- direct linked libraries;
- inventory, host and device IR, device-IR audit, compile/link logs, every LTO
  object, and the final binary.

The three runtime translation units are rebuilt into each arm's own output
directory without a planning pass loaded. The verifier rejects runtime objects
whose locator is not in that same build. It also requires
`GICC_COLLECTIVE_ONLY=1`, an unset ordinary `GICC_HINT_IN`, and the exact
collective hint for lower builds. This manifest is a prerequisite for v3
freezing and for every Gate-B-or-later runtime result.

After Gate A passes, `prepare_compiler_collective_v3.sh` performs discovery,
three prompt-view renders, all uniform builds, the mixed-policy canary, the
complete action-space audit, and the final freeze sequentially without calling
Flux or a model. It refuses an existing output directory and refuses to start
from a monitor that is not a clean `pdebug` Gate-A pass. The final
`gicc-collective-offline-freeze-v2` manifest also preserves and cross-checks
the raw Gate-A stdout/stderr, every nested build-provenance manifest, and the
union of source dependencies across builds.

`continue_compiler_collective_through_gate_c.sh` may wait on that Gate-A
monitor and execute the deterministic continuation. It has no provider path:
after the offline freeze it submits one `pdebug` Gate-B arm, waits for its
clean audited completion, and only then submits the next. It stops immediately
on any failed or stale monitor and stops unconditionally after writing the
Gate-C analysis; it cannot enter confirmatory or model evaluation.

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

Before any model or Gate-B runtime, compile one deterministic, compiler-owned
mixed-policy canary from the frozen graph. It must select non-anchor options in
all four message regions, lower to a real size-policy CFG in the unchanged
benchmark, and expose the exact composite candidate ID plus every branch target
through IR metadata. This is an offline materializer test, not performance
evidence and not a model output.
Also enumerate the complete declared joint action space through the strict
bridge. Every action must be accepted, every materialized composite candidate
ID must be unique, and the content hash of the complete ID set must be frozen.
This proves compiler-plan capacity only; it is not a performance result.

### Gate A: diagnostic runtime smoke

- build preflight: `GICC_COLLECTIVE_ONLY=1`, no `GICC_HINT_IN`, compiler logs
  report `scope=collective`, and the same-build device IR retains the
  proxy-ring operations used by the catalog kernels;
- topology: 2 nodes, 8 ranks/node, one rank/GCD;
- sizes: 1 KiB and 4 KiB;
- arm: `baseline_auto` only;
- one correctness call, no warmup, one timed call;
- pass: exit 0, `COLLECTIVE_DONE total_errors=0`, two unique result rows, no
  timeout/hang.

This is diagnostic and is not paper evidence.
The currently queued scope-fix smoke (`f5tmvdqmC5sd`) was compiled with
`GICC_COLLECTIVE_ONLY=1` and passed the device-IR reservation audit, but it
predates the v3 same-build provenance rule. Because it exercises only the
semantic anchor and invokes no model, it may qualify the repaired compiler
pipeline's runtime safety but cannot qualify a v3 graph, binary, or model
prompt.

### Gate B: catalog safety qualification

- same 2-node topology;
- regenerate the graph, prompt, controls, manifest, and materialized IR from
  the frozen two-node profile before the first Gate-B submission;
- require a verified same-build provenance manifest for every arm;
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

- three paired allocation blocks covering every surviving uniform arm;
- one two-node `pdebug` batch allocation at a time; within that allocation,
  execute all arms sequentially on the exact same nodes;
- rotate arm order by one position between allocation blocks;
- warmup 2, timed calls 7 per size;
- report rank-max median latency per job;
- analyze per-size winners, best uniform target, pointwise oracle, and the
  compiler-bin oracle restricted to the four frozen regions;
- exact paired bootstrap confidence intervals over the three complete
  same-allocation replicate-block log ratios;
- freeze graph, manifest, hints, source/catalog hashes, materialized-IR hashes,
  binary hashes, job IDs, node lists, and raw logs.

This same-allocation design is the meaning of "paired": merely assigning arm
jobs the same replicate number while allowing different nodes would not
support a paired confidence interval. `continue_compiler_collective_gate_d.sh`
waits for the machine-readable Gate-C pass, submits only three sequential
batch allocations, audits every per-arm log and exact Flux resource set, and
stops after `gicc-collective-confirmatory-analysis-v1`. It cannot invoke a
provider or enter Gate E.

The paper's action-space headroom is the compiler-bin oracle relative to the
semantic baseline and best uniform target, not an unconstrained source oracle.

### Gate E: model capability evaluation

Gate E stops without a provider call unless Gate D's preregistered
compiler-bin policy has a positive speedup and its paired 95% interval excludes
one. `prepare_compiler_collective_gate_e.py` regenerates Gate D from all three
raw allocation monitors before freezing an authorization request; it has no
provider or scheduler call path.
`run_compiler_collective_model_trials.py` additionally requires a
content-addressed authorization that binds the request, every prompt, the
system prompt and response schema, provider executable/version/model/effort,
and bounded transport retries. It runs calls sequentially with no tools or
session persistence, archives each raw envelope before bridge validation, and
cannot be launched by the Gate-A-through-D controllers.

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

The existing path-selection GBT is not yet an eligible collective comparator:
its labels distinguish proxy/IPC dispatch actions, not the eight collective
algorithms in this graph. A collective GBT requires a disjoint, pre-evaluation
training set. Gate-B or Gate-D labels must not be reused to train it; doing so
would turn the evaluation oracle into leaked training data. Until such a
dataset exists, report the GBT comparison as unavailable rather than fitting a
misleading model.

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
For each view, preregister the modal accepted policy as the stability estimate
and the best control-screen policy among its 20 responses as an explicitly
post-hoc capability upper bound. Deduplicate these representatives across
views before compilation and runtime measurement.

### Post-v3 exploratory topology scout

The completed two-node v3 screen is closed as a negative capacity result; its
thresholds are not relaxed and its Gate D/E remain stopped. Compiler source
inspection gives a separate, preregistered hypothesis: `baseline_auto` already
uses the strongest two-node direct/ring path, while `hierarchical_double_tree`
is designed to gain its logarithmic node-level advantage only at four or more
nodes. Test that hypothesis with exactly one four-node `pdebug` batch allocation
running the two existing frozen binaries sequentially on the same nodes, over
the nine frozen sizes, with one warmup and three timed calls.

This scout deliberately uses a two-node graph's binaries at four nodes, so it
is exploratory evidence only. A topology-matched v4 graph is warranted only if
both the baseline-to-pointwise geomean is at least `1.05` and one size has at
least `1.10x` headroom, with both arms winning at least one size. Otherwise no
four-node v4 or provider call is launched.

## Stop conditions

- Any source/catalog hash mismatch.
- Any non-`pdebug` queue request.
- More than one active or queued experiment job.
- Any correctness error, hang, stale graph, missing compiler metadata, or
  materializer mismatch.
- Any absent, stale, or unverifiable same-build provenance manifest.
- Any prompt containing a source fragment, source location, function name,
  target ID, or IR.
- Any provider call without new explicit authorization for the exact frozen
  prompt hash.
