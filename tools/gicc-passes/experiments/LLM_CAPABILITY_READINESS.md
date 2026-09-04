# Compiler-only LLM capability readiness

Status: derived readiness audit; no provider call, no provider authorization,
and no application-source modification.

The upgraded paper question has three distinct levels that must not be
collapsed into one claim:

1. **Compiler expressibility:** can LTO expose legal communication decisions
   with materially different generated code and runtime behavior?
2. **Stable oracle headroom:** does a preregistered compiler-owned action space
   beat its strongest simple compiler control reproducibly?
3. **LLM use of richer compiler context:** among graphs that pass level 2, can
   a source-free relational LLM policy approach the held-out compiler oracle
   and beat the deterministic compiler rule?

The seven-entry decision suite establishes level 1 across communication
route/schedule, communication coalescing/trigger placement, and collective
algorithm/size policy. It does not by itself establish levels 2 or 3.

## Current machine-derived result

`audit_compiler_llm_readiness.py` verifies the suite and prompt bytes, binds
each evidence file, checks the historical/current placement graph equivalence,
and fails closed at every transition. Its current output is:

| Entry | Current status | Allowed next stage |
|---|---|---|
| `coalescing_placement` | `closed_negative` | no model for this graph |
| `collective_n8` | `awaiting_scout` | wait for the existing N8 `pdebug` scout |
| `jacobi` | `awaiting_predecessor` | wait for N8, then the one queued-by-controller scout |
| `minimod` | `runtime_labels_missing` | establish a preregistered compiler oracle first |
| `mixed_lto` | `runtime_labels_missing` | establish a preregistered compiler oracle first |
| `mm_minimal` | `awaiting_predecessor` | run the frozen guarded-trigger scout after the serial campaign |
| `loop_lto` | `awaiting_predecessor` | run the frozen reused-descriptor scout after the serial campaign |

The report has readiness ID
`sha256:dfca08a9866bb433a13aa625500f5bf07b7cbb11399c0c43d341589bb44a396b`
and serialized SHA-256
`0b12758ff250adca5aa45f9ab1f902b2e7918d48df1479f0b3e4c0ee879eb462`.
It reports zero provider-protocol-permitted entries, zero authorized provider
calls, zero measured LLM policies, and
`paper_llm_performance_claim_ready=false`.

The placement result remains scientifically useful even though its LLM gate
failed. It demonstrates a `5.485628x` trigger-batch-to-factorized-oracle gap,
so compiler transformations can matter greatly. The strongest uniform arm is
only `1.017982x` behind the factorized oracle, however, and its paired 95%
interval crosses one. The experiment therefore supports compiler action-space
capacity but cannot support an LLM-selection benefit.

## Fail-closed transitions

- A negative preregistered runtime gate permanently closes that graph to a
  post-hoc model experiment.
- A positive exploratory N8 or producer-fission scout advances only to a
  separately frozen confirmatory compiler-oracle experiment. It never
  authorizes a provider request.
- For the already model-visible N8 collective graph, a positive confirmation
  advances to `provider_protocol_permitted` only after the readiness audit
  replays all three raw N8 allocation monitors and verifies the exact graph,
  transition, `pdebug` contract, artifact hashes, and confirmation gate. A
  positive result permits only request freezing; provider calls remain zero
  until a separate content-addressed authorization exists.
- The guarded early-trigger scout is subject to the same rule and remains
  absent from model-visible candidates while it waits or until confirmation.
- A positive producer-fission or guarded-trigger confirmation is also replayed
  from its three raw `pdebug` allocation monitors. Because both candidates are
  absent from the current frozen suite, a pass advances only to
  `graph_expansion_required`; it cannot make the current request eligible.
  The compiler graph, prompts, suite, and readiness evidence must all be
  regenerated and content-addressed before either candidate reaches a model.
- Producer fission now has those two additional machine-checked transitions.
  The graph-expansion preparer binds the passed confirmation back to the exact
  compiler features, template, dossier, and masked graph, preserves all nine
  existing candidate IDs, and exposes only the confirmed two-phase candidate.
  Its output is still `suite_refreeze_required`. A second preparer verifies the
  old suite and prompts, replaces only the Jacobi entry, and preserves every
  other entry. Only when the readiness audit replays both manifests and sees
  the refrozen suite does the entry reach `provider_protocol_permitted`.
  Provider-call authorization remains false at that state.
- The reused-loop-descriptor scout is a graph-expansion oracle. Its candidate
  remains explicitly model-invisible even after a positive scout; only an
  independently frozen confirmation may justify rebuilding the compiler graph.
- A graph without runtime labels remains a capability-inventory entry, not a
  performance test.
- Only a positive confirmatory headroom result may permit preparation of an
  exact content-addressed provider request. Sending that request still
  requires separate user authorization binding the prompt, system prompt,
  response schema, provider/model settings, and call count.
- Any later model response may contain only graph-bound compiler option IDs.
  The compiler independently validates and materializes them during LTO; no
  response may edit or generate application source.

This separation is also the correct interpretation of the richer LLM input.
The relational view contains compiler-proved topology, dependence, ordering,
resource, and cross-opportunity relations that the original per-row scalar GBT
does not contain. That creates a larger *reasoning surface*, not automatic
performance headroom. Runtime gates determine whether each reasoning problem
is worth evaluating at all.

## Retrospective Minimod route-only evidence

The older Minimod campaign contains a useful compiler-only subset, but it is
not promoted into the readiness table. Its `default`, `trigger`, and `proxy`
binaries differ through LTO route hints, and the strict historical parser can
revalidate all raw logs, route counters, checksums, and paired allocations.
Holding the hand-written schedule fixed gives the following descriptive
best-uniform-route versus per-topology route-oracle gaps:

| Grid | Fixed schedule | Oracle headroom | Largest cell | Route winners |
|---:|---|---:|---:|---|
| 400 | overlap | `1.007268x` | `1.022787x` | default, proxy |
| 800 | overlap | `1.007272x` | `1.023226x` | default, proxy |
| 400 | serial | `1.017184x` | `1.083567x` | default, proxy, trigger |
| 800 | serial | `1.015597x` | `1.069086x` | default, proxy |

Four of five topology winners repeat across the two grids under each fixed
schedule (eight of ten comparisons in total). This is real evidence that an
LTO route decision can be context-dependent, but its aggregate opportunity is
only about 0.7% under the faster overlap schedule. It therefore strengthens
the motivation for
compiler-level selection while also showing why richer scheduling candidates,
not merely a more powerful selector, are necessary.

The compatibility gate remains closed: historical facts use schema 4 while
the current graph uses schema 6, only the three uniform-route candidates map
semantically, six mixed-route candidates have no measurements, and the
serial/overlap axis was hand-written. The deterministic report consequently
sets `current_graph_runtime_labels_complete=false`,
`llm_performance_measured=false`, and
`provider_protocol_permitted=false`. Its result ID is
`sha256:3af3c14c9ecfb95653b2691aee8bdca2dd9b45fc968cf14aa17171390e1fc008`;
the serialized report SHA-256 is
`a02c3531a95b861042261a9a2d8587dd8217b9e52996c710bc2f3e60e3423706`.

## Retrospective LTO trace feasibility evidence

An independent historical campaign compared a loop of handwritten host DWQ
enqueues with the host trace synthesized by LTO from the communication loop in
device IR. The strict retrospective audit re-parses 40 raw logs: two batch
sizes, ten repeated process launches per arm and batch, and sixteen message
sizes. All LTO logs have exact enqueue counts and none of the logs contains a
correctness-failure marker.

Using the geometric mean of paired median-time ratios over message sizes and
then trials, handwritten/LTO was `0.951514x` at batch 4 (LTO was `5.0957%`
slower) and `1.030673x` at batch 64 (LTO was `2.9760%` faster). The reversal is
useful engineering evidence: LTO synthesis is in the same performance regime
as the handwritten fast path, but its overhead is workload/batch sensitive and
must be evaluated rather than assumed away. It also motivates compiler-visible
batch and schedule decisions beyond a static route label.

This is deliberately not confirmatory evidence. The driver intended all trials
to share one allocation, did not preserve the allocation identity, always ran
the handwritten arm before LTO, and did not content-address the executed
binaries. The result therefore cannot establish equivalence, populate current
decision-graph labels, or support an LLM performance claim. Its result ID is
`sha256:7e9be825e11ed89445a31c2fd67c14dbe73820f7b401f9e0340a59d38ecb5c0a`;
the serialized report SHA-256 is
`cc97f0aecff14af15786ed24546266558a5b3519270c047ab2943544f4d02a3c`.

The retrospective has now been converted into a prospective, compiler-only
oracle. LTO proves that every descriptor field is loop invariant, replaces six
runtime-sized host arrays with one scalar descriptor, and calls a runtime
helper that still enqueues exactly the original number of ordered network
operations. The application source and optimized device kernel are identical
between arms. The frozen N2 `pdebug` scout uses six balanced same-allocation
pairs at batches 4 and 64, validates all 24 enqueue counts, and is queued only
as a local successor to the existing serial campaign. Commit `06ab7fd` freezes
its protocol and fail-closed monitor. No scout result exists yet, so the
candidate is not present in the model-visible graph.

The readiness report is now consumed by the suite-level, provider-free
`audit_compiler_llm_capability_protocol.py` audit. That layer preregisters the
equal-authority information ablation and regret/stability metrics but freezes
no provider request while this report exposes zero eligible entries. See
`LLM_CAPABILITY_EVALUATION.md` for the distinction between the modal stable
policy, the post-hoc best-of-20 capability ceiling, and paired runtime evidence.

## Reproduction

From the repository root:

```sh
python3 tools/gicc-passes/experiments/audit_compiler_llm_readiness.py emit \
  --suite build_ofi/compiler_decision_suite_20260904/suite.json \
  --prompt-dir build_ofi/compiler_decision_suite_20260904/prompts \
  --placement-summary docs/experiments/compiler-comm-plan-capacity/runs/compiler-comm-plan-placement-v1/summary.json \
  --placement-historical-graph build_ofi/compiler_comm_plan_placement/generated/opportunity-graph.json \
  --placement-current-graph build_ofi/compiler_comm_plan_placement_current_20260904/generated/opportunity-graph.json \
  --collective-state build_ofi/compiler_collective_hierpipe_n8_scout_20260903.state \
  --collective-analysis build_ofi/compiler_collective_hierpipe_n8_scout_20260903/analysis.json \
  --collective-confirmation-state build_ofi/compiler_collective_n8_confirmation_20260904.state \
  --collective-confirmation-analysis build_ofi/compiler_collective_n8_confirmation_20260904/analysis.json \
  --producer-state build_ofi/producer_fission_oracle_scout_7687377_20260904.state \
  --producer-analysis build_ofi/producer_fission_oracle_scout_7687377_20260904/analysis.json \
  --producer-confirmation-state build_ofi/producer_fission_confirmation_7687377_20260904.state \
  --producer-confirmation-analysis build_ofi/producer_fission_confirmation_7687377_20260904/analysis.json \
  --producer-expansion-manifest build_ofi/producer_fission_graph_expansion_20260904/manifest.json \
  --producer-refreeze-manifest build_ofi/producer_fission_suite_refreeze_20260904/manifest.json \
  --guarded-state build_ofi/guarded_early_trigger_scout_77897d9_20260904.state \
  --guarded-analysis build_ofi/guarded_early_trigger_scout_77897d9_20260904/analysis.json \
  --guarded-confirmation-state build_ofi/guarded_early_trigger_confirmation_77897d9_20260904.state \
  --guarded-confirmation-analysis build_ofi/guarded_early_trigger_confirmation_77897d9_20260904/analysis.json \
  --reused-state build_ofi/reused_loop_descriptor_scout_aff76f9_20260904.state \
  --reused-analysis build_ofi/reused_loop_descriptor_scout_aff76f9_20260904/analysis.json \
  --out build_ofi/compiler_llm_readiness_20260904/report.json
```

Use `verify` with the same inputs and
`--report build_ofi/compiler_llm_readiness_20260904/report.json` to prove that
the report still matches the current external state. When a monitored runtime
stage changes, `verify` deliberately rejects the stale report and `emit`
derives a new one.

The retrospective route-only audit is reproduced separately with:

```sh
python3 tools/gicc-passes/experiments/analyze_minimod_route_only_retrospective.py emit \
  --dataset 400=docs/experiments/minimod-paper/paper-v1-grid400-analysis/measurements.csv \
  --dataset 800=docs/experiments/minimod-paper/paper-v1-analysis-n1-n4/measurements.csv \
  --legacy-features build_ofi/minimod_paper/meta/standard-features/features.json \
  --current-features build_ofi/compiler_fact_coverage_20260904/minimod_disjoint/meta/features.json \
  --graph build_ofi/compiler_fact_coverage_20260904/portfolio/minimod/group-graph.json \
  --binary-sha256 build_ofi/minimod_paper/binary-sha256.txt \
  --out build_ofi/minimod_route_only_retrospective_20260904/report.json
```

The historical LTO trace audit is reproduced with:

```sh
python3 tools/gicc-passes/experiments/analyze_lto_trace_retrospective.py emit \
  --input-dir build_ofi/variance_out \
  --driver build_ofi/variance_out/run_alloc.sh \
  --out build_ofi/lto_trace_retrospective_20260904/report.json
python3 tools/gicc-passes/experiments/analyze_lto_trace_retrospective.py verify \
  --input-dir build_ofi/variance_out \
  --driver build_ofi/variance_out/run_alloc.sh \
  --out build_ofi/lto_trace_retrospective_20260904/report.json
```
