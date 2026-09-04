# Compiler-only LLM capability evaluation contract

Status: locally audited protocol only. No provider request has been frozen, no
provider call is authorized, no model has been invoked, and no scheduler job is
created by this protocol.

## Question and boundary

The experiment asks whether an LLM can use source-free compiler relations to
choose an existing compiler/LTO communication transformation with low regret
against a runtime-confirmed compiler oracle. It does **not** ask the model to
rewrite application source, generate IR, invent a transform, or assert its own
legality.

Every response is restricted to IDs already present in the frozen compiler
graph. The strict bridge and LTO pass independently revalidate those IDs before
materialization. Invalid responses fall back atomically to the unchanged
compiler anchor. Evaluation timings, winners, and oracle labels are never part
of a model prompt.

## Why this is not “LLM versus GBT”

The frozen scalar GBT and the relational LLM do not have the same input or
transformation authority. The GBT remains a useful route baseline, but it is
not an action-controlled model comparison.

The controlled experiment is instead the three LLM views already frozen for
every suite entry:

- `relational`: the compiler graph with explicit relations and candidate
  semantics;
- `descriptors`: the same compiler entities and candidate IDs without explicit
  relation edges;
- `opaque`: the same candidate IDs with candidate semantics removed.

All three use the same model, decoding settings, response schema, strict
validator, and selectable compiler IDs. Therefore only compiler information,
not action authority, changes across the ablation.

## Eligibility

Prompt availability is not eligibility. An entry can advance only after an
independently preregistered runtime campaign confirms stable compiler-owned
headroom and the readiness audit explicitly sets
`provider_protocol_permitted=true`. A positive exploratory scout advances only
to confirmation. A model-invisible candidate must first pass confirmation,
expand the compiler graph, and be refrozen before it can appear in a prompt.

For each eligible entry, the conditional protocol specifies 20 independent,
stateless responses per view. Calls are sequential and use a deterministic
response-index-major rotating view order: trial 1 uses
relational/descriptors/opaque, trial 2 starts with descriptors, and trial 3
starts with opaque before the cycle repeats. This is 60 conditional calls for
one eligible graph, but
still zero permitted calls until a separate request binds the exact prompt,
schema, provider executable/version/model/effort, and retry policy and the user
explicitly authorizes that content-addressed request.

## Scoring and runtime validation

The intention-to-treat analysis assigns the compiler-anchor fallback to every
invalid response and reports:

- invalid-output rate;
- oracle-normalized geometric-mean regret;
- regret relative to the deterministic compiler control;
- exact-oracle policy rate and mean decision-slot accuracy;
- modal-policy rate, number of unique policies, and policy entropy.

The preregistered primary representative is the modal policy for each view.
The best accepted policy among 20 responses is a separate, explicitly post-hoc
capability upper bound; it cannot be presented as typical model behavior.
Offline scoring against already held-out oracle labels is only a policy screen,
not runtime speedup evidence.

The pure aggregation rules are implemented in
`python/gicc_llm_capability_metrics.py`. Family-specific adapters remain
responsible for verifying the provider archive, strict-bridge output, graph
identity, and held-out runtime-control labels before calling it. In particular,
the modal-policy tie break is the lexicographically smallest canonical policy
ID, never the policy with better held-out regret; only the explicitly post-hoc
best-of-20 selector may inspect regret when choosing a representative.

`python/gicc_compiler_policy_bridge.py` provides the shared response boundary
for all three suite families. It delegates to the existing authoritative
collective, communication-group, or structural bridge, revalidates the emitted
compiler IDs, verifies atomic fallback, and normalizes the result into one
policy slot map. The compiler hint may contain private materializer identities,
so it is explicitly downstream-only and can never become provider input.

Every representative used in a performance claim must be deduplicated,
compiled through the strict LTO bridge, audited in materialized IR, and measured
against the semantic anchor and deterministic compiler control in paired
same-allocation `pdebug` blocks. At most one campaign job may be active or
queued at a time.

The paper claims remain separated:

- the implemented suite already supports a compiler-only method and action-space
  claim;
- the modal representative can support a stable-policy claim only after paired
  runtime validation;
- best-of-20 can support only a labeled capability-ceiling claim;
- value from relational language understanding requires improvement over the
  equal-authority `descriptors` and `opaque` views;
- portfolio generalization requires at least two independent eligible entries
  spanning at least two compiler decision families.

## Current audited result

`audit_compiler_llm_capability_protocol.py` re-verifies the frozen decision
suite and content identities of the readiness and input-separation audits. It
also checks every entry/graph/prompt binding and refuses any source-visible,
already-authorized, already-measured, unequal-authority, or model-invisible
configuration.

The current report is
`build_ofi/compiler_llm_capability_protocol_20260904/report.json`:

- protocol ID:
  `sha256:eea09d325ae91805353067e9151bd8b827a225e3133f130f00841aae1f44ebeb`;
- serialized report SHA-256:
  `51efc4e64ed6499d124cd3e63e41bf97174691de753dea3d71bf772d23dc5ac3`;
- frozen metrics implementation SHA-256:
  `fcc5858af4498ad39daa0efcdce4934fc7dd470e3e654189a8bb29a5d4c1250e`;
- frozen unified bridge SHA-256:
  `68b46946928c86e7da2b1607056cd28e3649b5b805c4a23733bd933a056e9b7f`;
- status: `blocked_no_runtime_eligible_entries`;
- suite entries: 7;
- runtime-eligible entries: 0;
- provider requests frozen: 0;
- provider calls authorized or made: 0;
- paper LLM-performance claim ready: false.

The report additionally binds the exact protocol auditor and all three
authoritative family bridges, so changing validation or fallback semantics
invalidates the protocol identity even when the graph bytes do not change.

`prepare_compiler_llm_capability_request.py` is the common request-freezing
boundary for all three decision families. It re-runs this protocol audit from
the readiness and input-separation evidence and refuses an entry unless its
status is exactly `provider_protocol_permitted`. A successful future freeze
copies only the source-free prompts, exact response schema, and system prompt
into the provider-visible bundle; the compiler graph remains a private
downstream bridge input. The generated request starts with zero permitted calls
and still requires a separate authorization bound to its exact request ID and
provider settings. It also content-addresses the common trial runner and
analysis implementation.

`run_compiler_llm_capability_trials.py` supports all three graph families via
the unified compiler-policy bridge. Before inspecting a provider executable it
re-verifies eligibility, every request-bundle byte, and a separate authorization
ID binding the exact prompts, schema, rotating order, provider version/model,
fresh-session setting, and transport-only retry limits. Calls are strictly
sequential. Every raw attempt, extracted response, private compiler hint, and
normalized policy is hash-bound and re-verifiable. A syntactically or
semantically invalid successful response is recorded once and receives the
atomic compiler-anchor fallback; it is never retried to search for a better
answer. With the current zero-eligible report, no request can be created and the
runner cannot reach a provider call.

`analyze_compiler_llm_capability_trials.py` closes the archive-to-metrics path
without calling the provider or compiler. It revalidates the complete 60-trial
archive and accepts held-out costs only through a content-addressed,
provider-invisible family adapter that covers every observed graph-bound
policy. It then applies the shared ITT fallback, modal-policy, entropy, regret,
and post-hoc best-of-20 rules and records the exact archive trials underlying
each representative. Offline screen results remain explicitly distinct from
paired runtime speedup evidence.

For `collective_n8`,
`collective/prepare_collective_llm_policy_screen.py` is the concrete family
adapter. The topology scout and three-allocation confirmation measure only the
three preregistered hierarchy/pipeline policies, while the model-facing graph
retains eight legal algorithms in each of four compiler-owned message bins
(4096 joint policies). The adapter therefore refuses to extrapolate those
three curves to the larger action space. After an exact 60-trial archive is
complete, `collective/continue_compiler_collective_llm_controls.sh` may submit
one and only one additional N8 `pdebug` allocation. Within that allocation it
runs all eight uniform compiler arms in three rotated sequential blocks. It
does not call a provider and does not inspect or modify application source.

The adapter replays the passed N8 confirmation, all archived responses, the
frozen compiler bundle, all raw full-catalog logs, their correctness results,
and the exact common allocation. It then assigns each graph-bound size policy
the pooled latency of its selected uniform arm at each measured message size.
Bin weights come only from the frozen compiler graph. The oracle is the
minimum-geometric-mean arm independently within each frozen bin; the anchor is
the compiler's atomic semantic fallback; and the deterministic comparator is
the preregistered source-free topology/pipeline heuristic. Missing algorithms,
sizes, blocks, raw logs, a non-`pdebug` job, an incomplete archive, or a failed
confirmation all stop the screen. These composed costs select policies for
later validation and are explicitly not runtime measurements of the composed
LLM policies.

This negative readiness result is important: it prevents rich prompts alone
from being counted as LLM evidence. The existing N8 collective, Jacobi
producer-fission, guarded early-trigger, and reused-descriptor campaigns must
finish their serial runtime gates before the report can advance.

## Historical feasibility versus the upgraded claim

The earlier `llm-zero-shot-v1` archive is now independently re-audited in
`HISTORICAL_COMPILER_LLM_CEILING.md`. Its 20 source-free responses were all
accepted, collapsed to five legal route policies, and were materialized by LTO
from unchanged application source. The frequency-selected policy was 1.021570x
over compiler default with paired interval [0.980814, 1.048598], so it does not
support a stable speedup claim. The post-hoc best policy was observed at
1.054465x with interval [1.038120, 1.071642], but appeared once in 20 responses,
has only four allocation replicates (exact sign p=0.125), and was selected after
comparing all five runtime campaigns.

That archive therefore supplies compiler-only feasibility and an explicitly
post-hoc capability ceiling. It does not supply the relational/descriptors/
opaque information ablation, cross-program generalization, or current `pdebug`
runtime evidence. Every historical job ledger names `pci`, and the audit leaves
the upgraded suite at zero eligible graphs and zero permitted provider calls.
The audit ID is
`sha256:0ae3dd141caff01b179788483c13b0f6d1d3081ee66cdb8f9b0c5397784a1aba`.

## Reproduction

```sh
python3 tools/gicc-passes/experiments/audit_compiler_llm_capability_protocol.py emit \
  --suite build_ofi/compiler_decision_suite_20260904/suite.json \
  --prompt-dir build_ofi/compiler_decision_suite_20260904/prompts \
  --readiness build_ofi/compiler_llm_readiness_20260904/report.json \
  --input-separation build_ofi/compiler_input_separation_20260904/report.json \
  --out build_ofi/compiler_llm_capability_protocol_20260904/report.json

python3 tools/gicc-passes/experiments/audit_compiler_llm_capability_protocol.py verify \
  --suite build_ofi/compiler_decision_suite_20260904/suite.json \
  --prompt-dir build_ofi/compiler_decision_suite_20260904/prompts \
  --readiness build_ofi/compiler_llm_readiness_20260904/report.json \
  --input-separation build_ofi/compiler_input_separation_20260904/report.json \
  --report build_ofi/compiler_llm_capability_protocol_20260904/report.json
```
