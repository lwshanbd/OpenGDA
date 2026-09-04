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

`COMPILER_ACTION_AUTHORITY.md` verifies the output-side difference rather
than merely asserting it: the frozen GBT emits one of three route labels,
whereas the suite includes structural-transform candidate IDs and collective
algorithm/size-policy option IDs.  That wider authority comes from the
compiler interface, not from LLM identity; a structured ML baseline could use
the same interface.  Consequently, no result may attribute the interface's
extra actions to language reasoning.

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
- exact-oracle and best-of-20 rates must be interpreted against the analytic
  20-draw chance calibration in `LLM_SAMPLING_NULL.md`; a single oracle hit is
  already rare in a 4096-policy graph but expected in the 2–9-policy graphs;
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

The post-N8-cancellation report is
`build_ofi/compiler_llm_capability_protocol_after_n8_cancel_20260904/report.json`:

- protocol ID:
  `sha256:fa2ce44d7e41702b6ec9b3894afb2087892c46e2abb0a4d4ab074ffce0ce2956`;
- serialized report SHA-256:
  `9ccc45d6a623b7c56e8bf54cc1e117726d1bd0d77a68fb6a47b5e4a89acaed46`;
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
the readiness, input-separation, and analytic sampling-null evidence and
refuses an entry unless its status is exactly
`provider_protocol_permitted`. A successful future freeze
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
each representative. It also converts each view's exact-oracle rate back to an
integer hit count and reports the exact binomial tail under the frozen
action-space-specific uniform null. Offline screen results and chance
calibration remain explicitly distinct from paired runtime speedup evidence.

For the frozen `collective_n8` entry,
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

That N8 path is currently unavailable: the only eighth `pdebug` node has been
drained since 2026-07-21, so the scout was cancelled without producing runtime
rows. The N8-specific adapter and confirmation cannot consume N6 evidence. A
separately frozen N6/48-rank graph retains the 4096-policy compiler action
interface and is queued last in the serial recovery campaign. Even if its
scout passes, it requires a topology-matched N6 confirmation, policy-screen
adapter, suite refreeze, readiness audit, and new request ID before any model
call can be considered.

This zero-eligible readiness result is important: it prevents rich prompts,
an infrastructure failure, or an offline N6 graph from being counted as LLM
evidence. The reused-descriptor and recovery campaigns must finish their
serial runtime gates before the report can advance.

The topology-matched N6 post-authorization path is now implemented but not
launched.  `collective/continue_compiler_collective_n6_after_authorization.sh`
first requires the final priority selection to name `collective_n6`, binds its
request ID to a separate exact authorization, and delegates the 60 stateless
provider calls to the common strictly serial runner.  A complete archive then
advances serially through one hidden full-catalog control allocation,
provider-free capability analysis, deduplication and LTO materialization of at
most nine representatives, and three independent paired `pdebug` allocations.
At no point does a model receive source, IR, runtime labels, compiler-private
hints, or permission to generate code.

Before the first inference,
`preflight_compiler_llm_capability_authorization.py` independently regenerates
the request from the final suite/readiness/protocol evidence, validates the
authorization's exact 60-call delivery and no-source/no-tools boundary, and
checks the authorized provider CLI version.  It emits a content-addressed
record with `provider_inference_invoked=false`; only after that record is
frozen does the post-authorization controller enter the model archive stage.

An exact authorization is never inferred from a general instruction to keep
working.  After the final priority request is frozen,
`prepare_compiler_llm_capability_authorization_proposal.py emit` can create an
offline, content-addressed proposal for the provider executable/version, model,
effort, delivery, and retry limits.  The proposal's outer schema is deliberately
rejected by the trial runner and permits zero provider calls.  It separately
states the number of semantic trials and the worst-case provider-process
invocation count implied by transport retries.  Materializing
the embedded runner-compatible authorization requires a later explicit `grant`
operation whose two arguments exactly repeat both the proposal ID and proposed
authorization ID; any evidence or provider-setting change requires a new
proposal.  Emitting or verifying a proposal neither executes the provider CLI
nor grants permission for inference.

`collective/analyze_collective_n6_llm_runtime_validation.py` replays the three
raw allocation monitors, exact jobspecs, node sets, compiler artifacts and
rotated policy orders.  It reports each modal and post-hoc representative
against both the semantic anchor and deterministic compiler control, plus its
weighted runtime cost regret to the materialized compiler oracle.  Only the
relational modal policy can satisfy the preregistered stable-policy gate; the
best-of-20 representative remains a labeled capability ceiling.  A single N6
family result cannot establish portfolio generalization or complete the paper
claim, regardless of its speedup.

The post-authorization controller finishes by running
`audit_collective_n6_llm_paper_claims.py`.  That auditor regenerates the
capability analysis from the raw provider archive and hidden policy screen,
regenerates the runtime analysis from the LTO bundle and three raw Flux
monitors, and then emits a conservative claim matrix.  It separately reports
different-family entries that have a stable compiler oracle and entries that
also have a complete hidden-screen/LTO/paired-runtime adapter.  The current
route/schedule family has no such post-archive adapter, so eligibility alone
does not recommend freezing another request and never authorizes one.

The family-independent first stage of that follow-up adapter is now provided
by `prepare_communication_llm_policy_catalog.py`. Only after a complete model
archive exists, it enumerates the full Cartesian product of the selected
communication graph's existing candidate IDs, revalidates every policy through
the compiler bridge, and writes contained private LTO hints. It is an offline
catalog operation: it has no application-source path and invokes no provider,
compiler, scheduler, runtime, or oracle. This stage alone does not mark the
route/schedule family executable; the application-specific LTO audit, hidden
runtime screen, and paired validation must still be attached for whichever
different-family candidate passes its preregistered runtime gate.

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
  --readiness build_ofi/compiler_llm_readiness_after_n8_cancel_20260904/report.json \
  --input-separation build_ofi/compiler_input_separation_20260904/report.json \
  --sampling-null build_ofi/llm_sampling_null_20260904/report.json \
  --out build_ofi/compiler_llm_capability_protocol_after_n8_cancel_20260904/report.json

python3 tools/gicc-passes/experiments/audit_compiler_llm_capability_protocol.py verify \
  --suite build_ofi/compiler_decision_suite_20260904/suite.json \
  --prompt-dir build_ofi/compiler_decision_suite_20260904/prompts \
  --readiness build_ofi/compiler_llm_readiness_after_n8_cancel_20260904/report.json \
  --input-separation build_ofi/compiler_input_separation_20260904/report.json \
  --sampling-null build_ofi/llm_sampling_null_20260904/report.json \
  --report build_ofi/compiler_llm_capability_protocol_after_n8_cancel_20260904/report.json
```
