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
rotating view order. This is 60 conditional calls for one eligible graph, but
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
  `sha256:7b699d0e0cbe2570388b9ef7b2ad1b5e1afc328e8eb8121bd1947ce00b56cfd6`;
- serialized report SHA-256:
  `83918559c2d0c1b19205906268988d7291c68f8fbb238a106f9b1ea5f4240fd5`;
- frozen metrics implementation SHA-256:
  `fcc5858af4498ad39daa0efcdce4934fc7dd470e3e654189a8bb29a5d4c1250e`;
- status: `blocked_no_runtime_eligible_entries`;
- suite entries: 7;
- runtime-eligible entries: 0;
- provider requests frozen: 0;
- provider calls authorized or made: 0;
- paper LLM-performance claim ready: false.

This negative readiness result is important: it prevents rich prompts alone
from being counted as LLM evidence. The existing N8 collective, Jacobi
producer-fission, guarded early-trigger, and reused-descriptor campaigns must
finish their serial runtime gates before the report can advance.

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
