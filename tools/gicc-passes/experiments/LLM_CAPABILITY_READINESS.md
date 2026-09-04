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
| `mm_minimal` | `runtime_labels_missing` | establish a preregistered compiler oracle first |
| `loop_lto` | `runtime_labels_missing` | establish a preregistered compiler oracle first |

The report has readiness ID
`sha256:544ec38a9eee16e52a50c49c2a75a072a711893c89209e09ac05cd3abbb80325`
and serialized SHA-256
`151716c8dba1054e4bfc3c25973ff7d2d8df85039df4ffbdafe6d1e805c3c752`.
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
  --producer-state build_ofi/producer_fission_oracle_scout_90b9123_20260904.state \
  --producer-analysis build_ofi/producer_fission_oracle_scout_90b9123_20260904/analysis.json \
  --out build_ofi/compiler_llm_readiness_20260904/report.json
```

Use `verify` with the same inputs and
`--report build_ofi/compiler_llm_readiness_20260904/report.json` to prove that
the report still matches the current external state. When a monitored runtime
stage changes, `verify` deliberately rejects the stale report and `emit`
derives a new one.
