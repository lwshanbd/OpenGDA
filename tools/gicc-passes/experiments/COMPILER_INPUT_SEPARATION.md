# Compiler input separation: scalar GBT versus relational LLM views

Status: machine-audited input contract; no model/provider call, no runtime
claim, and no application-source modification.

The GBT and LLM are **not** fed the same information. The frozen GBT consumes
exactly seven scalar values for one action prediction:

1. log2 message bytes;
2. log2 logical operation count;
3. log2 total bytes;
4. route-is-proxy;
5. log2 trigger batch size or a sentinel;
6. log2 proxy producer count or a sentinel;
7. log2 proxy worker count or a sentinel.

Its own report records that `descriptor_reusable`, `coalescable`, and
`flops_to_first_use` are present in the compiler dossier but are not mapped
into the historical training grid.

The primary LLM view instead exposes source-free compiler graphs. Across the
seven frozen decision tasks, the audit finds nine semantic families:

- operation-order and cross-opportunity relations;
- symbolic argument-expression structure;
- dependence and legality proofs;
- symbolic transfer intervals;
- launch and resource structure;
- completion and compute-distance facts;
- compiler-generated candidate semantics and effects;
- collective topology and message-size policy;
- masked-transform reasons.

This is the intended role of the LLM: reason over relationships already
proved and bounded by the compiler. It does not receive source and cannot
invent a transformation, legality fact, materializer, or code.

## Controlled information ablation

Each LLM task has three prompt views: `relational`, `descriptors`, and
`opaque`. The audit parses all 21 content-addressed prompts and proves that the
three views of every task contain the exact same selectable ID set:

| Task | Selectable IDs | Relational edge kinds |
|---|---:|---|
| `coalescing_placement` | 24 | equal facts, numeric order, shared candidate semantics |
| `collective_n8` | 32 | increasing message size, shared compiler property |
| `jacobi` | 9 | operation order, transfer-argument relation |
| `loop_lto` | 2 | operation order, transfer-argument relation |
| `minimod` | 9 | operation order, transfer-argument relation |
| `mixed_lto` | 9 | operation order, transfer-argument relation |
| `mm_minimal` | 3 | operation order, transfer-argument relation |

Consequently, a future comparison among these three LLM views is
action-controlled: only available semantic information changes. The old GBT
remains a useful scalar route baseline, but a direct “GBT versus LLM” result
would not by itself isolate model intelligence because the upgraded compiler
suite also contains structural scheduling and collective choices outside the
GBT's route-only action contract.

The audit therefore establishes richer LLM *input*, not better LLM
performance. A performance claim still requires stable compiler-oracle
headroom, held-out labels, the exact frozen prompt protocol, explicit provider
authorization, and runtime validation of the compiler-materialized plan.

## Frozen result

- audit ID:
  `sha256:af214761ac85e72bf872ee4165b803f92d811bf1e1abf0feddd521b339aa2859`;
- serialized report SHA-256:
  `6c7fc4f63b07e480f96ab0e9473a8abdb65f2de90b92527b9ba8b9b9e5446fb4`;
- suite entries: 7;
- scalar GBT features: 7;
- relational compiler semantic families: 9;
- identical selectable IDs across every LLM information ablation: true;
- application source visible or modified: false;
- model/provider invoked: false;
- performance superiority claimed: false.

## Reproduction

From the repository root:

```sh
python3 tools/gicc-passes/experiments/audit_compiler_input_separation.py emit \
  --suite build_ofi/compiler_decision_suite_20260904/suite.json \
  --prompt-dir build_ofi/compiler_decision_suite_20260904/prompts \
  --gbt-report build_ofi/compiler_lto_eval/generated/gbt-history-report.json \
  --out build_ofi/compiler_input_separation_20260904/report.json

python3 tools/gicc-passes/experiments/audit_compiler_input_separation.py verify \
  --suite build_ofi/compiler_decision_suite_20260904/suite.json \
  --prompt-dir build_ofi/compiler_decision_suite_20260904/prompts \
  --gbt-report build_ofi/compiler_lto_eval/generated/gbt-history-report.json \
  --report build_ofi/compiler_input_separation_20260904/report.json
```
