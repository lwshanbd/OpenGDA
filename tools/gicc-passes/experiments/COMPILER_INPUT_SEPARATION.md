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

The audit checks this boundary from the serialized provider surface, not only
from declarations. For every prompt it rejects source/IR filenames, absolute
filesystem paths, raw LLVM IR syntax, source/debug-location keys, private
site/target identities, and materializer fields. It also parses every response
schema, requires every object to set `additionalProperties=false`, binds
`graph_id` to the suite, and proves that the authoritative choice fields
enumerate exactly the IDs visible in the corresponding prompts.

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

`COMPILER_ACTION_AUTHORITY.md` separately audits the output side.  It proves
that the upgraded compiler-policy interface contains structural and collective
actions absent from the frozen route GBT, while explicitly recording that this
width comes from the interface rather than the identity of the model.  A
structured ML method could use the same compiler candidate IDs; only an
equal-authority evaluation can isolate LLM reasoning quality.

## Frozen result

- audit ID and serialized report SHA-256: regenerate with the commands below
  after selecting the terminal N8 or confirmed N6 suite;
- suite entries: 7;
- scalar GBT features: 7;
- relational compiler semantic families: 9;
- identical selectable IDs across every LLM information ablation: true;
- application source visible or modified: false;
- source locations, raw LLVM IR, filesystem paths, and private materializers
  visible: false;
- response schemas closed and graph-bound: true;
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
