# Compiler action authority: route GBT versus compiler-policy interface

Status: machine-audited interface comparison; no model/provider/compiler/
scheduler invocation, no application-source access, and no performance claim.

This audit separates two questions that must not be conflated in the paper:

1. Does the upgraded compiler interface expose decisions that the frozen
   scalar GBT evaluator cannot name?
2. Does an LLM select those decisions better than an equal-authority control?

The answer to the first question is yes.  The second remains unmeasured.  The
wider authority belongs to the compiler-policy interface, not intrinsically
to an LLM; a structured ML model could be connected to the same candidate-ID
interface and would be a valid equal-authority baseline.

## Audited result

The frozen GBT report has seven scalar features and exactly one
`chosen_action` route label per prediction.  Its complete legal vocabulary is
`default`, `proxy`, and `trigger`; it has no candidate-ID, collective
size-policy, or compiler-transform output field.

The current compiler suite contains two independently counted decision spaces
with additional semantics:

| Independent entry | All policies | Route/anchor-only policies | Policies using wider compiler authority |
|---|---:|---:|---:|
| `coalescing_placement` | 4096 | 64 | 4032 (98.4375%) |
| `collective_n8` | 4096 | 1 | 4095 (99.9756%) |

The first row contains 12 candidate IDs selecting `COALESCE_LOOP` or
`COALESCE_LOOP_EARLY`.  The second contains 28 non-anchor option IDs spanning
seven algorithms, three communication-graph families, and pipeline depths
1/4/8.  These counts are per independent entry and are not summed or
multiplied into a fictitious global space.

The audit also records the limiting counterexample.  All 32 currently visible
communication route/group candidates are semantically composed from the same
three route actions available to the GBT.  Of those, 27 are atomic two-site
candidate-ID decisions and 18 choose mixed routes, so their output granularity
is absent from the frozen one-label GBT schema; that does not prove a
redesigned structured GBT or other ML model could not emit them.

Finally, the three conditional compiler transforms—producer-frontier fission,
guarded early trigger, and loop-descriptor reuse—are outside the GBT route
vocabulary, but all remain runtime-unconfirmed and model-invisible.  They are
not counted as current LLM authority or performance evidence.

The content-addressed report ID is
`sha256:581ec95ba62736ca95c7b5ef90cad46742239e1958cc6c76355c5d2fb47ca17d`.
It binds the exact suite, GBT report, input-separation audit, conditional
frontier, all seven private graphs, and the auditor itself.

## Paper interpretation

This result supports the claim that the upgraded work studies a materially
wider compiler/LTO decision interface than the historical route-only GBT.  It
does not support “LLM beats GBT” or “LLM is required.”  Model intelligence must
be evaluated with equal transformation authority: the three LLM information
views share one candidate set, and future comparisons should include the
already frozen deterministic compiler controls plus any feasible structured
ML baseline using the same graph-bound IDs.

## Reproduction

From the repository root:

```sh
python3 tools/gicc-passes/experiments/audit_compiler_action_authority.py \
  --suite build_ofi/compiler_decision_suite_20260904/suite.json \
  --prompt-dir build_ofi/compiler_decision_suite_20260904/prompts \
  --gbt-report build_ofi/compiler_lto_eval/generated/gbt-history-report.json \
  --input-separation build_ofi/compiler_input_separation_20260904/report.json \
  --communication jacobi=build_ofi/compiler_fact_coverage_20260904/portfolio/jacobi/group-graph.json \
  --communication minimod=build_ofi/compiler_fact_coverage_20260904/portfolio/minimod/group-graph.json \
  --communication mixed_lto=build_ofi/compiler_fact_coverage_20260904/portfolio/mixed_lto/group-graph.json \
  --communication mm_minimal=build_ofi/compiler_fact_coverage_20260904/portfolio/mm_minimal/group-graph.json \
  --communication loop_lto=build_ofi/compiler_fact_coverage_20260904/portfolio/loop_lto/group-graph.json \
  --structural-graph build_ofi/compiler_comm_plan_placement_current_20260904/generated/opportunity-graph.json \
  --collective-graph build_ofi/compiler_collective_capacity_n8_hierpipe_v1_20260903/discovery/graph.json \
  --conditional-frontier build_ofi/compiler_action_frontier_20260904/report.json \
  --out build_ofi/compiler_action_authority_20260904/report.json
```
