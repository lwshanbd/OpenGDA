# Compiler-only conditional action frontier

Status: machine-audited compiler expressibility result; zero runtime-confirmed
conditional candidates, zero model/provider calls, and no performance claim.

`audit_compiler_action_frontier.py` answers a deliberately pre-runtime
question: do the compiler facts and already implemented LTO materializers
support legal action-space extensions beyond the seven-entry frozen suite?
It verifies the current suite and prompts, regenerates the dormant
producer-fission compiler case, checks the guarded-trigger and
reused-descriptor compiler metadata and final-IR audits, and runs the three
pure graph expansions in memory.

The result is an exact one-candidate superset for each affected independent
task:

| Entry | Current policies | Conditional policies | Compiler transform |
|---|---:|---:|---|
| `jacobi` | 9 | 10 | `PRODUCER_FRONTIER_TWO_PHASE` |
| `mm_minimal` | 3 | 4 | `GUARDED_EARLY_TRIGGER` |
| `loop_lto` | 2 | 3 | `REUSE_LOOP_DESCRIPTOR` |

All old candidate IDs are preserved.  The counts are per independent entry
and are neither summed nor multiplied into a global action-space number.  The
three new candidates remain runtime-unconfirmed and model-invisible; they can
enter a prompt only after their own positive confirmation and the existing
content-addressed graph/suite-refreeze chain.

The report ID is
`sha256:c08c872ad29466bad9e37e29de8af6d40b9113fae10c76ce3265a04c73bbc488`.
Its boundary records that the audit read no application source, wrote no
expanded graph, invoked no compiler, scheduler, model, or provider, and used
no runtime result.  This supports the paper claim that additional
compiler-owned communication transformations are already expressible.  It
does not support a speedup claim or an LLM-over-control claim.

## Reproduction

From the repository root:

```sh
python3 tools/gicc-passes/experiments/audit_compiler_action_frontier.py \
  --suite build_ofi/compiler_decision_suite_20260904/suite.json \
  --prompt-dir build_ofi/compiler_decision_suite_20260904/prompts \
  --producer-dossier build_ofi/compiler_fact_coverage_20260904/portfolio/jacobi/dossier.json \
  --producer-template build_ofi/compiler_fact_coverage_20260904/jacobi_current/meta/_Z18jacobi_step_kernelILi32ELi32EEvPN4gicc9DeviceCtxEPfPKfS3_iiibiiiiimmmmm.json \
  --producer-graph build_ofi/compiler_fact_coverage_20260904/portfolio/jacobi/group-graph.json \
  --producer-coverage-report build_ofi/compiler_fact_coverage_20260904/coverage-report.json \
  --guarded-dossier build_ofi/compiler_fact_coverage_20260904/portfolio/mm_minimal/dossier.json \
  --guarded-template build_ofi/compiler_fact_coverage_20260904/mm_minimal_disjoint/meta/_Z18matmul_step_kernelPN4gicc9DeviceCtxEPKfS3_Pfiiiiiim.json \
  --guarded-graph build_ofi/compiler_fact_coverage_20260904/portfolio/mm_minimal/group-graph.json \
  --guarded-binary-dir build_ofi/guarded_early_trigger_oracle_77897d9 \
  --reused-dossier build_ofi/compiler_fact_coverage_20260904/portfolio/loop_lto/dossier.json \
  --reused-template build_ofi/compiler_fact_coverage_20260904/bench_pingpong_lto/meta/_Z15dwq_loop_kernelPN4gicc9DeviceCtxEiimi.json \
  --reused-graph build_ofi/compiler_fact_coverage_20260904/portfolio/loop_lto/group-graph.json \
  --reused-binary-dir build_ofi/reused_loop_descriptor_oracle_aff76f9_20260904 \
  --out build_ofi/compiler_action_frontier_20260904/report.json
```
