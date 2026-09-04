# Guarded early-trigger scout-to-confirmation transition

Status: preregistered before the pending `mm_minimal` scout runs. This
protocol applies only to compiler/LTO-generated schedules over the unchanged
application source. It contains no provider or source-edit path.

## No post-hoc size selection

The scout passes when either `N=4096` or `N=8192` has at least three of four
paired wins and median speedup of at least `1.02`. That gate decides only
whether confirmation runs. Both sizes remain in the confirmation estimand,
regardless of which size passed the exploratory gate.

## Confirmation contract

After a passed, regenerated scout, the transition preparer binds its raw
monitor, analysis, binaries, unchanged source, hints, compiler metadata, and
final-IR audit into a content-addressed plan. It independently regenerates
the eight scout speedups, all rank checksums, and the exact 161-versus-483
kernel-launch attestations.

Confirmation uses three independent two-node `pdebug` allocations, submitted
and completed strictly one at a time. Every allocation uses 16 ranks, eight
ranks/GPUs per node, eight CPU cores per rank, both sizes, and one `AB` plus
one `BA` block. The primary metric is the geometric mean of paired
baseline/guarded speedups across both sizes and both blocks, clustered by
allocation.

The compiler oracle is confirmed only if:

- all 12 pairs have identical 16-rank result checksums;
- baseline reports 161 and guarded reports 483 successful kernel launches on
  every rank, proving the guarded schedule ran rather than falling back;
- aggregate speedup is at least `1.02`;
- the exact allocation-cluster paired-bootstrap 95% lower bound is above one;
- at least two of three allocation-level geometric means exceed one.

Failure keeps the transform absent from model-visible candidates. Success
permits creation of a new content-addressed compiler graph containing the
pre-authored candidate ID; it does not mutate a frozen graph or authorize a
provider call. One positive application establishes compiler action-space
headroom, not an LLM-selection claim.

## Confirmation-gated graph expansion

After—and only after—the confirmation gate passes, replay the raw evidence and
generate a separate compiler graph bundle:

```sh
python3 tools/gicc-passes/experiments/guarded_early_trigger/prepare_confirmed_guarded_early_graph.py prepare \
  --confirmation-analysis build_ofi/guarded_early_trigger_confirmation_77897d9_20260904/analysis.json \
  --dossier build_ofi/compiler_fact_coverage_20260904/portfolio/mm_minimal/dossier.json \
  --template build_ofi/compiler_fact_coverage_20260904/mm_minimal_disjoint/meta/_Z18matmul_step_kernelPN4gicc9DeviceCtxEPKfS3_Pfiiiiiim.json \
  --graph build_ofi/compiler_fact_coverage_20260904/portfolio/mm_minimal/group-graph.json \
  --output-dir build_ofi/guarded_early_trigger_graph_expansion_20260904
```

The preparer verifies the confirmation, its runtime-guard/correctness gates,
the baseline and device-attested compiler metadata, and the final-IR audit. It
requires the old graph to regenerate exactly, enriches its dossier with only
the four confirmed guarded facts, preserves all three route candidate IDs,
and adds one compiler-owned `GUARDED_EARLY_TRIGGER` candidate. The unchanged
application source is hash-checked by the confirmation transition but is not
included in any model view. The bundle does not refreeze the suite, invoke a
model, or authorize a provider call.
