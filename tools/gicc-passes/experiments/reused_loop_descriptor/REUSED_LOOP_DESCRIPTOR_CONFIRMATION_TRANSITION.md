# Reused loop-descriptor scout-to-confirmation transition

Status: preregistered while the serial compiler-headroom campaign is still
waiting for its N8 predecessor. This protocol is compiler/LTO-only. It contains
no model, provider, source-edit, or immediate scheduler action.

## Fixed estimand

The exploratory scout asks whether replacing caller-side descriptor-array
construction with one compiler-proved loop-invariant descriptor template has
headroom. A positive scout may advance only to this confirmation; it does not
make the transform model-visible.

Confirmation retains both preregistered loop batches, `4` and `64`, and the
first ten canonical message sizes from `1B` through `64KB`. No batch or size
may be selected after seeing the scout result.

## Allocation and order contract

Confirmation consists of three independent allocations, completed one at a
time:

- queue `pdebug` only;
- 2 nodes and 2 ranks, one rank/GPU per node, 64 CPU cores per rank;
- two order-balanced blocks per allocation: `AB` then `BA`;
- both batches and all 16 canonical message sizes in every block;
- baseline is A and compiler-materialized descriptor reuse is B;
- exactly `(10 warmup + 21 measured) * batch * 16 sizes` enqueues per arm;
- at most one active or queued job owned by this campaign.

There are 12 paired batch/block observations and 120 mechanism-targeted
small-message speedups. Every run must retain all original network operations
and their order. The frozen final-IR audit must continue to prove four original
kernel-launch call sites in both arms, identical device kernels, descriptor
arrays only in baseline, and the repeated-descriptor helper only in reuse.

## Frozen primary analysis

For each allocation, compute the geometric mean of baseline/reuse median-time
speedups over both blocks, both batches, and the ten fixed small-message sizes.
The primary point estimate is the geometric mean of those three allocation
values. Its interval is the exact 3-out-of-3 paired bootstrap over independent
allocations (27 resamples).

The compiler oracle is confirmed only if all conditions hold:

1. the primary point estimate is at least `1.01`;
2. the exact allocation-cluster bootstrap 95% lower bound is strictly above
   one;
3. at least two of three allocation-level geometric means exceed one;
4. for each batch, the median over its six block/allocation small-message
   geometric means is at least `0.98`;
5. all scheduler, artifact, IR-shape, output, and enqueue-count audits pass.

Failure keeps `REUSE_LOOP_DESCRIPTOR` absent from the model-visible compiler
graph. Success permits a separate content-addressed graph expansion that adds
only the already compiler-defined `trigger_reused_descriptor_loop` candidate.
It still does not authorize a provider call. A single positive benchmark is
compiler action-space evidence, not an LLM-selection result.

After a positive scout, freeze (but do not execute) the transition with:

```sh
python3 tools/gicc-passes/experiments/reused_loop_descriptor/prepare_reused_loop_descriptor_confirmation.py prepare \
  --monitor build_ofi/reused_loop_descriptor_scout_aff76f9_20260904/monitor.json \
  --analysis build_ofi/reused_loop_descriptor_scout_aff76f9_20260904/analysis.json \
  --binary-dir build_ofi/reused_loop_descriptor_oracle_aff76f9_20260904 \
  --out build_ofi/reused_loop_descriptor_confirmation_transition_aff76f9_20260904.json
```

`verify-contained` replays the scout analysis from the raw monitor, rechecks
the original source hash, compiler facts, two distinct binaries, and final-IR
mechanism attestation, and verifies every file record and the transition ID.
The preparer itself has no scheduler path.

Only after that transition exists and re-verifies, the separate controller is
eligible to run:

```sh
bash tools/gicc-passes/experiments/reused_loop_descriptor/continue_reused_loop_descriptor_confirmation.sh \
  build_ofi/reused_loop_descriptor_confirmation_transition_aff76f9_20260904.json \
  build_ofi/reused_loop_descriptor_oracle_aff76f9_20260904 \
  build_ofi/reused_loop_descriptor_confirmation_aff76f9_20260904
```

The controller refuses an existing output, waits for an idle account before
each submission, submits exactly one N2 `pdebug` allocation, waits for and
audits it, and only then considers the next allocation. It never cancels or
modifies another job. The analyzer reparses every raw log and independently
recomputes all 120 fixed-stratum speedups and the three-allocation gate.

For unattended serial continuation, the standalone successor waits for the
frozen scout state and performs no submission itself. It skips confirmation
after a `negative` or `failed` scout. Only a `promising` scout lets it freeze
and re-verify the transition before invoking the controller above:

```sh
bash tools/gicc-passes/experiments/reused_loop_descriptor/continue_reused_loop_descriptor_after_scout.sh \
  build_ofi/reused_loop_descriptor_scout_aff76f9_20260904.state \
  build_ofi/reused_loop_descriptor_scout_aff76f9_20260904 \
  build_ofi/reused_loop_descriptor_oracle_aff76f9_20260904 \
  build_ofi/reused_loop_descriptor_confirmation_transition_aff76f9_20260904.json \
  build_ofi/reused_loop_descriptor_confirmation_aff76f9_20260904
```

The successor hashes all confirmation code and frozen binaries before waiting,
then rechecks those hashes before either transition preparation or execution.
Its own state is written beside the confirmation output with the suffix
`.chain.state`; it contains no provider or model path.
