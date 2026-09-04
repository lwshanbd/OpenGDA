# N8 hierarchy-pipeline scout-to-confirmation transition

Status: preregistered before the pending N8 scout produced any runtime result.
This file is separate from the immutable scout protocol and is not part of the
pending scout's artifact set.

The transition is compiler-only. It cannot invoke a model, compiler, scheduler,
or provider, and it cannot modify application source. A failed N8 capacity
gate produces no confirmation plan. A passed gate is mapped deterministically
to existing option IDs in graph
`sha256:ce569e2575cfdc924004631dcf96a2d81077520c3198c8f78edde3a4405cf806`.

## Frozen selector

For each of the graph's four immutable compiler bins, compute each hierarchy
pipeline arm's geometric mean over the pooled scout median latencies at the
sizes in that bin, then select the minimum:

| Compiler bin | Scout sizes used |
|---|---|
| `message-bin-0`, at most 4 KiB | 1 KiB, 4 KiB |
| `message-bin-1`, at most 256 KiB | 8 KiB, 64 KiB, 256 KiB |
| `message-bin-2`, at most 8 MiB | 1 MiB, 4 MiB, 8 MiB |
| `message-bin-3`, above 8 MiB | 16 MiB |

Exact ties use the fixed priority unpipelined, pipe4, pipe8. The same rule over
all nine sizes regenerates the scout's best uniform comparator. Both policies
are converted to graph-bound option IDs and passed through the existing strict
collective bridge. The already frozen source-free heuristic `(1,1,4,8)` is the
second comparator. No algorithm name, threshold, or target chosen by this
transition bypasses the compiler graph.

If the derived bin policy equals either comparator, that equality is recorded;
the transition closes as `closed_no_incremental_policy`, submits no
confirmation, and must not be described as extra LLM headroom.

## Confirmation contract

The transition manifest specifies three independent eight-node `pdebug`
allocations, submitted and completed one at a time. Each allocation uses 64
ranks, eight ranks/GPUs per node, eight CPU cores per rank, the same nine
message sizes, two warmups, and seven timed calls. The three arms are the
derived bin policy, the scout best-uniform arm, and the frozen structural
heuristic, with a fixed Latin rotation across allocations.

The scout best-uniform arm and frozen heuristic are preregistered co-primary
comparators; neither is selected after seeing confirmation data. Confirmation
requires all correctness checks and, against *both* comparators, a
derived-policy point-estimate speedup of at least `1.03` with an exact
paired-bootstrap 95% lower bound strictly above one. Scout and confirmation
labels remain unavailable to any later model.

A confirmation pass may permit preparation of a separate content-addressed
model request. It does not authorize a provider call. A confirmation failure
closes this N8 graph to a model-performance claim.

## Deferred use

Only after the current scout has passed:

```sh
python3 tools/gicc-passes/experiments/collective/prepare_compiler_collective_n8_confirmation.py prepare \
  --graph build_ofi/compiler_collective_capacity_n8_hierpipe_v1_20260903/discovery/graph.json \
  --scout build_ofi/compiler_collective_hierpipe_n8_scout_20260903/analysis.json \
  --heuristic-decision build_ofi/compiler_decision_suite_20260904/controls/collective-n8-heuristic-decision.json \
  --output-dir build_ofi/compiler_collective_n8_confirmation_transition_20260904
```

The preparer intentionally rejects the current state while the scout result is
missing or negative. It creates compiler decisions and hints only; it does not
build or submit the confirmation runtime experiment.

The preparer materializes three strict-bridge hints: the derived bin policy,
the regenerated scout-best-uniform policy, and the frozen structural heuristic.
If and only if the transition reports `confirmation_plan_ready`, the following
controller builds all three from the same compiler/LTO inputs.  It then submits
one allocation, waits for its clean audited completion, and only then submits
the next allocation:

```sh
bash tools/gicc-passes/experiments/collective/continue_compiler_collective_n8_confirmation.sh \
  build_ofi/compiler_collective_capacity_n8_hierpipe_v1_20260903 \
  build_ofi/compiler_collective_hierpipe_n8_scout_20260903 \
  build_ofi/compiler_decision_suite_20260904/controls/collective-n8-heuristic-decision.json \
  build_ofi/compiler_collective_n8_confirmation_transition_20260904 \
  build_ofi/compiler_collective_n8_confirmation_20260904
```

This command is documented, not launched by the transition preparer. The
controller and analyzer reject any queue other than `pdebug`, any shared job
ID across allocations, an altered arm order, changed artifacts, nonzero
correctness errors, or a failure to satisfy both co-primary comparisons.
