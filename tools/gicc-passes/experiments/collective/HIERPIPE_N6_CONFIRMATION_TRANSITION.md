# N6 hierarchy-pipeline scout-to-confirmation transition

Status: preregistered before the N6 scout produced any runtime result.

This transition is compiler-owned and conditional. It never invokes a model
or provider and cannot modify application source. A failed N6 capacity gate
ends the experiment. A passed gate allows this tool to derive exactly one
graph-bound size policy using the fixed rule below; it still does not authorize
a provider request.

## Frozen derivation rule

For each of the four compiler message bins, compute each scout arm's geometric
mean over the preregistered measured sizes in that bin, using the pooled median
from the three rotated scout blocks. Choose the minimum-cost arm, breaking ties
in this fixed order:

1. `hierarchical_double_tree`;
2. `hierarchical_double_tree_pipe4`;
3. `hierarchical_double_tree_pipe8`.

The exact size membership is:

- bin 0: 1 KiB, 4 KiB;
- bin 1: 8 KiB, 64 KiB, 256 KiB;
- bin 2: 1 MiB, 4 MiB, 8 MiB;
- bin 3: 16 MiB.

The compiler bridge maps each selected algorithm back to one existing option
ID in graph
`sha256:a7aa1f681a64574251aa8e2511550fb8d46b2ea08b5a4aae35fd3865b860a0c0`.
It must reject any missing bin, changed interval, unknown option, graph change,
or source/code/IR-generating output.

The derived policy advances only if it differs from both the scout's best
uniform hierarchy/pipeline arm and the frozen source-free structural heuristic.
The N6 heuristic is generated before scout results with control ID
`sha256:41d13984a0b843dd32a2c7621ef7cf429a8c2a743660ca45129537f3e8cdba5e`.
Equality to either simple comparator closes the conditional confirmation.

## Confirmation design

If enabled, run three independent six-node `pdebug` allocations sequentially,
never more than one active or queued. Every allocation uses 48 ranks, eight
ranks/GPUs per node, eight CPU cores per rank, the same nine message sizes,
two warmups, and seven timed calls. Compare:

1. the derived per-bin policy;
2. the scout's best uniform hierarchy/pipeline arm;
3. the frozen structural heuristic.

Rotate order as a Latin cycle across the three allocations. All correctness
checks must report zero errors.

For each comparator, compute one paired geometric-mean speedup within each
allocation across all sizes. Use the exact 27-sample, three-out-of-three paired
cluster bootstrap over allocations. The derived policy passes only if, for
both co-primary comparisons, the point estimate is at least `1.03` and the
bootstrap 95% lower bound is strictly greater than `1.0`.

Failure closes this N6 graph to an LLM-performance claim. Passing supplies
stable compiler-oracle headroom only; N6 still requires a policy-screen
adapter, suite refreeze, readiness audit, exact request freeze, and separate
content-addressed authorization before any provider call.

`continue_compiler_collective_n6_after_scout.sh` implements the conditional
transition. It waits without submitting, checks the scout gate and frozen
artifacts, and skips both transition and runtime work after a negative/failed
scout. For a positive scout it still skips runtime when the derived policy is
equal to either simple comparator. Only an incremental policy may reach the
three sequential `pdebug` confirmation allocations.
