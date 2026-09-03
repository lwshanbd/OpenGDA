# Frozen n8 hierarchy-pipeline capacity protocol

Status before runtime: preregistered; no model or provider call.

## Compiler boundary and frozen capacity

This experiment changes neither the benchmark source nor its ABI. All actions
are existing compiler-catalog targets with runtime-checked preconditions. Any
future model may return only graph-bound option IDs that the LTO bridge lowers;
it cannot emit code, IR, source locations, or source edits.

The exact eight-node graph is
`sha256:ce569e2575cfdc924004631dcf96a2d81077520c3198c8f78edde3a4405cf806`.
The exact offline bundle is
`sha256:4ed11ecb6a6049b07e18ccf4e88cc7f54f7c1247c119ed1d37a133d6b017f361`.
It contains eight topology-legal catalog options in each of four message
regions, for `8^4 = 4096` joint compiler policies. The strict bridge accepted
all 4096 with unique composite candidate IDs; all uniform controls and one
mixed-policy canary passed materialized host/device-IR and same-build
provenance audits. `hierarchical_direct` and `hierarchical_ring` are masked
because their catalog implementations support exactly two nodes and would
fall back at eight nodes.

The two-node primitive calibration is only a source-free link/issue prior. It
is not an eight-node collective label or latency measurement.

## Hypothesis

At n4, pipe4 was only 3.6% slower than the unpipelined hierarchical tree at
16 MiB, while pipeline fixed costs dominated smaller messages. At n8 the node
tree has another level, so chunking may hide the additional large-message
critical path. The preregistered qualitative prediction is therefore:

- the unpipelined hierarchical tree wins at least one size at or below 256 KiB;
- pipe4 or pipe8 wins at least one size at or above 4 MiB.

This is an exploratory compiler-capacity test. A discovered policy is not an
LLM performance result and cannot be sent to a provider without a later,
separately frozen request.

## One-job runtime design

Submit exactly one `pdebug` batch allocation with eight nodes, 64 ranks, eight
ranks per node (one rank per GCD), eight CPU cores and one GPU per rank. Within
that allocation run three complete paired blocks, with two warmups and seven
timed calls at each of 1 KiB, 4 KiB, 8 KiB, 64 KiB, 256 KiB, 1 MiB, 4 MiB,
8 MiB, and 16 MiB. Rotate the exact arm order:

1. unpipelined, pipe4, pipe8;
2. pipe4, pipe8, unpipelined;
3. pipe8, unpipelined, pipe4.

All three blocks must share one job ID and exact node list. Every one of the 81
arm/size rows must report zero errors. The controller may submit only this one
job and must stop after content-addressed analysis.

## Capacity gate

The n8 pipeline hypothesis is promising only if every condition holds:

- the qualitative small/unpipelined and large/pipeline prediction both hold;
- pooled block medians have at least two distinct size winners;
- best-uniform-to-pointwise-oracle geometric-mean headroom is at least `1.05`;
- maximum single-size headroom is at least `1.10`;
- for every pooled size whose pipeline winner differs from the pooled
  best-uniform arm, that same winner beats the best-uniform arm in at least two
  of the three rotated blocks.

Failure closes n8 hierarchy pipeline depth as a model-worthy action and does
not trigger a model call. Passing permits only a separately preregistered
confirmation and frozen compiler-only prompt. It does not authorize a provider
call or a model-selected runtime job.

Post-run outcomes belong in `RESULTS.md`; this file must remain byte-identical
after its hash is archived by the runtime monitors.
