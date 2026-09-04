# Frozen n6 hierarchy-pipeline capacity protocol

Status before runtime: preregistered; no model or provider call.

## Compiler boundary and frozen capacity

This experiment changes neither the benchmark source nor its ABI. All actions
are existing compiler-catalog targets with runtime-checked preconditions. Any
future model may return only graph-bound option IDs that the LTO bridge lowers;
it cannot emit code, IR, source locations, or source edits.

The exact six-node graph is
`sha256:a7aa1f681a64574251aa8e2511550fb8d46b2ea08b5a4aae35fd3865b860a0c0`.
The exact offline bundle is
`sha256:dc243d4f358d3c6b64eb9ff26ff5851ef458f9eed2f7be079424657125edc113`.
It contains eight topology-legal catalog options in each of four message
regions, for `8^4 = 4096` joint compiler policies. The strict bridge accepted
all 4096 with unique composite candidate IDs; all uniform controls and one
mixed-policy canary passed materialized host/device-IR and same-build
provenance audits. `hierarchical_direct` and `hierarchical_ring` are masked
because their catalog implementations support exactly two nodes and would
fall back at six nodes.

The two-node primitive calibration is only a source-free link/issue prior. It
is not a six-node collective label or latency measurement.

## Why n6 replaces the blocked n8 scout

The `pdebug` partition exposes eight nodes, but `tioga41` has been drained
since 2026-07-21 with the scheduler reason `node falls out consistently`.
Consequently an eight-node request is unschedulable and has no scheduler start
estimate. Six nodes are the largest even topology that can run on the seven
currently usable nodes; choosing six rather than seven also avoids adding an
odd-node tree imbalance to the scale comparison. The cancelled n8 allocation
produced no runtime observations and is not evidence.

The read-only scheduler and bundle binding is independently machine-audited in
`../PDEBUG_COLLECTIVE_FEASIBILITY.md`, with audit ID
`sha256:88d726fcd7b87d5fc83a46881800a99682d49bceac3d81f5c3f2bb5a0a3b0d3c`.

## Hypothesis

At n4, pipe4 was only 3.6% slower than the unpipelined hierarchical tree at
16 MiB, while pipeline fixed costs dominated smaller messages. At n6 the node
tree has a longer/irregular path than n4, so chunking may hide part of the
additional large-message critical path. The preregistered qualitative
prediction is:

- the unpipelined hierarchical tree wins at least one size at or below 256 KiB;
- pipe4 or pipe8 wins at least one size at or above 4 MiB.

This is an exploratory compiler-capacity test. A discovered policy is not an
LLM performance result and cannot be sent to a provider without a later,
separately frozen request.

## One-job runtime design

Submit exactly one `pdebug` batch allocation with six nodes, 48 ranks, eight
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

The n6 pipeline hypothesis is promising only if every condition holds:

- the qualitative small/unpipelined and large/pipeline prediction both hold;
- pooled block medians have at least two distinct size winners;
- best-uniform-to-pointwise-oracle geometric-mean headroom is at least `1.05`;
- maximum single-size headroom is at least `1.10`;
- for every pooled size whose pipeline winner differs from the pooled
  best-uniform arm, that same winner beats the best-uniform arm in at least two
  of the three rotated blocks.

Failure closes n6 hierarchy pipeline depth as a model-worthy action and does
not trigger a model call. Passing permits only a separately preregistered
confirmation and frozen compiler-only prompt. It does not authorize a provider
call or a model-selected runtime job.

Post-run outcomes belong in `RESULTS.md`; this file must remain byte-identical
after its hash is archived by the runtime monitors.
