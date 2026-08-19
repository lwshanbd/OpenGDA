# Minimod paper experiments

This directory realizes the three paper experiments described after the
initial end-to-end ML result, using **Minimod only**:

1. strong baselines (`default`, best fixed, a preregistered hand rule,
   leave-one-scale-out GBT + compiler, and an empirical oracle);
2. fixed-problem scaling at 1/2/4/8 nodes with one and eight ranks per node,
   five randomized allocation-level paired replicates;
3. O5 compiler-only legality in a real Minimod binary: byte-identical
   descriptor-loop kernel bodies with and without a host-mirror annotation.

The measured candidate set for (1) and (2) is
`{IPC_OR_DWQ,DWQ_TRIGGER,CPU_PROXY_ENQUEUE} x {serial,overlap}`.  The transport
half of every arm is separately materialized by LTO.  The serial/overlap half
is an existing, hand-written benchmark schedule selected at runtime; it is a
cost/oracle axis, not evidence that the compiler or an LLM discovered or
implemented that schedule.  All six arms run in randomized order inside one
allocation so node placement is paired.  The primary wall-clock sample is the
maximum `Time kernel` over ranks, not rank 0 alone.

`build.sh` copies the external Minimod tree into an isolated `build_ofi`
directory before instrumenting it.  It refuses unexpected source hashes and
does not edit the external repository.  All jobs have a hard two- or
three-minute allocation limit and a 75-second limit per arm.

```bash
bash examples/minimod_paper/build.sh
bash examples/minimod_paper/submit.sh smoke
bash examples/minimod_paper/submit.sh o5-smoke
bash examples/minimod_paper/submit.sh matrix paper-v1
MINIMOD_STANDARD_GRID=400 \
  bash examples/minimod_paper/submit.sh matrix-pci paper-v1-grid400
bash examples/minimod_paper/submit.sh matrix4 paper-v1-n4-pdebug
bash examples/minimod_paper/submit.sh matrix8 paper-v1-n8
bash examples/minimod_paper/submit.sh o5 paper-v1-o5
MINIMOD_O5_GRID=800 MINIMOD_O5_STEPS=100 \
  bash examples/minimod_paper/submit.sh o5 paper-v1-o5-main
bash examples/minimod_paper/analyze.sh \
  --standard-nodes 1,2,4 \
  --standard docs/experiments/minimod-paper/paper-v1 \
  --standard docs/experiments/minimod-paper/paper-v1-n4-pci-rpn1-replace \
  --standard docs/experiments/minimod-paper/paper-v1-n4-pci-resume \
  --o5 docs/experiments/minimod-paper/paper-v1-o5 \
  --output docs/experiments/minimod-paper/paper-v1-analysis-n1-n4
bash examples/minimod_paper/analyze.sh \
  --o5 docs/experiments/minimod-paper/paper-v1-o5-main \
  --output docs/experiments/minimod-paper/paper-v1-o5-main-analysis
bash examples/minimod_paper/analyze.sh \
  --standard-nodes 1,2,4 \
  --standard docs/experiments/minimod-paper/paper-v1-grid400 \
  --output docs/experiments/minimod-paper/paper-v1-grid400-analysis
env -u PYTHONPATH PYTHONNOUSERSITE=1 /usr/tce/bin/python3 \
  examples/minimod_paper/analyze_cross_grid.py \
  --analysis 400=docs/experiments/minimod-paper/paper-v1-grid400-analysis \
  --analysis 800=docs/experiments/minimod-paper/paper-v1-analysis-n1-n4 \
  --output docs/experiments/minimod-paper/paper-v1-cross-grid-analysis
```

The strict analyzer writes per-allocation `measurements.csv`, kernel/comm/comp
median and IQR columns in `cell-medians.csv`, strong-scaling speedup and
efficiency in `scaling.csv`, and all policy decisions/scores in
`policies.json`, including direct paired-bootstrap comparisons between GBT
and the fixed, global, hand, and oracle policies.  It rejects missing ranks,
incomplete cells, wrong compiler routes, duplicate samples, and checksum
differences before scoring anything.

`analyze_cross_grid.py` then trains the same deterministic GBT on one
validated grid and scores it only on the other, excluding the one-node,
one-rank no-communication control cell.  It reports exact policy picks,
oracle regret, and paired-bootstrap comparisons with default, the best global
action, and the preregistered hand rule.

The full jobs are intentionally separate allocation-level replicates.  Do not
replace them with six back-to-back process repetitions in one allocation;
that would understate placement variance and invalidate the pairing design.
The account currently exposes only `pci` and `pdebug`: 1/2/4-node jobs use
`pci`, while 8-node jobs use `pdebug` and may remain queued if that partition
has unavailable nodes.  `MINIMOD_8_QUEUE` can override this when a larger
queue is granted to the account.  On 2026-08-17 a direct `pllm` submission
was rejected with `queue "pllm" not valid for user; valid queues for user:
pdebug,pci`, even though the global resource listing showed idle `pllm`
nodes.

`matrix4` also accepts `MINIMOD_4_RPN1_REPS` and
`MINIMOD_4_RPN8_REPS`.  This permits a queue migration without rerunning
completed allocation replicates.  An explicitly empty value skips that rank
layout, while an unset value retains all five repetitions.  For example:

```bash
MINIMOD_4_QUEUE=pci \
MINIMOD_4_RPN1_REPS="3 4 5" \
MINIMOD_4_RPN8_REPS="1 2 3 4 5" \
bash examples/minimod_paper/submit.sh matrix4 paper-v1-n4-pci-resume
```

The two valid pdebug repetitions in `paper-v1-n4-pdebug` are auxiliary only.
They are deliberately absent from the strict command above because rep 1 and
2 were repeated on PCI in `paper-v1-n4-pci-rpn1-replace`; including both
directories would correctly fail as duplicate sample keys.
