# Compiler-path calibration set

This directory contains training labels for a compiler-fact-only decision
model.  It is separate from the frozen evaluation workload and is not a
source-transformation experiment.

## Boundary

`compiler_lto_calibration.cpp` is a calibration harness compiled through the
same discover, feature-extract, decision-bridge, LTO-lowering, and link path as
the frozen evaluation.  The model never reads this source.  It receives only
the canonical schema-v6 dossier and the measured proxy/trigger costs associated
with each compiler-fact group.  Its output remains a
`gicc-llm-decision-v1` response; the pass validates and materializes it.

The frozen calibration contract has 26 decision sites in 18 scenarios and is
bound to:

- source SHA-256:
  `ac9a7f75da60dbac36f3871e1c0113ec8071b1f7d924ef3dd96e9fdb5888b37d`;
- dossier ID:
  `sha256:0c7a8d46d2fd554ea3404d5df3e15a1630012f785e283ccc21558977c8c7aec6`;
- dossier SHA-256:
  `7b8c38fe4ce49de979172458830a400e27256c9ca56aa9b55cfaad89f53319d1`;
- feature SHA-256:
  `84035e607860d7f06b5bb8a86731a41bc4b0bcffb4fa2425baaa48431cc164ba`.

Calibration message sizes are 512 B, 1 KiB, 2 KiB, 8 KiB, 16 KiB, 32 KiB,
64 KiB, 256 KiB, and 512 KiB.  The frozen evaluation sizes 256 B, 4 KiB, and
1 MiB are mechanically rejected if they appear in calibration.  Calibration
metadata also records that frozen evaluation results were not read.

This is deliberately matched-domain calibration, not a claim of unseen-program
generalization.  Its structural families were chosen to cover the same
compiler-fact schema (single, loop/reuse, adjacent, far-use, and static groups),
while exact evaluation sizes and all evaluation labels/results stay held out.

## Stabilized measurements

`compiler-lto-calibration-cuid-v2` contains four allocation-paired replicates
with proxy/trigger order alternated.  Every record passed exact payload hashes,
route-count checks, dossier binding, and one-binary-per-arm checks.  HIP CUID is
fixed per source so the output object name cannot silently change device symbol
identity.

Seventeen of eighteen scenarios have the same winning action in every
replicate.  All six single-operation scenarios prefer trigger; reuse,
adjacent-loop, far-use, static-two, and static-three scenarios prefer proxy.
The static-six scenario is intentionally retained as an instability finding:
its proxy median is bimodal (about 52 versus 104--107 us), so its per-replicate
winner is not stable.  The model excludes that scenario rather than learning a
noisy aggregate label.

The v2 GBT therefore trains on 17 scenarios (34 action-cost rows).  It uses only
compiler facts: message size, batch and total bytes, launch grid, site count,
loop membership, descriptor reuse, coalescability, issue-to-use FLOPs and
exactness, plus the candidate action.  Leave-one-scenario-out validation matches
14/17 actions at 1.0217x geometric-mean regret.  The held-out frozen evaluation
is read only after training and supplies no labels.

The stabilized calibration summary SHA-256 is
`e3e28e60127ae491d570a6743cd04acb46a40cb42d9d51da6013e260d9f6f8f2`.

## Reproduction

```bash
bash examples/proxy/build_compiler_lto_calibration.sh freeze
bash examples/proxy/build_compiler_lto_calibration.sh controls
bash examples/proxy/submit_compiler_lto_calibration.sh paired \
  compiler-lto-calibration-cuid-v2
python3 examples/proxy/analyze_compiler_lto_calibration.py \
  docs/experiments/compiler-lto-calibration/runs/compiler-lto-calibration-cuid-v2/raw/*.log \
  --dossier docs/experiments/compiler-lto-calibration/frozen-v1/dossier.json \
  --manifest docs/experiments/compiler-lto-calibration/frozen-v1/manifest.json \
  --expected-reps 1,2,3,4 \
  --json docs/experiments/compiler-lto-calibration/runs/compiler-lto-calibration-cuid-v2/summary.json

python3 examples/proxy/compiler_lto_calibrated_gbt.py \
  --calibration-dossier docs/experiments/compiler-lto-calibration/frozen-v1/dossier.json \
  --calibration-results docs/experiments/compiler-lto-calibration/runs/compiler-lto-calibration-cuid-v2/summary.json \
  --evaluation-dossier docs/experiments/compiler-lto-eval/frozen-v1/dossier.json \
  --output docs/experiments/compiler-lto-eval/models/gbt-calibrated-v2/response.json \
  --report docs/experiments/compiler-lto-eval/models/gbt-calibrated-v2/report.json
```
