# Frozen compiler-level LTO decision evaluation

This experiment evaluates decisions made from compiler facts and materialized
by the LTO passes. It is deliberately not a source-editing experiment.

## Boundary

Every arm is compiled from the unchanged
`examples/proxy/compiler_lto_eval.cpp` source. Its frozen SHA-256 is
`d707d5b6773a5299b2d8a19a6c832f82b719bc3be08b01f250a481500ecb486a`.
The build checks this hash before and after every facts/control/candidate
pipeline.

The only permissible model input is the canonical projection of real LTO
facts plus the hash-bound deployment profile:

- feature schema: 6
- decision sites: 10
- dossier ID:
  `sha256:2299725094d5d805e6f8732227a2b11c28c57ff3ea194cf32d8e66c65c3a1aa5`
- `features.json`:
  `a8991c66736f85cf43e4364904f5bfa73b0810c0845424bd2748b0a1610db190`
- `dossier.json`:
  `c77f4cff5e14552491c7a8cf8129724885c2f9a933898460375f7cadaca32274`
- `prompt.txt`:
  `51cb67f351a8f6e02e64186d1da4d0bff15fc1dcc1000960cb84df12a7b95b94`
- `profile.json`:
  `03c404978aca58b8120e27e73c9c8613f724a6139f2828b3c3019eab776f6105`

The bridge requires a response bound to that dossier ID, exact coverage of
all sites, and an action from each compiler-provided legal set. The LTO pass,
not the model, implements the lowering. Malformed, stale, incomplete, or
illegal responses fail closed. No provider was called and no LLM result is
reported here.

## Frozen workload

Seven separately timed scenarios cover ten real device-operation sites. The
compiler facts distinguish message size, loop batch, descriptor reuse,
coalescing opportunity, issue-to-use distance, launch geometry, and
host-knowability.

| scenario | compiler-visible shape | legal decision |
| --- | --- | --- |
| tiny-k1-grid1 | 256 B, one op, one block | proxy / trigger / default |
| reuse-k32-grid1 | 4 KiB, 32-op reusable descriptor loop | proxy / trigger |
| adjacent-k16-grid1 | 4 KiB, 16 adjacent loop ops | proxy / trigger |
| far-k64-grid8 | 4 KiB, 64 ops, long static distance, eight blocks | proxy / trigger |
| large-k1-grid8 | 1 MiB, one op, eight blocks | proxy / trigger / default |
| dynamic-k1-grid1 | device-loaded offsets, not host-knowable | proxy only |
| static4-k4-grid4 | four static 4 KiB sites, four blocks | proxy / trigger per site |

The dynamic site is a legality control. Since every policy materializes
proxy there, timing variation between separately linked binaries is not
decision regret and is excluded from the primary score.

## Control results

`compiler-lto-controls-v1` contains four allocation-paired replicates, with
the four arm orders rotated. Each arm has 100 timed iterations per scenario.
All 112 result records passed complete-region hashes and exact staged/pushed
route-count checks. Jobs `f5qkHLcrNhBd`, `f5qkHLkNF431`, `f5qkHLsDZjZR`,
and `f5qkHLz1vSX9` completed cleanly.

Geometric means of the four per-replicate medians are:

| scenario | default (us) | proxy (us) | trigger (us) | hand rule (us) | measured best action |
| --- | ---: | ---: | ---: | ---: | --- |
| tiny-k1-grid1 | 25.565 | 28.128 | 25.080 | 28.051 | trigger |
| reuse-k32-grid1 | 350.253 | 324.505 | 347.193 | 346.864 | proxy |
| adjacent-k16-grid1 | 182.239 | 169.512 | 181.874 | 178.910 | proxy |
| far-k64-grid8 | 698.929 | 634.673 | 697.521 | 692.361 | proxy |
| large-k1-grid8 | 75.222 | 78.759 | 74.773 | 78.265 | trigger |
| dynamic-k1-grid1 | 29.127 | 29.366 | 29.000 | 28.728 | forced proxy; excluded |
| static4-k4-grid4 | 56.567 | 41.136 | 56.649 | 40.592 | proxy |

The same winner appears in all four replicates for each of the six
decision-bearing scenarios. Relative to choosing the better uniform legal
action per scenario, the compiler default has 1.1031x geometric-mean regret
and matches two of six winners; the existing hand rule has 1.0604x regret
and matches one of six. This establishes a measurable decision problem. It
does not establish that a learned or language-model policy can solve it.

## Exact static-four action-space check

`compiler-lto-static4-oracle-v1` enumerates all `2^4` proxy/trigger
assignments for the four static sites in one bounded two-node allocation,
alternating order extremes to reduce monotonic drift. All 16 binaries have
distinct hashes. All records passed the complete payload hash and exact
route-count checks. Job `f5qkNmce2Dio` completed on `tioga30,32` with status
zero.

The exact action-space winner in this run is mask `1111` (all four sites use
proxy) at 39.532 us. The next configuration is 1.0981x slower; mask `0000`
(all trigger) is 56.755 us, or 1.4357x slower. This is a complete enumeration
of legal static-four assignments, but only one allocation-level replicate;
the four-replicate uniform controls above carry the stronger repeatability
claim.

## Source-free GBT transfer baseline

`compiler_lto_gbt.py` tests whether a conventional tabular model trained on
the pre-existing `grid_big.csv` microbenchmark can transfer to the real LTO
path. The training file is bound by the dossier to SHA-256
`c28e3cb69fb78aa4cf9223fb281cf71658a58abd1def661e29351caba0466836`.
The model reads the frozen dossier and that historical table only: it does not
accept source, frozen runtime logs, or oracle labels.

Only compatible axes are transferred: message bytes, logical op count,
trigger batch size, proxy producer count, and proxy worker count. Historical
distance is measured in microseconds while LTO reports instruction/FLOP
distance, so the GBT trains only on historical `D=0` rows instead of inventing
a conversion. Consequently, compiler-only semantic facts such as descriptor
reuse, coalescability, and exact issue-to-use FLOPs remain available to an LLM
but are not falsely encoded into the tabular baseline.

Two deterministic variants were generated:

- `history`: 621 historical rows, all historical sizes;
- `size-holdout`: 399 rows after excluding every frozen message size (256 B,
  4 KiB, and 1 MiB; 1 MiB was already outside the historical table).

Both independently emit the same ten-site action assignment and both pass the
strict decision bridge. They choose trigger for tiny, reuse, adjacent, far,
and all four static sites; proxy for large and the forced dynamic site. Since
their materialized actions are identical, only `history` is run as a distinct
policy; duplicating the size-holdout binary would add no decision evidence.

`compiler-lto-gbt-v1` compares four real LTO binaries—default, hand rule,
GBT, and a clearly labeled measured-oracle scoring control—inside four paired
allocations with rotated order. The oracle response explicitly records that
it reads control/oracle results and is never model input or model output.
Jobs `f5qkYLNd58tw`, `f5qkYLWYfJFq`, `f5qkYLdPyynF`, and `f5qkYLkKkd9h`
completed cleanly; all 112 records passed full-region hashes and exact
staged/pushed route checks. The analyzer also verifies that the logged arm,
replicate, and binary name agree and that each arm uses one SHA-256-identical
binary across all four replicates. Those four binary hashes are retained in
the machine-readable `summary.json`.

The history response SHA-256 is
`16256237df41058707d98f386a32f6803975a2e5a1c8727b3a2e094a171d1047`;
the size-held-out response is
`5bd79edd74b5e2a8ce8fe19aec503611888edb7cd250e2be75c88f2fe9a8e84a`.
The measured-oracle response is separately bound as
`a35274b0df139230da65bda20257d37ecc67e67eb6952136acd33f84be25d3c2`,
and the final run summary is
`1217a7ef5beb22c2fe16050e1494f2d52d3f700f8c07c57002ff47358c61eb6d`.

| policy | primary geomean regret | action matches | sum of scenario geomean medians |
| --- | ---: | ---: | ---: |
| measured oracle | 1.0000x | 6/6 | 1300.527 us |
| hand rule | 1.0709x | 1/6 | 1397.029 us |
| compiler default | 1.0957x | 2/6 | 1415.253 us |
| historical GBT | 1.1103x | 1/6 | 1416.562 us |

The proxy-only dynamic scenario is excluded from primary regret. GBT regret
is above 1.099x in every paired replicate. This is a negative ML result, not
an LLM result: the old runtime-oriented microbenchmark does not transfer to
the compiler-generated lowering path. Fitting the GBT to the frozen oracle
would leak the test labels, so the result is retained as-is. It shows why a
compiler-path calibration set and/or reasoning over richer compiler facts is
needed before claiming learned optimization.

## Compiler-path calibrated GBT

The follow-up model trains only on the separate
`compiler-lto-calibration-cuid-v2` compiler-fact dossier and its paired
proxy/trigger measurements. Calibration and evaluation message-size sets are
disjoint. Seventeen of eighteen calibration scenarios have a stable winner
across all four allocations; the one unstable static-six label is recorded but
excluded. The resulting GBT has 34 action-cost rows and uses only size, batch,
total bytes, launch grid, site count, loop membership, descriptor reuse,
coalescability, issue-to-use FLOPs/exactness, and the proposed action. It never
reads source, the historical runtime grid, frozen test logs, or frozen oracle
labels.

Leave-one-complete-scenario-out calibration validation matches 14/17 actions
at 1.0217x geometric-mean regret. Applied once to the frozen dossier, the
model chooses trigger for the tiny and large single operations, proxy for the
reuse, adjacent, far-use, and four-static-site groups, and the only legal proxy
action for the dynamic-offset control. This matches the separately measured
action oracle in all six decision-bearing scenarios and all nine such sites.

An initial run exposed a HIP build confound: Clang's default CUID hashes the
full compiler command, including the per-arm object name, so identical decision
responses produced different GPU code-object identities. The build now binds
all versions of this source to one explicit CUID. The calibrated response and
the measured-oracle response consequently produce byte-identical binaries
(SHA-256
`b0ace9bac935a2c5039f09c907353e0474b15ccc2ecdba9e3c4941a0dcc99ef9`).
The oracle is used only for action-agreement scoring; it is not a runtime
denominator.

`compiler-lto-gbt-calibrated-balanced-v3` runs five arms in ten allocations.
Five Williams-style orders and their reversals put every arm in every position
twice and every ordered arm pair adjacent twice. All 50 logs and 350 scenario
records pass complete-region hashes, exact route counts, provenance, and stable
per-arm binary checks. The proxy-only dynamic scenario is excluded from the
primary score. Speedup is the paired reference time divided by calibrated-GBT
time over the six decision-bearing scenarios:

| reference | calibrated GBT speedup | paired bootstrap 95% CI | faster allocations | exact sign p |
| --- | ---: | ---: | ---: | ---: |
| compiler default | **1.0690x** | [1.0302, 1.0946] | 9/10 | 0.0215 |
| hand rule | **1.0399x** | [1.0048, 1.0621] | 9/10 | 0.0215 |
| historical-grid GBT | **1.0807x** | [1.0468, 1.1028] | 9/10 | 0.0215 |

The byte-identical oracle duplicate measures residual order/allocation noise:
duplicate/candidate is 0.9927x with a 95% interval of [0.9463, 1.0385], which
contains one. This is the first positive learned result on the real compiler
path: compiler-path calibration reverses the old GBT's negative transfer and
beats both the existing hand rule and compiler default on this frozen suite.
It is an ML/GBT result, not an LLM result, and it does not yet establish
cross-program or cross-platform generalization: the calibration workload was
purpose-built to span the same compiler-fact families while holding out exact
sizes and every test label/result.

The calibrated response SHA-256 is
`d40c340105cc1fc3026d4bbef4954f9526761f09b926bb414296e494aeb3e75e`,
the model report is
`6092038f40b4ab1e83347e08341481094365d5b35b756c78ce9cc371f6c1080b`,
the accepted pass hint is
`47bfb028b9b9936e7c365f5314c8730aa912f678baa61b057d67ab42c4a0b5b9`,
and the final run summary is
`6fb8af459c0320b212d303572afe03cac95695cae807595b9437fe164c53174e`.

## Reproduction and gates

```bash
cmake --build tools/gicc-passes/build --target check-gicc-passes -j2
bash examples/proxy/build_compiler_lto_eval.sh freeze
bash examples/proxy/build_compiler_lto_eval.sh controls
bash examples/proxy/submit_compiler_lto_eval.sh paired compiler-lto-controls-v1
python3 examples/proxy/analyze_compiler_lto_eval.py \
  docs/experiments/compiler-lto-eval/runs/compiler-lto-controls-v1/raw/*.log

bash examples/proxy/build_compiler_lto_eval.sh oracle
bash examples/proxy/submit_compiler_lto_oracle.sh \
  compiler-lto-static4-oracle-v1
python3 examples/proxy/analyze_compiler_lto_oracle.py \
  docs/experiments/compiler-lto-eval/runs/compiler-lto-static4-oracle-v1/raw/*.log

python3 examples/proxy/compiler_lto_gbt.py \
  --dossier docs/experiments/compiler-lto-eval/frozen-v1/dossier.json \
  --grid docs/experiments/grid/grid_big.csv --policy history \
  --output docs/experiments/compiler-lto-eval/models/gbt-v1/history-response.json \
  --report docs/experiments/compiler-lto-eval/models/gbt-v1/history-report.json
bash examples/proxy/build_compiler_lto_eval.sh candidate \
  docs/experiments/compiler-lto-eval/models/gbt-v1/history-response.json \
  gbt-history
bash examples/proxy/build_compiler_lto_eval.sh measured-oracle
bash examples/proxy/submit_compiler_lto_eval.sh gbt compiler-lto-gbt-v1
python3 examples/proxy/analyze_compiler_lto_candidates.py \
  docs/experiments/compiler-lto-eval/runs/compiler-lto-gbt-v1/raw/*.log \
  --dossier docs/experiments/compiler-lto-eval/frozen-v1/dossier.json \
  --response gbt-history=docs/experiments/compiler-lto-eval/models/gbt-v1/history-response.json \
  --response measured-oracle=docs/experiments/compiler-lto-eval/models/measured-oracle-v1/measured-oracle-response.json \
  --oracle-arm measured-oracle

python3 examples/proxy/compiler_lto_calibrated_gbt.py \
  --calibration-dossier docs/experiments/compiler-lto-calibration/frozen-v1/dossier.json \
  --calibration-results docs/experiments/compiler-lto-calibration/runs/compiler-lto-calibration-cuid-v2/summary.json \
  --evaluation-dossier docs/experiments/compiler-lto-eval/frozen-v1/dossier.json \
  --output docs/experiments/compiler-lto-eval/models/gbt-calibrated-v2/response.json \
  --report docs/experiments/compiler-lto-eval/models/gbt-calibrated-v2/report.json
bash examples/proxy/build_compiler_lto_eval.sh candidate \
  docs/experiments/compiler-lto-eval/models/gbt-calibrated-v2/response.json \
  gbt-calibrated
bash examples/proxy/submit_compiler_lto_eval.sh calibrated-balanced \
  compiler-lto-gbt-calibrated-balanced-v3
python3 examples/proxy/analyze_compiler_lto_transfer.py \
  docs/experiments/compiler-lto-eval/runs/compiler-lto-gbt-calibrated-balanced-v3/raw/*.log \
  --dossier docs/experiments/compiler-lto-eval/frozen-v1/dossier.json \
  --response gbt-history=docs/experiments/compiler-lto-eval/models/gbt-v1/history-response.json \
  --response gbt-calibrated=docs/experiments/compiler-lto-eval/models/gbt-calibrated-v2/response.json \
  --candidate-arm gbt-calibrated \
  --oracle-response docs/experiments/compiler-lto-eval/models/measured-oracle-v1/measured-oracle-response.json \
  --expected-reps 1,2,3,4,5,6,7,8,9,10 \
  --json docs/experiments/compiler-lto-eval/runs/compiler-lto-gbt-calibrated-balanced-v3/summary.json
```

Candidate model responses use
`build_compiler_lto_eval.sh candidate RESPONSE.json NAME`. That command
strictly validates the response, lowers the same source through LTO, and
records its binary hash. Provider invocation is intentionally outside
the build and is not part of this evidence set.
