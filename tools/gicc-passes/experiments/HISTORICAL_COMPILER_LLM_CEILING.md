# Historical compiler-only LLM capability ceiling

Status: reproducible exploratory evidence from the earlier `pci` archive. This
audit makes no provider call, submits no job, and does not change eligibility
or authorization for the upgraded `pdebug` decision suite.

## What the archive establishes

The frozen `llm-zero-shot-v1` protocol sent a source-free compiler-fact dossier
to the decision model in 20 independent sessions. Application source,
evaluation results, calibration labels, and tools were absent. All 20 responses
passed the strict bridge without fallback or semantic retry. They collapsed to
five route-action policies; all five were lowered by LTO and runtime-tested
from the same frozen application source SHA-256.

This supports a narrow feasibility statement: an LLM can select legal
compiler-level communication decisions that the compiler materializes into an
unchanged-source program. It does not yet show that relational language input
outperforms descriptor-only or opaque compiler input, because the historical
study did not contain that equal-authority ablation.

## Runtime interpretation

| role | policy | response frequency | default/candidate | paired 95% CI | allocations | exact sign p |
| --- | --- | ---: | ---: | --- | ---: | ---: |
| frequency-selected representative | `llm-policy01` | 10/20 | 1.021570x | [0.980814, 1.048598] | 10 | 0.021484 |
| post-hoc best of 20 | `llm-policy05` | 1/20 | 1.054465x | [1.038120, 1.071642] | 4 | 0.125000 |

The modal policy is the honest representative of typical sampling behavior;
its bootstrap interval crosses parity. Policy 05 is selected only after
inspecting runtime results for all five materialized policies. Its interval is
therefore post-selection and unadjusted, and four allocation replicates give
an exact sign-test p-value of 0.125. It is a useful observed capability ceiling,
not confirmatory evidence or typical model performance.

`LLM_SAMPLING_NULL.md` adds a chance calibration without reinterpreting the
archive as a uniform sample from the unknown legal space. Conditional only on
uniform draws over the five *observed* policies, 20 draws hit a predesignated
best policy with 98.8471% probability and cover all five with 94.2719%
probability. The observed 10/20 modal count is less compatible with that narrow
null (1.2974% probability that any policy reaches at least ten), indicating a
response preference but not a runtime benefit.

Mapping each of the 20 responses to its policy's separately measured point
estimate gives a frequency-weighted geometric mean of 1.022550x over compiler
default. This number is descriptive only: the five campaigns did not share
allocations and policy 05 has four measured replicates while the others have
ten, so no paired confidence interval is valid for that aggregate.

Every archived job ledger names `pci`. The data may be cited only as historical
exploratory evidence; it cannot satisfy the current `pdebug` runtime contract.
All five policies also choose only existing `default`/`proxy`/`trigger` routes,
so they do not test the upgraded relational structural transformations.

## Machine audit

`audit_historical_compiler_llm_ceiling.py` fails closed unless it verifies:

- the source-free 20-trial protocol and frozen source hash;
- 20 accepted responses, zero fallback, zero semantic retry, and an exact
  partition into five policies;
- representative response, dossier, materialized binary, policy-frequency,
  paired-replicate, raw-log, and job-ledger identities;
- exact byte-for-byte reproduction of the archived trial analysis and all five
  runtime summaries from their raw inputs;
- route-only transformation scope and the historical queue boundary.

The current generated report is
`build_ofi/historical_compiler_llm_ceiling_20260904/report.json`:

- audit ID:
  `sha256:0ae3dd141caff01b179788483c13b0f6d1d3081ee66cdb8f9b0c5397784a1aba`;
- serialized report SHA-256:
  `fa17072c7e34342b03b00ddb4907f201fba8a8d44fe579bf4ab1e2495924719e`;
- claim flags: compiler-only feasibility true; stable modal speedup, current
  `pdebug` performance, relational-context value, generalization, and upgraded
  suite eligibility false;
- no provider, compiler, or scheduler is invoked by the auditor.

The upstream analyzers are rerun first into
`build_ofi/historical_compiler_llm_ceiling_20260904/regenerated`. Use archived
`analysis.json` as the `--trial-analysis` path for runtime regeneration: the
analyzer records that provenance path, and using the byte-identical regenerated
path would intentionally prevent an exact serialized comparison. Then emit or
verify the audit with:

```sh
python3 tools/gicc-passes/experiments/audit_historical_compiler_llm_ceiling.py emit \
  --protocol docs/experiments/compiler-lto-eval/llm-zero-shot-v1/protocol.json \
  --analysis docs/experiments/compiler-lto-eval/llm-zero-shot-v1/analysis.json \
  --manifest docs/experiments/compiler-lto-eval/frozen-v1/manifest.json \
  --source examples/proxy/compiler_lto_eval.cpp \
  --runtime-root docs/experiments/compiler-lto-eval/runs \
  --regenerated-dir build_ofi/historical_compiler_llm_ceiling_20260904/regenerated \
  --trial-analyzer examples/proxy/analyze_compiler_lto_llm_trials.py \
  --runtime-analyzer examples/proxy/analyze_compiler_lto_llm_runtime.py \
  --out build_ofi/historical_compiler_llm_ceiling_20260904/report.json
```
