# Analytic sampling null for the compiler-only LLM protocol

Status: machine-audited chance calibration; no model/provider/compiler/
scheduler invocation, no source access, and no runtime claim.

The capability protocol takes 20 samples per information view.  A
best-of-20 result is meaningful only relative to the size of the exact
graph-bound policy space.  `audit_llm_sampling_null.py` freezes the analytic
reference for independent uniform draws with exactly one predesignated oracle.
This is a chance baseline, not an assertion that an LLM samples uniformly.

## Current-suite calibration

| Entry | Legal policies | Probability that 20 uniform draws hit the oracle at least once | Exact-oracle hits needed for one-sided 5% tail |
|---|---:|---:|---:|
| `coalescing_placement` | 4096 | 0.4872% | 1 |
| `collective_n8` | 4096 | 0.4872% | 1 |
| `jacobi` | 9 | 90.5169% | 6 |
| `minimod` | 9 | 90.5169% | 6 |
| `mixed_lto` | 9 | 90.5169% | 6 |
| `mm_minimal` | 3 | 99.9699% | 11 |
| `loop_lto` | 2 | 99.9999% | 15 |

This changes how the paper must describe a capability ceiling.  One exact
oracle hit among 20 is already unusual in either 4096-policy task, but is
expected in the 2–9-policy tasks.  For the smaller spaces, repeated
exact-oracle selection, regret, and improvement over the equal-authority
opaque/descriptors views matter more than merely discovering the best policy
once.

The conditional compiler frontier is reported separately because none of its
candidates is model-visible yet.  If all three runtime gates and refreezes
pass, the corresponding action-space sizes become 10, 4, and 3; their
uniform-null oracle-hit probabilities in 20 draws are 87.8423%, 99.6829%, and
99.9699% respectively.

## Historical best-of-20 interpretation

The historical archive observed five distinct materialized policies in 20
accepted responses.  Conditional on uniform sampling over that *observed*
five-policy support—not the unknown full legal space:

- the probability of sampling a predesignated best policy at least once is
  98.8471%;
- the probability of observing all five policies is 94.2719%;
- the probability that any policy appears at least the observed 10 times is
  1.2974%.

Thus seeing all five policies and choosing their post-hoc runtime winner does
not by itself demonstrate compiler reasoning.  The 10/20 modal frequency does
show a nonuniform response preference under this narrow null, but its paired
runtime interval still crosses parity.  It supports neither stable speedup nor
relational-context value.

The report ID is
`sha256:ad2b9a225c5ff05a0f05965ec2100544d05eee66853a29e3fc69a177eba54b6e`.
It binds the suite, action-authority audit, conditional frontier, historical
archive audit, and the null auditor itself.  Exact rational probabilities are
stored alongside their decimal representations.

## Reproduction

From the repository root:

```sh
python3 tools/gicc-passes/experiments/audit_llm_sampling_null.py \
  --suite build_ofi/compiler_decision_suite_20260904/suite.json \
  --prompt-dir build_ofi/compiler_decision_suite_20260904/prompts \
  --action-authority build_ofi/compiler_action_authority_20260904/report.json \
  --conditional-frontier build_ofi/compiler_action_frontier_20260904/report.json \
  --historical build_ofi/historical_compiler_llm_ceiling_20260904/report.json \
  --out build_ofi/llm_sampling_null_20260904/report.json
```
