# Producer-fission runtime triage

This fail-closed path separates a compiler implementation repair from the
validity of the Jacobi candidate.  A failed scout may be repaired at the pass
level, but the candidate remains model-invisible unless the already-declared
numerical equivalence test passes.

The triage must bind the original `pdebug` failure, the fixed compiler build,
and a baseline/fission pair executed serially inside one two-node allocation.
It uses the scout tolerances unchanged (`rel_tol=1e-6`, `abs_tol=1e-7`).  If
the pair differs beyond either tolerance, timings are diagnostic only: no
confirmation is launched, no speedup is reported as paper evidence, and the
candidate stays masked from both GBT and LLM inputs.

The report may label the missing producer-to-trigger ordering as a semantic
inference, not as a measured performance claim.  Application source changes
are outside this protocol; the failed and fixed builds must hash the same
application source.

`continue_producer_fission_runtime_triage.sh` refuses to submit when any user
job is active, launches one `pdebug` allocation, and blocks in the auditor
until Flux records a clean event.  The runner executes the two arms serially;
the controller rechecks every frozen artifact before accepting the report.
