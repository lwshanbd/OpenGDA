# Reused loop descriptor oracle scout

## Frozen question

Does compiler-materialized reuse of one loop-invariant DWQ descriptor reduce
end-to-end time relative to the existing compiler-generated descriptor-array
batch, while preserving the original number and order of network operations?

This is an offline compiler oracle scout. It does not evaluate an LLM and
cannot support a paper performance claim by itself.

## Frozen artifact

- build commit: `aff76f948be393a05f4e103e2317e27f7425674b`
- artifact directory:
  `build_ofi/reused_loop_descriptor_oracle_aff76f9_20260904`
- unchanged application source SHA-256:
  `a116f2148923d5678c865e134c3c09e69e8b0645c80db1b160e1fde6aea2a143`
- baseline binary SHA-256:
  `2be1526aa6fb64c6f567d11fcdd4c15d227f969a7bc4db36cfe413d01a814559`
- reused binary SHA-256:
  `6797a87646f22d83c3e19e63610739766b2557fae21e327b6b6482582d307dfc`
- final IR audit SHA-256:
  `5a696217891099c672dbf559b303a7a88ed2fb2ae98c271b8e589e95d212912f`

The audit proves identical compiler facts and kernel metadata, four original
kernel-launch call sites in both arms, and byte-identical optimized device
kernel bodies. Baseline uses the array-batch helper; reuse uses the scalar
repeated-descriptor helper and has no descriptor arrays.

## Execution

- queue: `pdebug` only;
- one allocation: 2 nodes, 2 ranks, one rank/GPU per node;
- this scout may start only after every earlier experiment in the serial
  chain is terminal; it must never overlap another job owned by this study;
- immediately before submission, the successor waits until the account has
  no active or queued Flux job; it never cancels a job owned by another
  agent;
- runtime batches: 4 and 64, fixed before execution from the historical
  crossover evidence;
- six paired replicates per batch;
- orders: `AB, BA, BA, AB, AB, BA`, where A is baseline and B is reuse;
- every arm executes the canonical 16 message sizes already embedded in the
  unchanged application.

All 24 program invocations reuse the same allocation and node pair. No retry
is permitted inside a pair: a failed run fails the scout.

## Correctness and provenance gate

The monitor must reject unless:

1. the Flux jobspec and resource set match `pdebug`, N2, n2;
2. every content-addressed artifact still has its frozen hash;
3. each log has exactly the canonical 16 size rows and the requested batch;
4. every printed iteration count is `21 * batch`;
5. no `VERIFY-FAIL`, fatal, timeout, or scheduler exception appears;
6. each enqueue audit is exact:
   `(10 warmup + 21 measured) * batch * 16 sizes`;
7. every timing is finite and positive.

The final-IR audit binds arm identity; runtime enqueue counts bind preservation
of all `n` network operations. A fast result with missing work is invalid.

## Frozen descriptive scout gate

The mechanism-targeted small-message stratum is fixed as 1B through 64KB
(the first ten canonical sizes), where host descriptor preparation is not
hidden by multi-megabyte wire time.

Let each paired replicate contribute the geometric mean of
`baseline_median / reused_median` over that stratum. The oracle is promising
only if all conditions hold:

- at least one batch has at least 4/6 winning pairs and median paired speedup
  at least 1.02;
- the geometric mean across all 12 batch/replicate clusters is at least 1.01;
- neither batch has median paired speedup below 0.98.

Passing permits an independently frozen multi-allocation confirmation and a
new profile-gated compiler graph. It does not permit a provider/model call.
Failure keeps `reused_loop_descriptor` model-invisible.
