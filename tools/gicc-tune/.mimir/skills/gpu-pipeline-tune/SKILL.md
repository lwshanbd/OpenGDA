---
name: gpu-pipeline-tune
description: Tune a GiOMP program's pipelining (which put sites are pipelined, grain, distribute chunk, kernel split around the fence, teams, piece size) by proposing configurations, measuring them on the GPU cluster with the gtune tools, and revising from the measurements.
---

# Tuning GiOMP pipelines from measurements

You tune performance only. The compiler keeps the code correct: it reports
which sites are legal and which options each allows, and gtune_submit rejects
anything else before it is built. Every run is also checked against the
reference run's results; a candidate whose values differ is reported as
`check_failed` and must never be recommended.

## The tools

- `gtune_sites(app)`: the sites (S0, S1, ...) and kernels (K0, ...) with
  their legal options and their **default** (what the build does when the
  configuration says nothing), the run-time and build knobs, and the
  reference run. Each kernel carries `reference_gpu_s`, its GPU time in the
  reference run, and each site `kernel_ran_in_reference`: a kernel at 0 s
  and the sites on it cannot change the time, so leave them alone.
- `gtune_setup(app)`: once per application, when gtune_sites says there is no
  site report or no reference yet. Then `gtune_wait`.
- `gtune_submit(app, configs)`: up to 8 candidates in one round. They are
  built in parallel and run in one allocation, so batch the candidates of one
  hypothesis together instead of submitting them one by one.
- `gtune_wait(app, round)`: blocks up to 15 minutes. A round spends most of
  its time queued for nodes; call it again while it is `queued`, `running` or
  `building`. Do not submit another round while one is waiting.
- `gtune_history(app)`: the best trials so far, the failures, and
  `never_varied`: the choices no trial has moved off their default yet.

A candidate whose every choice equals the default is rejected: it would only
measure the default again. Submit `{}` when you want the default as a
control. In particular, a legal site is already pipelined by default, so
`"pipeline": true` changes nothing; the experiment is `"pipeline": false`.

## What the knobs do

- **Site `pipeline`** (puts after a kernel only): `true` lets the kernel send
  the halo itself as its chunks finish writing it; `false` leaves the put to
  the host after the kernel. A pipelined put to a cross-node peer pays a fixed
  counting cost of the order of 100 microseconds per launch, so it only pays
  off when the kernel writing the range runs long enough to hide the
  transfer, and the range is large (at least hundreds of KB).
- **Site `grain`**: `element` mirrors each store to a same-node peer and sends
  the rest at kernel start; `block` sends whole distribute blocks as each one
  finishes. A box site only allows `element`.
- **Kernel `dist_chunk`**: iterations per distribute chunk (power of two).
  Larger chunks cut per-chunk overhead; too large leaves teams idle at the
  end (load imbalance) and, for a pipelined kernel, delays the first send.
  Kernels marked `pipelined` count chunks, so their chunk also sets how
  finely the send overlaps the compute.
- **Kernel `split`** (only on kernels marked `splittable`; off by default):
  the fence before the kernel is deferred, and the kernel runs as two or
  three launches -- first the iterations that touch none of the data the
  fence waits for, overlapping the transfers, then the fence, then the rest.
  It pays when the transfers before the kernel take long and most of its
  iterations do not touch them (a stencil reading only an edge halo). It
  costs a launch per extra range (tens of microseconds even when tiny), an
  exchange of transfer lists among all ranks, and a rank with neighbours on
  both sides runs one launch more than an edge rank. A round with split on
  reports `split_stats` per rank: `before_s` is the wait for the other ranks'
  lists (includes waiting for slower ranks), `inside_s` the overlapped part,
  `fence_s` what was left of the transfers, `outside_s` the rest; counters of
  launches that could not split say why (`not_split_no_fence`: no fence was
  pending). Split composes with the sites' `pipeline`: a pipelined site sends
  during its own kernel, a split kernel hides what remains after it.
- **`runtime.OMP_NUM_TEAMS` / `OMP_TEAMS_THREAD_LIMIT`**: launch geometry for
  every kernel. A GH200 has 132 SMs; multiples of it are natural choices.
  The runtime's own choice (unset) has been at least as good as any fixed
  value on every program tuned so far, so one round on this axis is enough
  unless a kernel is visibly under-occupied.
- **`build.OMPX_PIPE_PIECE_SHIFT`**: a pipelined put is split into pieces of
  about 2^shift bytes (default 22, 4 MB). Smaller pieces start sending
  earlier but cost more messages.
- **`build.OMPX_PIPE_COUNT_MIN_LOG2`**: a pipelined put smaller than 2^n bytes
  (default 20, 1 MB) to a cross-node peer is not sent by the kernel but waits
  for the next quiet. Lowering it lets smaller halos be pipelined.

## The loop

1. **Read the program.** `gtune_sites`. Look at the reference breakdown: which
   kernels take the time, how large the time outside kernels is (that is
   communication and host work). State in one or two sentences where you
   think time can be saved and why.
   Before the first round, list the axes that can matter (the knobs of the
   kernels that take the time, the sites on them, the build knobs if any
   site is pipelined, the launch geometry) and plan to touch each of them
   at least once within the budget. Do not spend more than one round on an
   axis that has shown no gain, unless the breakdown points at it.
2. **Propose a round** that tests one hypothesis with 3 to 6 candidates:
   vary the knob you believe matters, keep the rest at the default, include
   one point on each side of the value you expect to win. Put the hypothesis
   in each config's `note`.
3. **Read the round.** Compare each candidate's `vs_reference` with its
   `spread`. A difference smaller than about twice the spread is noise, not a
   result. Look at the breakdown to see *where* a change came from: a
   pipelined site should shrink the time outside kernels; a chunk change
   should move the kernel it targets.
4. **Revise.** Check `never_varied` in `gtune_history` before planning the
   next round, and prefer an axis that is still untested over refining one
   that has already been mapped. Keep what won, drop what did not, and form the next hypothesis
   from the measurements, not from the previous guess. Combine winning
   settings only after each has won on its own, then confirm the combination
   in its own round.
5. **Confirm.** Before reporting a best configuration, run it again in a
   round with the default configuration `{}` alongside it, with `repeats` 3 or
   more, so the gain is measured against a reference timed in the same
   allocation.
6. **Stop** when two consecutive rounds bring no gain beyond noise, or after
   the budget the user gave you.

## The report

End with: the best configuration (as the JSON you submitted), its time and
gain against the reference measured in the confirmation round, the spread,
what each setting contributed, the hypotheses that turned out wrong, and any
candidate that failed its check (with the reason). Say plainly when no
configuration beats the default by more than the noise.
