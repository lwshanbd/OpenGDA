---
name: gpu-pipeline-tune
description: Tune a GiOMP program's pipelining (which put sites are pipelined, grain, distribute chunk, teams, piece size) by proposing configurations, measuring them on the GPU cluster with the gtune tools, and revising from the measurements.
---

# Tuning GiOMP pipelines from measurements

You tune performance only. The compiler keeps the code correct: it reports
which sites are legal and which options each allows, and gtune_submit rejects
anything else before it is built. Every run is also checked against the
reference run's results; a candidate whose values differ is reported as
`check_failed` and must never be recommended.

## The tools

- `gtune_sites(app)`: the sites (S0, S1, ...) and kernels (K0, ...) with
  their legal options, the run-time and build knobs, and the reference run.
- `gtune_setup(app)`: once per application, when gtune_sites says there is no
  site report or no reference yet. Then `gtune_wait`.
- `gtune_submit(app, configs)`: up to 8 candidates in one round. They are
  built in parallel and run in one allocation, so batch the candidates of one
  hypothesis together instead of submitting them one by one.
- `gtune_wait(app, round)`: blocks up to 15 minutes. A round spends most of
  its time queued for nodes; call it again while it is `queued`, `running` or
  `building`. Do not submit another round while one is waiting.
- `gtune_history(app)`: the best trials so far and the failures.

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
- **`runtime.OMP_NUM_TEAMS` / `OMP_TEAMS_THREAD_LIMIT`**: launch geometry for
  every kernel. A GH200 has 132 SMs; multiples of it are natural choices.
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
2. **Propose a round** that tests one hypothesis with 3 to 6 candidates:
   vary the knob you believe matters, keep the rest at the default, include
   one point on each side of the value you expect to win. Put the hypothesis
   in each config's `note`.
3. **Read the round.** Compare each candidate's `vs_reference` with its
   `spread`. A difference smaller than about twice the spread is noise, not a
   result. Look at the breakdown to see *where* a change came from: a
   pipelined site should shrink the time outside kernels; a chunk change
   should move the kernel it targets.
4. **Revise.** Keep what won, drop what did not, and form the next hypothesis
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
