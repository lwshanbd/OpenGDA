# Final compiler-only suite audit v2

This successor starts only after the versioned N6 scout/confirmation chain has
reached a terminal state.  It never submits a scheduler job and never invokes a
model or provider.

The producer-fission, guarded-early-trigger, and reused-loop-descriptor
candidates remain absent from the model-visible graphs.  Their distinct
terminal-negative records are replayed and bound by
`audit_compiler_terminal_negatives.py`; an interrupted scout is not relabeled
as a completed scout.

If and only if the topology-matched N6 confirmation state is `confirmed`, the
successor replaces `collective_n8` with `collective_n6`.  The replacement must
preserve all non-collective entries and the frozen 4,096-policy compiler action
space.  Every other N6 terminal state retains the original N8 suite and grants
no new authority.

The ordered offline stages are terminal-negative audit, input-separation
audit, readiness audit, action-authority audit, sampling-null audit, and
capability-protocol audit.  All reports state that provider-call authority is
zero.  A later successor may freeze one exact content-addressed request, but it
also has no provider-call path.
