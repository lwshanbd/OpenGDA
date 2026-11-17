#!/usr/bin/env python3
"""mkg.py BASE NAME BUILD_JSON START_JSON [SETUP_REPLACE]: write the GiOMP app
NAME for the second tuning campaign: BASE's run, metric and checks, the given
build, pipeline-only scope, the pipeline build and run-time knobs, and START
(C_mpi mapped onto this program's kernels) as its starting point."""
import json, sys, os
A = '/shared/data1/Users/l1065028/new-gicc/tools/gicc-tune/apps'
base, name, build, start = sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4])
g = json.load(open(f'{A}/{base}/app.json'))
m = {k: v for k, v in g.items() if k not in ('runtime_knobs', 'build_knobs', 'start', 'pipeline_only', 'compute_only')}
m['name'] = name
m['build'] = build
m['pipeline_only'] = True
if len(sys.argv) > 5:
    old, new = sys.argv[5].split('=>')
    m['run']['setup'] = m['run']['setup'].replace(old, new)
m['description'] = (g['description'].split(' Starting point:')[0] +
    " Starting point: the compiler's default pass choices, with each kernel's distribute chunk and launch "
    "geometry as tuned for the MPI version of the program (the same kernels).")
m['build_knobs'] = {
    "OMPX_PIPE_PIECE_SHIFT": {"min": 12, "max": 24, "default": 22, "note": "pipeline piece size is 2^shift bytes (at most 64 pieces per put)"},
    "OMPX_PIPE_COUNT_MIN_LOG2": {"min": 10, "max": 24, "default": 20, "note": "puts smaller than 2^n bytes to a cross-node peer wait for the next quiet instead of being sent by the kernel"},
    "OMPX_DYN_PIECE_SHIFT": {"min": 12, "max": 22, "default": 17, "note": "smallest piece run-time counting sends, 2^shift bytes"},
}
m['runtime_knobs'] = {
    "OMPX_SPLIT_CHAIN": {"min": 0, "max": 1, "default": 1, "note": "1: each split kernel runs only its window before the fence and the rests of all of them run after one fence; 0: every split kernel makes the fence itself"},
    "OMPX_SPLIT_STABLE": {"min": 0, "max": 1, "default": 0, "note": "1: once the per-epoch transfer records repeat (period 1 or 2), stop exchanging them and reuse them; the run aborts if a later epoch differs"},
    "OMPX_FENCE_NEIGHBORS": {"min": 0, "max": 1, "default": 1, "note": "1: with the records locked by OMPX_SPLIT_STABLE, a deferred fence waits only on the ranks it exchanges with; no effect without OMPX_SPLIT_STABLE=1"},
}
if start:
    m['start'] = start
os.makedirs(f'{A}/{name}', exist_ok=True)
json.dump(m, open(f'{A}/{name}/app.json', 'w'), indent=2)
open(f'{A}/{name}/app.json', 'a').write('\n')
print('wrote', name)
