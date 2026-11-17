#!/usr/bin/env python3
"""confirm.py OUTDIR LABEL=APP:TRIAL|APP:/exe[:ENV=VAL,...] ... [--reps N] [--nodes N]

Step 8 of the campaign: one allocation, every configuration run N times
(default 3), interleaved, each with the binary and environment of a trial the
tuner already built and ran (TRIAL from APP's ledger; 0 is the reference).
Extra ENV=VAL pairs are added to that trial's environment (the split control:
OMPX_SPLIT=0). Writes OUTDIR/confirm.sbatch; after the job, `confirm.py
OUTDIR --parse` prints median and spread per label from OUTDIR/confirm.out."""
import json, re, sys, os, glob, statistics, shlex
G = '/shared/data1/Users/l1065028/new-gicc/tools/gicc-tune'

def exe_cmd(app, exe):
    r = spec(app)['run']
    return f"env OMPX_SPLIT_STATS=1 {r['launch']} {exe} {r['args']}"

def trial_cmd(app, trial):
    for rs in sorted(glob.glob(f'{G}/runs/{app}/rounds/*/run.sh'), key=lambda p: int(p.split('/')[-2])):
        for line in open(rs):
            if re.search(rf'> t{trial}\.r0\.out ', line) and ' nsys ' not in line:
                return line.split(' > ')[0]
    raise SystemExit(f'{app}: trial {trial} not found')

def spec(app):
    return json.load(open(f'{G}/apps/{app}/app.json'))

def write(outdir, items, reps, nodes):
    os.makedirs(outdir, exist_ok=True)
    lines = ['#!/bin/bash', '#SBATCH -p maple', '#SBATCH -A app', f'#SBATCH -N {nodes}',
             '#SBATCH --ntasks-per-node=1', '#SBATCH --cpus-per-task=4', '#SBATCH --gres=gpu:1',
             '#SBATCH --mem=0', '#SBATCH -t 90',
             '# Confirmation round: every configuration in one allocation, interleaved.']
    lines.append(f'cd {outdir}')
    for r in range(reps):
        for label, app, trial, extra in items:
            cmd = exe_cmd(app, trial) if isinstance(trial, str) else trial_cmd(app, trial)
            if extra:
                cmd = cmd.replace('env ', 'env ' + ' '.join(extra) + ' ', 1)
            # Each run in a subshell with its own app's setup: two apps of one
            # program may differ only there (MiniMod's GICC_HALO_OVERLAP).
            setup = spec(app)['run']['setup']
            lines.append(f'echo "=== {label} r{r}"; ( {setup}; {cmd} ) 2>&1 | tail -60')
    lines.append('echo CONFIRM_DONE')
    open(f'{outdir}/confirm.sbatch', 'w').write('\n'.join(lines) + '\n')
    json.dump([[l, a, t, e] for l, a, t, e in items], open(f'{outdir}/items.json', 'w'))

def parse(outdir):
    items = json.load(open(f'{outdir}/items.json'))
    text = open(f'{outdir}/confirm.out').read()
    blocks = re.split(r'^=== (\S+) r\d+$', text, flags=re.M)
    vals, fails = {}, {}
    for i in range(1, len(blocks), 2):
        label, body = blocks[i], blocks[i + 1]
        app = next(a for l, a, _, _ in items if l == label)
        s = spec(app)
        m = re.search(s['metric']['regex'], body)
        ok = all(re.search(c['regex'], body) for c in s.get('checks', []))
        if m and ok:
            vals.setdefault(label, []).append(float(m.group(1)))
        else:
            fails[label] = fails.get(label, 0) + 1
    for l, a, t, e in items:
        v = vals.get(l, [])
        if v:
            med = statistics.median(v)
            print(f'{l:28s} median {med:.5g}  spread {(max(v)-min(v))/med*100:.1f}%  runs {v}' +
                  (f'  FAILED {fails[l]}' if l in fails else ''))
        else:
            print(f'{l:28s} no valid run (failed {fails.get(l, 0)})')

if __name__ == '__main__':
    outdir = sys.argv[1]
    if '--parse' in sys.argv:
        parse(outdir); sys.exit()
    reps = int(sys.argv[sys.argv.index('--reps') + 1]) if '--reps' in sys.argv else 3
    nodes = int(sys.argv[sys.argv.index('--nodes') + 1]) if '--nodes' in sys.argv else 2
    items = []
    for a in sys.argv[2:]:
        if a.startswith('--') or a.isdigit():
            continue
        label, rest = a.split('=', 1)
        parts = rest.split(':')
        # APP:TRIAL[:ENV,...], or APP:/path/to/exe[:ENV,...] for a binary the
        # tuner did not build (the step 6 cross-check), run as APP runs.
        t = parts[1] if parts[1].startswith('/') else int(parts[1])
        items.append((label, parts[0], t, parts[2].split(',') if len(parts) > 2 else []))
    write(outdir, items, reps, nodes)
    print(f'wrote {outdir}/confirm.sbatch')
