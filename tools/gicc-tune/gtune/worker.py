"""Runs one round in the background: build every candidate, then run them all
in one allocation.

    python -m gtune.worker <round dir>

The round directory holds plan.json:

    {"app": "miniweather", "kind": "setup" | "trial", "repeats": 2,
     "candidates": [{"trial": 7, "config": {...}}, ...]}

A setup round builds the default configuration once, records the sites the
compiler reports (sites.json) and the reference run (reference.json). A
trial round checks each candidate against the reference and appends it to
the ledger. Progress goes to status.json, the outcome to results.json.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from . import core


def ssh(app: core.App, cmd: str, log: Path, timeout: int) -> int:
    with open(log, "w") as f:
        try:
            return subprocess.run(["ssh", "-o", "BatchMode=yes", app.spec["host"], cmd],
                                  stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            f.write(f"\n[gtune] timed out after {timeout} s\n")
            return 124


def build(app: core.App, bd: Path, mat: dict) -> tuple[bool, str]:
    bd.mkdir(parents=True, exist_ok=True)
    (bd / "tune.json").write_text(json.dumps(mat["tune"], indent=1))
    sites = bd / "sites.jsonl"
    sites.unlink(missing_ok=True)
    env = dict(app.spec["build"].get("env", {}))
    env.update({"GICC_TUNE_IN": str(bd / "tune.json"), "GICC_SITES_OUT": str(sites),
                "GTUNE_DEFS": mat["defs"]})
    exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items())
    script = core.ROOT / app.spec["build"]["script"]
    cmd = f"cd {shlex.quote(str(bd))} && env {exports} bash {shlex.quote(str(script))} {bd / 'exe'}"
    rc = ssh(app, cmd, bd / "build.log", timeout=1800)
    ok = rc == 0 and (bd / "exe").exists()
    tail = "" if ok else "\n".join((bd / "build.log").read_text(errors="replace").splitlines()[-15:])
    return ok, tail


def run_script(app: core.App, rd: Path, jobs: list[dict], repeats: int) -> Path:
    r = app.spec["run"]
    lines = ["#!/bin/bash", r.get("setup", ""), f"cd {shlex.quote(str(rd))}"]
    # Repeats interleaved across candidates, so drift in the machine spreads
    # over all of them rather than landing on one.
    for rep in range(repeats):
        for j in jobs:
            env = " ".join(f"{k}={shlex.quote(v)}" for k, v in j["env"].items())
            lines.append(f"env {env} {r['launch']} {j['exe']} {r.get('args', '')} "
                         f"> {j['tag']}.r{rep}.out 2>&1 </dev/null")
    # One profiled run each, apart from the timed ones: Nsight Systems on every
    # rank, then rank 0's kernel summary as CSV. prof_args, when given, replace
    # args for this run (to leave out work the timed region does not cover).
    nsys = r.get("nsys")
    pargs = r.get("prof_args", r.get("args", ""))
    for j in jobs if nsys else []:
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in j["env"].items())
        rep = f"{j['tag']}.nsys.%q{{SLURM_PROCID}}"
        lines.append(f"env {env} {r['launch']} {nsys} profile -t cuda -o {rep} --force-overwrite true "
                     f"{j['exe']} {pargs} > {j['tag']}.prof.out 2>&1 </dev/null")
        lines.append(f"{nsys} stats -q -r cuda_gpu_kern_sum -f csv -o {j['tag']} --force-overwrite true "
                     f"{j['tag']}.nsys.0.nsys-rep > /dev/null 2>&1")
    p = rd / "run.sh"
    p.write_text("\n".join(lines) + "\n")
    return p


def main(rd: Path) -> None:
    plan = json.loads((rd / "plan.json").read_text())
    app = core.App.load(plan["app"])
    setup = plan["kind"] == "setup"
    repeats = int(plan.get("repeats") or app.spec["run"].get("repeats", 2))
    cands = plan["candidates"]

    # Builds, one per distinct compile.
    keys: dict[str, int] = {}
    for c in cands:
        c["build"] = keys.setdefault(core.build_key(app, c["config"]) if not setup else "ref", len(keys))
    core.write_status(rd, "building", builds=len(keys))
    mats = {}
    for c in cands:
        mats.setdefault(c["build"], core.materialize(app, c["config"]) if not setup
                        else {"tune": {}, "defs": "", "env": {}})
    with ThreadPoolExecutor(max_workers=int(app.spec["build"].get("parallel", 4))) as ex:
        built = dict(zip(mats, ex.map(lambda b: build(app, rd / f"b{b}", mats[b]), mats)))

    if setup:
        lines = (rd / "b0" / "sites.jsonl").read_text().splitlines() if (rd / "b0" / "sites.jsonl").is_file() else []
        (app.dir / "sites.json").write_text(json.dumps(core.parse_sites(lines), indent=1))

    jobs = []
    results = []
    for c in cands:
        ok, tail = built[c["build"]]
        if not ok:
            results.append({"trial": c["trial"], "config": c["config"], "status": "build_failed", "why": tail})
            continue
        env = core.materialize(app, c["config"])["env"] if not setup else {}
        jobs.append({"cand": c, "tag": f"t{c['trial']}", "exe": str(rd / f"b{c['build']}" / "exe"), "env": env})

    if jobs:
        script = run_script(app, rd, jobs, repeats)
        minutes = min(120, 5 + len(jobs) * (repeats + 1) * int(app.spec["run"].get("minutes_per_run", 1)))
        alloc = app.spec["run"]["alloc"].format(minutes=minutes)
        core.write_status(rd, "queued", runs=len(jobs) * (repeats + 1))
        ssh(app, f"{alloc} bash {shlex.quote(str(script))}", rd / "alloc.log", timeout=6 * 3600)
        core.write_status(rd, "parsing")

    ref = core.reference(app)
    metric = app.spec["metric"]["name"]
    for j in jobs:
        c = j["cand"]
        runs, errors = [], []
        for rep in range(repeats):
            f = rd / f"{j['tag']}.r{rep}.out"
            text = f.read_text(errors="replace") if f.is_file() else ""
            got = core.parse_run(app, text)
            if metric not in got:
                errors.append(f"run {rep}: no {metric} in the output: " + " | ".join(text.splitlines()[-6:]))
                continue
            runs.append(got)
        row = {"trial": c["trial"], "config": c["config"], "runs": runs, "summary": core.summarize(app, runs)}
        if not runs:
            row.update(status="run_failed", why="; ".join(errors))
        else:
            bad = [] if setup or ref is None else core.check(app, runs[0], ref["values"])
            row.update(status="check_failed" if bad else "ok", why="; ".join(bad + errors))
        # The kernels' GPU time from the profiled run, set against the timed
        # runs' median: the profiler slows the run it watches.
        row["breakdown"] = core.kernel_breakdown(rd / f"{j['tag']}_cuda_gpu_kern_sum.csv",
                                                 row["summary"].get(metric))
        results.append(row)

    if setup:
        ok = [r for r in results if r["status"] == "ok"]
        if ok:
            vals = {k: v for k, v in ok[0]["runs"][0].items() if k != metric}
            core_ref = {"summary": ok[0]["summary"], "values": vals, "breakdown": ok[0].get("breakdown")}
            (app.dir / "reference.json").write_text(json.dumps(core_ref, indent=1))
    else:
        core.append_ledger(app, results)
    (rd / "results.json").write_text(json.dumps(results, indent=1))
    core.write_status(rd, "done", ok=sum(r["status"] == "ok" for r in results), total=len(results))


if __name__ == "__main__":
    rd = Path(sys.argv[1])
    try:
        main(rd)
    except Exception:
        core.write_status(rd, "failed", error=traceback.format_exc()[-2000:])
        raise
