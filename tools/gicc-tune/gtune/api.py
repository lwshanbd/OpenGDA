"""The operations a tuning agent calls (the MCP server is a thin layer over these)."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from typing import Any

from . import core


def _start(app: core.App, kind: str, candidates: list[dict], repeats: int | None) -> int:
    n, rd = core.new_round(app)
    (rd / "plan.json").write_text(json.dumps(
        {"app": app.name, "kind": kind, "repeats": repeats, "candidates": candidates}, indent=1))
    core.write_status(rd, "starting")
    with open(rd / "worker.log", "w") as log:
        subprocess.Popen([sys.executable, "-m", "gtune.worker", str(rd)], cwd=core.ROOT,
                         stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    return n


def setup(app_name: str) -> dict[str, Any]:
    """Build the default configuration, record its sites, and time it."""
    app = core.App.load(app_name)
    n = _start(app, "setup", [{"trial": 0, "config": {}}], None)
    return {"round": n, "next": f"gtune_wait(app={app_name!r}, round={n})"}


def sites(app_name: str) -> dict[str, Any]:
    app = core.App.load(app_name)
    out = core.describe_sites(app)
    out["app"] = app.spec.get("description", app_name)
    out["reference"] = core.reference(app)
    return out


def submit(app_name: str, configs: list[dict], repeats: int | None = None) -> dict[str, Any]:
    app = core.App.load(app_name)
    if core.reference(app) is None:
        return {"error": "no reference run yet: call gtune_setup first and wait for it"}
    if not configs:
        return {"error": "give at least one configuration"}
    if len(configs) > 8:
        return {"error": "at most 8 candidates per round"}
    rejected, cands = [], []
    trial = core.next_trial(app)
    seen = set()
    for cfg in configs:
        errs = core.validate(app, cfg)
        if not errs:
            # A candidate that only restates defaults measures the default
            # again; the plain {} is the one way to ask for that.
            moved = core.changes(app, cfg)
            body = {k: v for k, v in cfg.items() if k != "note" and v}
            if body and not moved:
                errs.append("every choice here equals the default, so this is the default configuration "
                            "again; submit {} if a control is wanted (gtune_sites lists each default)")
            key = json.dumps(core.materialize(app, cfg), sort_keys=True)
            if key in seen:
                errs.append("the same configuration as another candidate in this round")
            seen.add(key)
        if errs:
            rejected.append({"config": cfg, "errors": errs})
        else:
            cands.append({"trial": trial, "config": cfg})
            trial += 1
    if not cands:
        return {"submitted": 0, "rejected": rejected}
    n = _start(app, "trial", cands, repeats)
    return {"round": n, "submitted": [c["trial"] for c in cands], "rejected": rejected,
            "next": f"gtune_wait(app={app_name!r}, round={n})"}


def _compact(app: core.App, r: dict[str, Any], ref: dict[str, Any] | None) -> dict[str, Any]:
    metric = app.spec["metric"]["name"]
    out = {"trial": r["trial"], "status": r["status"], "config": r["config"]}
    if r.get("why"):
        out["why"] = r["why"][:600]
    s = r.get("summary") or {}
    if metric in s:
        out[metric] = s[metric]
        out["spread"] = s.get("spread")
        if ref and metric in ref.get("summary", {}):
            out["vs_reference"] = f"{(ref['summary'][metric] / s[metric] - 1) * 100:+.1f}%"
    if r.get("breakdown"):
        out["breakdown"] = r["breakdown"]
    return out


def wait(app_name: str, round_no: int, max_wait: int = 900) -> dict[str, Any]:
    """Block until the round finishes or *max_wait* seconds pass."""
    app = core.App.load(app_name)
    rd = app.dir / "rounds" / str(round_no)
    if not rd.is_dir():
        return {"error": f"no round {round_no}"}
    deadline = time.time() + max(0, min(max_wait, 1100))
    while True:
        st = core.read_status(rd)
        if st["phase"] in ("done", "failed") or time.time() >= deadline:
            break
        time.sleep(10)
    if st["phase"] == "failed":
        return {"round": round_no, "phase": "failed", "error": st.get("error", "")}
    if st["phase"] != "done":
        alloc = rd / "alloc.log"
        if st["phase"] == "queued" and alloc.is_file() and "Granted" in alloc.read_text(errors="replace"):
            st["phase"] = "running"
        return {"round": round_no, "phase": st["phase"], "since": st.get("time"),
                "next": f"gtune_wait(app={app_name!r}, round={round_no}) again"}
    results = json.loads((rd / "results.json").read_text())
    ref = core.reference(app)
    out = {"round": round_no, "phase": "done",
           "results": [_compact(app, r, ref) for r in results]}
    if ref:
        out["reference"] = ref["summary"]
        if ref.get("breakdown"):
            out["reference_breakdown"] = ref["breakdown"]
    return out


def history(app_name: str, top: int = 10) -> dict[str, Any]:
    return core.history(core.App.load(app_name), top)
