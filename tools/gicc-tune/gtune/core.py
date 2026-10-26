"""Tuning state for one application: its tunable sites, the trials run so far.

Everything lives under <tune root>/runs/<app>/:

    sites.json        the sites the reference build reported, with short aliases
    reference.json    the reference run: its time and the values the checks compare
    rounds/<n>/       one round of candidates: plan.json, per-candidate builds,
                      logs, status.json, results.json
    ledger.jsonl      one line per candidate, appended as each round finishes

A configuration names sites and kernels by alias (S0, K3) so a model never
copies the compiler's mangled names:

    {"sites":   {"S0": {"pipeline": false}},
     "kernels": {"K3": {"dist_chunk": 16384}},
     "build":   {"OMPX_PIPE_PIECE_SHIFT": 18},
     "runtime": {"OMP_NUM_TEAMS": 528}}

validate() rejects any choice the compiler did not report as legal, so an
invalid configuration never reaches a build.
"""

from __future__ import annotations

import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def runs_dir(app: str) -> Path:
    return Path(os.environ.get("GTUNE_RUNS", ROOT / "runs")) / app


@dataclass
class App:
    name: str
    spec: dict[str, Any]

    @classmethod
    def load(cls, name: str) -> "App":
        path = ROOT / "apps" / name / "app.json"
        if not path.is_file():
            known = sorted(p.name for p in (ROOT / "apps").iterdir() if (p / "app.json").is_file())
            raise ValueError(f"no app {name!r}; known: {', '.join(known)}")
        return cls(name, json.loads(path.read_text()))

    @property
    def dir(self) -> Path:
        d = runs_dir(self.name)
        d.mkdir(parents=True, exist_ok=True)
        return d


# ---- sites ------------------------------------------------------------------

def parse_sites(lines: list[str]) -> dict[str, Any]:
    """The GICC_SITES_OUT records of one build, deduplicated, with aliases."""
    sites: dict[str, dict] = {}
    kernels: dict[str, dict] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        rec.pop("chosen", None)
        if rec.get("kind") == "site":
            sites.setdefault(rec["id"], rec)
        elif rec.get("kind") == "kernel":
            k = kernels.setdefault(rec["id"], rec)
            k.update({key: v for key, v in rec.items() if key not in k})
    out: dict[str, Any] = {"sites": {}, "kernels": {}}
    for i, sid in enumerate(sorted(sites)):
        out["sites"][f"S{i}"] = sites[sid]
    for i, kid in enumerate(sorted(kernels)):
        out["kernels"][f"K{i}"] = kernels[kid]
    return out


def short_kernel(name: str) -> str:
    """The source function and line in an offload kernel's name."""
    line = re.search(r"_l(\d+)$", name)
    m = re.search(r"_Z(\d+)", name)
    fn = name[m.end():m.end() + int(m.group(1))] if m else name
    return f"{fn}:{line.group(1)}" if line else fn


def describe_sites(app: App) -> dict[str, Any]:
    s = load_sites(app)
    sites = []
    for alias, rec in s["sites"].items():
        sites.append({"alias": alias, "kernel": short_kernel(rec["kernel"]), "form": rec.get("form"),
                      "verdict": rec["verdict"], "why": rec.get("why", ""),
                      "options": rec.get("options", {}), "box": rec.get("box")})
    kernels = []
    for alias, rec in s["kernels"].items():
        kernels.append({"alias": alias, "kernel": short_kernel(rec["id"]),
                        "pipelined": bool(rec.get("pipelined")),
                        "default_chunk": rec.get("default", "clang default (one element per thread)")})
    return {"sites": sites, "kernels": kernels,
            "runtime_knobs": app.spec.get("runtime_knobs", {}),
            "build_knobs": app.spec.get("build_knobs", {})}


def load_sites(app: App) -> dict[str, Any]:
    p = app.dir / "sites.json"
    if not p.is_file():
        raise RuntimeError("no site report yet: run the setup round first (gtune_setup)")
    return json.loads(p.read_text())


# ---- configurations -----------------------------------------------------------

def _is_pow2(n: int) -> bool:
    return n > 0 and n & (n - 1) == 0


def validate(app: App, cfg: dict[str, Any]) -> list[str]:
    """Every reason *cfg* is not a legal configuration; empty when it is."""
    errs: list[str] = []
    if not isinstance(cfg, dict):
        return ["a configuration is a JSON object"]
    unknown = set(cfg) - {"sites", "kernels", "build", "runtime", "note"}
    if unknown:
        errs.append(f"unknown sections {sorted(unknown)}; use sites, kernels, build, runtime, note")
    s = load_sites(app)
    for alias, choice in (cfg.get("sites") or {}).items():
        rec = s["sites"].get(alias)
        if rec is None:
            errs.append(f"site {alias}: no such site (have {', '.join(s['sites']) or 'none'})")
            continue
        if rec["verdict"] != "LEGAL":
            errs.append(f"site {alias}: the compiler found it {rec['verdict']}, nothing can be chosen")
            continue
        opts = rec.get("options", {})
        for key, val in (choice or {}).items():
            if key not in ("pipeline", "grain"):
                errs.append(f"site {alias}: unknown choice {key!r}")
            elif val not in opts.get(key, []):
                errs.append(f"site {alias}: {key}={val!r} is not legal here; legal: {opts.get(key, [])}")
    for alias, choice in (cfg.get("kernels") or {}).items():
        if alias not in s["kernels"]:
            errs.append(f"kernel {alias}: no such kernel")
            continue
        for key, val in (choice or {}).items():
            if key != "dist_chunk":
                errs.append(f"kernel {alias}: unknown choice {key!r}")
            elif not isinstance(val, int) or not (_is_pow2(val) and 32 <= val <= 1 << 22):
                errs.append(f"kernel {alias}: dist_chunk must be a power of two in [32, 4194304]")
    for section, knobs in (("build", app.spec.get("build_knobs", {})),
                           ("runtime", app.spec.get("runtime_knobs", {}))):
        for key, val in (cfg.get(section) or {}).items():
            k = knobs.get(key)
            if k is None:
                errs.append(f"{section} knob {key!r} is not tunable here; have {sorted(knobs)}")
            elif not isinstance(val, int) or not (k["min"] <= val <= k["max"]):
                errs.append(f"{section} knob {key}={val!r}: must be an integer in [{k['min']}, {k['max']}]")
    return errs


def materialize(app: App, cfg: dict[str, Any]) -> dict[str, Any]:
    """The tuning file, -D flags and run-time environment *cfg* stands for."""
    s = load_sites(app)
    tune = {"sites": {}, "kernels": {}}
    for alias, choice in (cfg.get("sites") or {}).items():
        tune["sites"][s["sites"][alias]["id"]] = choice
    for alias, choice in (cfg.get("kernels") or {}).items():
        tune["kernels"][s["kernels"][alias]["id"]] = choice
    defs = []
    for key, val in (cfg.get("build") or {}).items():
        if key == "OMPX_PIPE_COUNT_MIN_LOG2":
            defs.append(f"'-DOMPX_PIPE_COUNT_MIN=((size_t)1<<{val})'")
        else:
            defs.append(f"-D{key}={val}")
    env = {k: str(v) for k, v in (cfg.get("runtime") or {}).items() if v}
    return {"tune": tune, "defs": " ".join(defs), "env": env}


def build_key(app: App, cfg: dict[str, Any]) -> str:
    """Candidates that differ only at run time share a build."""
    m = materialize(app, cfg)
    return json.dumps({"tune": m["tune"], "defs": m["defs"]}, sort_keys=True)


# ---- results ------------------------------------------------------------------

def parse_run(app: App, text: str) -> dict[str, Any]:
    r: dict[str, Any] = {}
    m = re.findall(app.spec["metric"]["regex"], text)
    if m:
        r[app.spec["metric"]["name"]] = float(m[-1])
    for c in app.spec.get("checks", []):
        m = re.findall(c["regex"], text)
        if m:
            try:
                r[c["name"]] = float(m[-1])
            except ValueError:
                r[c["name"]] = m[-1]
    return r


def check(app: App, got: dict[str, Any], ref: dict[str, Any]) -> list[str]:
    """Why a run's values disagree with the reference run's; empty when they agree."""
    bad = []
    for c in app.spec.get("checks", []):
        n = c["name"]
        if n not in got:
            bad.append(f"{n} missing from the output")
            continue
        if n not in ref or not isinstance(got[n], float):
            continue
        a, b = got[n], ref[n]
        if "abs_tol" in c and abs(a - b) > c["abs_tol"]:
            bad.append(f"{n}={a:g}, reference {b:g} (abs tol {c['abs_tol']:g})")
        if "rel_tol" in c and abs(a - b) > c["rel_tol"] * max(abs(b), 1e-300):
            bad.append(f"{n}={a:g}, reference {b:g} (rel tol {c['rel_tol']:g})")
    return bad


def kernel_breakdown(csv_path: Path, wall: float | None, top: int = 8) -> dict[str, Any] | None:
    """Where rank 0's time went, from Nsight Systems' cuda_gpu_kern_sum: GPU time
    per kernel, and the rest of the measured time *wall* (communication and host
    work that no kernel covers)."""
    import csv
    if not csv_path.is_file():
        return None
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return None
    ker = []
    for r in rows:
        ns = float(r.get("Total Time (ns)", 0) or 0)
        name = r.get("Name", "?")
        m = re.search(r"__omp_offloading\w+", name)
        ker.append((short_kernel(m.group(0)) if m else name[:60], ns / 1e9, int(r.get("Instances", 0) or 0)))
    ker.sort(key=lambda k: -k[1])
    total = sum(k[1] for k in ker)
    out = {"kernels_total_s": round(total, 4),
           "top_kernels": [{"kernel": n, "seconds": round(s, 4), "launches": c,
                            "avg_us": round(s / c * 1e6, 1) if c else None} for n, s, c in ker[:top]]}
    if wall:
        out["outside_kernels_s"] = round(wall - total, 4)
    return out


def summarize(app: App, runs: list[dict[str, Any]]) -> dict[str, Any]:
    metric = app.spec["metric"]["name"]
    vals = [r[metric] for r in runs if metric in r]
    if not vals:
        return {}
    return {metric: round(statistics.median(vals), 5), "runs": [round(v, 5) for v in vals],
            "spread": round((max(vals) - min(vals)) / statistics.median(vals), 4)}


# ---- ledger -------------------------------------------------------------------

def ledger(app: App) -> list[dict[str, Any]]:
    p = app.dir / "ledger.jsonl"
    if not p.is_file():
        return []
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def append_ledger(app: App, rows: list[dict[str, Any]]) -> None:
    with open(app.dir / "ledger.jsonl", "a") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")


def history(app: App, top: int = 10) -> dict[str, Any]:
    metric = app.spec["metric"]["name"]
    rows = ledger(app)
    ref = reference(app)
    ok = [r for r in rows if r.get("status") == "ok" and metric in r.get("summary", {})]
    ok.sort(key=lambda r: r["summary"][metric])
    def row(r):
        out = {"trial": r["trial"], "config": r["config"], metric: r["summary"][metric],
               "spread": r["summary"].get("spread")}
        if ref and metric in ref.get("summary", {}):
            out["vs_reference"] = f"{(ref['summary'][metric] / r['summary'][metric] - 1) * 100:+.1f}%"
        return out
    failed = [{"trial": r["trial"], "status": r["status"], "why": r.get("why", "")[:300], "config": r["config"]}
              for r in rows if r.get("status") != "ok"]
    return {"reference": ref.get("summary") if ref else None, "trials": len(rows),
            "best": [row(r) for r in ok[:top]], "failed": failed[-top:]}


def reference(app: App) -> dict[str, Any] | None:
    p = app.dir / "reference.json"
    return json.loads(p.read_text()) if p.is_file() else None


def new_round(app: App) -> tuple[int, Path]:
    d = app.dir / "rounds"
    d.mkdir(exist_ok=True)
    n = max([int(p.name) for p in d.iterdir() if p.name.isdigit()] + [0]) + 1
    rd = d / str(n)
    rd.mkdir()
    return n, rd


def next_trial(app: App) -> int:
    rows = ledger(app)
    return max([r["trial"] for r in rows] + [0]) + 1


def write_status(rd: Path, phase: str, **extra: Any) -> None:
    st = {"phase": phase, "time": time.strftime("%Y-%m-%d %H:%M:%S"), **extra}
    (rd / "status.json").write_text(json.dumps(st))


def read_status(rd: Path) -> dict[str, Any]:
    p = rd / "status.json"
    return json.loads(p.read_text()) if p.is_file() else {"phase": "unknown"}
