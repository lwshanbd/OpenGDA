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
            # One record per option the compiler offers on the kernel.
            k = kernels.setdefault(rec["id"], {"kind": "kernel", "id": rec["id"], "options": []})
            opt = rec.pop("option", None)
            if opt and opt not in k["options"]:
                k["options"].append(opt)
            if opt == "split":
                k["splittable"] = bool(rec.get("splittable"))
                if not k["splittable"]:
                    k["split_why"] = rec.get("why", "")
            else:
                k.update({key: v for key, v in rec.items() if key not in ("kind", "id", "why")})
    out: dict[str, Any] = {"sites": {}, "kernels": {}}
    for i, sid in enumerate(sorted(sites)):
        out["sites"][f"S{i}"] = sites[sid]
    for i, kid in enumerate(sorted(kernels)):
        out["kernels"][f"K{i}"] = kernels[kid]
    return out


def short_kernel(name: str) -> str:
    """The source function and line in an offload kernel's name."""
    line = re.search(r"_l(\d+)$", name)
    m = re.search(r"_ZL?(\d+)", name)
    fn = name[m.end():m.end() + int(m.group(1))] if m else name
    return f"{fn}:{line.group(1)}" if line else fn


def start(app: App) -> dict[str, Any]:
    """The app's starting point: tuning-file entries every build begins from
    (by compiler id, "*" for every site), under the candidate's own choices.
    Without one, the start is what the compiler does on its own."""
    return app.spec.get("start", {})


def site_default(app: App, rec: dict[str, Any]) -> dict[str, Any]:
    """What a build does at a site when the configuration says nothing: the
    app's start, else the compiler's own choice (a legal site is pipelined,
    at element grain where that is legal)."""
    if rec.get("verdict") != "LEGAL":
        return {}
    grains = rec.get("options", {}).get("grain", [])
    d = {"pipeline": True, "grain": "element" if "element" in grains else (grains or [None])[0]}
    st = start(app).get("sites", {})
    d.update(st.get("*", {}))
    d.update(st.get(rec["id"], {}))
    return d


def kernel_default(app: App, rec: dict[str, Any]) -> dict[str, Any]:
    """What a build does on a kernel when the configuration says nothing."""
    d: dict[str, Any] = {"dist_chunk": rec.get("default")}
    if rec.get("splittable"):
        st = start(app).get("kernels", {})
        d["split"] = bool({**st.get("*", {}), **st.get(rec["id"], {})}.get("split", False))
    return d


def knob_defaults(app: App) -> dict[str, dict[str, Any]]:
    """The default of every choice a configuration can make, by section."""
    s = load_sites(app)
    return {
        "sites": {a: site_default(app, r) for a, r in s["sites"].items()},
        "kernels": {a: kernel_default(app, r) for a, r in s["kernels"].items()},
        "build": {k: v.get("default") for k, v in app.spec.get("build_knobs", {}).items()},
        "runtime": {k: None for k in app.spec.get("runtime_knobs", {})},
    }


def changes(app: App, cfg: dict[str, Any]) -> list[str]:
    """The choices in *cfg* that differ from the default, as section.alias.key."""
    d = knob_defaults(app)
    out = []
    for section in ("sites", "kernels"):
        for alias, choice in (cfg.get(section) or {}).items():
            for key, val in (choice or {}).items():
                if val != d[section].get(alias, {}).get(key):
                    out.append(f"{section}.{alias}.{key}")
    for section in ("build", "runtime"):
        for key, val in (cfg.get(section) or {}).items():
            if val != d[section].get(key) and not (section == "runtime" and not val):
                out.append(f"{section}.{key}")
    return out


def describe_sites(app: App) -> dict[str, Any]:
    s = load_sites(app)
    sites = []
    for alias, rec in s["sites"].items():
        sites.append({"alias": alias, "kernel": short_kernel(rec["kernel"]), "form": rec.get("form"),
                      "verdict": rec["verdict"], "why": rec.get("why", ""),
                      "options": rec.get("options", {}), "default": site_default(app, rec),
                      "box": rec.get("box")})
    # GPU time each kernel took in the reference run; a kernel that never ran
    # there cannot matter.
    ref = reference(app) or {}
    # (Older references kept the full name before the line.)
    spent = {short_kernel(k["kernel"].rsplit(":", 1)[0]) if "_l" in k["kernel"] else k["kernel"]: k for k in (ref.get("breakdown") or {}).get("top_kernels", [])}
    kernels = []
    for alias, rec in s["kernels"].items():
        k = {"alias": alias, "kernel": short_kernel(rec["id"]), "pipelined": bool(rec.get("pipelined")),
             "default_chunk": rec.get("default", "clang default (one element per thread)"),
             "splittable": bool(rec.get("splittable"))}
        if rec.get("splittable"):
            k["default_split"] = kernel_default(app, rec)["split"]
        elif rec.get("split_why"):
            k["not_splittable_because"] = rec["split_why"]
        hit = spent.get(k["kernel"])
        k["reference_gpu_s"] = hit["seconds"] if hit else 0.0
        if hit:
            k["reference_launches"] = hit["launches"]
        kernels.append(k)
    ran = {k["kernel"] for k in kernels if k["reference_gpu_s"] > 0}
    for site in sites:
        site["kernel_ran_in_reference"] = site["kernel"] in ran
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
            if key == "split":
                if not s["kernels"][alias].get("splittable"):
                    errs.append(f"kernel {alias}: the compiler cannot split it: "
                                f"{s['kernels'][alias].get('split_why', 'not reported')}")
                elif not isinstance(val, bool):
                    errs.append(f"kernel {alias}: split must be true or false")
                elif val:
                    # A pipelined site makes its kernel send by itself, and
                    # then no fence is deferred for any kernel.
                    kname = short_kernel(s["kernels"][alias]["id"])
                    on = [a for a, rec in s["sites"].items() if short_kernel(rec["kernel"]) == kname
                          and {**site_default(app, rec), **((cfg.get("sites") or {}).get(a) or {})}.get("pipeline")]
                    if on:
                        errs.append(f"kernel {alias}: split needs its sites unpipelined, but "
                                    f"{', '.join(on)} pipeline (set \"pipeline\": false on them)")
            elif key != "dist_chunk":
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
    st = start(app)
    tune = {"sites": {k: dict(v) for k, v in st.get("sites", {}).items()},
            "kernels": {k: dict(v) for k, v in st.get("kernels", {}).items()}}
    if cfg.get("sites") or cfg.get("kernels"):
        s = load_sites(app)
        for alias, choice in (cfg.get("sites") or {}).items():
            tune["sites"].setdefault(s["sites"][alias]["id"], {}).update(choice)
        for alias, choice in (cfg.get("kernels") or {}).items():
            tune["kernels"].setdefault(s["kernels"][alias]["id"], {}).update(choice)
    defs = []
    for key, val in (cfg.get("build") or {}).items():
        if key == "OMPX_PIPE_COUNT_MIN_LOG2":
            # No spaces, no quotes: the build scripts split the flags on
            # whitespace and never re-parse them.
            defs.append(f"-DOMPX_PIPE_COUNT_MIN=((size_t)1<<{val})")
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
        if m and c.get("exact"):
            r[c["name"]] = m[-1]
        elif m:
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
        if n not in ref:
            continue
        if c.get("exact"):
            if got[n] != ref[n]:
                bad.append(f"{n}={got[n]}, reference {ref[n]}")
            continue
        if not isinstance(got[n], float):
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


def split_stats(text: str) -> list[dict[str, Any]]:
    """Each rank's OMPX_SPLIT_STATS line: launches split, the seconds before the
    first part (waiting for the other ranks' transfer lists), in the part that
    overlaps the transfers, in the fence, and in the rest."""
    pat = (r"rank (\d+): (\d+) split launches \((\d+) run whole before the fence; not split: "
           r"(\d+) with no fence deferred, (\d+) unknown, (\d+) with nothing left\): before ([\d.]+) s, "
           r"inside ([\d.]+) s, fence ([\d.]+) s, outside ([\d.]+) s")
    out = []
    for m in re.finditer(pat, text):
        g = m.groups()
        out.append({"rank": int(g[0]), "split": int(g[1]), "whole_before_fence": int(g[2]),
                    "not_split_no_fence": int(g[3]), "not_split_unknown": int(g[4]),
                    "not_split_nothing_left": int(g[5]), "before_s": float(g[6]),
                    "inside_s": float(g[7]), "fence_s": float(g[8]), "outside_s": float(g[9])})
    return sorted(out, key=lambda r: r["rank"])


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
    # The choices no trial has moved off the default yet.
    tried = {c for r in rows for c in changes(app, r.get("config") or {})}
    d = knob_defaults(app)
    every = [f"{sec}.{a}.{k}" for sec in ("sites", "kernels") for a, ch in d[sec].items() for k in ch
             if not (sec == "sites" and k == "grain" and len(load_sites(app)["sites"][a].get("options", {}).get("grain", [])) < 2)]
    every += [f"{sec}.{k}" for sec in ("build", "runtime") for k in d[sec]]
    return {"reference": ref.get("summary") if ref else None, "trials": len(rows),
            "best": [row(r) for r in ok[:top]], "failed": failed[-top:],
            "never_varied": [c for c in every if c not in tried]}


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
