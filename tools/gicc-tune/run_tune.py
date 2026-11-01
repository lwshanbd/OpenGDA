"""Run the tuning loop unattended: MIMIR as the agent, DeepSeek as its model.

    DEEPSEEK_KEY=... python run_tune.py miniweather --rounds 4

Uses MIMIR's Anthropic backend against DeepSeek's Anthropic-compatible
endpoint. Every tool is approved without prompting (the gtune tools are the
only ones that reach the cluster, and they reject configurations the
compiler did not report as legal). The transcript of events goes to
runs/<app>/agent-<time>.jsonl and the final answer to runs/<app>/report-<time>.md.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# The servers the loop needs: the tuning tools, skills, and reading files.
SERVERS = ("gtune", "mimir_api", "search", "todo")


def configure_backend(model: str) -> None:
    key = os.environ.get("DEEPSEEK_KEY")
    if not key:
        sys.exit("DEEPSEEK_KEY is not set")
    os.environ.setdefault("LLM_BACKEND", "anthropic")
    os.environ.setdefault("ANTHROPIC_BASE_URL", "https://api.deepseek.com/anthropic")
    os.environ["ANTHROPIC_API_KEY"] = key
    os.environ.setdefault("MIMIR_DEFAULT_MODEL", model)


def query_for(app: str, rounds: int, extra: str) -> str:
    skill = (ROOT / ".mimir/skills/gpu-pipeline-tune/SKILL.md").read_text()
    skill = skill.split("---", 2)[-1].strip()
    spec = json.loads((ROOT / "apps" / app / "app.json").read_text())
    if spec.get("pipeline_only"):
        extra = ("This run tunes only how the puts are pipelined: site pipeline and grain, "
                 "kernel dynamic and split, the chunk of pipelined kernels, and the piece-size "
                 "build knobs. Launch geometry (teams, threads) and run-time knobs are not "
                 "choices here; gtune_submit rejects them. " + extra)
    return (
        f"Tune the application '{app}' with the gtune tools, following the method below. "
        f"You have a budget of {rounds} trial rounds after the reference (gtune_setup does not count), "
        f"including the confirmation round. Work autonomously: do not ask the user anything. "
        f"{extra}\n\n{skill}"
    )


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("app")
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--model", default="deepseek-flash")
    ap.add_argument("--extra", default="", help="more instructions for this run")
    args = ap.parse_args()

    configure_backend(args.model)
    os.chdir(ROOT)  # the workspace MIMIR's servers and .mimir/ resolve against
    from mimir.client.agent_core import MimirAgent
    from mimir.client.extensions import all_servers

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = ROOT / "runs" / args.app
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / f"agent-{stamp}.jsonl", "w")

    def on_event(ev: dict) -> None:
        log.write(json.dumps(ev, default=str) + "\n")
        log.flush()
        t = ev.get("type")
        if t == "tool_call":
            print(f"[tool] {ev.get('name')} {json.dumps(ev.get('detail') or '')[:160]}", flush=True)
        elif t == "tool_result":
            print(f"[done] {ev.get('name')} ok={ev.get('ok')} {ev.get('duration_ms')} ms", flush=True)

    agent = MimirAgent(model=args.model)
    agent.set_approval_mode("auto_all")
    registry = all_servers()
    missing = [n for n in SERVERS if n not in registry]
    if missing:
        sys.exit(f"servers not found: {missing} (is .mimir/servers/server_gtune.py in {ROOT}?)")
    await agent.connect_servers({n: registry[n] for n in SERVERS})
    agent.seed_classification_from_caps()
    try:
        answer = await agent.run(query_for(args.app, args.rounds, args.extra), max_steps=0,
                                 mode="agent", event_callback=on_event)
    finally:
        await agent.cleanup()
        log.close()
    (out / f"report-{stamp}.md").write_text(str(answer))
    print("\n===== final answer =====\n" + str(answer))


if __name__ == "__main__":
    asyncio.run(main())
