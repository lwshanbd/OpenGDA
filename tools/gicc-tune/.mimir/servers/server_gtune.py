"""gtune: compiler-checked tuning of GiOMP pipelines on a GPU cluster.

Loaded by MIMIR from <workspace>/.mimir/servers/ when the workspace is
tools/gicc-tune. Each tool is a thin layer over gtune.api; see the
gpu-pipeline-tune skill for the loop they make up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

from mcp.server.fastmcp import FastMCP
from pydantic import Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gtune import api  # noqa: E402

try:
    from mimir.servers._shared.capabilities import READ, tool_caps
except ImportError:  # outside MIMIR: plain tools
    READ = "read"

    def tool_caps(**_: Any) -> dict:
        return {}

mcp = FastMCP("gtune")

App = Annotated[str, Field(description="Application name, a directory under apps/ (e.g. 'miniweather').")]


def _dump(obj: Any) -> str:
    return json.dumps(obj, indent=1, default=str)


@mcp.tool(**tool_caps(caps=[READ], label="Reading tunable sites"))
def gtune_sites(app: App) -> str:
    """The program's tunable sites as the compiler reported them, with the legal
    options of each, the run-time and build knobs, and the reference run (time and
    where the time goes). Read this before proposing any configuration."""
    try:
        return _dump(api.sites(app))
    except Exception as e:
        return _dump({"error": str(e)})


@mcp.tool(**tool_caps(reversibility="recoverable", label="Starting the reference build",
                      timeout_secs=60))
def gtune_setup(app: App) -> str:
    """Build the default configuration, record its tunable sites and time it on the
    cluster. Run once per application before any trial; then gtune_wait on the round."""
    try:
        return _dump(api.setup(app))
    except Exception as e:
        return _dump({"error": str(e)})


@mcp.tool(**tool_caps(reversibility="recoverable", label="Submitting candidate configurations",
                      timeout_secs=60))
def gtune_submit(
    app: App,
    configs: Annotated[list[dict], Field(description=(
        "1 to 8 candidate configurations, built in parallel and run in one allocation. "
        "Each is {\"sites\": {\"S0\": {\"pipeline\": false, \"grain\": \"block\"}}, "
        "\"kernels\": {\"K3\": {\"dist_chunk\": 16384, \"split\": true}}, \"build\": {\"OMPX_PIPE_PIECE_SHIFT\": 18}, "
        "\"runtime\": {\"OMP_NUM_TEAMS\": 528}, \"note\": \"the hypothesis it tests\"}; "
        "omit anything left at its default. {} is the default configuration."))],
    repeats: Annotated[int | None, Field(description="Timed runs per candidate (default from the app, usually 2).")] = None,
) -> str:
    """Validate candidate configurations against the compiler's legal options and
    start one round: build each, run each on the cluster, check its results against
    the reference. Returns at once with a round number; follow with gtune_wait.
    Invalid configurations are rejected with the reason and never built."""
    try:
        return _dump(api.submit(app, configs, repeats))
    except Exception as e:
        return _dump({"error": str(e)})


@mcp.tool(**tool_caps(caps=[READ], label="Waiting for a tuning round", timeout_secs=1150))
def gtune_wait(
    app: App,
    round: Annotated[int, Field(description="The round number gtune_setup or gtune_submit returned.")],
    max_wait: Annotated[int, Field(description="Seconds to wait at most (up to 1100).")] = 900,
) -> str:
    """Wait for a round to finish. Returns each candidate's median time, its change
    against the reference, run-to-run spread, the check verdict and where its time
    went; or the round's phase (building, queued, running) if it is still going."""
    try:
        return _dump(api.wait(app, round, max_wait))
    except Exception as e:
        return _dump({"error": str(e)})


@mcp.tool(**tool_caps(caps=[READ], label="Reading tuning history"))
def gtune_history(
    app: App,
    top: Annotated[int, Field(description="How many of the best trials to list.")] = 10,
) -> str:
    """Every trial so far: the best configurations with their times against the
    reference, and the ones that failed (build, run or check) with why."""
    try:
        return _dump(api.history(app, top))
    except Exception as e:
        return _dump({"error": str(e)})


if __name__ == "__main__":
    mcp.run()
