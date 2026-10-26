# gicc-tune

LLM-guided tuning of GiOMP pipelines. The compiler (gicc-passes) reports
every tunable site and the options that are legal there
(`GICC_SITES_OUT`); a model proposes configurations; each is validated
against that report, built with the choices (`GICC_TUNE_IN`), run on the
cluster, checked against the reference run's results, and timed, with a
per-kernel breakdown from Nsight Systems. The model revises from the
measurements. The model only tunes performance; the compiler keeps the
code correct.

    gtune/core.py      sites, validation, ledger
    gtune/worker.py    one round: parallel builds, one allocation, parsing
    gtune/api.py       setup / sites / submit / wait / history
    apps/<app>/        app.json (build, run, metric, checks, knobs), build.sh
    .mimir/servers/server_gtune.py          the tools, for MIMIR
    .mimir/skills/gpu-pipeline-tune/SKILL.md  the method
    run_tune.py        unattended loop: MIMIR + DeepSeek

Run from this directory with MIMIR's Python:

    DEEPSEEK_KEY=... python run_tune.py miniweather --rounds 4

Results go to `runs/<app>/` (ignored by git).
