#!/bin/bash
# tune.sh APP [ROUNDS] [EXTRA]: run the agent (MIMIR + DeepSeek) on APP with a budget of
# ROUNDS trial rounds (default 5), logging to runs/APP/loop.log.
cd /shared/data1/Users/l1065028/new-gicc/tools/gicc-tune
A=$1; R=${2:-5}
mkdir -p runs/$A
source <(grep DEEPSEEK_KEY ~/.bashrc.user.common); export DEEPSEEK_KEY
PYTHONUNBUFFERED=1 /shared/data1/Users/l1065028/MIMIR/.venv-linux-x86_64/bin/python run_tune.py $A --rounds $R ${3:+--extra "$3"} >> runs/$A/loop.log 2>&1
echo "loop $A exit $?"
