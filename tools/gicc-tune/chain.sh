#!/bin/bash
# chain.sh APP [ROUNDS]: once APP's reference round is done, run the agent on it.
cd "$(dirname "$0")"
A=$1; R=${2:-4}
until grep -qE '"phase": "(done|failed)"' runs/$A/rounds/1/status.json; do sleep 30; done
[ -f runs/$A/reference.json ] || { echo "reference failed"; exit 1; }
source <(grep DEEPSEEK_KEY ~/.bashrc.user.common); export DEEPSEEK_KEY
PYTHONUNBUFFERED=1 /shared/data1/Users/l1065028/MIMIR/.venv-linux-x86_64/bin/python run_tune.py $A --rounds $R > runs/$A/loop1.log 2>&1
echo "loop exit $?"
