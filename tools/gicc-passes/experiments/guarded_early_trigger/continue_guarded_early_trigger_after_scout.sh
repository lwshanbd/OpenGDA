#!/usr/bin/env bash
# Wait for the frozen scout, then conditionally prepare/run its confirmation.
set -euo pipefail

if [[ $# -ne 5 ]]; then
    echo "usage: $0 SCOUT_STATE SCOUT_DIR BINARY_DIR TRANSITION OUT_DIR" >&2
    exit 2
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/../../../.." && pwd)
case $1 in /*) scout_state=$1 ;; *) scout_state="$repo_root/$1" ;; esac
case $2 in /*) scout_dir=$2 ;; *) scout_dir="$repo_root/$2" ;; esac
binary_dir=$(cd -- "$3" && pwd)
case $4 in /*) transition=$4 ;; *) transition="$repo_root/$4" ;; esac
case $5 in /*) output_dir=$5 ;; *) output_dir="$repo_root/$5" ;; esac
preparer="$script_dir/prepare_guarded_early_trigger_confirmation.py"
protocol="$script_dir/GUARDED_EARLY_CONFIRMATION_TRANSITION.md"
runner="$script_dir/run_guarded_early_trigger_confirmation.sh"
controller="$script_dir/continue_guarded_early_trigger_confirmation.sh"
monitor="$script_dir/monitor_guarded_early_trigger_confirmation.py"
analyzer="$script_dir/analyze_guarded_early_trigger_confirmation.py"
state_path="$output_dir.chain.state"
events_path="$output_dir.chain.events"
lock_path="$output_dir.chain.lock"

mkdir -p "$(dirname -- "$output_dir")"
exec 9>"$lock_path"
if ! flock -n 9; then echo "another guarded confirmation successor is active" >&2; exit 2; fi
set_state() {
    local state=$1 detail=${2:-} temporary="$state_path.tmp.$$"
    printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$state" "$detail" >"$temporary"
    mv -- "$temporary" "$state_path"
    printf '%s\t%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$state" "$detail" >>"$events_path"
}
on_exit() { local status=$?; [[ $status -eq 0 ]] || set_state failed "successor_exit=$status"; }
trap on_exit EXIT

artifacts=(
    "$binary_dir/baseline/mm_minimal" "$binary_dir/guarded/mm_minimal"
    "$binary_dir/BUILD_PROVENANCE.txt" "$preparer" "$protocol" "$runner"
    "$controller" "$monitor" "$analyzer"
)
hashes=()
for artifact in "${artifacts[@]}"; do
    [[ -f $artifact ]] || { echo "missing frozen artifact: $artifact" >&2; exit 2; }
    digest=$(sha256sum "$artifact"); hashes+=("${digest%% *}")
done
verify_artifacts() {
    local index actual
    for index in "${!artifacts[@]}"; do
        actual=$(sha256sum "${artifacts[$index]}"); actual=${actual%% *}
        [[ $actual == "${hashes[$index]}" ]] || {
            echo "artifact changed while waiting: ${artifacts[$index]}" >&2; exit 2;
        }
    done
}

set_state waiting_scout "$scout_state"
phase=
while [[ -z $phase ]]; do
    [[ -f $scout_state ]] || { sleep 5; continue; }
    phase=$(awk 'NR == 1 {print $2}' "$scout_state")
    case $phase in promising|negative|failed) ;; *) phase=; sleep 5 ;; esac
done
verify_artifacts
if [[ $phase != promising ]]; then
    set_state skipped "scout_state=$phase"
    trap - EXIT
    exit 0
fi
set_state preparing_transition "$transition"
if [[ -e $transition ]]; then
    python3 "$preparer" verify-contained --report "$transition"
else
    python3 "$preparer" prepare --monitor "$scout_dir/monitor.json" \
        --analysis "$scout_dir/analysis.json" --binary-dir "$binary_dir" \
        --out "$transition"
fi
set_state running_confirmation "$output_dir"
bash "$controller" "$transition" "$binary_dir" "$output_dir"
confirmation_phase=$(awk 'NR == 1 {print $2}' "$output_dir.state")
case $confirmation_phase in
    confirmed) set_state confirmed "$output_dir/analysis.json" ;;
    negative) set_state negative "$output_dir/analysis.json" ;;
    *) echo "unexpected confirmation state: $confirmation_phase" >&2; exit 2 ;;
esac
trap - EXIT
