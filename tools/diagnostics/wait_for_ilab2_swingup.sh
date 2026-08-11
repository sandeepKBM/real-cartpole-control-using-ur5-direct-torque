#!/usr/bin/env bash
# Polls the energy-shaping swing-up search running on ilab2 until it finishes,
# then prints its final result. Meant to be launched in the background (via
# Claude Code's Bash tool with run_in_background:true, or manually with
# `nohup ./wait_for_ilab2_swingup.sh &`) -- Claude Code auto-notifies on a
# backgrounded Bash command's completion, which is the closest real
# equivalent to "restart me when the computation is done": there is no
# separate external hook that restarts the assistant, but a background wait
# that exits on completion gets the same practical effect.
set -euo pipefail

LOG=/tmp/gain_sched_rigor_and_single_ilab4.log
POLL_INTERVAL_S=30

echo "Waiting for ilab4 x_transport_gain_scheduling_newpose.py (single-gain search stage) to finish..."
while ssh -o ConnectTimeout=5 ilab4 "pgrep -f '[x]_transport_gain_scheduling_newpose.py'" >/dev/null 2>&1; do
    sleep "$POLL_INTERVAL_S"
done

echo "=== Search finished. Final result: ==="
grep -v "differential_evolution step\|UserWarning\|DifferentialEvolutionSolver" "$LOG" | tail -20
