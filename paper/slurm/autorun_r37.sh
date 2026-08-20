#!/bin/bash
# R3.7 auto-run: wait until the decoding sweeps finish (no evo2_generate_sweep
# processes remain), then compute the sweep metrics and the comparison figure.
BASE="${NONBDNA_BASE:-/work/11034/atzanakak/ls6}"
SCRIPTS="${REVISION_SCRIPTS:-$BASE/nonbdna/revisions/scripts}"
LOG=$BASE/logs/r37_autorun.log
echo "$(date) r37 autorun armed; waiting for sweeps to finish" >> "$LOG"

# wait for all four sweep halves to exit
while pgrep -f evo2_generate_sweep.py >/dev/null 2>&1; do
  sleep 300
done

echo "$(date) sweeps finished; scoring lowtemp + nucleus vs natural" >> "$LOG"
cd /tmp || exit 1
python3 "$SCRIPTS/run_sweep_metrics.py" \
    --configs lowtemp nucleus >> "$LOG" 2>&1
python3 "$SCRIPTS/plot_sweep_metrics.py" >> "$LOG" 2>&1
echo "$(date) R3.7 metrics + figure complete" >> "$LOG"
echo "$(date) outputs: revisions/results/sweep_metrics_{per_window,summary,config_summary}.csv + figures/sweep_metrics_comparison.png" >> "$LOG"
