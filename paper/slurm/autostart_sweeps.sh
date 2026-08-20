#!/bin/bash
# Auto-start the 300 kb decoding sweeps (lowtemp + nucleus) once the
# gene-completion job has fully finished and the GPUs are free.
#
# Trigger = run_gc_turboquant.py absent AND every GPU has >= THRESH MiB free,
# sustained for CONSEC consecutive checks (avoids the per-panel respawn gaps).
# Sweeps are resumable (skip finished windows), so re-launching is safe.
BASE="${NONBDNA_BASE:-/work/11034/atzanakak/ls6}"
REV="${NONBDNA_REVISIONS:-$BASE/nonbdna/revisions}"
SCRIPTS="${REVISION_SCRIPTS:-$REV/scripts}"
SWEEP=$REV/decoding_sweep
LOG=$BASE/logs/autostart.log
THRESH=85000      # MiB min free required on EVERY GPU
CONSEC=4          # consecutive OK checks (x INTERVAL) before firing
INTERVAL=180      # seconds between checks
module load tacc-apptainer 2>/dev/null || true

launch() {  # $1=gpu $2=config $3=manifest $4=extra-args $5=logsuffix
  nohup apptainer exec --nv \
    --env CUDA_VISIBLE_DEVICES="$1" \
    --env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -B "$BASE/code/evo2:$BASE/code/evo2" \
    -B "$BASE/huggingface:/root/.cache/huggingface" \
    -B "$BASE:$BASE" \
    "$BASE/containers/evo2.sif" bash -lc "
      export PYTHONNOUSERSITE=1
      cd $BASE/code/evo2
      pip install -e . >/dev/null 2>&1 || true
      export TRANSFORMER_ENGINE_FP8=0 EVO2_PRECISION=bf16
      python3 $SCRIPTS/evo2_generate_sweep.py \
        --manifest $SWEEP/$3 --config-name $2 --model evo2_7b $4 \
        --batch-size 1 --outdir $SWEEP/generated
    " > "$BASE/logs/decsweep_$5.log" 2>&1 &
  echo "$(date) launched $2 on GPU$1 (PID $!)" >> "$LOG"
}

echo "$(date) autostart watcher started (THRESH=${THRESH} CONSEC=${CONSEC})" >> "$LOG"
ok=0
while true; do
  if pgrep -f run_gc_turboquant.py >/dev/null 2>&1; then
    ok=0
    echo "$(date) gene-completion running -> wait" >> "$LOG"
    sleep "$INTERVAL"; continue
  fi
  minfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
  if [ -n "$minfree" ] && [ "$minfree" -ge "$THRESH" ]; then
    ok=$((ok+1))
  else
    ok=0
  fi
  echo "$(date) no gene-completion; minfree=${minfree}MiB ok=${ok}/${CONSEC}" >> "$LOG"
  if [ "$ok" -ge "$CONSEC" ]; then
    echo "$(date) TRIGGER -> launching 300kb sweeps" >> "$LOG"
    launch 0 lowtemp seed_manifest_A.csv "--temperature 0.7 --top_k 4" lowtemp_A
    launch 1 lowtemp seed_manifest_B.csv "--temperature 0.7 --top_k 4" lowtemp_B
    launch 2 nucleus  seed_manifest_A.csv "--temperature 1.0 --top_p 0.9 --top_k 0" nucleus_A
    launch 3 nucleus  seed_manifest_B.csv "--temperature 1.0 --top_p 0.9 --top_k 0" nucleus_B
    echo "$(date) all 4 sweep halves launched; watcher exiting" >> "$LOG"
    break
  fi
  sleep "$INTERVAL"
done
