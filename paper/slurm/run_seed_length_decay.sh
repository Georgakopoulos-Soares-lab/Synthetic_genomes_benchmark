#!/bin/bash
# Interactive (non-SLURM) launcher for the seed-length context-decay experiment.
# Run this directly on a GPU node with >=3 free GPUs.
#
#   bash run_seed_length_decay.sh
#
# Launches 3 parallel workers: GPU0=seed3k, GPU1=seed10k, GPU2=seed20k.
# Human windows only (5 windows x 3 seed lengths = 15 generations).
# Per-GPU logs go to $REV/logs/seedlen_<cfg>.log; monitor with:
#   tail -f "$REV"/logs/seedlen_seed3k.log

set -eo pipefail
set +u
module purge 2>/dev/null || true
module load tacc-apptainer 2>/dev/null || true
set -u

BASE="${NONBDNA_BASE:-/work/11034/atzanakak/ls6}"
IMG="$BASE/containers/evo2.sif"
HF_CACHE="$BASE/huggingface"
CODEDIR="$BASE/code/evo2"
REV="${NONBDNA_REVISIONS:-$BASE/nonbdna/revisions}"
SCRIPTS="${REVISION_SCRIPTS:-$REV/scripts}"
MANIFEST="${MANIFEST:-$REV/decoding_sweep/seed_manifest_long.csv}"
OUTROOT="${OUTROOT:-$REV/decoding_sweep/winlen_generated}"
MODEL="${MODEL:-evo2_7b}"
ONLY_TAGS="${ONLY_TAGS:-Publish_Human}"
CHUNK_TOKENS="${CHUNK_TOKENS:-8192}"
FORCE_PROMPT_THRESHOLD="${FORCE_PROMPT_THRESHOLD:-8192}"
# Amount of NEW sequence to generate after each seed (bp). 100 kb captures the
# decay onset + full steep rise (AUROC 0.68@seed-end -> ~0.89@100kb); the paper's
# curve only reaches its 0.93 plateau at ~175 kb, which is not needed to answer
# "does a longer prefix delay onset / change the rate". total_target = seed_len + NEW_LEN.
NEW_LEN="${NEW_LEN:-100000}"
# Windows generated in parallel per generate() call (all windows on a GPU share
# the same seed_len => identical n_tokens => no wasted compute when batched).
# 20k seed uses batch=1: its prefill needs a 16GB FFT workspace that won't fit
# alongside two sequences' KV cache on an H100 80GB.
BATCH_SIZE="${BATCH_SIZE:-2}"
BATCH_SIZE_20K="${BATCH_SIZE_20K:-1}"
LOGDIR="$REV/logs"

mkdir -p "$HF_CACHE" "$OUTROOT" "$LOGDIR"

# Free bytecode cache to keep inodes available for generated .syn.fa files.
find "$BASE/nonbdna" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$BASE/nonbdna" -name "*.pyc" -delete 2>/dev/null || true

echo "==> nvidia-smi"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true

# worker(gpu_id, config_name, seed_len, use_force_prompt, batch_size)
#   Generates exactly NEW_LEN new bp after a seed_len-bp prefix (total target =
#   seed_len + NEW_LEN), so every seed length gets the SAME amount of fresh
#   sequence -> directly comparable distance-from-seed curves.
#   batch_size is per-worker: 3k/10k seeds fit 2 windows/call on an H100 80GB,
#   but the 20k-seed prefill needs a 16GB FFT workspace that only fits at batch=1.
run_worker() {
  local gpu="$1" cfg="$2" seedlen="$3" useforce="$4" bs="$5"
  local target=$(( seedlen + NEW_LEN ))
  local extra="--seed-len-override ${seedlen} --target-len-override ${target}"
  [[ "$useforce" == "1" ]] && extra="$extra --force-prompt-threshold ${FORCE_PROMPT_THRESHOLD}"
  echo "==> GPU$gpu: $cfg  seed_len=${seedlen} new=${NEW_LEN} target=${target} batch=${bs}  (log: $LOGDIR/seedlen_${cfg}.log)"
  CUDA_VISIBLE_DEVICES="$gpu" apptainer exec --nv \
    -B "${CODEDIR}:${CODEDIR}" \
    -B "${HF_CACHE}:/root/.cache/huggingface" \
    -B "${BASE}:${BASE}" \
    "${IMG}" bash -lc "
      set -e
      export PYTHONNOUSERSITE=1
      export PYTHONPATH='${CODEDIR}'
      export TRANSFORMER_ENGINE_FP8=0
      export EVO2_PRECISION=bf16
      # Reduce allocator fragmentation (~20GB was reserved-but-unallocated at
      # batch=2); lets the longer-prompt prefill fit for seed10k/seed20k.
      export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
      python3 ${SCRIPTS}/evo2_generate_sweep.py \
        --manifest ${MANIFEST} \
        --config-name ${cfg} \
        --model ${MODEL} \
        --temperature 1.0 --top_k 4 \
        ${extra} \
        --batch-size ${bs} \
        --chunk-tokens ${CHUNK_TOKENS} \
        --outdir ${OUTROOT} \
        --only-tags ${ONLY_TAGS}
    " > "$LOGDIR/seedlen_${cfg}.log" 2>&1 &
}

# GPU, config, seed_len, use_force_prompt, batch_size
PIDS=()
run_worker 0 seed3k   3000 0 "${BATCH_SIZE}"    ; PIDS+=($!)
run_worker 1 seed10k 10000 1 "${BATCH_SIZE}"    ; PIDS+=($!)
run_worker 2 seed20k 20000 1 "${BATCH_SIZE_20K}"; PIDS+=($!)

echo "==> 3 workers launched (PIDs: ${PIDS[*]}); waiting..."
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || { echo "[error] worker PID $pid failed"; FAIL=1; }
done

echo "==================================================================="
if [[ "$FAIL" -eq 0 ]]; then
  echo "==> All 3 seed-length generations complete. Output: $OUTROOT"
else
  echo "==> One or more workers FAILED — check $LOGDIR/seedlen_*.log"
fi
ls -1 "$OUTROOT"/seed3k "$OUTROOT"/seed10k "$OUTROOT"/seed20k 2>/dev/null || true
