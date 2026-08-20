#!/bin/bash
#SBATCH -J evo2_seedlen_decay
#SBATCH -A BCS25073
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 24:00:00
#SBATCH -o /work/11034/atzanakak/ls6/logs/%x-%j.out
#SBATCH -e /work/11034/atzanakak/ls6/logs/%x-%j.err

# Seed-length context-decay experiment (Reviewer #1 minor task).
# Generates synthetic sequences with 3 kb / 10 kb / 20 kb seeds in PARALLEL
# across three GPUs, Human windows only (5 windows x 3 seed lengths = 15 runs).
# After generation finishes, downstream analysis runs context_decay_auroc.py
# (CPU) to compute distance-from-seed AUROC curves.
#
# Override scope:
#   sbatch --export=ALL,ONLY_TAGS="Publish_Human",MODEL="evo2_7b" seed_length_decay.sbatch

set -eo pipefail
set +u
module purge
module load tacc-apptainer
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
# force_prompt_threshold chunks the long-prompt prefill through the Hyena FFT,
# preventing OOM for seed_len >= 10k on H100 80 GB. Applied for 10k and 20k.
FORCE_PROMPT_THRESHOLD="${FORCE_PROMPT_THRESHOLD:-8192}"

mkdir -p "$HF_CACHE" "$BASE/logs" "$OUTROOT"

echo "==> nvidia-smi"
apptainer exec --nv "${IMG}" bash -lc 'nvidia-smi || true'

# Free bytecode cache to ensure enough inodes for generated .syn.fa files.
echo "==> Clearing .pyc caches to free inodes"
find "$BASE/nonbdna" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "$BASE/nonbdna" -name "*.pyc" -delete 2>/dev/null || true

echo "==> Launching 3 parallel GPU workers (GPU 0=seed3k, 1=seed10k, 2=seed20k)"
PIDS=()

# ── GPU 0: 3 kb seed (no force_prompt_threshold needed) ────────────────────
CUDA_VISIBLE_DEVICES=0 apptainer exec --nv \
  -B "${CODEDIR}:${CODEDIR}" \
  -B "${HF_CACHE}:/root/.cache/huggingface" \
  -B "${BASE}:${BASE}" \
  "${IMG}" bash -lc "
    set -e
    export PYTHONNOUSERSITE=1
    export PYTHONPATH='${CODEDIR}'
    export TRANSFORMER_ENGINE_FP8=0
    export EVO2_PRECISION=bf16
    python3 ${SCRIPTS}/evo2_generate_sweep.py \
      --manifest ${MANIFEST} \
      --config-name seed3k \
      --model ${MODEL} \
      --temperature 1.0 --top_k 4 \
      --seed-len-override 3000 \
      --chunk-tokens ${CHUNK_TOKENS} \
      --outdir ${OUTROOT} \
      --only-tags ${ONLY_TAGS}
  " &
PIDS+=($!)

# ── GPU 1: 10 kb seed (force_prompt_threshold to chunk long prefill) ────────
CUDA_VISIBLE_DEVICES=1 apptainer exec --nv \
  -B "${CODEDIR}:${CODEDIR}" \
  -B "${HF_CACHE}:/root/.cache/huggingface" \
  -B "${BASE}:${BASE}" \
  "${IMG}" bash -lc "
    set -e
    export PYTHONNOUSERSITE=1
    export PYTHONPATH='${CODEDIR}'
    export TRANSFORMER_ENGINE_FP8=0
    export EVO2_PRECISION=bf16
    python3 ${SCRIPTS}/evo2_generate_sweep.py \
      --manifest ${MANIFEST} \
      --config-name seed10k \
      --model ${MODEL} \
      --temperature 1.0 --top_k 4 \
      --seed-len-override 10000 \
      --force-prompt-threshold ${FORCE_PROMPT_THRESHOLD} \
      --chunk-tokens ${CHUNK_TOKENS} \
      --outdir ${OUTROOT} \
      --only-tags ${ONLY_TAGS}
  " &
PIDS+=($!)

# ── GPU 2: 20 kb seed (force_prompt_threshold required; no override needed) ─
CUDA_VISIBLE_DEVICES=2 apptainer exec --nv \
  -B "${CODEDIR}:${CODEDIR}" \
  -B "${HF_CACHE}:/root/.cache/huggingface" \
  -B "${BASE}:${BASE}" \
  "${IMG}" bash -lc "
    set -e
    export PYTHONNOUSERSITE=1
    export PYTHONPATH='${CODEDIR}'
    export TRANSFORMER_ENGINE_FP8=0
    export EVO2_PRECISION=bf16
    python3 ${SCRIPTS}/evo2_generate_sweep.py \
      --manifest ${MANIFEST} \
      --config-name seed20k \
      --model ${MODEL} \
      --temperature 1.0 --top_k 4 \
      --force-prompt-threshold ${FORCE_PROMPT_THRESHOLD} \
      --chunk-tokens ${CHUNK_TOKENS} \
      --outdir ${OUTROOT} \
      --only-tags ${ONLY_TAGS}
  " &
PIDS+=($!)

# ── Wait for all three GPU workers ──────────────────────────────────────────
echo "==> Waiting for GPU workers (PIDs: ${PIDS[*]})"
FAIL=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || { echo "[error] worker PID $pid exited non-zero"; FAIL=1; }
done
[[ "$FAIL" -eq 0 ]] || { echo "[fatal] one or more seed-length runs failed"; exit 1; }

echo "==================================================================="
echo "==> All seed-length generations complete."
echo "    Output: $OUTROOT"
echo ""
echo "    Next step (CPU, run from login node):"
echo "    python3 ${SCRIPTS}/context_decay_auroc.py \\"
echo "      --winlen-dir $OUTROOT \\"
echo "      --manifest $MANIFEST \\"
echo "      --outdir $REV/results/context_decay"
echo ""
echo "    python3 ${SCRIPTS}/plot_context_decay.py \\"
echo "      --results $REV/results/context_decay/context_decay_auroc.csv \\"
echo "      --outdir $REV/figures"
