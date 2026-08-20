#!/bin/bash
#SBATCH -J evo2_winlen
#SBATCH -A BCS25073
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -o /work/11034/atzanakak/ls6/logs/%x-%j.out
#SBATCH -e /work/11034/atzanakak/ls6/logs/%x-%j.err

# Conditioning-window-length experiment (Reviewer #1 minor #2): does a longer
# seed/prompt delay context decay? Generates with 3 kb / 10 kb / 20 kb seeds at
# the paper's baseline decoding (T=1, top_k=4), then leaves metrics to
# run_sweep_metrics.py (point --generated at this OUTROOT).
#
# Override: sbatch --export=ALL,SEEDLENS="3000 10000 20000",ONLY_TAGS="Publish_Human" window_length.sbatch

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
SEEDLENS="${SEEDLENS:-3000 10000 20000}"
ONLY_TAGS="${ONLY_TAGS:-}"
CHUNK_TOKENS="${CHUNK_TOKENS:-8192}"

mkdir -p "$HF_CACHE" "$BASE/logs" "$OUTROOT"

echo "==> nvidia-smi"
apptainer exec --nv "${IMG}" bash -lc 'nvidia-smi || true'

for sl in $SEEDLENS; do
  cfg="seed$(( sl / 1000 ))k"
  EXTRA=""
  [[ -n "$ONLY_TAGS" ]] && EXTRA="$EXTRA --only-tags $ONLY_TAGS"
  echo "==================================================================="
  echo "==> seed_len=$sl  config=$cfg (baseline decoding T=1 top_k=4)"
  apptainer exec --nv \
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
        --config-name ${cfg} \
        --model ${MODEL} \
        --temperature 1.0 --top_k 4 \
        --seed-len-override ${sl} \
        --chunk-tokens ${CHUNK_TOKENS} \
        --outdir ${OUTROOT} \
        ${EXTRA}
    "
done

echo "==> window-length sweep complete: $OUTROOT"
