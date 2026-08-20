#!/bin/bash
#SBATCH -J evo2_decsweep
#SBATCH -A BCS25073
#SBATCH -p h100
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH -o /work/11034/atzanakak/ls6/logs/%x-%j.out
#SBATCH -e /work/11034/atzanakak/ls6/logs/%x-%j.err

# Alternative-decoding generation sweep for the Evo2 revision (R3.7, R1 minor #2).
# Generates synthetic sequences for each window in the seed manifest under
# several decoding configurations, then leaves metric computation to the CPU
# pipeline (run_sweep_metrics.sh).
#
# Override scope via:  sbatch --export=ALL,CONFIGS="baseline lowtemp nucleus",ONLY_TAGS="Publish_Human" decoding_sweep.sbatch

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
MANIFEST="${MANIFEST:-$REV/decoding_sweep/seed_manifest.csv}"
OUTROOT="${OUTROOT:-$REV/decoding_sweep/generated}"
MODEL="${MODEL:-evo2_7b}"
CONFIGS="${CONFIGS:-lowtemp nucleus}"
ONLY_TAGS="${ONLY_TAGS:-}"
CHUNK_TOKENS="${CHUNK_TOKENS:-8192}"
SEED_LEN_OVERRIDE="${SEED_LEN_OVERRIDE:-}"

mkdir -p "$HF_CACHE" "$BASE/logs" "$OUTROOT"

# Decoding-config definitions: name -> "T topk topp"
declare -A CFG_T CFG_TOPK CFG_TOPP
# paper baseline (already generated separately; included for completeness)
CFG_T[baseline]=1.0 ; CFG_TOPK[baseline]=4 ; CFG_TOPP[baseline]=
# lower temperature, same top-k
CFG_T[lowtemp]=0.7 ; CFG_TOPK[lowtemp]=4 ; CFG_TOPP[lowtemp]=
# nucleus / top-p sampling (reviewer-requested)
CFG_T[nucleus]=1.0 ; CFG_TOPK[nucleus]= ; CFG_TOPP[nucleus]=0.9
# greedy-ish very low temperature
CFG_T[greedy]=0.1 ; CFG_TOPK[greedy]=1 ; CFG_TOPP[greedy]=

echo "==> nvidia-smi"
apptainer exec --nv "${IMG}" bash -lc 'nvidia-smi || true'

for cfg in $CONFIGS; do
  T="${CFG_T[$cfg]:-}"; TOPK="${CFG_TOPK[$cfg]:-}"; TOPP="${CFG_TOPP[$cfg]:-}"
  if [[ -z "$T" ]]; then echo "[skip] unknown config $cfg"; continue; fi
  EXTRA=""
  [[ -n "$TOPK" ]] && EXTRA="$EXTRA --top_k $TOPK"
  [[ -n "$TOPP" ]] && EXTRA="$EXTRA --top_p $TOPP"
  [[ -n "$ONLY_TAGS" ]] && EXTRA="$EXTRA --only-tags $ONLY_TAGS"
  [[ -n "$SEED_LEN_OVERRIDE" ]] && EXTRA="$EXTRA --seed-len-override $SEED_LEN_OVERRIDE"
  echo "==================================================================="
  echo "==> config=$cfg  T=$T top_k=${TOPK:-none} top_p=${TOPP:-none}"
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
        --temperature ${T} \
        --chunk-tokens ${CHUNK_TOKENS} \
        --outdir ${OUTROOT} \
        ${EXTRA}
    "
done

echo "==> decoding sweep complete: $OUTROOT"
