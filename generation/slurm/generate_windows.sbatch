#!/bin/bash
#SBATCH -J nonbdna-gen
#SBATCH -p h100
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=gpu:4
#SBATCH -t 48:00:00
#SBATCH -A *********
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
set -euo pipefail

module load tacc-apptainer

WORKDIR="runs/${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

python -m generation.src.nonbdna_gen.cli run \
  --config generation/configs/example.yaml \
  --workdir "$WORKDIR"
