#!/bin/bash
#SBATCH --job-name=stance-compare
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=00:45:00         # inference only: 3 models x 3,675 short comments, plus the baseline
#SBATCH --output=logs/compare.out   # must NOT match logs/s*.out, which compare.py reads

# Needs two things first:
#   1. python compare.py --download   (on a node with internet; compute nodes are offline)
#   2. train.sbatch finished, so logs/s*.out holds our model's scores
# Submit with a dependency to handle (2) automatically -- see README.

cd "$SLURM_SUBMIT_DIR" || exit 1
[ -f .venv/bin/activate ] && source .venv/bin/activate
export HF_HOME="$SLURM_SUBMIT_DIR/.hf"
export HF_HUB_OFFLINE=1

srun python compare.py --device cpu --full
