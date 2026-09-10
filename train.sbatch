#!/bin/bash
#SBATCH --job-name=stance
#SBATCH --array=0-4
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/fold_%a.out

# One fold per array task -- the 5 folds are independent, so wall clock is one fold, not five.
# finetune.py reads SLURM_CPUS_PER_TASK and sets OMP/MKL threads to match.
# Do NOT add dataloader workers: on CPU they steal cores from the GEMM threads.

source .venv/bin/activate
export HF_HOME="${SLURM_TMPDIR:-/tmp}/hf"   # avoid 5 tasks racing on a shared model cache

srun python finetune.py --fold "$SLURM_ARRAY_TASK_ID" --device cpu --seed 0
