#!/bin/bash
#SBATCH --job-name=stance
#SBATCH --array=0-14           # 5 folds x 3 seeds; task = seed*5 + fold
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=01:30:00        # measured 24 min/fold on 18 cores; 32 cores should be faster
#SBATCH --output=logs/s%a.out

# Folds and seeds are independent, so wall clock is one fold, not fifteen.
# finetune.py reads SLURM_CPUS_PER_TASK and sets OMP/MKL threads to match.
# Do NOT add dataloader workers: on CPU they steal cores from the GEMM threads.

cd "$SLURM_SUBMIT_DIR" || exit 1
source .venv/bin/activate

# Warm this cache on the login node first (see README), then the compute nodes never
# need the network. Without it, 15 tasks each try to download the model at once.
export HF_HOME="$SLURM_SUBMIT_DIR/.hf"
export HF_HUB_OFFLINE=1

# No task writes model/ -- these runs are the measurement, not the artifact.
# Use `python finetune.py --all` for the model you actually deploy.
srun python finetune.py \
  --fold "$((SLURM_ARRAY_TASK_ID % 5))" \
  --seed "$((SLURM_ARRAY_TASK_ID / 5))" \
  --device cpu
