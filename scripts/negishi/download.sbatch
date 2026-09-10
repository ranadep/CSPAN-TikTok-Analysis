#!/bin/bash -l
#SBATCH --job-name=cspan-tiktok
#SBATCH --account=bjdietri
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=24:00:00
#SBATCH --output=tiktok-%j.log

set -euo pipefail
module load conda
  cd /depot/bjdietri/data/tiktok/downloads

conda run --no-capture-output -p "$HOME/.conda/envs/tiktok" \
  bash download_tiktok_cspan.sh