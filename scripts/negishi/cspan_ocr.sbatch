#!/bin/bash -l
#SBATCH --job-name=cspan-ocr
#SBATCH --account=bjdietri
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=cspan-ocr-%j.log
set -euo pipefail
cd /depot/bjdietri/data/tiktok/downloads
exec 9>.cspan-ocr.lock
flock -n 9 || { echo 'An OCR job is already running.'; exit 1; }
module load conda
export OMP_THREAD_LIMIT=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
OCR_ENV="$HOME/.conda/envs/cspan-ocr"
if [ ! -x "$OCR_ENV/bin/tesseract" ] || [ ! -x "$OCR_ENV/bin/python" ]; then
    conda create -y -p "$OCR_ENV" -c conda-forge python=3.12 tesseract pillow
fi
conda run --no-capture-output -p "$OCR_ENV" python cspan_ocr.py
