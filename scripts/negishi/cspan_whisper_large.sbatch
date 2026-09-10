#!/bin/bash -l
#SBATCH --job-name=cspan-whisper-large
#SBATCH --account=bjdietri
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --array=0-1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=96
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=cspan-whisper-large-%A_%a.log

set -euo pipefail
cd /depot/bjdietri/data/tiktok/downloads
exec 9>".cspan-whisper-large-${SLURM_ARRAY_TASK_ID}.lock"
flock -n 9 || { echo 'Another transcription job is running.'; exit 1; }
module load conda
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Separate environment so the running downloader is unaffected.
WHISPER_ENV="$HOME/.venvs/cspan-whisper"
exec 8>"$HOME/.cspan-whisper-setup.lock"
flock 8
if [ ! -x "$WHISPER_ENV/bin/python" ]; then
    "$HOME/.conda/envs/tiktok/bin/python" -m venv "$WHISPER_ENV"
fi
if ! "$WHISPER_ENV/bin/python" -c 'import faster_whisper' >/dev/null 2>&1; then
    "$WHISPER_ENV/bin/python" -m pip install faster-whisper
fi
flock -u 8
exec 8>&-
"$WHISPER_ENV/bin/python" cspan_transcribe_large.py
