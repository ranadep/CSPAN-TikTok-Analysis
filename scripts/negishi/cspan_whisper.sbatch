#!/bin/bash -l
#SBATCH --job-name=cspan-whisper
#SBATCH --account=bjdietri
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=cspan-whisper-%j.log

set -euo pipefail
cd /depot/bjdietri/data/tiktok/downloads
exec 9>.cspan-whisper.lock
flock -n 9 || { echo 'Another transcription job is running.'; exit 1; }
module load conda
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Separate environment so the running downloader is unaffected.
WHISPER_ENV="$HOME/.venvs/cspan-whisper"
if [ ! -x "$WHISPER_ENV/bin/python" ]; then
    "$HOME/.conda/envs/tiktok/bin/python" -m venv "$WHISPER_ENV"
fi
if ! "$WHISPER_ENV/bin/python" -c 'import faster_whisper' >/dev/null 2>&1; then
    "$WHISPER_ENV/bin/python" -m pip install faster-whisper
fi
"$WHISPER_ENV/bin/python" cspan_transcribe.py
