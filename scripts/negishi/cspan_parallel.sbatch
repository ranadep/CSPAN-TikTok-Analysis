#!/bin/bash -l
#SBATCH --job-name=cspan-parallel
#SBATCH --account=bjdietri
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=cspan-parallel-%j.log

set -euo pipefail
cd /depot/bjdietri/data/tiktok/downloads
# Prevent overlapping copies of THIS job. Do not run the old downloader alongside it.
exec 9>.cspan-parallel.lock
flock -n 9 || { echo 'Another parallel download job is already running.'; exit 1; }
module load conda
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
conda run --no-capture-output -p "$HOME/.conda/envs/tiktok" python - <<'PYTHON'
import concurrent.futures
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

root = Path.cwd()
state = root / 'parallel-state'
state.mkdir(exist_ok=True)
(root / 'videos').mkdir(exist_ok=True)
workers = 32
base = [sys.executable, '-m', 'yt_dlp', '--ignore-config']
run = state / ('run-' + os.environ.get('SLURM_JOB_ID', str(int(time.time()))))
run.mkdir(exist_ok=False)
print('Logs and batches:', run, flush=True)

# Discover once per run; do not silently reuse an incomplete profile listing.
print('Discovering C-SPAN video URLs. This may take several minutes.', flush=True)
with (run / 'profile.json').open('w') as out, (run / 'discovery.log').open('w') as err:
    result = subprocess.run(base + [
        '--flat-playlist', '--dump-single-json', '--skip-download',
        '--abort-on-error', '--socket-timeout', '30', '--retries', '3',
        '--extractor-retries', '3', '--sleep-requests', '1',
        'https://www.tiktok.com/@cspanofficial'], stdout=out, stderr=err)
if result.returncode:
    raise SystemExit('Profile discovery failed. Read ' + str(run / 'discovery.log'))
data = json.loads((run / 'profile.json').read_text())
entries = data.get('entries') or []
ids = list(dict.fromkeys(str(e['id']) for e in entries
                        if e and re.fullmatch(r'\d+', str(e.get('id', '')))))
if not ids:
    raise SystemExit('No video IDs discovered; inspect profile.json and discovery.log.')
print(f'Discovered {len(ids)} unique video IDs; this is not proof of profile completeness.', flush=True)
(run / 'all-urls.txt').write_text(''.join(f'https://www.tiktok.com/@cspanofficial/video/{i}\n' for i in ids))

# Gather receipts from both the original downloader and earlier parallel runs.
def receipts():
    lines = set()
    for path in [root / '.downloaded-archive.txt', *state.glob('worker-*.archive')]:
        if path.exists():
            lines.update(line.strip() for line in path.read_text().splitlines() if line.strip())
    return lines

def completed_ids(lines):
    return {line.split()[-1] for line in lines if line.lower().startswith('tiktok ')}

seed = receipts()
done = completed_ids(seed)
pending = [i for i in ids if i not in done]
print(f'Already archived: {len(set(ids) & done)}; remaining: {len(pending)}', flush=True)
if not pending:
    raise SystemExit(0)

for worker in range(workers):
    archive = state / f'worker-{worker}.archive'
    temporary = archive.with_suffix('.tmp')
    temporary.write_text(''.join(s + '\n' for s in sorted(seed)))
    temporary.replace(archive)
    (run / f'worker-{worker}.urls').write_text(''.join(
        f'https://www.tiktok.com/@cspanofficial/video/{i}\n' for i in pending[worker::workers]))

# Separate archives/manifests; disjoint IDs mean workers never target the same video.
def download(worker):
    log = run / f'worker-{worker}.log'
    cmd = base + [
        '--no-playlist', '--continue', '--no-overwrites',
        '--socket-timeout', '30', '--retries', '3', '--fragment-retries', '3',
        '--extractor-retries', '3', '--retry-sleep', 'http:exp=5:60',
        '--sleep-requests', '1', '--sleep-interval', '2', '--max-sleep-interval', '5',
        '--restrict-filenames', '--merge-output-format', 'mp4', '--write-info-json',
        '--no-progress', '--newline',
        '--download-archive', str(state / f'worker-{worker}.archive'),
        '--print-to-file', 'after_move:%(webpage_url)s\t%(filepath)s',
        str(run / f'worker-{worker}.manifest.tsv'),
        '-o', 'videos/%(upload_date>%Y-%m-%d)s_%(id)s.%(ext)s',
        '--batch-file', str(run / f'worker-{worker}.urls')]
    with log.open('w') as out:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        blocked = 0
        for line in proc.stdout:
            out.write(line)
            out.flush()
            if re.search(r'HTTP Error (403|429)', line):
                blocked += 1
            if blocked >= 5:
                print(f'Worker {worker}: repeated HTTP blocking; stopping. See {log}', flush=True)
                proc.terminate()
                break
        try:
            return proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            return proc.wait()

started = time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
    futures = [pool.submit(download, w) for w in range(workers)]
    while True:
        finished = all(f.done() for f in futures)
        current = completed_ids(receipts())
        newly_done = len((current - done) & set(ids))
        minutes = max((time.monotonic() - started) / 60, 0.01)
        print(f'New downloads: {newly_done}/{len(pending)}; average {newly_done / minutes:.2f}/minute', flush=True)
        if finished:
            codes = [f.result() for f in futures]
            break
        time.sleep(60)
remaining = [i for i in ids if i not in completed_ids(receipts())]
(run / 'remaining-urls.txt').write_text(''.join(f'https://www.tiktok.com/@cspanofficial/video/{i}\n' for i in remaining))
print(f'Finished attempts. Unarchived videos: {len(remaining)}. Worker exit codes: {codes}', flush=True)
print('Rerun this same job to retry; retain parallel-state and the original archive.', flush=True)
raise SystemExit(1 if remaining or any(codes) else 0)
PYTHON
