#!/bin/bash -l
#SBATCH --job-name=cspan-diagnose
#SBATCH --account=bjdietri
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=cspan-diagnose-%j.log
set -euo pipefail
cd /depot/bjdietri/data/tiktok/downloads
module load conda
export PYTHONUNBUFFERED=1
"$HOME/.conda/envs/tiktok/bin/python" - <<'PY'
import json, os, re, subprocess, sys
from pathlib import Path
root = Path.cwd()
old = root / 'parallel-state/run-43024049'
new = root / 'parallel-state/run-43033028'
out = root / ('diagnosis-' + os.environ['SLURM_JOB_ID'])
out.mkdir(exist_ok=False)
def urls(p):
    return set(p.read_text().splitlines())
a = urls(old / 'all-urls.txt'); b = urls(new / 'all-urls.txt')
failed_before = urls(old / 'remaining-urls.txt'); failed_now = urls(new / 'remaining-urls.txt')
success_now = b - failed_now
summary = {
 'old_discovered':len(a), 'new_discovered':len(b),
 'added_to_discovery':len(b-a), 'absent_from_new_discovery':len(a-b),
 'previous_failures_now_archived':len(failed_before & success_now),
 'previous_failures_still_unarchived':len(failed_before & failed_now),
 'previous_failures_absent_from_new_discovery':len(failed_before-b),
 'newly_discovered_archived':len((b-a) & success_now),
 'current_remaining':len(failed_now)}
(out/'reconciliation.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2),flush=True)
failed = sorted(failed_now)
if not failed: raise SystemExit('No remaining URLs to diagnose.')
indexes = sorted(set([0, len(failed)//3, 2*len(failed)//3, len(failed)-1]))
sample = [('failed',failed[i]) for i in indexes]
if success_now: sample.append(('control',sorted(success_now)[0]))
(out/'sample.json').write_text(json.dumps(sample,indent=2))
# Isolated nightly environment leaves the working downloader untouched.
venv = out/'nightly-env'
subprocess.run([sys.executable,'-m','venv',str(venv)],check=True)
nightly = str(venv/'bin/python')
with (out/'install.log').open('w') as log:
    subprocess.run([nightly,'-m','pip','install','-U','--pre','yt-dlp[default,curl-cffi]'],stdout=log,stderr=subprocess.STDOUT,check=True)
for label, python in [('installed',sys.executable),('nightly',nightly)]:
    for kind, url in sample:
        vid = url.rstrip('/').rsplit('/',1)[-1]
        if not re.fullmatch(r'\d+',vid): raise SystemExit('Unexpected URL in sample.')
        dest = out/f'{label}-{kind}-{vid}.log'
        with dest.open('w') as log:
            try:
                result = subprocess.run([python,'-m','yt_dlp','--ignore-config','--verbose',
                  '--no-playlist','--list-formats','--socket-timeout','20','--retries','1',
                  '--extractor-retries','1','--sleep-requests','3',url],
                  stdout=log,stderr=subprocess.STDOUT,timeout=90)
                print(f'{label} {kind} {vid}: exit={result.returncode}; log={dest.name}',flush=True)
            except subprocess.TimeoutExpired:
                print(f'{label} {kind} {vid}: timed out; log={dest.name}',flush=True)
print(f'Diagnosis finished. Details: {out}',flush=True)
print('Format availability is not a completed download. Existing archives and videos were not changed.',flush=True)
PY
