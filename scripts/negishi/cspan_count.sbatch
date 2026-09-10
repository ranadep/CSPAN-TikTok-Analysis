#!/bin/bash -l
#SBATCH --job-name=cspan-count
#SBATCH --account=bjdietri
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=cspan-count-%j.log
set -euo pipefail
cd /depot/bjdietri/data/tiktok/downloads
module load conda
export PYTHONUNBUFFERED=1
conda run --no-capture-output -p "$HOME/.conda/envs/tiktok" python - <<'PY'
import json, os, re, subprocess, sys
from pathlib import Path
root=Path.cwd()
out=root/('count-'+os.environ['SLURM_JOB_ID'])
out.mkdir(exist_ok=False)
base=[sys.executable,'-m','yt_dlp','--ignore-config']
subprocess.run(base+['--version'],check=True)
old=set()
for p in sorted(root.glob('parallel-state/run-*/all-urls.txt')):
    ids={line.rstrip('/').rsplit('/',1)[-1] for line in p.read_text().splitlines() if line.strip()}
    old.update(ids)
    print(f'Previous {p.parent.name}: {len(ids)} unique IDs',flush=True)
print(f'Union of previous discovery runs: {len(old)}',flush=True)
fresh=[]
for attempt in (1,2):
    print(f'Starting fresh listing {attempt}/2; no media downloads.',flush=True)
    with (out/f'profile-{attempt}.json').open('w') as data, (out/f'discovery-{attempt}.log').open('w') as log:
        result=subprocess.run(base+['--flat-playlist','--dump-single-json','--skip-download',
          '--abort-on-error','--socket-timeout','30','--retries','3','--extractor-retries','3',
          '--sleep-requests','1','https://www.tiktok.com/@cspanofficial'],stdout=data,stderr=log)
    if result.returncode:
        raise SystemExit(f'Listing {attempt} failed; inspect {out}/discovery-{attempt}.log. Do not treat as a complete count.')
    data=json.loads((out/f'profile-{attempt}.json').read_text())
    ids={str(e['id']) for e in data.get('entries',[]) if e and re.fullmatch(r'\d+',str(e.get('id','')))}
    if not ids: raise SystemExit('Empty listing; cannot establish count.')
    fresh.append(ids)
    print(f'Fresh listing {attempt}: {len(ids)} unique IDs',flush=True)
    print(f'playlist_count field (extractor metadata, not independent verification): {data.get("playlist_count")}',flush=True)
union=old|fresh[0]|fresh[1]
archives=[root/'.downloaded-archive.txt',*root.glob('parallel-state/worker-*.archive')]
done=set()
for p in archives:
    if p.exists():
        done.update(line.split()[-1] for line in p.read_text().splitlines() if line.lower().startswith('tiktok '))
summary={'fresh_pass_1':len(fresh[0]),'fresh_pass_2':len(fresh[1]),
 'ids_differing_between_fresh_passes':len(fresh[0]^fresh[1]),
 'previous_union':len(old),'all_runs_union':len(union),
 'new_to_previous_runs':len(union-old),'previous_ids_absent_from_both_fresh_passes':len(old-(fresh[0]|fresh[1])),
 'archived_in_union':len(union&done),'unarchived_in_union':len(union-done)}
(out/'summary.json').write_text(json.dumps(summary,indent=2))
for name,ids in [('all-observed-urls',union),('unarchived-urls',union-done),('newly-observed-urls',union-old)]:
    (out/f'{name}.txt').write_text(''.join(f'https://www.tiktok.com/@cspanofficial/video/{i}\n' for i in sorted(ids)))
print(json.dumps(summary,indent=2),flush=True)
print('These are discovered post IDs, not independently verified video counts. Matching passes do not prove complete coverage.',flush=True)
print(f'Results saved to {out}',flush=True)
PY
