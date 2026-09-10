#!/bin/bash -l
#SBATCH --job-name=cspan-photos
#SBATCH --account=bjdietri
#SBATCH --partition=cpu
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --output=cspan-photos-%j.log
set -euo pipefail
cd /depot/bjdietri/data/tiktok/downloads
exec 9>.cspan-photos.lock
flock -n 9 || { echo 'A photo job is already running.'; exit 1; }
module load conda
export PYTHONUNBUFFERED=1
PHOTO_ENV="$HOME/.venvs/cspan-photos"
if [ ! -x "$PHOTO_ENV/bin/python" ]; then
    "$HOME/.conda/envs/tiktok/bin/python" -m venv "$PHOTO_ENV"
fi
"$PHOTO_ENV/bin/python" -m pip install -U gallery-dl
"$PHOTO_ENV/bin/python" - <<'PY'
import json, os, re, subprocess, sys, time
from pathlib import Path
root=Path.cwd(); photos=root/'photos';photos.mkdir(exist_ok=True)
source=root/'parallel-state/run-43033028/remaining-urls.txt'
urls=list(dict.fromkeys(source.read_text().splitlines()))
known='https://www.tiktok.com/@cspanofficial/video/7331135274807807275'
urls=[known]+[u for u in urls if u!=known]
job=os.environ['SLURM_JOB_ID']
version=subprocess.check_output([sys.executable,'-m','gallery_dl','--version'],text=True).strip()
print(f'gallery-dl {version}; {len(urls)} candidate posts; first test must succeed.',flush=True)
good=bad=0
for index,url in enumerate(urls):
    vid=url.rstrip('/').rsplit('/',1)[-1]
    if not re.fullmatch(r'\d+',vid): raise SystemExit('Invalid post URL: '+url)
    folder=photos/vid;folder.mkdir(exist_ok=True)
    marker=folder/'collection-complete.json'
    if marker.exists():
        saved=json.loads(marker.read_text())
        if saved.get('files') and all((folder/f).is_file() and (folder/f).stat().st_size for f in saved['files']):
            good+=1;continue
    log=folder/f'gallery-{job}.log'
    cmd=[sys.executable,'-m','gallery_dl','--config-ignore','--no-input',
      '--write-metadata','--write-info-json','--retries','2','--http-timeout','25',
      '--sleep-request','2','--sleep','1-3','--directory',str(folder),
      '--filename','{id}_{type}_{num:03}.{extension}',url]
    with log.open('w') as stream:
        try:
            code=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=180).returncode
        except subprocess.TimeoutExpired:
            code=124
    media=[p for p in folder.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.avif','.mp3','.m4a','.mp4','.webm','.ogg'} and p.stat().st_size]
    metadata=[]
    for p in folder.glob('*.json'):
        if p==marker:continue
        try:
            value=json.loads(p.read_text())
            if isinstance(value,dict) and str(value.get('id',''))==vid:metadata.append((p,value))
        except (ValueError,OSError):pass
    images=[p for p in media if p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.avif'}]
    types={d.get('post_type') for _,d in metadata}
    expected=max([len(d.get('imagePost',{}).get('images',[])) for _,d in metadata] or [0])
    complete=code==0 and bool(media) and bool(metadata) and (
        ('image' in types and expected > 0 and len(images)==expected) or
        ('video' in types and any(p.suffix.lower() in {'.mp4','.webm'} for p in media)))
    if index==0 and not (complete and images):
        raise SystemExit(f'Photo test not verified. Inspect {log}; downloaded media={len(media)}, metadata={len(metadata)}. Bulk run stopped.')
    if complete:
        record={'source_url':url,'post_id':vid,'gallery_dl_version':version,
          'retrieved_at_unix':time.time(),'post_types':sorted(t for t in types if t),
          'expected_images':expected,'saved_images':len(images),
          'files':[p.name for p in media]+[p.name for p,_ in metadata],
          'note':'Original media plus exposed metadata. Any available audio saved separately; no MP4 synthesized.'}
        temp=marker.with_suffix('.tmp');temp.write_text(json.dumps(record,indent=2));temp.replace(marker);good+=1
    else:bad+=1
    print(f'{index+1}/{len(urls)} posts processed; verified collections={good}; unresolved={bad}; last={vid} exit={code} images={len(images)} expected={expected}',flush=True)
print('Photo recovery finished. Existing video archives were not modified.',flush=True)
raise SystemExit(1 if bad else 0)
PY
