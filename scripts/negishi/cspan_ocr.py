"""English OCR of images from verified photo collections; restartable snapshot."""
import concurrent.futures as cf
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
from PIL import Image, ImageOps

ROOT=Path('/depot/bjdietri/data/tiktok/downloads')
OUT=ROOT/'photo-text'
VERSION=''

def atomic(path,text):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(text,encoding='utf-8');tmp.replace(path)

def process(item):
    path,record=item
    dest=OUT/path.parent.name;dest.mkdir(exist_ok=True)
    marker=dest/(path.name+'.json')
    try:
        if marker.exists() and (dest/(path.name+'.txt')).exists():
            saved=json.loads(marker.read_text())
            if saved.get('source_size')==path.stat().st_size and saved.get('source_mtime_ns')==path.stat().st_mtime_ns:
                return 'skipped',str(path),None
        with tempfile.TemporaryDirectory() as tmp:
            png=Path(tmp)/'input.png';prefix=Path(tmp)/'ocr'
            with Image.open(path) as im:
                im=ImageOps.exif_transpose(im).convert('RGB')
                width,height=im.size;im.save(png)
            command=['tesseract',str(png),str(prefix),'-l','eng','--oem','1','--psm','3','txt','tsv']
            result=subprocess.run(command,capture_output=True,text=True,timeout=180)
            if result.returncode:raise RuntimeError(result.stderr[-2000:])
            text=prefix.with_suffix('.txt').read_text()
            words=[]
            for r in csv.DictReader(io.StringIO(prefix.with_suffix('.tsv').read_text()),delimiter='\t',quoting=csv.QUOTE_NONE):
                if r['level']=='5' and r.get('text','').strip():
                    words.append({'text':r['text'],'confidence':float(r['conf']),
                        **{k:int(r[k]) for k in ('left','top','width','height','block_num','par_num','line_num','word_num')}})
        stat=path.stat()
        data={'post_id':path.parent.name,'source_url':record['source_url'],'source_image':str(path),
          'source_size':stat.st_size,'source_mtime_ns':stat.st_mtime_ns,'engine':VERSION,
          'language':'eng','page_segmentation_mode':3,'image_width':width,'image_height':height,
          'coordinate_system':'Pixels in EXIF-oriented image','text':text,'words':words,
          'needs_review':not words or any(w['confidence']<60 for w in words),
          'note':'OCR of image pixels only, not TikTok caption. Confidence is an engine score, not probability of correctness.'}
        atomic(dest/(path.name+'.txt'),text)
        atomic(marker,json.dumps(data,ensure_ascii=False,indent=2))
        return 'completed',str(path),None
    except Exception as exc:return 'failed',str(path),repr(exc)

def main():
    global VERSION
    OUT.mkdir(exist_ok=True)
    VERSION=subprocess.check_output(['tesseract','--version'],text=True).splitlines()[0]
    langs=subprocess.check_output(['tesseract','--list-langs'],text=True)
    if 'eng' not in langs.split():raise SystemExit('English OCR language data missing.')
    items=[]
    for marker in sorted((ROOT/'photos').glob('*/collection-complete.json')):
        record=json.loads(marker.read_text())
        for name in record['files']:
            path=marker.parent/name
            if path.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.avif'} and path.is_file():
                items.append((path,record))
    items=list({str(p):(p,r) for p,r in items}.values())
    if not items:raise SystemExit('No verified photo images yet. Run after photo collection finishes.')
    print(f'{VERSION}; {len(items)} verified image files in snapshot.',flush=True)
    counts={'completed':0,'skipped':0,'failed':0};started=time.monotonic()
    with (OUT/f'errors-{os.environ.get("SLURM_JOB_ID","local")}.jsonl').open('w') as errors:
        with cf.ThreadPoolExecutor(max_workers=min(16,int(os.environ.get('SLURM_CPUS_PER_TASK','16')))) as pool:
            futures=[pool.submit(process,item) for item in items]
            for f in cf.as_completed(futures):
                status,path,error=f.result();counts[status]+=1
                if error:errors.write(json.dumps({'file':path,'error':error})+'\n');errors.flush()
                if sum(counts.values())%10==0:print(counts,flush=True)
    print(f'Finished in {(time.monotonic()-started)/60:.1f} minutes: {counts}. Rerun for newly collected photos.',flush=True)
    raise SystemExit(1 if counts['failed'] else 0)

if __name__=='__main__':main()
