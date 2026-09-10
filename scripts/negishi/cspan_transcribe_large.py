"""Two disjoint large-v3 shards of a persisted input snapshot."""
import concurrent.futures as cf
import importlib.metadata
import json
import multiprocessing
import os
from pathlib import Path
import time

ROOT = Path('/depot/bjdietri/data/tiktok/downloads')
OUT = ROOT / 'transcripts-large-v3'
MODEL = None

def initialize(model_path):
    global MODEL
    from faster_whisper import WhisperModel
    MODEL = WhisperModel(model_path, device='cpu', compute_type='int8', cpu_threads=8, num_workers=1)

def atomic(path, text):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(path)

def transcribe(item):
    video, url = item
    video = Path(video)
    try:
        segments, info = MODEL.transcribe(str(video), language='en', task='transcribe',
            beam_size=5, vad_filter=True, condition_on_previous_text=False)
        rows = [{'start': s.start, 'end': s.end, 'text': s.text,
                 'avg_logprob': s.avg_logprob, 'no_speech_prob': s.no_speech_prob}
                for s in segments]
        result = {'source_file': str(video), 'source_url': url, 'model': 'large-v3',
                  'compute_type': 'int8', 'language': 'en', 'beam_size': 5,
                  'vad_filter': True, 'condition_on_previous_text': False,
                  'faster_whisper_version': importlib.metadata.version('faster-whisper'),
                  'duration_seconds': info.duration, 'segments': rows,
                  'review_note': 'Automatic transcript; review accuracy. Empty segments may indicate no detected speech.'}
        # JSON is written last and serves as the completion marker.
        atomic(OUT / (video.stem + '.txt'), '\n'.join(r['text'].strip() for r in rows) + '\n')
        atomic(OUT / (video.stem + '.json'), json.dumps(result, ensure_ascii=False, indent=2))
        return str(video), info.duration, None
    except Exception as exc:
        return str(video), 0, repr(exc)

def inputs():
    files = {}
    manifests = [ROOT / 'video-manifest.tsv', *ROOT.glob('parallel-state/run-*/*.manifest.tsv')]
    for manifest in manifests:
        if not manifest.is_file():
            continue
        for line in manifest.read_text().splitlines():
            if '\t' not in line:
                continue
            url, name = line.split('\t', 1)
            path = Path(name)
            if not path.is_absolute():
                path = ROOT / path
            path = path.resolve()
            if path.parent == (ROOT / 'videos').resolve() and path.is_file() and path.stat().st_size:
                files[str(path)] = url
    return sorted(files.items())

def main():
    OUT.mkdir(exist_ok=True)
    import fcntl
    snapshot = ROOT / 'large-v3-inputs.json'
    with (ROOT / '.large-v3-inputs.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not snapshot.exists():
            records = inputs()
            if not records:
                raise SystemExit('No completed videos available for snapshot.')
            atomic(snapshot, json.dumps(records))
        all_files = json.loads(snapshot.read_text())
    # Partition the full stable list BEFORE filtering completed transcripts.
    shard = int(os.environ.get('SLURM_ARRAY_TASK_ID', '0'))
    if shard not in (0, 1):
        raise SystemExit('Expected array task 0 or 1')
    all_files = all_files[shard::2]
    pending = [(p, u) for p, u in all_files if not (OUT / (Path(p).stem + '.json')).is_file()]
    print(f'Completed video files in manifests: {len(all_files)}; transcripts to create: {len(pending)}', flush=True)
    if not all_files:
        raise SystemExit('No completed videos found in download manifests.')
    if not pending:
        return
    from faster_whisper.utils import download_model
    print('Preparing large-v3 model (first run downloads weights).', flush=True)
    model_path = download_model('large-v3')
    workers = min(12, max(1, int(os.environ.get('SLURM_CPUS_PER_TASK', '96')) // 8), len(pending))
    print(f'Starting {workers} workers, 8 CPU threads each.', flush=True)
    started = time.monotonic()
    good = bad = 0
    job = os.environ.get('SLURM_JOB_ID', str(int(time.time())))
    with (OUT / f'errors-{job}.jsonl').open('w') as errors:
        with cf.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                    initializer=initialize, initargs=(model_path,)) as pool:
            futures = {pool.submit(transcribe, item) for item in pending}
            while futures:
                finished, futures = cf.wait(futures, timeout=60, return_when=cf.FIRST_COMPLETED)
                for future in finished:
                    path, duration, error = future.result()
                    if error:
                        bad += 1
                        errors.write(json.dumps({'file': path, 'error': error}) + '\n')
                        errors.flush()
                    else:
                        good += 1
                minutes = max((time.monotonic() - started) / 60, 0.001)
                if not finished or (good + bad) % 10 == 0 or not futures:
                    rate = good / minutes
                    eta = len(futures) / rate if rate else None
                    print(f'Transcribed {good}/{len(pending)}; errors {bad}; {rate:.2f} videos/min; estimated minutes left: {round(eta, 1) if eta else "unknown"}', flush=True)
    print('Shard finished. Rerun to retry unfinished transcripts in large-v3-inputs.json.', flush=True)
    raise SystemExit(1 if bad else 0)

if __name__ == '__main__':
    main()
