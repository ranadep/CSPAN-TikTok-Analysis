# Negishi collection and text extraction

Scripts for C-SPAN TikTok collection, photo recovery, video transcription,
and photo OCR on Purdue Negishi.

These scripts currently use:
- Account: bjdietri
- Working directory: /depot/bjdietri/data/tiktok/downloads
- User environments under $HOME

They are copies of the working scripts, not a portable installation package.
Review paths, environments, and resource requests before running elsewhere.

Main jobs:
- cspan_parallel.sbatch: parallel video download
- cspan_count.sbatch: compare profile discovery counts
- cspan_diagnose.sbatch: compare installed/nightly extraction
- cspan_photos.sbatch: recover images and metadata
- cspan_whisper_large.sbatch: large-v3 transcription, two array tasks
- cspan_ocr.sbatch: image text extraction

Python files must be copied alongside their corresponding submission scripts
in the working directory.

The large-v3 array requests 192 CPUs in total. Other jobs may wait for capacity.
Small-model transcription scripts are retained as an alternative.

Downloaded media, metadata, logs, and generated text are not included.
