#!/bin/bash
# Runs once queue_step0.sh is done: the roberta-large tune.py replay with CPU-offloaded AdamW.
source /d/cspan-ml/env.sh
cd /c/Users/anubh/CSPAN-TikTok-Analysis
until [ -f logs/queue_step0.done ]; do sleep 60; done
set -x
$PY -u repro_finetune.py --device dml --offload-optim --experiment repro_tune_large_lr1e-5_ep3 \
    --model cardiffnlp/twitter-roberta-large-2022-154m --lr 1e-5 --epochs 3 --folds 0 1 2 --seeds 0 >> logs/repro_tune_large.log 2>&1
echo DONE > logs/queue_after.done
