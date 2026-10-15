#!/bin/bash
# Sequential GPU queue for Step 0 (+ first Step 1 experiment). Every script resumes from
# results/runs.csv, so after a crash just rerun this file.
source /d/cspan-ml/env.sh
cd /c/Users/anubh/CSPAN-TikTok-Analysis
set -x
# reproduce reported RoBERTa (0.625 / 0.652): finetune.train_fold unchanged, original folds, 3 seeds
$PY -u repro_finetune.py --device dml >> logs/repro_finetune.log 2>&1
# "71%" hypothesis: tune.py selection stage = folds 0-2, seed 0, roberta-large configs
$PY -u repro_finetune.py --device dml --experiment repro_tune_large_lr1e-5_ep3 \
    --model cardiffnlp/twitter-roberta-large-2022-154m --lr 1e-5 --epochs 3 --folds 0 1 2 --seeds 0 >> logs/repro_tune_large.log 2>&1
# shuffled-label sanity check, transformer
$PY -u train.py --device dml --experiment shuffled_roberta --shuffle-labels --seeds 0 >> logs/shuffled_roberta.log 2>&1
# reference: comment-only RoBERTa on the locked protocol
$PY -u train.py --device dml --experiment E0_comment --seeds 0 1 2 >> logs/E0_comment.log 2>&1
# Step 1.1: comment + video description sentence pair
$PY -u train.py --device dml --experiment E1_pair --input pair --seeds 0 1 2 >> logs/E1_pair.log 2>&1
echo QUEUE_DONE > logs/queue_step0.done
