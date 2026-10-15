#!/bin/bash
# Step 1, ideas 1-2 on top of E1_pair (current best). Each is one change vs E1_pair. Resumable.
source /d/cspan-ml/env.sh
cd /c/Users/anubh/CSPAN-TikTok-Analysis
until [ -f logs/queue_after.done ]; do sleep 60; done
set -x
# 1b: more description context (48.6% of pairs exceed 160 tokens)
$PY -u train.py --device dml --experiment E1b_pair_len256 --input pair --max-len 256 --seeds 0 1 2 >> logs/E1b_pair_len256.log 2>&1
# 2: learning-rate grid {1e-5, 3e-5} (2e-5 is E1_pair)
$PY -u train.py --device dml --experiment E2a_pair_lr1e-5 --input pair --lr 1e-5 --seeds 0 1 2 >> logs/E2a_pair_lr1e-5.log 2>&1
$PY -u train.py --device dml --experiment E2b_pair_lr3e-5 --input pair --lr 3e-5 --seeds 0 1 2 >> logs/E2b_pair_lr3e-5.log 2>&1
echo DONE > logs/queue_step1a.done
