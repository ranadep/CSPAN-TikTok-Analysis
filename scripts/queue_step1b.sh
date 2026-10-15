#!/bin/bash
# Step 1, reordered: E1b (already running) -> longer training -> lr grid. One change each vs E1_pair. Resumable.
source /d/cspan-ml/env.sh
cd /c/Users/anubh/CSPAN-TikTok-Analysis
while tasklist //FI "PID eq 17600" | grep -q python; do sleep 30; done   # E1b_pair_len256 finishing
set -x
# resume E1b if it was interrupted (no-op when complete)
$PY -u train.py --device dml --experiment E1b_pair_len256 --input pair --max-len 256 --seeds 0 1 2 >> logs/E1b_pair_len256.log 2>&1
# user request: train much longer; early stopping on val F1 still picks the epoch (5/15 E1 runs peaked at the 6-epoch cap)
$PY -u train.py --device dml --experiment E3_pair_ep20_pat3 --input pair --epochs 20 --patience 3 --seeds 0 1 2 >> logs/E3_pair_ep20_pat3.log 2>&1
# learning-rate grid {1e-5, 3e-5} (2e-5 is E1_pair)
$PY -u train.py --device dml --experiment E2a_pair_lr1e-5 --input pair --lr 1e-5 --seeds 0 1 2 >> logs/E2a_pair_lr1e-5.log 2>&1
$PY -u train.py --device dml --experiment E2b_pair_lr3e-5 --input pair --lr 3e-5 --seeds 0 1 2 >> logs/E2b_pair_lr3e-5.log 2>&1
echo DONE > logs/queue_step1b.done
