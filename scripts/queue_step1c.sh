#!/bin/bash
# Step 1 queue (replaces 1b): E1b -> longer training -> + dropout -> lr grid. Resumable.
source /d/cspan-ml/env.sh
cd /c/Users/anubh/CSPAN-TikTok-Analysis
# wait for the running E1b process (Windows PID 40136); ps -W is reliable where tasklist was not
while ps -W | awk '{print $4}' | grep -qx 40136; do sleep 30; done
set -x
$PY -u train.py --device dml --experiment E1b_pair_len256 --input pair --max-len 256 --seeds 0 1 2 >> logs/E1b_pair_len256.log 2>&1
# user request: train much longer; early stopping on val F1 still picks the epoch
$PY -u train.py --device dml --experiment E3_pair_ep20_pat3 --input pair --epochs 20 --patience 3 --seeds 0 1 2 >> logs/E3_pair_ep20_pat3.log 2>&1
# user request: more dropout for long training (one change vs E3: encoder dropout 0.1 -> 0.2)
$PY -u train.py --device dml --experiment E3d_pair_ep20_pat3_drop0.2 --input pair --epochs 20 --patience 3 --dropout 0.2 --seeds 0 1 2 >> logs/E3d_pair_ep20_pat3_drop0.2.log 2>&1
# learning-rate grid {1e-5, 3e-5} (2e-5 is E1_pair)
$PY -u train.py --device dml --experiment E2a_pair_lr1e-5 --input pair --lr 1e-5 --seeds 0 1 2 >> logs/E2a_pair_lr1e-5.log 2>&1
$PY -u train.py --device dml --experiment E2b_pair_lr3e-5 --input pair --lr 3e-5 --seeds 0 1 2 >> logs/E2b_pair_lr3e-5.log 2>&1
echo DONE > logs/queue_step1c.done
