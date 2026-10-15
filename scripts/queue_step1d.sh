#!/bin/bash
# Step 1 queue, one process per (seed, fold): DirectML does not free a finished fold's model memory, so a
# long-lived process creeps up to the 12 GB limit, spills into system RAM and slows ~15x (or OOMs).
# Each invocation resumes from results/runs.csv, so completed folds are skipped.
source /d/cspan-ml/env.sh
cd /c/Users/anubh/CSPAN-TikTok-Analysis
run() {  # run <experiment> <train.py args...>
    local exp=$1; shift
    for s in 0 1 2; do for k in 0 1 2 3 4; do
        $PY -u train.py --device dml --experiment "$exp" --seeds $s --folds $k "$@" >> "logs/$exp.log" 2>&1
    done; done
    echo "finished $exp"
}
set -x
# E1b_pair_len256 (max_len 256) stopped after 6/15 runs: +0.001 paired F1 vs E1 (noise), and on DirectML it
# needs ~11 GB VRAM + spill to system RAM, which slowed folds from ~6 min to 90+ min. Not pursued.
# user request: train much longer; early stopping on val F1 still picks the epoch
run E3_pair_ep20_pat3 --input pair --epochs 20 --patience 3
# user request: more dropout for long training (one change vs E3: encoder dropout 0.1 -> 0.2)
run E3d_pair_ep20_pat3_drop0.2 --input pair --epochs 20 --patience 3 --dropout 0.2
# learning-rate grid {1e-5, 3e-5} (2e-5 is E1_pair)
run E2a_pair_lr1e-5 --input pair --lr 1e-5
run E2b_pair_lr3e-5 --input pair --lr 3e-5
echo DONE > logs/queue_step1d.done
