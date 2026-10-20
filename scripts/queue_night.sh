#!/bin/bash
# Overnight Step 1 queue. One process per (seed, fold) -- DirectML does not free a finished fold's memory.
# Each config is ONE change vs the current best (E1_pair). Every train.py call resumes from results/runs.csv,
# so rerunning this file after a crash only redoes the fold that was interrupted.
# Stall guard: a fold that runs past FOLD_TIMEOUT is killed and retried once (memory pressure makes DirectML
# page to disk; a fresh process usually recovers). After each experiment, results are committed and pushed.
source /d/cspan-ml/env.sh
cd /c/Users/anubh/CSPAN-TikTok-Analysis
FOLD_TIMEOUT=2400   # 40 min; a healthy 20-epoch pair fold takes ~6-12 min

kill_trainers() {  # the venv launcher spawns the real interpreter; kill every train.py process tree
    powershell.exe -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train.py*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }"
    sleep 10
}

fold() {  # fold <experiment> <seed> <k> <args...>
    local exp=$1 s=$2 k=$3; shift 3
    for attempt in 1 2; do
        timeout "$FOLD_TIMEOUT" "$PY" -u train.py --device dml --experiment "$exp" --seeds "$s" --folds "$k" "$@" >> "logs/$exp.log" 2>&1
        local rc=$?
        [ $rc -ne 124 ] && return $rc
        echo "STALL: $exp seed $s fold $k exceeded ${FOLD_TIMEOUT}s (attempt $attempt)" | tee -a "logs/$exp.log"
        kill_trainers
    done
}

run() {  # run <experiment> <train.py args...>
    local exp=$1; shift
    for s in 0 1 2; do for k in 0 1 2 3 4; do fold "$exp" "$s" "$k" "$@"; done; done
    echo "finished $exp"
    git add results/runs.csv results/curves.csv >/dev/null 2>&1
    git commit -q -m "Step 1 results: $exp

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" >/dev/null 2>&1 && git push -q >/dev/null 2>&1 && echo "pushed $exp"
}

set -x
# user request: train much longer; early stopping on val F1 still picks the epoch (5/15 E1 runs peaked at the 6-epoch cap)
run E3_pair_ep20_pat3 --input pair --epochs 20 --patience 3
# user request: more dropout for long training (one change vs E3: encoder dropout 0.1 -> 0.2)
run E3d_pair_ep20_pat3_drop0.2 --input pair --epochs 20 --patience 3 --dropout 0.2
# idea 3: noisy-label robustness (E1 curves: val loss rises from epoch 2-3 while F1 still climbs = overconfidence)
run E4a_pair_ls0.1 --input pair --label-smoothing 0.1
# idea 2: learning-rate grid {1e-5, 3e-5} (2e-5 is E1_pair)
run E2a_pair_lr1e-5 --input pair --lr 1e-5
run E2b_pair_lr3e-5 --input pair --lr 3e-5
# idea 3: class-weighted CE (neutral is the weak class, F1 0.51)
run E4b_pair_weighted --input pair --loss weighted
# idea 4: backbones. bertweet has 130 position embeddings, so max_len 128 is forced (second, unavoidable change)
run E5a_pair_bertweet --input pair --backbone vinai/bertweet-base --max-len 128
run E5b_pair_debertav3 --input pair --backbone microsoft/deberta-v3-base
echo DONE > logs/queue_night.done
