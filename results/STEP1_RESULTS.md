# Step 1 results (running log)

Locked protocol, near-dup v2 (nd2), dev CV only: 52 videos, 5 video-grouped folds, 3 seeds each. The frozen
holdout is untouched. Macro-F1 is the mean ± std over the 15 (fold × seed) runs; accuracy is the mean per-fold
accuracy. Noise floor: fold std ≈ 0.05. Every run is in `results/runs.csv`; curves are in `results/plots/`.

| # | experiment | change vs current best | macro-F1 | accuracy | seed-ensemble F1 / acc | verdict |
|---|---|---|---|---|---|---|
| — | E0_comment | reference: comment only | 0.607 ± 0.054 | 0.625 | 0.617 / 0.633 | reference |
| 1 | **E1_pair** | + video description as a sentence pair (max_len 160) | **0.642 ± 0.054** | **0.673** | **0.660 / 0.691** | **adopted (current best)** |

## E1: comment + video description

* Paired against E0 on identical (fold, seed): **+0.036 macro-F1, better in 13 of 15 runs** (paired std 0.032).
  The per-fold mean gain is +0.054, +0.026, +0.001, +0.052 and +0.046. Unpaired, the gain is below the 0.054 fold std.
  It is adopted on the paired evidence plus the prior (GPT-4o labeled each comment *with* the description).
  The same experiment under nd1 gave the same result (0.642 ± 0.046), so the gain is reproducible.
* Seed-ensemble, pooled CV test folds (n = 2,977):

  | class | precision | recall | F1 |
  |---|---|---|---|
  | left | 0.752 | 0.767 | 0.759 |
  | neutral | 0.531 | 0.496 | 0.513 |
  | right | 0.702 | 0.713 | 0.708 |

  Confusion (rows = true): left 977 / 150 / 147, neutral 146 / 316 / 175, right 177 / 129 / 760.
  Neutral is the weak class, as expected from a labeling prompt that tells GPT-4o to assume political intent.
* Slices (ensemble): **C-SPAN videos 0.699 F1 / 0.753 acc** (E0: 0.613 / 0.651); non-C-SPAN 0.630 / 0.646;
  emoji-only 0.701 / 0.784; text 0.652 / 0.680. The description helps the target domain most.
* Curves: val loss bottoms out at epochs 2–3 while val F1 peaks at epochs 4–5. Mean best epoch is 4.8, and 13 of 15
  runs went 6 epochs, so the model is still overconfident. Label smoothing (idea 3) targets this.

## Queued next (one change each vs E1_pair)

1b longer description context (max_len 256) · 2 lr 1e-5 · 2 lr 3e-5 · then 3 label smoothing 0.1, class-weighted CE ·
4 bertweet-base, deberta-v3-base, (desktop only) roberta-large · 5 seed ensemble · if gains flatten: BiLSTM head /
dropout.
