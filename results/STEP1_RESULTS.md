# Step 1 results (running log)

Locked protocol, near-dup v2 (nd2), dev CV only: 52 videos, 5 video-grouped folds, 3 seeds each. The frozen
holdout is untouched. Macro-F1 is the mean ± std over the 15 (fold × seed) runs; accuracy is the mean per-fold
accuracy. Noise floor: fold std ≈ 0.05. Every run is in `results/runs.csv`; curves are in `results/plots/`.

| # | experiment | change vs current best | macro-F1 | accuracy | seed-ensemble F1 / acc | verdict |
|---|---|---|---|---|---|---|
| — | E0_comment | reference: comment only | 0.607 ± 0.054 | 0.625 | 0.617 / 0.633 | reference |
| 1 | **E1_pair** | + video description as a sentence pair (max_len 160) | **0.642 ± 0.054** | **0.673** | **0.660 / 0.691** | **adopted (current best)** |
| 3 | E3_pair_ep20_pat3 | epochs 6 → 20, patience 2 → 3 (user request) | 0.643 ± 0.037 | 0.677 | 0.664 / 0.695 | **noise** (paired +0.001 ± 0.030, better 6/15) |
| 3d | E3d_pair_ep20_pat3_drop0.2 | E3 + encoder dropout 0.1 → 0.2 (user request) | 0.634 ± 0.048 | 0.664 | 0.654 / 0.682 | **noise, slightly worse** (paired −0.009 vs E1, −0.010 vs E3) |
| 1b | E1b_pair_len256 | max_len 160 → 256 (48.6% of pairs were truncated) | 0.649 ± 0.048 (6/15 runs) | 0.687 | — | **stopped early: noise** (paired +0.001 ± 0.046 vs E1 on the same 6 runs) |

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

## E1b: longer description context — stopped

Paired with E1 on the 6 completed (seed, fold) runs: +0.078, +0.006, −0.065, −0.010, +0.007, −0.008 → mean +0.001.
It was stopped there, not finished, for two reasons: the effect is indistinguishable from zero, and at 256 tokens
DirectML needs ~11 GB VRAM plus spill into system RAM, which slowed folds from ~6 min to 90+ min once other apps
were using memory. Reported here, as with every run; not pursued.

Infrastructure fixes made while running it (no effect on results): one process per (seed, fold), because DirectML
does not release a finished fold's memory; batches padded to multiples of 32 tokens (identical predictions, max logit
difference 7e-8).

## E3: train up to 20 epochs (patience 3)

No gain: paired +0.001 macro-F1 vs E1 (better in 6/15), accuracy +0.004. Best epochs ranged 2–13 (median 6), runs
stopped at 5–16 epochs. The mean curve shows memorization, not learning: train loss 0.004 at epoch 16, val loss rising
from 1.13 (epoch 3) to 3.25, val macro-F1 flat at ~0.60 from epoch 5 and declining after 9. Fold std narrowed
(0.054 → 0.037) and the seed ensemble reached 0.664 / 0.695, but that is not a reason to adopt it. E1 (6 epochs)
stays the current best; E3d tests whether more dropout changes the late-epoch picture.

## E3d: 20 epochs + dropout 0.2

Paired vs E1: −0.009 macro-F1 (better in 4/15); vs E3: −0.010 (6/15). Dropout did slow memorization — train loss 0.25
at epoch 9 (E3: 0.09), val loss rising later — but val macro-F1 peaks at the same ~0.60–0.61 (epochs 6–8). Conclusion
from E3 + E3d: **overfitting is not what limits the score.** The ceiling comes from ~1,900 training comments with
noisy GPT-4o labels; regularization delays memorization without adding signal. E1 stays the current best.

## Queued next (one change each vs E1_pair)

E3 up to 20 epochs, patience 3 (5/15 E1 runs peaked at the 6-epoch cap) · E3d same + dropout 0.2 · 2 lr 1e-5 · 2 lr 3e-5 · then 3 label smoothing 0.1, class-weighted CE ·
4 bertweet-base, deberta-v3-base, (desktop only) roberta-large · 5 seed ensemble · if gains flatten: BiLSTM head /
dropout.
