# Step 0 report: evaluation locked

Machine: desktop, i5-12600KF, RX 7700 XT via **DirectML** (`torch-directml`, torch 2.4.1, Python 3.11 venv on D:).
All numbers below come from `results/runs.csv` (`python analyze.py table`). Macro-F1 is 3-class (left/neutral/right),
mean ± std over every (fold × seed) run; accuracy is the mean of per-fold accuracies.

## 1. Reproductions (original protocol: `data.load()`, all 65 videos, 5 video-grouped folds)

| run | seeds | macro-F1 | accuracy | reported | verdict |
|---|---|---|---|---|---|
| TF-IDF + LogReg (`baseline.py`, unchanged) | 1 | 0.560 ± 0.030 | 0.572 | 0.555 ± 0.051, 0.571 | reproduced |
| twitter-roberta-base, `finetune.train_fold` unchanged | 3 | 0.626 ± 0.022 | 0.653 | 0.625 ± 0.048, 0.652 | reproduced |
| "~71% accuracy" run: roberta-large, folds 0–2 only | 1 | 0.681 (3 folds) | **0.706** (3 folds) | ~71% | reproduced; **different protocol** (§2) |

Fold sizes here (719/742/743/766/705) differ from the reported per-fold F1s, so the fold *assignment* differs
slightly (pandas / scikit-learn versions; `compare.py` already warns about this). Pooled scores match within noise.
The only code differences for the RoBERTa reproduction: model weights loaded from a local safetensors copy
(torch 2.4 cannot `torch.load` the `.bin`), and on DirectML `torch.optim.AdamW` is replaced by `SimpleAdamW`,
which is the same update rule (verified equal to torch AdamW to 4e-8 on CPU).

## 2. Where the "~71%" comes from

There is no 71% anywhere in the repo or its git history; `results/tuning.csv` (which `tune.py` writes) was never
committed. The candidates, in order of likelihood:

1. **`tune.py` selection stage.** It scores 12 configs on **folds 0–2 only, seed 0**, and 7 of the 12 sampled configs
   are **roberta-large**. The leaderboard's top line is a best-of-12 on 3 folds with 1 seed: selection-biased and
   not comparable to the 5-fold × 3-seed number. Replay: `repro_tune_large_lr1e-5_ep3` (see table below).
2. **A single easy fold.** Under the exact default protocol, fold 2 alone scores 0.699 / 0.704 / 0.707 accuracy
   across the three seeds; folds 0–2 average 0.667–0.679. A run that reported fold 2, or the best of
   several configs on folds 0–2, lands at ~0.70–0.71.

What changed relative to the honest protocol in either case: subset of folds (3 of 5, including the easy fold 2),
one seed, and (case 1) max-over-configs on the same folds that are reported. Not a better model.

**Replay result (reproduced).** `finetune.train_fold` unchanged, roberta-large (`twitter-roberta-large-2022-154m`),
lr 1e-5, 3 epochs, folds 0–2, seed 0 — the first large config `tune.py` samples:

| run (original folds, seed 0) | fold 0 | fold 1 | fold 2 | folds 0–2 macro-F1 | folds 0–2 accuracy |
|---|---|---|---|---|---|
| base, default (lr 2e-5, 4 ep) | 0.632 / 0.655 | 0.646 / 0.655 | 0.640 / 0.699 | 0.639 | 0.670 |
| **large, lr 1e-5, 3 ep** | 0.680 / 0.704 | 0.681 / 0.691 | 0.681 / 0.724 | **0.681** | **0.706** |

(cells: macro-F1 / accuracy). So the ~71% is roberta-large on `tune.py`'s three selection folds with one seed.
It splits into a **real model gain** (+0.04 macro-F1, +0.04 accuracy vs base on the identical folds and seed) and a
**fold-subset effect** (base scores 0.670 accuracy on folds 0–2 vs 0.652 on all five). It is not comparable to the
5-fold × 3-seed 0.652, and it was never checked on held-out folds 3–4 (`tune.py --final`). roberta-large needs ~6 GB
for weights + grads + Adam state; it OOMs on the 12 GB card under DirectML, so this replay used `OffloadAdamW`
(same update, moments on CPU; ~50 min per fold). It does not fit the laptop's 4 GB at all.

## 3. Locked protocol (`protocol.py`, frozen in commit 755393a)

* **Final holdout**: 13 of 65 videos (6 C-SPAN + 7 other; seed 0, chosen by account only, no labels looked at),
  698 comments (19.0%). IDs in `splits/final_holdout_videos.txt`. Only `train.py --final-holdout` scores it, once;
  it writes `results/FINAL_HOLDOUT_DONE` and refuses to run again.
* **Dev CV**: 52 videos (22 C-SPAN), 2,977 comments, `StratifiedGroupKFold(5, seed 0)` by video.
* **Inner validation**: in each CV training split, 1/6 of the videos (7–8 videos, ~400 comments) are a grouped
  validation set for early stopping (patience 2 on val macro-F1, max 6 epochs). The CV test fold is touched once,
  after the best epoch is restored.
* **Near-duplicate guard**: normalize (NFKC, lowercase, drop U+FE0F, collapse runs of the same non-word char,
  strip whitespace and punctuation). Training rows whose normalized text appears in val or test are dropped
  (75–90 per fold); val rows matching test are dropped from val; test rows are never altered. Then
  `check_disjoint` asserts exact-text, video and normalized-text disjointness for train/val, train/test, val/test
  (and train/val/holdout for the final run). Across all 65 videos, 74 normalized texts (266 comments) occur on
  more than one video — mostly single emoji ("💙", "🤡") and short slogans.
* Training rows per fold drop from ~2,940 (original) to ~1,880 because of the holdout and the val split, so
  locked-protocol scores start lower than the original-protocol ones.

## 4. Sanity checks

| check | macro-F1 | accuracy | chance ≈ 0.33 | verdict |
|---|---|---|---|---|
| TF-IDF, training labels permuted (3 seeds × 5 folds) | 0.326 ± 0.031 | 0.347 | yes | no leak |
| RoBERTa, training labels permuted (1 seed × 5 folds) | 0.279 ± 0.078 | 0.388 | below (collapses to one class) | no leak |

## 5. Starting point for Step 1 (locked protocol, dev CV)

| run | seeds | macro-F1 | accuracy | mean best epoch |
|---|---|---|---|---|
| TF-IDF + LogReg | 1 | 0.540 ± 0.046 | 0.550 | — |
| TF-IDF + LogReg, + description TF-IDF | 1 | 0.529 ± 0.050 | 0.551 | — |
| **E0** twitter-roberta-base, comment only (reference) | 3 | **0.602 ± 0.047** | 0.624 | 3.9 |
| E0 seed-ensemble (mean logits) | 3 | 0.616 ± 0.037 | 0.638 | — |

E0 curves (`results/plots/curves_E0_comment.png`): val loss is lowest at epoch 2 and rises afterward while val
macro-F1 keeps creeping up to epoch 4–6: classic overconfidence. Early stopping is on F1, as specified.

Noise floor for "is this an improvement": fold std ≈ 0.047.

## Open items / caveats

* DirectML is not ROCm (the handoff's warning was about ROCm); it is an officially supported Windows backend,
  ~7× faster than this CPU. If any result looks off, rerun on CPU or the laptop's CUDA: the code paths are the same.
* Two Windows bugchecks occurred overnight (0x7E at 01:36 before any of this work started; 0x1A
  MEMORY_MANAGEMENT at 03:06 during training). Every script resumes from `results/runs.csv`; four
  pre-crash reproduction folds were rerun (their scores are preserved in `logs/repro_finetune_CRASHED_0306.log`).
