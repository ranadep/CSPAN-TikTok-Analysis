# Step 0 report: evaluation locked

Machine: desktop, i5-12600KF, RX 7700 XT via **DirectML** (`torch-directml`, torch 2.4.1, Python 3.11 venv on D:).
All numbers below come from `results/runs.csv` (`python analyze.py table`). Macro-F1 is 3-class (left/neutral/right),
mean ± std over every (fold × seed) run; accuracy is the mean of per-fold accuracies.

## 1. Reproductions (original protocol: `data.load()`, all 65 videos, 5 video-grouped folds)

| run | seeds | macro-F1 | accuracy | reported | verdict |
|---|---|---|---|---|---|
| TF-IDF + LogReg (`baseline.py`, unchanged) | 1 | 0.560 ± 0.030 | 0.572 | 0.555 ± 0.051, 0.571 | reproduced |
| twitter-roberta-base, `finetune.train_fold` unchanged | 3 | 0.626 ± 0.022 | 0.653 | 0.625 ± 0.048, 0.652 | reproduced |
| "~71% accuracy" candidate: roberta-large, folds 0–2 only | 1 | 0.681 (3 folds) | **0.706** (3 folds) | ~71% | consistent with ~71%; **different protocol** (§2) |

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

**Replay result (consistent with ~71%; no original artifact exists to compare against).** `finetune.train_fold` unchanged, roberta-large (`twitter-roberta-large-2022-154m`),
lr 1e-5, 3 epochs, folds 0–2, seed 0 — the first large config `tune.py` samples:

| run (original folds, seed 0) | fold 0 | fold 1 | fold 2 | folds 0–2 macro-F1 | folds 0–2 accuracy |
|---|---|---|---|---|---|
| base, default (lr 2e-5, 4 ep) | 0.632 / 0.655 | 0.646 / 0.655 | 0.640 / 0.699 | 0.639 | 0.670 |
| **large, lr 1e-5, 3 ep** | 0.680 / 0.704 | 0.681 / 0.691 | 0.681 / 0.724 | **0.681** | **0.706** |

(cells: macro-F1 / accuracy). The most likely source of the ~71% is therefore roberta-large on `tune.py`'s three selection folds with one seed.
The gap decomposes into a **suggestive model gain** (+0.04 macro-F1, +0.04 accuracy vs base on the identical folds and
seed; 3 folds × 1 seed, and below the 0.05 noise floor, so not established) and a **fold-subset effect** (base scores
0.670 accuracy on folds 0–2 vs 0.652 on all five). Caveat: fold assignment differs from the original environment
(there fold 1 was the hard one at 0.555 F1), so which folds are "easy" may not carry over exactly. It is not comparable to the
5-fold × 3-seed 0.652, and it was never checked on held-out folds 3–4 (`tune.py --final`). roberta-large needs ~6 GB
for weights + grads + Adam state; it OOMs on the 12 GB card under DirectML, so this replay used `OffloadAdamW`
(same update, moments on CPU; ~50 min per fold). It does not fit the laptop's 4 GB at all.

## 3. Locked protocol (`protocol.py`, frozen in commit 755393a)

* **Final holdout**: 13 of 65 videos (6 C-SPAN + 7 other; seed 0, chosen by account only, no labels looked at),
  698 comments (19.0%). IDs in `splits/final_holdout_videos.txt`. Among the new-protocol scripts only
  `train.py --final-holdout` scores it; it writes the committed `results/FINAL_HOLDOUT_DONE` marker *before* the first
  prediction, refuses any other run afterwards, and `analyze.py` refuses holdout analysis until the marker says
  COMPLETE. **Exception, by necessity:** the original-protocol reproductions in §1–2 (`baseline_orig`,
  `repro_finetune_default`, `repro_tune_large_*`) train and test on all 65 videos, holdout included, because that is
  the protocol being reproduced. Their predictions (`results/preds/repro_*`) are never used for error analysis,
  model selection or any Step 1 decision.
* **Dev CV**: 52 videos (22 C-SPAN), 2,977 comments, `StratifiedGroupKFold(5, seed 0)` by video.
* **Inner validation**: in each CV training split, 1/6 of the videos (7–8 videos, ~400 comments) are a grouped
  validation set for early stopping (patience 2 on val macro-F1, max 6 epochs). The CV test fold is touched once,
  after the best epoch is restored.
* **Near-duplicate guard (nd2)**: normalize (NFKC, lowercase, drop U+FE0F, strip whitespace and punctuation,
  collapse repeated emoji grapheme clusters and short repeated emoji patterns — skin tones, flags, `💙🇺🇸💙🇺🇸`;
  words are never collapsed). Training rows whose normalized text appears in val or test are dropped
  (86–103 per fold); val rows matching test are dropped from val; test rows are never altered. Then
  `check_disjoint` asserts exact-text, video and normalized-text disjointness for train/val, train/test, val/test
  (and train/val/holdout for the final run). Across all 65 videos, 87 normalized texts (310 comments) occur on
  more than one video — mostly single emoji ("💙", "🤡") and short slogans.
* Training rows per fold drop from ~2,940 (original) to ~1,880 because of the holdout and the val split, so
  locked-protocol scores start lower than the original-protocol ones.

## 4. Sanity checks

| check | macro-F1 | accuracy | chance ≈ 0.33 | verdict |
|---|---|---|---|---|
| TF-IDF, training labels permuted (3 seeds × 5 folds) | 0.326 ± 0.026 | 0.345 | yes | no leak |
| RoBERTa, training **and val** labels permuted (3 seeds × 5 folds) | 0.315 ± 0.045 | 0.352 | yes | no leak |

History, all kept in `runs.csv`: the first RoBERTa check (nd1, 1 seed) gave 0.279 ± 0.078 — 2 of 5 folds collapsed
to one class, the review rightly called it weak. A rerun that permuted only the training labels gave 0.335 ± 0.076
with folds up to 0.43 (`shuffled_roberta_valtrue`, stopped after 7 runs): early stopping picked the epoch with the best
*true-label* validation F1, which lets a little genuine signal back in. That is not a test leak, but it is not a clean
null either, so the final check permutes the validation labels too.

## 5. Starting point for Step 1 (locked protocol, dev CV)

| run | seeds | macro-F1 | accuracy | mean best epoch |
|---|---|---|---|---|
| TF-IDF + LogReg | 1 | 0.540 ± 0.045 | 0.550 | — |
| TF-IDF + LogReg, + description TF-IDF | 1 | 0.526 ± 0.051 | 0.550 | — |
| **E0** twitter-roberta-base, comment only (reference) | 3 | **0.607 ± 0.054** | 0.625 | 3.8 |
| E0 seed-ensemble (mean logits) | 3 | 0.617 ± 0.049 | 0.633 | — |

All rows are nd2. The nd1 versions (E0 0.602 ± 0.047) are within noise. **E1_pair_nd1** (comment + description,
0.642 ± 0.046, acc 0.675) was run before the review — a Step 1 run done early; it is being rerun under nd2.

E0 curves (`results/plots/curves_E0_comment.png`): val loss is lowest at epoch 2 and rises afterward while val
macro-F1 keeps creeping up to epoch 4–6: classic overconfidence. Early stopping is on F1, as specified.

Noise floor for "is this an improvement": fold std ≈ 0.05.

## Independent review

A separate review agent checked the protocol code, every number in this report against `runs.csv`, and the holdout
handling: **no leak found**, verdict "OK with minor issues". Fixes applied: report wording (71% is "consistent with",
holdout exception for the original-protocol reproductions), grapheme-level near-dup normalizer (nd2; all CV runs
redone), committed + early-written holdout marker with an `analyze.py` check, 3-seed shuffled check with permuted
val labels.

## Open items / caveats

* DirectML is not ROCm (the handoff's warning was about ROCm); it is an officially supported Windows backend,
  ~7× faster than this CPU. If any result looks off, rerun on CPU or the laptop's CUDA: the code paths are the same.
* Two Windows bugchecks occurred overnight (0x7E at 01:36 before any of this work started; 0x1A
  MEMORY_MANAGEMENT at 03:06 during training). Every script resumes from `results/runs.csv`; four
  pre-crash reproduction folds were rerun (their scores are preserved in `logs/repro_finetune_CRASHED_0306.log`).
