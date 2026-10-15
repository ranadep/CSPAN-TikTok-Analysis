# Handoff: improve the C-SPAN TikTok comment stance classifier

Put this file in the repo root, then start Claude Code there with the kickoff prompt at the bottom.

## Goal

Get the highest honest score on political stance classification (left / neutral / right) of TikTok comments. "Honest" means the score holds on videos the model never saw. A higher number produced by leakage, test-set tuning, or a cherry-picked seed is a failure, not a win. Primary metric: 3-class macro-F1, with accuracy also reported.

## Context (read before touching code)

* Repo: `data.py`, `baseline.py`, `finetune.py`, `train.sbatch`, `requirements.txt`, README. Read them all first.
* Data source: the labeled CSV is `comments_with_labels_for_all_three_group_anonymized.csv` from [raye22/tiktok_personalization](https://github.com/raye22/tiktok_personalization) (Yan & Wu, ICWSM 2026). It contains 84,246 rows, which shrink to 5,358 unique comments (the same comments were re-scraped by 22 audit accounts at many timestamps), and then to 3,675 unique labeled comments across 65 videos. 28 of the 65 videos are from @cspanofficial, our target domain. Video metadata, including `video_description` and `username`, is in that repo's `data/test_video_info.csv`.
* Labels: GPT-4o produced them, not humans, using 5 classes: Pro-Democrat, Anti-Republican, Pro-Republican, Anti-Democrat, Neutral. Collapse to 3 as left = Pro-Dem + Anti-Rep, right = Pro-Rep + Anti-Dem, neutral. The model trains a 5-class head and collapses at argmax. The authors report GPT-4o accuracy 0.84 and κ 0.79 against human labels, so the labels are noisy and the realistic ceiling is somewhere below that. Known prompt biases: it says to assume political intent and to be "bold and decisive," and it maps 💙 → Pro-Democrat and ❤️ → Pro-Republican. Expect neutral to be under-labeled and emoji to be shortcut features.
* Results so far:
  * TF-IDF + logistic regression baseline: macro-F1 0.555.
  * RoBERTa (`cardiffnlp/twitter-roberta-base-2022-154m`): macro-F1 0.625 ± 0.048, accuracy 0.652 (5-fold grouped by video, 3 seeds).
  * A later run reportedly reached about 71% accuracy. This is unverified. Reproduce it before building on it.
* Leakage history: a naive row-level split gave 0.882 accuracy; deduplicating plus random folds gave 0.641; deduplicating plus grouping by video gave 0.586. `check_no_leak()` in `data.py` asserts disjoint texts and disjoint videos per fold. Never remove it.

## Hardware (local, Windows)

* Laptop: RTX 3050 Ti, 4 GB VRAM. Use CUDA with fp16. Choose max length from the data's 95th-percentile token length (comments are short, so 64–128 is likely enough). Use batch size 16 or less, with gradient accumulation if needed.
* Desktop: RX 7700 XT. ROCm on Windows isn't a reliable path for this card, so don't spend time on it. Use CPU there, or use the laptop.
* Pick the device automatically (cuda → mps → cpu), and keep `--device` overridable. Keep `train.sbatch` working so the same code still runs on the cluster.

## Step 0: lock the evaluation before any tuning

1. Reproduce the baseline and the "71%" run under the exact protocol above (same folds, same dedup, same 3-class collapse). Record both accuracy and macro-F1. If the 71% doesn't reproduce, or came from a protocol change, write down exactly what changed, and don't move on until it's clear.
2. Carve out a final holdout: with a fixed seed, pick about 20% of videos (13 videos, including roughly 6 C-SPAN), save their IDs to `splits/final_holdout_videos.txt`, and commit that file. Nothing touches the holdout until the very last run. No tuning, no early stopping, no peeking at errors.
3. Model selection uses 5-fold cross-validation grouped by video on the remaining ~52 videos. Inside each training fold, set aside a grouped validation split of videos for early stopping. Never early-stop on the CV test fold.
4. Add a near-duplicate check on top of `check_no_leak()`. Normalize text (lowercase, strip punctuation and whitespace, collapse repeated emoji), then assert that no normalized comment appears on both sides of any split. Copy-pasted comments across videos are common.
5. Shuffled-label sanity check: train once with permuted labels. Macro-F1 must fall to chance (about 0.33). If it doesn't, something is leaking.

## Step 1: experiments, in this order (stop when gains flatten)

Run each idea as one change against the current best. Use 3 seeds and report mean ± std across folds and seeds.

1. Add video context: encode the comment and the video description as a sentence pair (`[comment] </s> [description]`). The labels were made with the description, and comments like "Exactly!" can't be labeled without it. This is likely the biggest win.
2. Training hyperparameters: a small grid of learning rate {1e-5, 2e-5, 3e-5}, epochs up to 6 with early stopping (patience 2) on validation macro-F1, warmup 6–10%, weight decay 0.01. Keep the grid small.
3. Noisy-label robustness: label smoothing (0.1), and either class-weighted cross-entropy or focal loss for the neutral class.
4. Backbones (if time allows): `vinai/bertweet-base` and `microsoft/deberta-v3-base`. A large model won't fit in 4 GB. Don't try it locally.
5. Seed ensemble: average the logits of the 3 seeds as a final option.

Overall budget: about 20 configurations. More than that and you start overfitting to the CV folds through selection.

## Overfitting and honesty rules

* Log every run (config, git hash, seed, per-fold scores, train loss vs. validation loss) to `results/runs.csv`. Report all runs, not just the best one.
* Plot train vs. validation loss and macro-F1 per epoch. A widening gap means stopping earlier or regularizing more.
* If a change gains less than the fold standard deviation (about 0.05), treat it as noise, not an improvement.
* Never: split by rows, drop or merge the neutral class to raise the score, change the dedup step, tune on the holdout, report the best seed alone, or report accuracy without macro-F1.
* Run the holdout once, at the end, with the chosen config. If the holdout score is far below CV, report it as is. Don't go back and tune.

## Required outputs

1. A results table covering the baseline, the reproduced "71%" run, every experiment (macro-F1 mean ± std, accuracy), and the final holdout score.
2. Per-class precision, recall, and F1 plus a confusion matrix, especially for neutral.
3. Breakdowns: C-SPAN vs. non-C-SPAN videos, emoji-only vs. text comments, and a leave-C-SPAN-out run (train on the 37 non-C-SPAN videos, test on the 28 C-SPAN ones).
4. 30 example errors, with the comment, video description, true label, and predicted label.
5. The best checkpoint plus a short `MODEL_CARD.md`: data source, label provenance and its biases, protocol, scores, and limits. In particular, the scores measure agreement with GPT-4o labels, not human ground truth.

## Guardrails

* Don't commit data files, checkpoints, or any API keys. Add them to `.gitignore`.
* Ask before pushing to the remote or rewriting shared scripts that teammates use.
* Keep changes as flags or new files, so the original pipeline still reproduces its old numbers.

## Kickoff prompt for Claude Code

Read `HANDOFF_comment_classifier.md` and every script in this repo. Then do Step 0 only: reproduce the baseline and the ~71% run, create the frozen final holdout, and add the near-duplicate and shuffled-label checks. Report back with a short summary and the results table before starting any Step 1 experiments.
