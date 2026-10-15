# Model card: C-SPAN TikTok comment stance classifier

**Status: draft — final scores are filled in after the one holdout run.**

## What it does

Classifies a TikTok comment as **left / neutral / right**, given the comment and (in the chosen configuration) the
video's description. Trains a 5-class head (Pro-Democrat, Anti-Republican, Pro-Republican, Anti-Democrat, Neutral)
and collapses to 3 classes at argmax: left = Pro-Dem + Anti-Rep, right = Pro-Rep + Anti-Dem.

## Data

* Source: `comments_with_labels_for_all_three_group_anonymized.csv` and `data/test_video_info.csv` from
  [raye22/tiktok_personalization](https://github.com/raye22/tiktok_personalization) (Yan & Wu, ICWSM 2026).
* 84,246 rows → 5,358 unique comments (re-scraped by 22 audit accounts at many timestamps) → **3,675 unique labeled
  comments on 65 videos**; 28 videos (1,540 comments) are from @cspanofficial. Dedup is `data.load()`: one row per
  unique text, label and video by mode.
* Not used as features: `group`, `likes`, `user_id` (unavailable on newly scraped comments).

## Label provenance and its biases — read this before trusting any number

The labels are **GPT-4o output, not human judgments**. The authors report GPT-4o accuracy 0.84 / κ 0.79 against
human labels, so roughly 1 in 6 "gold" labels is wrong and **every score below measures agreement with GPT-4o, not
with people**. The labeling prompt (`Comment_label_prompt` upstream) tells the model to "boldly and decisively"
classify, to "assume that comments are likely to carry political intent", and maps 💙 → Pro-Democrat and
❤️ → Pro-Republican. Expected consequences, all visible in the error analysis: neutral is under-labeled; heart
emoji are shortcut features; comments whose stance depends on the video ("Exactly!") are labeled using the video
description, which is why the description is an input.

## Evaluation protocol

`protocol.py` (see `results/STEP0_REPORT.md`):

* Frozen **final holdout**: 13 videos (6 C-SPAN), 698 comments, `splits/final_holdout_videos.txt`, scored once.
* Model selection: 5-fold CV grouped by video on the other 52 videos; early stopping on a grouped validation split
  inside each training fold; the CV test fold is touched once per run.
* Near-duplicate guard: normalized text (lowercase, no punctuation/whitespace, repeated emoji collapsed) never
  appears on both sides of any split.
* Sanity: permuted-label training scores chance (TF-IDF 0.326, RoBERTa 0.315 macro-F1).
* Every run is in `results/runs.csv`; ~20 configurations total to limit selection overfitting on the CV folds.

## Scores

FINAL_SCORES_TABLE

## Limits

* Agreement with GPT-4o labels, which inherit the prompt's biases (above). Human-level stance accuracy is unknown.
* 65 videos from 8 accounts, Sept–Oct 2024 election coverage. Fold-to-fold std ≈ 0.05 macro-F1: differences
  smaller than that are noise.
* Comments on topics or accounts not in the data, or after the 2024 election, are out of distribution.
* Emoji-only comments score well largely because of the 💙/❤️ shortcut, which the labels themselves encode.
* Not for moderation or decisions about individuals.
