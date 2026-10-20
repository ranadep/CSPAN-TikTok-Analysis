# Error audit: model mistakes vs label mistakes

40 random errors of the E1_pair seed-ensemble, pooled CV test folds (dev videos only, never the holdout;
`analyze.frame_for("E1_pair", True)`, `random_state=1`). Each was judged by reading the comment with its video
description: who is right, the GPT-4o label or the model?

| verdict | count | share |
|---|---|---|
| model wrong, GPT-4o label right | 27 | 68% |
| GPT-4o label wrong, model right | 7 | 18% |
| genuinely ambiguous | 6 | 15% |

Label errors found: "Women need to be protected from guys like Trump💙💙💙" → Pro-Republican; "Kamala Harris is a great
leader!" → Pro-Republican; a spam-bot comment → Pro-Democrat; "🥵🥵🥵" → Anti-Democrat; a defense of the Trump-rally
comedian → Anti-Republican; "of course she didn't show" (Harris skipping the Al Smith dinner) → Neutral; a critique of
the Trump camp's comedian → Anti-Democrat.

Clear model errors include "Trump is right as always. crooked incompetent joe gang is a disgrace" → left (on a video
critical of Trump), "Ok Little Marco" → right, "Finally he's got a job he's qualified for" (Trump at McDonald's) →
right, "W trump THE MEDIA IS THE VIRUS" → left, "Latinos for trump is the same thing as roaches for raid" → right.

## Reading

* Most errors (~2/3) are comments a person labels instantly but that need world knowledge (nicknames, who a
  comedian is) or sarcasm. Several suggest the pair model leans on the video's slant over the comment.
* Label noise is real but secondary: ~1 in 5 errors. With ~31% of comments misclassified, perfect labels would
  add at most ~5–6 points of accuracy (plus up to ~4 from the ambiguous cases).
* Other evidence. TF-IDF learning curve on the same folds, by fraction of training videos: 25% 0.455 → 50% 0.517 →
  75% 0.530 → 100% 0.540 macro-F1. More data still helps, with diminishing returns. Regularization (E3d) and longer
  training (E3) did not move the score. Re-scraped duplicates agree 99.2%, which says GPT-4o labeled once and the
  label was copied; it says nothing about noise.

A 40-error sample has wide error bars (±~13 points on each share).
