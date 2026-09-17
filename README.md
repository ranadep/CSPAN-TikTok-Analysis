# Political stance classification for C-SPAN TikTok comments - Ayush and Anubhav

Classifies a comment as **left / neutral / right**. Fine-tunes
[`cardiffnlp/twitter-roberta-base-2022-154m`](https://huggingface.co/cardiffnlp/twitter-roberta-base-2022-154m)
on CPU, with a TF-IDF baseline as the gate.

## The thing that will bite you

The source CSV has **84,246 rows but only 5,358 unique comments** — the same comments were
re-scraped across 64 timestamp snapshots. After dropping nulls and unlabeled rows there are
**3,675 unique labeled comments**. That is the real dataset size.

Two leaks follow from this, and `data.py` closes both:

| Split | Accuracy | |
|---|---|---|
| Row-level (naive `train_test_split`) | 0.882 | 91% of test rows are copies of training rows |
| Deduplicated, random folds | 0.641 | still leaks video topic |
| Deduplicated + grouped by `video_id` | 0.586 | honest — this is the number |

`check_no_leak()` asserts disjoint texts *and* disjoint videos on every fold. Keep it.

## Usage

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
# on the Linux cluster, install torch from the CPU index (see requirements.txt)

.venv/bin/python data.py          # sanity: 3675 comments, 65 videos, no leaks
.venv/bin/python baseline.py      # the gate, ~3 seconds
.venv/bin/python finetune.py --device mps    # local dev loop (Apple Silicon)
sbatch train.sbatch                          # cluster: 5 folds as a job array

echo "Trump 2024" | .venv/bin/python finetune.py --predict
```

## Design notes

- **Trains a 5-class head, collapses to 3 at argmax.** The 5-class distribution is far more
  balanced (1.55:1 vs 2.16:1), so it dissolves the imbalance for free. Collapse by argmax —
  summing the pro/anti probabilities measured worse.
- **`max_length=64`, dynamic length-grouped padding.** p95 is 29 words; the default 512 would
  be ~8x wasted compute. Dynamic padding on top of that is worth a further ~12% (measured on
  Apple Silicon: 2.00 vs 2.27 s/step — less than the 3-4x you'd naively expect, because short
  sequences don't amortize per-layer overhead).
- **No dataloader workers on CPU.** They steal cores from the GEMM threads. Same reason
  `bf16` and `gradient_checkpointing` are off.
- **Emoji matter.** 9% of comments are pure emoji (💙 is a strong left signal). RoBERTa's
  byte-level BPE round-trips them; DeBERTa-v3's SentencePiece vocab can emit `<unk>`.
  `finetune.py` asserts this at startup.
- **`group`, `likes`, `user_id` are not features.** They don't exist on newly scraped comments.

## Decision rule

Transformer must beat the baseline by **≥4 macro-F1 points on the same folds, averaged over
3 seeds**. Per-fold std is ±0.030, so anything smaller is noise. If it doesn't clear that,
ship `baseline.py` — it trains in 3 seconds with no torch dependency.
