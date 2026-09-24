"""Score off-the-shelf Hugging Face political-leaning classifiers against ours.

The HF models are used as-is (zero-shot). None of them trains on our labels, so each one
predicts all 3,675 comments and is scored per fold on the same video-grouped folds as
baseline.py and finetune.py. Compare only against runs from the same machine: the fold
assignment depends on the pandas / scikit-learn versions.
"""
import argparse, glob, re
from finetune import make_batches, pick_device, NCPU  # first: sets thread env vars before torch loads
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import data, baseline

# name -> (tokenizer repo or None, model output index -> our index: 0 left, 1 neutral, 2 right)
# Label orders checked against each model's config.json and card, not assumed.
MODELS = {
    "matous-volf/political-leaning-politics": ("launch/POLITICS", [0, 1, 2]),
    "bucketresearch/politicalBiasBERT": ("bert-base-cased", [0, 1, 2]),
    "m-newhauser/distilbert-political-tweets": (None, [2, 0]),  # 0 Republican, 1 Democrat; no neutral
}


@torch.no_grad()
def zero_shot(repo, tok_repo, mapping, texts, device):
    tok = AutoTokenizer.from_pretrained(tok_repo or repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).to(device).eval()
    out = np.empty(len(texts), dtype=int)
    for c, enc, _ in make_batches(texts, None, tok, False, None):
        enc = {k: v.to(device) for k, v in enc.items()}
        out[c] = model(**enc).logits.argmax(-1).cpu().numpy()
    return np.array(mapping)[out]


def summarize(d, pred):
    """macro-F1 per fold (mean, std), pooled accuracy, and accuracy on left/right comments only.
    The last column is the fair one for the binary tweet model, which cannot say neutral."""
    per_fold = [f1_score(d.y3[d.fold == k], pred[(d.fold == k).values], average="macro",
                         labels=[0, 1, 2], zero_division=0) for k in sorted(d.fold.unique())]
    lr = (d.y3 != 1).values
    return np.mean(per_fold), np.std(per_fold), (pred == d.y3.values).mean(), (pred[lr] == d.y3.values[lr]).mean()


def finetuned_from_logs(d, pattern="logs/s*.out"):
    """Our fine-tuned model's cross-validation scores, read from the Slurm logs. The logs hold
    per-fold scores only, not predictions, so the left/right-only column is not available."""
    runs = {}  # fold -> list of (macro-F1, accuracy) across seeds
    for path in glob.glob(pattern):
        m = re.search(r"folds \[(\d+)\], seed \d+\).*?macro-F1 ([\d.]+) \+/- [\d.]+\s+accuracy ([\d.]+)",
                      open(path).read(), re.S)
        if m:
            runs.setdefault(int(m[1]), []).append((float(m[2]), float(m[3])))
    if sorted(runs) != sorted(d.fold.unique()):
        return None
    f1 = [np.mean([r[0] for r in runs[k]]) for k in sorted(runs)]
    n = d.fold.value_counts()
    acc = sum(np.mean([r[1] for r in runs[k]]) * n[k] for k in runs) / len(d)
    seeds = min(len(v) for v in runs.values())
    return f"twitter-roberta fine-tuned ({seeds} seed{'s' * (seeds > 1)})", np.mean(f1), np.std(f1), acc, float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="auto")
    p.add_argument("--models", nargs="*", default=list(MODELS), choices=list(MODELS))
    p.add_argument("--full", action="store_true", help="also print each model's per-class report")
    p.add_argument("--download", action="store_true",
                   help="cache every model and tokenizer, then exit (run on a node with internet)")
    a = p.parse_args()

    if a.download:
        for name, (tok_repo, _) in MODELS.items():
            AutoTokenizer.from_pretrained(tok_repo or name)
            AutoModelForSequenceClassification.from_pretrained(name)
            print(f"cached {name}" + (f" + tokenizer {tok_repo}" if tok_repo else ""))
        return
    device = pick_device(a.device)
    torch.set_num_threads(NCPU)

    d = data.load()
    data.check_no_leak(d)
    rows = [("TF-IDF baseline (trained)",) + summarize(d, baseline.run(d))]
    ft = finetuned_from_logs(d)
    rows.append(ft) if ft else print("no complete logs/s*.out found -- run train.sbatch to include our model")

    for name in a.models:
        print(f"scoring {name} ...", flush=True)
        pred = zero_shot(name, *MODELS[name], d.comment_text.values, device)
        if a.full:
            data.report(d, pred, name)
        rows.append((f"{name} (zero-shot)",) + summarize(d, pred))

    print(f"\n{'model':<58} {'macro-F1':>15} {'accuracy':>9} {'L/R acc':>8}")
    for name, f1, sd, acc, lr in sorted(rows, key=lambda r: -r[1]):
        lr_s = "   n/a" if np.isnan(lr) else f"{lr:.3f}"
        print(f"{name:<58} {f1:.3f} +/- {sd:.3f} {acc:>9.3f} {lr_s:>8}")
    print("\nmacro-F1: mean over the 5 video-grouped folds.  L/R acc: accuracy on left/right comments only.")


if __name__ == "__main__":
    main()
