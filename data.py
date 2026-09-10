"""Load the labeled comments, deduplicate, and assign leak-free CV folds.

The raw CSV has 84k rows but only 5,358 unique comments -- the same comments were
re-scraped across 64 timestamp snapshots. Splitting on rows puts copies of a comment
in both train and test (measured: 0.88 accuracy vs 0.64 honest). Two rules fix it:
dedup to unique text, and group folds by video_id so a held-out video is genuinely unseen.
"""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

CSV = "/Users/ayushagarwal/Downloads/Comments with Labels Anonymized.csv"

LABELS5 = ["Anti-Democrat", "Anti-Republican", "Neutral", "Pro-Democrat", "Pro-Republican"]
LABELS3 = ["left", "neutral", "right"]
COLLAPSE = [2, 0, 1, 0, 2]  # 5-class index -> 3-class index


def load(n_folds=5, seed=0):
    df = pd.read_csv(CSV).dropna(subset=["comment_text", "label"])
    mode = lambda s: s.mode().iat[0]  # ponytail: ties break alphabetically; only 49 texts disagree
    d = (df.groupby("comment_text", sort=False)
           .agg(label=("label", mode), video_id=("video_id", mode))
           .reset_index())
    d["y5"] = d.label.map(LABELS5.index)
    d["y3"] = d.y5.map(COLLAPSE.__getitem__)
    # emoji/punctuation-only comments -- reported as a separate slice, they behave differently
    d["emoji_only"] = ~d.comment_text.str.contains(r"\w", regex=True, na=False)

    d["fold"] = -1
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, test) in enumerate(splitter.split(d, d.y3, groups=d.video_id)):
        d.loc[test, "fold"] = k
    assert (d.fold >= 0).all()
    return d


def check_no_leak(d):
    """The whole plan rests on this. Run it every fold."""
    for k in sorted(d.fold.unique()):
        tr, te = d[d.fold != k], d[d.fold == k]
        assert not (set(tr.comment_text) & set(te.comment_text)), f"text leak in fold {k}"
        assert not (set(tr.video_id) & set(te.video_id)), f"video leak in fold {k}"


if __name__ == "__main__":
    d = load()
    check_no_leak(d)
    print(f"{len(d)} unique labeled comments, {d.video_id.nunique()} videos, no leaks")
    print(d.y3.map(LABELS3.__getitem__).value_counts().to_string())
    print(f"emoji-only: {d.emoji_only.sum()}")
    print(d.groupby("fold").size().to_string())


def report(d, pred3, name):
    """Shared scoring so baseline and transformer are comparable. Macro-F1 is the headline:
    the majority class is 45%, so accuracy flatters a model that ignores Neutral."""
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    import numpy as np

    per_fold = [f1_score(d.y3[d.fold == k], pred3[d.fold == k], average="macro")
                for k in sorted(d.fold.unique())]
    print(f"\n=== {name} ===")
    print("fold macro-F1:", " ".join(f"{f:.3f}" for f in per_fold))
    print(f"macro-F1 {np.mean(per_fold):.3f} +/- {np.std(per_fold):.3f}   "
          f"accuracy {(pred3 == d.y3).mean():.3f}")
    print(classification_report(d.y3, pred3, target_names=LABELS3, digits=3, zero_division=0))
    print("confusion (rows=true):")
    print(pd.DataFrame(confusion_matrix(d.y3, pred3), index=LABELS3, columns=LABELS3).to_string())
    for slice_name, m in [("emoji-only", d.emoji_only), ("has text", ~d.emoji_only)]:
        print(f"{slice_name:>10} (n={m.sum():4d}): "
              f"macro-F1 {f1_score(d.y3[m], pred3[m], average='macro', zero_division=0):.3f}")
    return float(np.mean(per_fold))
