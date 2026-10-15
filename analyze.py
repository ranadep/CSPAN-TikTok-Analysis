"""Turn results/runs.csv + results/preds/ into the tables and plots the handoff asks for.

    python analyze.py table                     # every experiment, mean +/- std over folds x seeds
    python analyze.py detail E1_pair [--ensemble]   # per-class P/R/F1, confusion, slices
    python analyze.py curves E1_pair            # train vs val loss / val F1 per epoch (png)
    python analyze.py errors E1_pair --n 30     # sampled CV-test errors with video description
    python analyze.py ensemble E1_pair          # seed-ensemble (mean logits) CV scores

All CV numbers are on the dev videos only; the frozen holdout appears only in rows with
protocol=final_holdout.
"""
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import data
import protocol as P

ROOT = Path(__file__).parent
RES = ROOT / "results"


def runs():
    return pd.read_csv(RES / "runs.csv")


def table(md=True):
    r = runs()
    g = r.groupby(["experiment", "protocol"], sort=False)
    t = pd.DataFrame({
        "runs": g.size(),
        "seeds": g.seed.nunique(),
        "folds": g.fold.nunique(),
        "macro_f1": g.macro_f1.mean(),
        "f1_std": g.macro_f1.std(ddof=0),
        "accuracy": g.accuracy.mean(),
        "acc_std": g.accuracy.std(ddof=0),
        "best_epoch": g.best_epoch.mean(),
    }).reset_index()
    out = t.round(3)
    if md:
        print(out.to_markdown(index=False))
    return t


def load_preds(exp, seeds=None, holdout=False):
    """{seed: (logits, idx, y3)} concatenated over folds (or holdout)."""
    d = RES / "preds" / exp
    files = sorted(d.glob("s*_holdout.npz" if holdout else "s*_f*.npz"))
    by = {}
    for f in files:
        s = int(f.stem.split("_")[0][1:])
        if seeds is not None and s not in seeds:
            continue
        z = np.load(f)
        fold = -1 if holdout else int(f.stem.split("_f")[1])
        by.setdefault(s, []).append((z["logits"], z["idx"], z["y3"], np.full(len(z["idx"]), fold)))
    return {s: tuple(np.concatenate(x) for x in zip(*v)) for s, v in by.items()}


def ensemble(exp, holdout=False, quiet=False):
    """Mean logits over seeds, aligned by row index. Returns (frame with pred, per-fold F1s)."""
    P_ = load_preds(exp, holdout=holdout)
    seeds = sorted(P_)
    base = pd.DataFrame({"idx": P_[seeds[0]][1], "y3": P_[seeds[0]][2], "fold": P_[seeds[0]][3]}).set_index("idx")
    acc = np.zeros((len(base), 5))
    for s in seeds:
        lg, idx = P_[s][0], P_[s][1]
        acc += pd.DataFrame(lg, index=idx).loc[base.index].values
    # softmax-average would be similar; logit averaging is what the handoff specifies
    base["pred"] = np.array(data.COLLAPSE)[acc.argmax(1)]
    f1s = [f1_score(g.y3, g.pred, average="macro") for _, g in base.groupby("fold")]
    if not quiet:
        print(f"{exp} seed-ensemble ({len(seeds)} seeds): macro-F1 {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}  "
              f"acc(mean over folds) {np.mean([(g.y3 == g.pred).mean() for _, g in base.groupby('fold')]):.3f}  "
              f"folds {' '.join(f'{f:.3f}' for f in f1s)}")
    return base, f1s


def frame_for(exp, use_ensemble, holdout=False, seed=0):
    ref = P.load_holdout() if holdout else P.load_dev()
    if use_ensemble:
        b, _ = ensemble(exp, holdout=holdout, quiet=True)
    else:
        lg, idx, y3, fold = load_preds(exp, seeds=[seed], holdout=holdout)[seed]
        b = pd.DataFrame({"idx": idx, "y3": y3, "fold": fold, "pred": np.array(data.COLLAPSE)[lg.argmax(1)]}).set_index("idx")
    m = ref.join(b[["pred", "fold"]].rename(columns={"fold": "cv_fold"}), how="inner")
    assert (m.y3.values == b.loc[m.index, "y3"].values).all(), "prediction rows misaligned"
    return m


def detail(exp, use_ensemble, holdout=False):
    m = frame_for(exp, use_ensemble, holdout)
    y, pr = m.y3.values, m.pred.values
    print(f"\n### {exp} ({'seed-ensemble' if use_ensemble else 'seed 0'}; {'HOLDOUT' if holdout else 'pooled CV test folds'}, n={len(m)})")
    print("```")
    print(classification_report(y, pr, target_names=data.LABELS3, digits=3, zero_division=0))
    print("confusion (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y, pr, labels=[0, 1, 2]), index=data.LABELS3, columns=data.LABELS3).to_string())
    print("```")
    rows = []
    for name, mask in [("C-SPAN videos", m.is_cspan), ("non-C-SPAN videos", ~m.is_cspan),
                       ("emoji-only comments", m.emoji_only), ("text comments", ~m.emoji_only)]:
        mm = mask.values
        rows.append((name, int(mm.sum()), f1_score(y[mm], pr[mm], average="macro", labels=[0, 1, 2], zero_division=0),
                     (y[mm] == pr[mm]).mean()))
    print(pd.DataFrame(rows, columns=["slice", "n", "macro_f1", "accuracy"]).round(3).to_markdown(index=False))
    return m


def curves(exp):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = pd.read_csv(RES / "curves.csv")
    c = c[c.experiment == exp]
    g = c.groupby("epoch").agg(train_loss=("train_loss", "mean"), val_loss=("val_loss", "mean"),
                               val_f1=("val_macro_f1", "mean"), n=("val_loss", "size"))
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for (s, k), h in c.groupby(["seed", "fold"]):
        ax[0].plot(h.epoch, h.train_loss, color="tab:blue", alpha=.15)
        ax[0].plot(h.epoch, h.val_loss, color="tab:orange", alpha=.15)
        ax[1].plot(h.epoch, h.val_macro_f1, color="tab:green", alpha=.15)
    ax[0].plot(g.index, g.train_loss, color="tab:blue", lw=2, label="train loss")
    ax[0].plot(g.index, g.val_loss, color="tab:orange", lw=2, label="val loss")
    ax[1].plot(g.index, g.val_f1, color="tab:green", lw=2, label="val macro-F1")
    for a_ in ax:
        a_.set_xlabel("epoch"); a_.legend(); a_.grid(alpha=.3)
    ax[0].set_title(f"{exp}: loss (thin = individual runs)")
    ax[1].set_title(f"{exp}: validation macro-F1")
    out = RES / "plots" / f"curves_{exp}.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print(g.round(3).to_markdown())
    print(f"saved {out}")


def errors(exp, n, use_ensemble, holdout=False, seed=0):
    m = frame_for(exp, use_ensemble, holdout)
    e = m[m.y3 != m.pred]
    e = e.sample(min(n, len(e)), random_state=seed)
    out = e.assign(true=e.y3.map(data.LABELS3.__getitem__), predicted=e.pred.map(data.LABELS3.__getitem__),
                   label5=e.label, account=e.username,
                   description=e.video_description.str.slice(0, 160).str.replace("\n", " "))
    cols = ["comment_text", "true", "label5", "predicted", "account", "description"]
    out = out[cols]
    path = RES / f"errors_{exp}{'_holdout' if holdout else ''}.csv"
    out.to_csv(path, index=False, encoding="utf-8")
    print(f"{len(e)} sampled of {int((m.y3 != m.pred).sum())} errors -> {path}")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["table", "detail", "curves", "errors", "ensemble"])
    p.add_argument("exp", nargs="?")
    p.add_argument("--ensemble", action="store_true")
    p.add_argument("--holdout", action="store_true")
    p.add_argument("--n", type=int, default=30)
    a = p.parse_args()
    if a.cmd == "table":
        table()
    elif a.cmd == "detail":
        detail(a.exp, a.ensemble, a.holdout)
    elif a.cmd == "curves":
        curves(a.exp)
    elif a.cmd == "errors":
        print(errors(a.exp, a.n, a.ensemble, a.holdout).to_markdown(index=False))
    elif a.cmd == "ensemble":
        ensemble(a.exp, holdout=a.holdout)


if __name__ == "__main__":
    main()
