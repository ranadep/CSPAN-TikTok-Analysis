"""TF-IDF + LogReg on the locked protocol (dev 52 videos, protocol.cv_split folds).

Same training rows as train.py (purged CV training split; the val videos are unused here),
so the numbers are directly comparable. --shuffle-labels is the leakage sanity check.
--pair appends the video description (separate TF-IDF block) to mirror `train.py --input pair`.
"""
import argparse
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

import protocol as P
from baseline import TOKENS
from exp_utils import log_run


def fit_predict(tr, te, y, pair):
    vc = TfidfVectorizer(ngram_range=(1, 2), min_df=2, token_pattern=TOKENS, sublinear_tf=True)
    Xtr, Xte = vc.fit_transform(tr.comment_text), vc.transform(te.comment_text)
    if pair:
        vd = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=20000)
        Xtr = sp.hstack([Xtr, vd.fit_transform(tr.video_description)]).tocsr()
        Xte = sp.hstack([Xte, vd.transform(te.video_description)]).tocsr()
    clf = LogisticRegression(class_weight="balanced", max_iter=3000).fit(Xtr, y)
    return clf.predict(Xte)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shuffle-labels", action="store_true")
    p.add_argument("--pair", action="store_true")
    p.add_argument("--seeds", type=int, nargs="+", default=[0])
    a = p.parse_args()
    name = "tfidf_lr" + ("_pair" if a.pair else "") + ("_shuffled" if a.shuffle_labels else "")
    dev = P.load_dev()
    for seed in a.seeds:
        f1s, accs = [], []
        for k in range(P.N_FOLDS):
            tr, va, te, n = P.cv_split(dev, k)
            y = tr.y3.values
            if a.shuffle_labels:
                y = np.random.default_rng(10_000 + seed).permutation(y)
            pr = fit_predict(tr, te, y, a.pair)
            f1, acc = f1_score(te.y3, pr, average="macro"), (pr == te.y3.values).mean()
            f1s.append(f1); accs.append(acc)
            log_run(run_id=f"{name}-s{seed}-f{k}", experiment=name,
                    protocol="dev52_5fold" + ("_shuffled" if a.shuffle_labels else ""),
                    config={"model": "tfidf_logreg", "pair": a.pair}, seed=seed, fold=k,
                    n_train=len(tr), n_test=len(te), macro_f1=round(f1, 4), accuracy=round(acc, 4),
                    notes=f"near_dup_purged={n}")
        print(f"{name} seed {seed}: macro-F1 {np.mean(f1s):.3f} +/- {np.std(f1s):.3f}  "
              f"acc {np.mean(accs):.3f}   folds {' '.join(f'{f:.3f}' for f in f1s)}")


if __name__ == "__main__":
    main()
