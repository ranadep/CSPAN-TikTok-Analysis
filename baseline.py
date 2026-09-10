"""TF-IDF + logistic regression. The gate: if the transformer can't beat this by >=4 macro-F1
points on the same folds, ship this instead -- it trains in seconds with no torch dependency."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import data

# default sklearn token_pattern drops emoji and single chars; 9% of comments are pure emoji
TOKENS = r"(?u)\w+|[^\w\s]"


def run(d, ngram=(1, 2)):
    pred = np.empty(len(d), dtype=int)
    for k in sorted(d.fold.unique()):
        tr, te = d.fold != k, d.fold == k
        pipe = make_pipeline(
            TfidfVectorizer(ngram_range=ngram, min_df=2, token_pattern=TOKENS, sublinear_tf=True),
            LogisticRegression(class_weight="balanced", max_iter=2000),
        )
        pipe.fit(d.comment_text[tr], d.y3[tr])
        pred[te.values] = pipe.predict(d.comment_text[te])
    return pred


if __name__ == "__main__":
    d = data.load()
    data.check_no_leak(d)
    data.report(d, run(d), "TF-IDF + LogReg (grouped by video)")
