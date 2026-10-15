"""Log baseline.py (unchanged) per-fold scores on the original protocol to results/runs.csv."""
import sys; sys.path.insert(0, ".")
from sklearn.metrics import f1_score
import data, baseline
from exp_utils import log_run
d = data.load(); data.check_no_leak(d); pred = baseline.run(d)
for k in sorted(d.fold.unique()):
    m = (d.fold == k).values
    log_run(run_id=f"baseline_orig-s0-f{k}", experiment="baseline_orig", protocol="orig_65vid_5fold",
            config={"model": "tfidf_logreg"}, seed=0, fold=int(k), n_train=int((~m).sum()), n_test=int(m.sum()),
            macro_f1=round(f1_score(d.y3[m], pred[m], average="macro"), 4), accuracy=round((pred[m] == d.y3[m]).mean(), 4))
print("logged")
