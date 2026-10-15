"""Locked evaluation protocol (Step 0 of HANDOFF_comment_classifier.md).

Built on data.load() -- the dedup step is unchanged. On top of it:

  * final holdout: 13 of the 65 videos (6 C-SPAN + 7 other, fixed seed), ids frozen in
    splits/final_holdout_videos.txt. Only `train.py --final-holdout` reads those rows' labels.
  * dev CV: 5 folds grouped by video over the remaining 52 videos.
  * inner validation: inside each CV training split, a grouped set of videos is held out for
    early stopping. The CV test fold is never used for stopping or selection-within-a-run.
  * near-duplicate guard: comments are normalized (lowercase, no punctuation/whitespace,
    repeated emoji collapsed). Any *training* comment whose normalized form appears in the
    val or test side is dropped from training, and val rows matching the test side are dropped
    from val (test/holdout rows are never altered); then
    check_no_near_dup() asserts disjointness. check_no_leak()-style exact text/video checks
    run on every split too.

    python protocol.py            # create (first time) / verify the holdout, print split stats
"""
import re, unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import data

ROOT = Path(__file__).parent
VIDEO_INFO = ROOT / "test_video_info.csv"   # raye22/tiktok_personalization data/test_video_info.csv
HOLDOUT_FILE = ROOT / "splits" / "final_holdout_videos.txt"
CSPAN = "cspanofficial"
HOLDOUT_SEED, HOLDOUT_CSPAN, HOLDOUT_OTHER = 0, 6, 7
CV_SEED, N_FOLDS, VAL_SPLITS = 0, 5, 6   # val = 1/6 of each CV training split's videos

_REPEAT = re.compile(r"(\W)\1+")  # runs of the same non-word char (emoji, !!!) -> one


def normalize(text):
    t = unicodedata.normalize("NFKC", str(text)).lower().replace("️", "")
    t = _REPEAT.sub(r"\1", t)
    t = "".join(ch for ch in t if not ch.isspace() and not unicodedata.category(ch).startswith("P"))
    return _REPEAT.sub(r"\1", t)  # again: removing spaces can create new runs ("💙 💙")


def load_all():
    """All 3,675 deduplicated comments with video context. `fold` from data.load is dropped."""
    d = data.load().drop(columns="fold")
    v = pd.read_csv(VIDEO_INFO, usecols=["video_id", "username", "video_description"])
    d = d.merge(v.drop_duplicates("video_id"), on="video_id", how="left", validate="m:1")
    assert d.username.notna().all(), "video without metadata"
    d["video_description"] = d.video_description.fillna("").astype(str)
    d["is_cspan"] = d.username.eq(CSPAN)
    d["norm"] = d.comment_text.map(normalize)
    return d


def make_holdout(d):
    """Run once. Fixed seed; picks videos by account only (no labels are looked at)."""
    vids = d.drop_duplicates("video_id")[["video_id", "is_cspan"]].sort_values("video_id")
    rng = np.random.default_rng(HOLDOUT_SEED)
    pick = np.concatenate([
        rng.choice(vids.video_id[vids.is_cspan].values, HOLDOUT_CSPAN, replace=False),
        rng.choice(vids.video_id[~vids.is_cspan].values, HOLDOUT_OTHER, replace=False)])
    HOLDOUT_FILE.parent.mkdir(exist_ok=True)
    HOLDOUT_FILE.write_text("\n".join(str(v) for v in sorted(pick)) + "\n")
    return set(int(v) for v in pick)


def holdout_videos():
    return {int(l) for l in HOLDOUT_FILE.read_text().split()}


def load_dev(n_folds=N_FOLDS, seed=CV_SEED):
    """The 52 dev videos with a `fold` column (video-grouped, stratified on y3)."""
    d = load_all()
    dev = d[~d.video_id.isin(holdout_videos())].reset_index(drop=True)
    dev["fold"] = -1
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for k, (_, te) in enumerate(sgkf.split(dev, dev.y3, groups=dev.video_id)):
        dev.loc[te, "fold"] = k
    assert (dev.fold >= 0).all()
    return dev


def load_holdout():
    d = load_all()
    return d[d.video_id.isin(holdout_videos())].reset_index(drop=True)


def check_disjoint(a, b, what):
    assert not (set(a.comment_text) & set(b.comment_text)), f"text leak {what}"
    assert not (set(a.video_id) & set(b.video_id)), f"video leak {what}"
    check_no_near_dup(a, b, what)


def check_no_near_dup(a, b, what=""):
    both = set(a.norm) & set(b.norm)
    assert not both, f"near-duplicate leak {what}: {len(both)} e.g. {sorted(both)[:3]}"


def purge(train, *others):
    """Drop training rows whose normalized text appears in any held-out side."""
    bad = set().union(*(set(o.norm) for o in others))
    keep = ~train.norm.isin(bad)
    return train[keep], int((~keep).sum())


def val_split(train, seed):
    """Grouped (by video), label-stratified validation split of a training set."""
    sgkf = StratifiedGroupKFold(n_splits=VAL_SPLITS, shuffle=True, random_state=seed)
    tr, va = next(sgkf.split(train, train.y3, groups=train.video_id))
    return train.iloc[tr], train.iloc[va]


def cv_split(dev, k, seed=CV_SEED):
    """(train, val, test) for CV fold k, near-duplicate purged and asserted leak-free.
    The val split depends only on (k, seed) -- identical across model seeds and configs."""
    test = dev[dev.fold == k]
    train, val = val_split(dev[dev.fold != k], seed=1000 + 10 * seed + k)
    val, _ = purge(val, test)          # val only steers early stopping; test is never altered
    train, n_purged = purge(train, val, test)
    check_disjoint(train, test, f"train/test fold {k}")
    check_disjoint(train, val, f"train/val fold {k}")
    check_disjoint(val, test, f"val/test fold {k}")
    return train, val, test, n_purged


def final_split(seed=CV_SEED):
    """(train, val, holdout) for the one final run: train on all dev videos minus a grouped
    val split for early stopping; score the frozen holdout."""
    dev, hold = load_dev(), load_holdout()
    train, val = val_split(dev, seed=2000 + seed)
    val, _ = purge(val, hold)
    train, n_purged = purge(train, val, hold)
    check_disjoint(train, hold, "train/holdout")
    check_disjoint(train, val, "train/val")
    check_disjoint(val, hold, "val/holdout")
    return train, val, hold, n_purged


def leave_cspan_out_split(scope="dev", seed=CV_SEED):
    """Train on non-C-SPAN videos, test on C-SPAN videos. scope="dev": the 52 dev videos only
    (30 -> 22). scope="all": all 65 (37 -> 28) -- touches holdout videos, so it is only
    allowed after the one final holdout run has been recorded."""
    if scope == "all":
        assert (ROOT / "results" / "FINAL_HOLDOUT_DONE").exists(), "scope=all would touch the holdout"
        d = load_all()
    else:
        d = load_dev()
    test = d[d.is_cspan]
    train, val = val_split(d[~d.is_cspan], seed=3000 + seed)
    val, _ = purge(val, test)
    train, n_purged = purge(train, val, test)
    check_disjoint(train, test, "train/test lcso")
    check_disjoint(train, val, "train/val lcso")
    check_disjoint(val, test, "val/test lcso")
    return train, val, test, n_purged


if __name__ == "__main__":
    d = load_all()
    if not HOLDOUT_FILE.exists():
        make_holdout(d)
        print(f"created {HOLDOUT_FILE}")
    hv = holdout_videos()
    assert len(hv) == HOLDOUT_CSPAN + HOLDOUT_OTHER and hv <= set(d.video_id)
    hold = d[d.video_id.isin(hv)]
    print(f"holdout: {len(hv)} videos ({hold.drop_duplicates('video_id').is_cspan.sum()} C-SPAN), "
          f"{len(hold)} comments ({len(hold) / len(d):.1%})")
    dev = load_dev()
    print(f"dev: {dev.video_id.nunique()} videos ({dev.drop_duplicates('video_id').is_cspan.sum()} C-SPAN), "
          f"{len(dev)} comments")
    print("dev labels:", dev.y3.map(data.LABELS3.__getitem__).value_counts().to_dict())
    for k in range(N_FOLDS):
        tr, va, te, n = cv_split(dev, k)
        print(f"fold {k}: train {len(tr)} ({tr.video_id.nunique()} vids, {n} near-dup purged)  "
              f"val {len(va)} ({va.video_id.nunique()} vids)  test {len(te)} ({te.video_id.nunique()} vids)")
    tr, va, ho, n = final_split()
    print(f"final: train {len(tr)} ({n} purged)  val {len(va)}  holdout {len(ho)}  -- splits leak-free")
    # how much near-duplication exists across videos at all (informational)
    g = d.groupby("norm").video_id.nunique()
    print(f"normalized texts on >1 video: {(g > 1).sum()} ({d.norm.isin(g[g > 1].index).sum()} comments)")
