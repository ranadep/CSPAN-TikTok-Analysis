"""Hyperparameter search for finetune.py, with an honest final estimate.

Selection: each config trains on 4 folds and is scored on one of folds 0-2 (3 runs per config).
Final:     the winner and the current default are compared on folds 3-4, which selection never saw.
Why split: the best of N configs scored on the same folds you report is biased upward -- part
of "best" is luck. Results append to results/tuning.csv, so a stopped search resumes where it left off.
"""
import argparse, csv, os, random
import finetune as F  # first: sets thread env vars before torch loads
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
import data

SPACE = {
    "model_name": [F.MODEL, "cardiffnlp/twitter-roberta-large-2022-154m"],
    "lr": [1e-5, 2e-5, 3e-5, 5e-5],
    "epochs": [3, 4, 6],
    "label_smoothing": [0.0, 0.1],
}
DEFAULT = {"model_name": F.MODEL, "lr": F.LR, "epochs": F.EPOCHS, "label_smoothing": 0.0}
SELECT, FINAL = [0, 1, 2], [3, 4]
OUT = "results/tuning.csv"
FIELDS = ["stage", "config", "fold", "seed", "macro_f1", "accuracy"]


def key(cfg):
    return f"{cfg['model_name'].split('/')[-1]}|lr={cfg['lr']}|ep={cfg['epochs']}|ls={cfg['label_smoothing']}"


def parse(k):
    name, lr, ep, ls = k.split("|")
    return {"model_name": f"cardiffnlp/{name}", "lr": float(lr[3:]), "epochs": int(ep[3:]),
            "label_smoothing": float(ls[3:])}


def done():
    if not os.path.exists(OUT):
        return []
    with open(OUT) as f:
        return list(csv.DictReader(f))


def run(d, cfg, fold, seed, stage, device, toks):
    if any(r["stage"] == stage and r["config"] == key(cfg) and int(r["fold"]) == fold and int(r["seed"]) == seed
           for r in done()):
        return  # already in the CSV -- resume
    tok = toks.setdefault(cfg["model_name"], AutoTokenizer.from_pretrained(cfg["model_name"]))
    print(f"[{stage}] {key(cfg)} fold {fold} seed {seed}", flush=True)
    _, pred, te = F.train_fold(d, fold, tok, device, seed, **cfg)
    y = d.y3.values[te]
    row = {"stage": stage, "config": key(cfg), "fold": fold, "seed": seed,
           "macro_f1": round(f1_score(y, pred, average="macro"), 4), "accuracy": round((pred == y).mean(), 4)}
    new = not os.path.exists(OUT)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"    macro-F1 {row['macro_f1']:.3f}  accuracy {row['accuracy']:.3f}", flush=True)


def leaderboard(stage, folds):
    by = {}
    for r in done():
        if r["stage"] == stage:
            by.setdefault(r["config"], []).append(r)
    rows = [(k, np.mean([float(r["macro_f1"]) for r in v]), np.mean([float(r["accuracy"]) for r in v]), len(v))
            for k, v in by.items() if {int(r["fold"]) for r in v} >= set(folds)]  # complete configs only
    return sorted(rows, key=lambda r: -r[1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=12, help="configs to try, the default always included")
    p.add_argument("--final", action="store_true", help="compare the winner vs the default on folds 3-4")
    p.add_argument("--seeds", type=int, default=3, help="seeds per fold in the final comparison")
    p.add_argument("--device", default="auto")
    a = p.parse_args()

    device = F.pick_device(a.device)
    torch.set_num_threads(F.NCPU)
    d = data.load()
    data.check_no_leak(d)
    toks = {}

    if not a.final:
        grid = [dict(zip(SPACE, v)) for v in __import__("itertools").product(*SPACE.values())]
        trials = [DEFAULT] + [c for c in random.Random(0).sample(grid, len(grid)) if c != DEFAULT][:a.trials - 1]
        for cfg in trials:
            for fold in SELECT:
                run(d, cfg, fold, 0, "select", device, toks)
        print(f"\n{'config':<62} {'macro-F1':>9} {'accuracy':>9}   (folds {SELECT}, seed 0)")
        for k, f1, acc, _ in leaderboard("select", SELECT):
            print(f"{k:<62} {f1:>9.3f} {acc:>9.3f}" + ("   <- default" if k == key(DEFAULT) else ""))
        return

    board = leaderboard("select", SELECT)
    assert board, "no complete selection results -- run without --final first"
    best = parse(board[0][0])
    pairs = [best] if best == DEFAULT else [best, DEFAULT]
    for cfg in pairs:
        for fold in FINAL:
            for seed in range(a.seeds):
                run(d, cfg, fold, seed, "final", device, toks)
    print(f"\nHeld-out comparison on folds {FINAL} ({a.seeds} seeds) -- these folds played no part in choosing:")
    for k, f1, acc, n in leaderboard("final", FINAL):
        tag = "winner" if k == key(best) else "default"
        print(f"  {tag:<8} {k:<62} macro-F1 {f1:.3f}  accuracy {acc:.3f}  ({n} runs)")


if __name__ == "__main__":
    main()
