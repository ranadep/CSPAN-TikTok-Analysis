"""Step 0 reproduction of the reported RoBERTa number (0.625 +/- 0.048 macro-F1, 0.652 acc).

Runs finetune.train_fold *unchanged* on the original protocol (data.load(): all 65 videos,
5 video-grouped folds, seed 0 fold assignment), 3 seeds. Only the plumbing differs:
the model is loaded from a local safetensors copy and, on DirectML, AdamW is swapped for the
numerically equivalent SimpleAdamW. Every fold/seed is logged to results/runs.csv.

    python repro_finetune.py --seeds 0 1 2
"""
import argparse, time
import finetune as F  # first: sets thread env vars before torch loads
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoTokenizer
import data
from exp_utils import pick_device, is_dml, SimpleAdamW, resolve_model, log_run, done_runs, ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--experiment", default="repro_finetune_default")
    # tune.py's search dimensions, to replay its configs under the same protocol
    p.add_argument("--model", default=F.MODEL)
    p.add_argument("--lr", type=float, default=F.LR)
    p.add_argument("--epochs", type=int, default=F.EPOCHS)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    a = p.parse_args()

    device = pick_device(a.device)
    if is_dml(device):
        F.torch.optim.AdamW = SimpleAdamW
    torch.set_num_threads(F.NCPU)
    name = resolve_model(a.model)
    tok = AutoTokenizer.from_pretrained(name)
    d = data.load()
    data.check_no_leak(d)
    folds = a.folds or sorted(int(k) for k in d.fold.unique())
    cfg = {"model": a.model, "lr": a.lr, "epochs": a.epochs, "label_smoothing": a.label_smoothing, "max_len": F.MAX_LEN,
           "batch": F.BATCH, "warmup": F.WARMUP, "input": "comment"}
    print(f"device={device} model={name} folds={folds} seeds={a.seeds}", flush=True)

    pred_dir = ROOT / "results" / "preds" / a.experiment
    pred_dir.mkdir(parents=True, exist_ok=True)
    for seed in a.seeds:
        pred = np.full(len(d), -1)
        for k in folds:
            rid, npz = f"{a.experiment}-s{seed}-f{k}", pred_dir / f"s{seed}_f{k}.npz"
            if rid in done_runs() and npz.exists():  # resume after a crash
                pred[np.load(npz)["te"]] = np.load(npz)["pred3"]
                continue
            t0 = time.time()
            _, pk, te = F.train_fold(d, k, tok, device, seed, model_name=name, lr=a.lr,
                                     epochs=a.epochs, label_smoothing=a.label_smoothing)
            pred[te] = pk
            np.savez(npz, pred3=pk, te=te)
            y = d.y3.values[te]
            f1, acc = f1_score(y, pk, average="macro"), (pk == y).mean()
            print(f"seed {seed} fold {k}: macro-F1 {f1:.3f} acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
            log_run(run_id=rid, experiment=a.experiment,
                    protocol="orig_65vid_5fold", config=cfg, seed=seed, fold=k,
                    n_train=int((~te).sum()), n_test=int(te.sum()), macro_f1=round(f1, 4),
                    accuracy=round(acc, 4), epochs_run=a.epochs)
        if set(folds) == set(d.fold.unique()):
            data.report(d, pred, f"twitter-roberta 5->3 repro (seed {seed})")


if __name__ == "__main__":
    main()
