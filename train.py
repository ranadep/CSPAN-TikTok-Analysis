"""Experiment trainer on the locked protocol (protocol.py). One config per invocation.

Each (seed, CV fold) run: train on the purged training videos, early-stop on the grouped
validation videos (macro-F1, patience), restore the best epoch, then score the CV test fold
once. Logs every run to results/runs.csv, per-epoch curves to results/curves.csv, and test
logits to results/preds/<experiment>/ (for seed ensembles and error analysis).

    python train.py --experiment pair --input pair --seeds 0 1 2
    python train.py --experiment shuffled --shuffle-labels --seeds 0
    python train.py --experiment FINAL --final-holdout ...      # once, at the very end

Defaults reproduce finetune.py's settings (twitter-roberta-base, lr 2e-5, batch 16,
warmup 0.1, wd 0.01) except that epochs are capped at 6 with early stopping.
"""
import argparse, json, os, time
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, \
    get_linear_schedule_with_warmup

import data
import protocol as P
from exp_utils import pick_device, is_dml, make_optimizer, resolve_model, log_run, done_runs, ROOT

COLLAPSE = torch.tensor(data.COLLAPSE)
CURVES = ROOT / "results" / "curves.csv"


# ---------------------------------------------------------------------------- model
class LSTMHead(nn.Module):
    """Encoder -> dropout -> BiLSTM over token states -> masked mean+max pool -> dropout -> linear.
    Only used if the plain classification head plateaus (--head lstm)."""

    def __init__(self, name, n_labels, dropout, hidden=256):
        super().__init__()
        self.enc = AutoModel.from_pretrained(name)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(h, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(4 * hidden, n_labels)

    def forward(self, input_ids, attention_mask, **kw):
        x = self.drop(self.enc(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state)
        x, _ = self.lstm(x)
        m = attention_mask.unsqueeze(-1).to(x.dtype)
        mean = (x * m).sum(1) / m.sum(1).clamp(min=1)
        mx = x.masked_fill(m == 0, -1e4).max(1).values
        return self.out(self.drop(torch.cat([mean, mx], -1)))


def build_model(a):
    name = resolve_model(a.backbone)
    if a.head == "lstm":
        return LSTMHead(name, 5, 0.1 if a.dropout is None else a.dropout)
    kw = {}
    if a.dropout is not None:
        kw = {"hidden_dropout_prob": a.dropout, "attention_probs_dropout_prob": a.dropout}
    m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=5, **kw)
    return m


def logits_of(model, enc):
    out = model(**enc)
    return out.logits if hasattr(out, "logits") else out


# ---------------------------------------------------------------------------- data
def encode(tok, df, a):
    first = df.comment_text.astype(str).tolist()
    if a.input == "pair":
        second = [" ".join(s.split()) for s in df.video_description]
        # longest_first: the (usually longer) description is trimmed first; a rare 300-token
        # comment cannot overflow the way only_second would let it
        return tok(first, second, truncation="longest_first", max_length=a.max_len)
    return tok(first, truncation=True, max_length=a.max_len)


def batches(enc, labels, bs, shuffle, rng, pad_id):
    """Length-grouped dynamic padding, as in finetune.make_batches."""
    n = len(enc["input_ids"])
    lens = np.array([len(x) for x in enc["input_ids"]])
    order = np.arange(n)
    if shuffle:
        rng.shuffle(order)
        mega = 50 * bs
        order = np.concatenate([order[i:i + mega][np.argsort(lens[order[i:i + mega]], kind="stable")]
                                for i in range(0, n, mega)])
    else:
        order = order[np.argsort(lens, kind="stable")]
    chunks = [order[i:i + bs] for i in range(0, n, bs)]
    if shuffle:
        rng.shuffle(chunks)
    for c in chunks:
        L = lens[c].max()
        ids = np.full((len(c), L), pad_id, dtype=np.int64)
        mask = np.zeros((len(c), L), dtype=np.int64)
        for r, j in enumerate(c):
            ids[r, :lens[j]] = enc["input_ids"][j]
            mask[r, :lens[j]] = 1
        yield c, {"input_ids": torch.from_numpy(ids), "attention_mask": torch.from_numpy(mask)}, \
            (None if labels is None else torch.from_numpy(labels[c]))


@torch.no_grad()
def predict_logits(model, enc, device, bs, pad_id):
    model.eval()
    out = np.zeros((len(enc["input_ids"]), 5), dtype=np.float32)
    for c, x, _ in batches(enc, None, 64, False, None, pad_id):
        x = {k: v.to(device) for k, v in x.items()}
        out[c] = logits_of(model, x).float().cpu().numpy()
    return out


def collapse(logits5):
    return np.array(data.COLLAPSE)[logits5.argmax(-1)]


# ---------------------------------------------------------------------------- loss
def make_loss(a, y_train, device):
    w = None
    if a.loss in ("weighted", "focal_weighted"):
        cnt = np.bincount(y_train, minlength=5).astype(float)
        w = torch.tensor(cnt.sum() / (5 * np.maximum(cnt, 1)), dtype=torch.float32).to(device)

    def loss(logits, y):
        if a.loss.startswith("focal"):
            logp = Fn.log_softmax(logits, -1)
            ce = Fn.nll_loss(logp, y, weight=w, reduction="none")
            pt = logp.gather(1, y[:, None]).squeeze(1).exp()
            fl = ((1 - pt) ** a.focal_gamma) * ce
            if a.label_smoothing:
                fl = fl * (1 - a.label_smoothing) - a.label_smoothing * logp.mean(-1)
            return fl.mean()
        return Fn.cross_entropy(logits, y, weight=w, label_smoothing=a.label_smoothing)
    return loss


# ---------------------------------------------------------------------------- one run
def run_one(a, tok, device, train, val, test, seed, fold, n_purged, tag):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    y_tr = train.y5.values.astype(np.int64)
    if a.shuffle_labels:  # sanity check: permute training labels only; val/test keep true labels
        y_tr = np.random.default_rng(10_000 + seed).permutation(y_tr)
    e_tr, e_va, e_te = encode(tok, train, a), encode(tok, val, a), encode(tok, test, a)
    pad = tok.pad_token_id

    model = build_model(a).to(device)
    opt = make_optimizer(model.parameters(), device, a.lr, a.weight_decay, a.optim)
    steps_per_epoch = -(-len(y_tr) // (a.batch * a.grad_accum))
    total = a.epochs * steps_per_epoch
    sched = get_linear_schedule_with_warmup(opt, int(a.warmup * total), total)
    lossf = make_loss(a, y_tr, device)
    use_amp = a.fp16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    y_va3 = val.y3.values
    y_va5 = torch.from_numpy(val.y5.values.astype(np.int64))
    best = (-1.0, None, None, 0)  # f1, state, val_loss, epoch
    bad, curves, tr_loss = 0, [], float("nan")
    for epoch in range(1, a.epochs + 1):
        model.train()
        tot, nb = 0.0, 0
        for i, (_, x, y) in enumerate(batches(e_tr, y_tr, a.batch, True, rng, pad)):
            x = {k: v.to(device) for k, v in x.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                l = lossf(logits_of(model, x).float(), y.to(device)) / a.grad_accum
            scaler.scale(l).backward()
            tot += l.item() * a.grad_accum
            nb += 1
            if (i + 1) % a.grad_accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        if nb % a.grad_accum:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()
        tr_loss = tot / nb
        lv = predict_logits(model, e_va, device, a.batch, pad)
        va_loss = float(Fn.cross_entropy(torch.from_numpy(lv), y_va5))
        va_f1 = f1_score(y_va3, collapse(lv), average="macro")
        curves.append((epoch, tr_loss, va_loss, va_f1))
        print(f"  [{tag}] seed {seed} fold {fold} ep {epoch}: train_loss {tr_loss:.3f} "
              f"val_loss {va_loss:.3f} val_F1 {va_f1:.3f}", flush=True)
        if va_f1 > best[0]:
            state = {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
            best, bad = (va_f1, state, va_loss, epoch), 0
        else:
            bad += 1
            if bad >= a.patience:
                break

    model.load_state_dict(best[1])
    lt = predict_logits(model, e_te, device, a.batch, pad)
    return model, lt, best, curves, tr_loss


def save_curves(experiment, seed, fold, curves):
    new = not CURVES.exists()
    with open(CURVES, "a", encoding="utf-8") as f:
        if new:
            f.write("experiment,seed,fold,epoch,train_loss,val_loss,val_macro_f1\n")
        for e, tl, vl, vf in curves:
            f.write(f"{experiment},{seed},{fold},{e},{tl:.4f},{vl:.4f},{vf:.4f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--backbone", default="cardiffnlp/twitter-roberta-base-2022-154m")
    p.add_argument("--input", choices=["comment", "pair"], default="comment")
    p.add_argument("--max-len", type=int, default=None, help="default: 64 comment / 160 pair")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--warmup", type=float, default=0.1)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--loss", choices=["ce", "weighted", "focal", "focal_weighted"], default="ce")
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--head", choices=["linear", "lstm"], default="linear")
    p.add_argument("--dropout", type=float, default=None, help="encoder dropout override")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--folds", type=int, nargs="+", default=None)
    p.add_argument("--device", default="auto")
    p.add_argument("--optim", choices=["auto", "scratch", "offload"], default="auto",
                   help="auto: torch AdamW (SimpleAdamW on DirectML); scratch/offload: for large models on DML")
    p.add_argument("--fp16", action="store_true", help="CUDA only (ignored elsewhere)")
    p.add_argument("--shuffle-labels", action="store_true")
    p.add_argument("--final-holdout", action="store_true",
                   help="THE one holdout run: train on all dev videos, score splits/final_holdout_videos.txt")
    p.add_argument("--leave-cspan-out", choices=["dev", "all"], default=None,
                   help="train non-C-SPAN videos, test C-SPAN videos (all = 37->28, only after holdout)")
    p.add_argument("--smoke", action="store_true", help="tiny subsets, 1 epoch: plumbing test only")
    p.add_argument("--save", default=None, help="dir to save the (last seed's) best checkpoint")
    a = p.parse_args()
    if a.max_len is None:
        a.max_len = 160 if a.input == "pair" else 64

    device = pick_device(a.device)
    torch.set_num_threads(min(10, os.cpu_count() or 1))
    tok = AutoTokenizer.from_pretrained(resolve_model(a.backbone))
    cfg = {k: v for k, v in vars(a).items() if k not in ("seeds", "folds", "device", "save", "experiment")}
    pred_dir = ROOT / "results" / "preds" / a.experiment
    pred_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} cfg={json.dumps(cfg)}", flush=True)

    if a.final_holdout:
        # Run-once guard, committed to git so it travels with the repo. Written BEFORE the first
        # holdout prediction; a crash mid-run may only finish the *same* run's remaining seeds.
        marker = ROOT / "results" / "FINAL_HOLDOUT_DONE"
        if marker.exists():
            state = marker.read_text().split()
            assert state[:2] == ["STARTED", a.experiment], \
                f"holdout already used ({marker}: {' '.join(state)}); it is scored exactly once"
            print(f"resuming interrupted final run {a.experiment}", flush=True)
        else:
            marker.write_text(f"STARTED {a.experiment}\n")
        for seed in a.seeds:
            if f"{a.experiment}-s{seed}-holdout" in done_runs():
                continue
            train, val, hold, n_purged = P.final_split(seed=0)
            t0 = time.time()
            model, lt, best, curves, tr_loss = run_one(a, tok, device, train, val, hold, seed, -1, n_purged, "FINAL")
            pr, y = collapse(lt), hold.y3.values
            f1, acc = f1_score(y, pr, average="macro"), (pr == y).mean()
            print(f"FINAL seed {seed}: holdout macro-F1 {f1:.4f} acc {acc:.4f} ({time.time() - t0:.0f}s)", flush=True)
            np.savez(pred_dir / f"s{seed}_holdout.npz", logits=lt, idx=hold.index.values, y3=y)
            save_curves(a.experiment, seed, -1, curves)
            log_run(run_id=f"{a.experiment}-s{seed}-holdout", experiment=a.experiment, protocol="final_holdout",
                    config=cfg, seed=seed, fold=-1, n_train=len(train), n_test=len(hold),
                    macro_f1=round(f1, 4), accuracy=round(acc, 4), best_epoch=best[3],
                    epochs_run=len(curves), train_loss_last=round(tr_loss, 4),
                    val_loss_best=round(best[2], 4), val_f1_best=round(best[0], 4),
                    notes=f"near_dup_purged={n_purged}")
            if a.save:
                Path(a.save).mkdir(parents=True, exist_ok=True)
                sd = Path(a.save) / f"seed{seed}"
                (model.save_pretrained(sd) if hasattr(model, "save_pretrained")
                 else torch.save(model.state_dict(), sd.with_suffix(".pt")))
                tok.save_pretrained(sd)
            del model
        marker.write_text(f"COMPLETE {a.experiment}\n")
        return

    if a.leave_cspan_out:
        for seed in a.seeds:
            if f"{a.experiment}-s{seed}-lcso-{a.leave_cspan_out}" in done_runs():
                continue
            train, val, test, n_purged = P.leave_cspan_out_split(a.leave_cspan_out)
            model, lt, best, curves, tr_loss = run_one(a, tok, device, train, val, test, seed, -2, n_purged, "LCSO")
            pr, y = collapse(lt), test.y3.values
            f1, acc = f1_score(y, pr, average="macro"), (pr == y).mean()
            print(f"[{a.experiment}] LCSO-{a.leave_cspan_out} seed {seed}: macro-F1 {f1:.4f} acc {acc:.4f}", flush=True)
            np.savez(pred_dir / f"s{seed}_lcso_{a.leave_cspan_out}.npz", logits=lt, idx=test.index.values, y3=y)
            save_curves(a.experiment, seed, -2, curves)
            log_run(run_id=f"{a.experiment}-s{seed}-lcso-{a.leave_cspan_out}", experiment=a.experiment,
                    protocol=f"leave_cspan_out_{a.leave_cspan_out}_{P.NEAR_DUP_VERSION}", config=cfg, seed=seed, fold=-2,
                    n_train=len(train), n_test=len(test), macro_f1=round(f1, 4), accuracy=round(acc, 4),
                    best_epoch=best[3], epochs_run=len(curves), train_loss_last=round(tr_loss, 4),
                    val_loss_best=round(best[2], 4), val_f1_best=round(best[0], 4),
                    notes=f"near_dup_purged={n_purged}")
            del model
        return

    dev = P.load_dev()
    folds = a.folds if a.folds is not None else list(range(P.N_FOLDS))
    for seed in a.seeds:
        for k in folds:
            if not a.smoke and f"{a.experiment}-s{seed}-f{k}" in done_runs()                     and (pred_dir / f"s{seed}_f{k}.npz").exists():
                print(f"[{a.experiment}] seed {seed} fold {k}: already done, skipping", flush=True)
                continue
            train, val, test, n_purged = P.cv_split(dev, k)
            if a.smoke:
                train, val, test = train.iloc[:64], val.iloc[:32], test.iloc[:32]
                a.epochs = 1
            t0 = time.time()
            model, lt, best, curves, tr_loss = run_one(a, tok, device, train, val, test, seed, k, n_purged, a.experiment)
            pr, y = collapse(lt), test.y3.values
            f1, acc = f1_score(y, pr, average="macro"), (pr == y).mean()
            print(f"[{a.experiment}] seed {seed} fold {k}: macro-F1 {f1:.4f} acc {acc:.4f} "
                  f"best_ep {best[3]} ({time.time() - t0:.0f}s)", flush=True)
            if a.smoke:
                continue  # plumbing test: never logged
            np.savez(pred_dir / f"s{seed}_f{k}.npz", logits=lt, idx=test.index.values, y3=y)
            save_curves(a.experiment, seed, k, curves)
            log_run(run_id=f"{a.experiment}-s{seed}-f{k}", experiment=a.experiment,
                    protocol="dev52_5fold" + ("_shuffled" if a.shuffle_labels else "") + "_" + P.NEAR_DUP_VERSION,
                    config=cfg, seed=seed, fold=k, n_train=len(train), n_test=len(test),
                    macro_f1=round(f1, 4), accuracy=round(acc, 4), best_epoch=best[3],
                    epochs_run=len(curves), train_loss_last=round(tr_loss, 4),
                    val_loss_best=round(best[2], 4), val_f1_best=round(best[0], 4),
                    notes=f"near_dup_purged={n_purged}")
            if a.save and k == folds[-1] and seed == a.seeds[-1]:
                Path(a.save).mkdir(parents=True, exist_ok=True)
                model.save_pretrained(a.save); tok.save_pretrained(a.save)
            del model


if __name__ == "__main__":
    main()
