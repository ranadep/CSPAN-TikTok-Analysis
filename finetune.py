"""Fine-tune twitter-roberta for 3-class political stance on CPU.

Model choice: cardiffnlp/twitter-roberta-base-2022-154m. Byte-level BPE round-trips every
emoji (9% of comments are pure emoji, and that is the *highest*-accuracy slice), and 154M
tweets is the closest pretraining match to TikTok comment register. DeBERTa-v3's SentencePiece
vocab can emit <unk> on emoji and its disentangled attention is ~1.5-2x slower on CPU.

Trains a 5-class head and collapses to 3 at argmax: the 5-class distribution is far more
balanced (1.55:1 vs 2.16:1), so it dissolves the imbalance for free. Output is still 3-class.
"""
import argparse, os, sys

# must precede the torch import
NCPU = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count())
for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, str(NCPU))
os.environ.setdefault("OMP_PROC_BIND", "close")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
import data

MODEL = "cardiffnlp/twitter-roberta-base-2022-154m"
MAX_LEN, BATCH, LR, EPOCHS, WARMUP = 64, 16, 2e-5, 4, 0.1
OUT = "model"


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def make_batches(texts, labels, tok, shuffle, rng):
    """Dynamic padding + length grouping. Worth ~12% on CPU (measured: 2.00 vs 2.27 s/step).
    Less than the 3-4x you would expect from the token count -- short sequences do not
    amortize the per-layer overhead."""
    order = np.arange(len(texts))
    if shuffle:
        rng.shuffle(order)
        # sort by length within megabatches so batches are length-homogeneous but still varied
        mega = 50 * BATCH
        order = np.concatenate([sorted(order[i:i + mega], key=lambda j: len(texts[j]))
                                for i in range(0, len(order), mega)])
    else:
        order = np.array(sorted(order, key=lambda j: len(texts[j])))
    chunks = [order[i:i + BATCH] for i in range(0, len(order), BATCH)]
    if shuffle:
        rng.shuffle(chunks)
    for c in chunks:
        enc = tok([texts[j] for j in c], truncation=True, max_length=MAX_LEN,
                  padding=True, return_tensors="pt")
        yield c, enc, (torch.tensor(labels[c]) if labels is not None else None)


def train_fold(d, fold, tok, device, seed):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tr, te = (d.fold != fold).values, (d.fold == fold).values
    x_tr, y_tr = d.comment_text.values[tr], d.y5.values[tr]

    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * -(-len(x_tr) // BATCH)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)

    for epoch in range(EPOCHS):
        model.train()
        for _, enc, y in make_batches(x_tr, y_tr, tok, True, rng):
            enc = {k: v.to(device) for k, v in enc.items()}
            loss = model(**enc, labels=y.to(device)).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()
        print(f"  fold {fold} epoch {epoch + 1}/{EPOCHS} done", flush=True)

    return model, predict(model, d.comment_text.values[te], tok, device), te


@torch.no_grad()
def predict(model, texts, tok, device):
    """Returns 3-class predictions (5-class argmax collapsed -- measurably better than
    summing the pro/anti probabilities)."""
    model.eval()
    out = np.empty(len(texts), dtype=int)
    for c, enc, _ in make_batches(texts, None, tok, False, None):
        enc = {k: v.to(device) for k, v in enc.items()}
        out[c] = model(**enc).logits.argmax(-1).cpu().numpy()
    return np.array([data.COLLAPSE[i] for i in out])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, default=None, help="single fold (for a Slurm array)")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--all", action="store_true",
                   help="train on all 3675 comments and save to model/ -- no score, no test set")
    p.add_argument("--predict", metavar="TEXT", nargs="*", help="score text with saved model/")
    a = p.parse_args()

    device = pick_device(a.device)
    torch.set_num_threads(NCPU)
    torch.set_num_interop_threads(1)
    tok = AutoTokenizer.from_pretrained(MODEL)

    if a.predict is not None:
        texts = a.predict or [l.strip() for l in sys.stdin if l.strip()]
        model = AutoModelForSequenceClassification.from_pretrained(OUT).to(device)
        for t, y in zip(texts, predict(model, np.array(texts, dtype=object), tok, device)):
            print(f"{data.LABELS3[y]}\t{t}")
        return

    assert tok.tokenize("💙💙💙") and tok.unk_token not in tok.tokenize("💙💙💙🇺🇸"), \
        "tokenizer maps emoji to <unk> -- wrong model for this data"

    d = data.load(n_folds=a.folds, seed=0)
    data.check_no_leak(d)
    folds = [a.fold] if a.fold is not None else sorted(int(f) for f in d.fold.unique())
    print(f"device={device} threads={NCPU} folds={folds}", flush=True)

    if a.all:
        # fold -1 matches no row, so the train set is everything and the test set is empty.
        # This is the production model. Cross-validation measures the method; this is the artifact.
        model, _, _ = train_fold(d, -1, tok, device, a.seed)
        model.save_pretrained(OUT); tok.save_pretrained(OUT)
        print(f"saved model trained on all {len(d)} comments -> {OUT}/")
        return

    pred = np.full(len(d), -1)
    for k in folds:
        _, pk, te = train_fold(d, k, tok, device, a.seed)
        pred[te] = pk

    scored = d[pred >= 0].reset_index(drop=True)
    data.report(scored, pred[pred >= 0], f"twitter-roberta 5->3 (folds {folds}, seed {a.seed})")


if __name__ == "__main__":
    main()
