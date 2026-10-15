"""Shared helpers for the experiment scripts (repro_finetune.py, train.py).

Kept separate so the original pipeline (data.py / baseline.py / finetune.py) is untouched
apart from device selection.
"""
import csv, datetime, json, os, subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).parent
RUNS = ROOT / "results" / "runs.csv"
RUN_FIELDS = ["run_id", "time", "git", "experiment", "protocol", "config", "seed", "fold",
              "n_train", "n_test", "macro_f1", "accuracy", "best_epoch", "epochs_run",
              "train_loss_last", "val_loss_best", "val_f1_best", "notes"]

# Local safetensors copies of HF repos that ship only pytorch_model.bin. torch-directml pins
# torch 2.4, and transformers refuses torch.load on torch < 2.6 (CVE-2025-32434).
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "D:/cspan-ml/models"))


def resolve_model(name):
    local = MODEL_DIR / name.split("/")[-1]
    return str(local) if (local / "config.json").exists() else name


def pick_device(name="auto"):
    """cuda -> mps -> directml -> cpu. Pass --device to override."""
    if name == "dml":
        import torch_directml
        return torch_directml.device()
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    try:
        import torch_directml
        if torch_directml.is_available():
            return torch_directml.device()
    except ImportError:
        pass
    return torch.device("cpu")


def is_dml(device):
    return device.type == "privateuseone"


class SimpleAdamW(torch.optim.Optimizer):
    """AdamW built from mul_/add_/addcdiv_ only. torch-directml lacks lerp_, so torch's AdamW
    falls back to CPU every step (measured 0.59 -> 0.24 s/step on an RX 7700 XT)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            b1, b2 = g["betas"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if not st:
                    st["t"], st["m"], st["v"] = 0, torch.zeros_like(p), torch.zeros_like(p)
                st["t"] += 1
                t = st["t"]
                if g["weight_decay"]:
                    p.mul_(1 - g["lr"] * g["weight_decay"])
                st["m"].mul_(b1).add_(p.grad, alpha=1 - b1)
                st["v"].mul_(b2).addcmul_(p.grad, p.grad, value=1 - b2)
                denom = (st["v"] / (1 - b2 ** t)).sqrt_().add_(g["eps"])
                p.addcdiv_(st["m"], denom, value=-g["lr"] / (1 - b1 ** t))


def make_optimizer(params, device, lr, weight_decay):
    cls = SimpleAdamW if is_dml(device) else torch.optim.AdamW
    return cls(params, lr=lr, weight_decay=weight_decay)


def git_hash():
    try:
        h = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "*.py"], cwd=ROOT)
        return h + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def log_run(**row):
    """Append one (config, seed, fold) result to results/runs.csv. Every run is logged."""
    RUNS.parent.mkdir(exist_ok=True)
    row.setdefault("time", datetime.datetime.now().isoformat(timespec="seconds"))
    row.setdefault("git", git_hash())
    if isinstance(row.get("config"), dict):
        row["config"] = json.dumps(row["config"], sort_keys=True)
    new = not RUNS.exists()
    with open(RUNS, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, RUN_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)
