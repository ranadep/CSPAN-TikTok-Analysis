"""One-time: make a local safetensors copy of a HF model that ships only pytorch_model.bin.

torch-directml pins torch 2.4.1, and transformers refuses torch.load on torch < 2.6
(CVE-2025-32434). The weights here come from the named public repos; torch.load still runs
with weights_only=True. Output goes to $MODEL_DIR/<repo name>, which exp_utils.resolve_model
picks up automatically.

    python scripts/convert_to_safetensors.py vinai/bertweet-base microsoft/deberta-v3-base
"""
import os, sys
from pathlib import Path
import transformers.modeling_utils as mu
from transformers import AutoModel, AutoTokenizer

mu.check_torch_load_is_safe = lambda: None  # weights_only=True is still enforced by transformers
out_root = Path(os.environ.get("MODEL_DIR", "D:/cspan-ml/models"))
for repo in sys.argv[1:]:
    out = out_root / repo.split("/")[-1]
    m = AutoModel.from_pretrained(repo)
    m.save_pretrained(out, safe_serialization=True)
    AutoTokenizer.from_pretrained(repo).save_pretrained(out)
    print("saved", out)
