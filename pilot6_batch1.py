from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from pilot5 import LENGTHS, PLACEMENTS, REPO, NoPELlama, bits
from pilot6 import FillerTask, score

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
model = NoPELlama(hf_hub_download(REPO, "model.safetensors"), dev)
k = json.loads(Path("results6/tune.json").read_text())["chosen_k"]
task = FillerTask(tok, k)
res = {}
for i, pl in enumerate(PLACEMENTS):
    data = task.paired_set(200, pl, seed=40_000 + i)
    for L in LENGTHS:
        r, f = score(model, task, data[L], batch=1)
        res[f"none/{pl}/{L}"] = {
            "restricted": r.float().mean().item(),
            "full": f.float().mean().item(),
            "correct": bits(r),
        }
        print(pl, L, round(r.float().mean().item(), 3), flush=True)
Path("results6/none_batch1.json").write_text(json.dumps(res))
print("wrote results6/none_batch1.json")
