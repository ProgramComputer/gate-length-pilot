from __future__ import annotations

import torch, pilot5
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

tok = Tokenizer.from_file(hf_hub_download(pilot5.REPO, "tokenizer.json"))
task = pilot5.Task(tok)
model = pilot5.NoPELlama(hf_hub_download(pilot5.REPO, "model.safetensors"), torch.device("cuda"))
g = torch.Generator().manual_seed(7)
for n in (10, 50, 100, 200, 400):
    N = 50
    order = torch.rand(N, len(task.keys), generator=g).argsort(1)[:, :n]
    ks = task.keys[order]
    vs = task.values[torch.randint(512, (N, n), generator=g)]
    t = torch.randint(n, (N,), generator=g)
    data = {"seq": task.build(ks, vs, t), "answer": vs.gather(1, t[:, None])[:, 0]}
    r, f = pilot5.accuracy(model, task, data)
    T = data["seq"].shape[1]
    print(
        "records",
        n,
        "tokens",
        T,
        "restricted",
        round(r.float().mean().item(), 2),
        "full",
        round(f.float().mean().item(), 2),
        flush=True,
    )
s = data["seq"][0]
logits = model(s[None].cuda())[0]
print(
    "tail:",
    repr(tok.decode(s[-9:].tolist())),
    "| answer:",
    repr(tok.decode([int(data["answer"][0])])),
    "| top5:",
    [tok.decode([i]) for i in logits.topk(5).indices.tolist()],
)
