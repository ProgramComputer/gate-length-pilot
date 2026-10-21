from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gatepilot.correct import Intervention, evaluate
from gatepilot.model import Model
from gatepilot.probe import probe_accuracies, record_stream
from gatepilot.task import EVAL_LENS, PLACEMENTS, paired_set

TEMPS = (1.0, 1.2, 1.5, 1.8)

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--runs", default="runs")
p.add_argument("--out", default="results2")
p.add_argument("--n-test", type=int, default=1000)
p.add_argument("--n-probe", type=int, default=1024)
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

ckpt = torch.load(Path(args.runs) / f"seed{args.seed}" / "model.pt", map_location="cpu")
pos = ckpt["args"].get("pos", "nope") if "args" in ckpt else "nope"
model = Model(pos=pos)
model.load_state_dict(ckpt["state"])
model.eval().to(torch.device(args.device))

test = {
    pl: paired_set(args.n_test, EVAL_LENS, pl, seed=40_000 + i) for i, pl in enumerate(PLACEMENTS)
}
probe_seqs = paired_set(args.n_probe, (1024,), "uniform", seed=50_000)[1024]["seq"]
n_train = args.n_probe * 3 // 4

res = {
    "seed": args.seed,
    "pos": pos,
    "temps": TEMPS,
    "acc": {},
    "correct": {},
    "distance": {},
    "probe": {},
}
for pl in PLACEMENTS:
    for L in EVAL_LENS:
        res["distance"][f"{pl}/{L}"] = test[pl][L]["distance"].tolist()

t0 = time.time()
for c in TEMPS:
    iv = Intervention(uniform_temp=c)
    for L in EVAL_LENS:
        for pl in PLACEMENTS:
            correct, _ = evaluate(model, test[pl][L], iv)
            res["acc"][f"{c}/{pl}/{L}"] = round(correct.float().mean().item(), 4)
            res["correct"][f"{c}/{pl}/{L}"] = "".join("1" if k else "0" for k in correct.tolist())
    res["probe"][str(c)] = probe_accuracies(
        *record_stream(model, probe_seqs, uniform_temp=c), n_train=n_train
    )
    best = max(
        range(1, len(model.blocks) + 1), key=lambda layer: res["probe"][str(c)][layer]["refit"][0]
    )
    print(
        f"{pos} seed {args.seed} temp {c}: acc@1024 uniform={res['acc'][f'{c}/uniform/1024']:.3f} "
        f"before={res['acc'][f'{c}/before/1024']:.3f} after={res['acc'][f'{c}/after/1024']:.3f}; "
        f"probe layer {best} refit by window {res['probe'][str(c)][best]['refit']}"
        f"  ({time.time() - t0:.0f}s)",
        flush=True,
    )

out = Path(args.out)
out.mkdir(exist_ok=True)
(out / f"{pos}_seed{args.seed}.json").write_text(json.dumps(res))
print("wrote", out / f"{pos}_seed{args.seed}.json")
