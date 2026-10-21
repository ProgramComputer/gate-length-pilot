from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from gatepilot.correct import (
    CONDITIONS,
    Intervention,
    collect,
    discrepancy,
    evaluate,
    make_condition,
)
from gatepilot.model import Model
from gatepilot.task import EVAL_LENS, PLACEMENTS, QUERY, TRAIN_LEN, paired_set

SITES = ("gate", "gate_post", "up", "resid")

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--baseline-only", action="store_true")
p.add_argument("--n-test", type=int, default=1000)
p.add_argument("--n-cal", type=int, default=512)
p.add_argument("--runs", default="runs")
p.add_argument("--out", default="results")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

dev = torch.device(args.device)
ckpt = torch.load(Path(args.runs) / f"seed{args.seed}" / "model.pt", map_location="cpu")
model = Model(pos=ckpt.get("args", {}).get("pos", "nope"))
model.load_state_dict(ckpt["state"])
model.eval().to(dev)
pairs = [(site, layer) for site in SITES for layer in range(len(model.blocks))]
conditions = ("none",) if args.baseline_only else CONDITIONS

cal = paired_set(args.n_cal, EVAL_LENS, "uniform", seed=20_000)
chk = paired_set(args.n_cal, EVAL_LENS, "uniform", seed=30_000)
test = {
    pl: paired_set(args.n_test, EVAL_LENS, pl, seed=40_000 + i) for i, pl in enumerate(PLACEMENTS)
}

res = {
    "seed": args.seed,
    "n_params": ckpt["n_params"],
    "conditions": conditions,
    "acc": {},
    "logp": {},
    "correct": {},
    "distance": {},
    "drift": {},
    "hist": {},
    "manip": {},
    "size": {},
}
for pl in PLACEMENTS:
    for L in EVAL_LENS:
        res["distance"][f"{pl}/{L}"] = test[pl][L]["distance"].tolist()


def query_acts(
    data: dict[str, torch.Tensor],
    iv: Intervention,
) -> dict[tuple[str, int, int], torch.Tensor]:
    return collect(model, data["seq"], iv, pairs, record_from=0, record_roles=(QUERY,))


def r4(t: torch.Tensor) -> list[float]:
    return [round(v, 4) for v in t.tolist()]


t0 = time.time()
if not args.baseline_only:
    short = query_acts(cal[TRAIN_LEN], Intervention())
    for L in EVAL_LENS:
        long = short if L == TRAIN_LEN else query_acts(cal[L], Intervention())
        for site, layer in pairs:
            xs, xl = short[site, layer, QUERY], long[site, layer, QUERY]
            res["drift"][f"{L}/{site}/{layer}"] = {
                "smd": r4((xl.mean(0) - xs.mean(0)) / (xs.std(0) + 1e-6)),
                "log_sd_ratio": r4(((xl.std(0) + 1e-6) / (xs.std(0) + 1e-6)).log()),
                "w2": round(discrepancy(short, long, site, layer).mean().item(), 4),
            }
        for layer in range(len(model.blocks)):
            key = ("attn_entropy", layer, QUERY)
            res["drift"][f"{L}/attn_entropy/{layer}"] = {
                "mean_short": round(short[key].mean().item(), 4),
                "mean_long": round(long[key].mean().item(), 4),
            }
        if L == EVAL_LENS[-1]:
            for layer in range(len(model.blocks)):
                c = int(torch.tensor(res["drift"][f"{L}/gate/{layer}"]["smd"]).abs().argmax())
                for site in ("gate", "gate_post"):
                    res["hist"][f"{site}/{layer}"] = {
                        "channel": c,
                        "short": r4(short[site, layer, QUERY][:, c]),
                        "long": r4(long[site, layer, QUERY][:, c]),
                    }
    print(f"drift measured ({time.time() - t0:.0f}s)", flush=True)
    chk_short = query_acts(chk[TRAIN_LEN], Intervention())

for L in EVAL_LENS:
    for cond in conditions:
        iv = make_condition(cond, model, cal[TRAIN_LEN]["seq"], cal[L]["seq"], L)
        for pl in PLACEMENTS:
            correct, logp = evaluate(model, test[pl][L], iv)
            res["acc"][f"{cond}/{pl}/{L}"] = round(correct.float().mean().item(), 4)
            res["logp"][f"{cond}/{pl}/{L}"] = round(logp.mean().item(), 4)
            res["correct"][f"{cond}/{pl}/{L}"] = "".join(
                "1" if c else "0" for c in correct.tolist()
            )
        if not args.baseline_only:
            chk_long = chk_short if L == TRAIN_LEN else query_acts(chk[L], iv)
            for site, layer in pairs:
                res["manip"][f"{cond}/{L}/{site}/{layer}"] = round(
                    discrepancy(chk_short, chk_long, site, layer).mean().item(), 4
                )
            for layer in range(len(model.blocks)):
                res["manip"][f"{cond}/{L}/attn_entropy/{layer}"] = round(
                    chk_long["attn_entropy", layer, QUERY].mean().item(), 4
                )
            res["size"][f"{cond}/{L}"] = iv.size
        print(
            f"L={L:5d} {cond:9s} acc "
            + " ".join(f"{pl}={res['acc'][f'{cond}/{pl}/{L}']:.3f}" for pl in PLACEMENTS)
            + f"  ({time.time() - t0:.0f}s)",
            flush=True,
        )

out = Path(args.out)
out.mkdir(exist_ok=True)
name = f"baseline_seed{args.seed}.json" if args.baseline_only else f"seed{args.seed}.json"
(out / name).write_text(json.dumps(res))
print("wrote", out / name)
