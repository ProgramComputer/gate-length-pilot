from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from gatepilot.model import Model
from gatepilot.task import TRAIN_LEN, n_records, paired_set, train_batch

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--steps", type=int, default=20000)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--warmup", type=int, default=500)
p.add_argument("--eval-every", type=int, default=500)
p.add_argument(
    "--pos",
    choices=("nope", "rope"),
    default="nope",
    help="positional scheme (rope = pilot 2 control)",
)
p.add_argument(
    "--curriculum",
    type=float,
    default=0.0,
    help="fraction of training over which the max number of records ramps from 8 to 62",
)
p.add_argument("--out", default="runs")
p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = p.parse_args()

torch.manual_seed(args.seed)
gen = torch.Generator().manual_seed(args.seed)
dev = torch.device(args.device)
model = Model(pos=args.pos).to(dev)
n_params = sum(p.numel() for p in model.parameters())
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), weight_decay=0.01)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt,
    lambda s: min(1, (s + 1) / args.warmup)
    * (0.55 + 0.45 * math.cos(math.pi * min(s, args.steps) / args.steps)),
)
val = paired_set(1000, [TRAIN_LEN], "uniform", seed=10_000)[TRAIN_LEN]
val_short = dict(
    zip(("seq", "answer"), train_batch(1000, torch.Generator().manual_seed(10_001), max_records=8))
)

out = Path(args.out) / f"seed{args.seed}"
out.mkdir(parents=True, exist_ok=True)
log = open(out / "train_log.jsonl", "w")
print(f"seed {args.seed}: {n_params:,} parameters on {dev}")


@torch.no_grad()
def validate(data: dict[str, torch.Tensor]) -> tuple[float, int]:
    model.eval()
    pred = torch.cat(
        [model(data["seq"][i : i + 250].to(dev)).argmax(-1).cpu() for i in range(0, 1000, 250)]
    )
    model.train()
    return (pred == data["answer"]).float().mean().item(), len(pred.unique())


t0 = time.time()
for step in range(1, args.steps + 1):
    ramp = min(1.0, step / (args.curriculum * args.steps)) if args.curriculum else 1.0
    seq, ans = train_batch(
        args.batch, gen, max_records=round(8 + ramp * (n_records(TRAIN_LEN) - 8))
    )
    loss = F.cross_entropy(model(seq.to(dev)), ans.to(dev))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()
    opt.step()
    sched.step()
    if step in (1, 10, 100) or step % args.eval_every == 0 or step == args.steps:
        acc, spread = validate(val)
        row = dict(
            step=step,
            loss=round(loss.item(), 4),
            val_acc128=round(acc, 4),
            pred_spread=spread,
            val_acc_short=round(validate(val_short)[0], 4),
            grad_norm=round(grad_norm, 3),
            lr=sched.get_last_lr()[0],
            sec=round(time.time() - t0, 1),
        )
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(row, flush=True)

torch.save(
    {"state": model.state_dict(), "args": vars(args), "n_params": n_params}, out / "model.pt"
)
