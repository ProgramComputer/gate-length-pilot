from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from gatepilot.correct import Intervention, evaluate
from gatepilot.probe import BUCKETS, StreamRecorder, probe_accuracies
from gatepilot.task import paired_set
from pilot3 import Hooks, load

OFFSETS = {"none": None, "patch": 1, "sham": 3}


def with_patch(
    iv: Intervention,
    heads_by_layer: dict[int, list[int]],
    offset: int | None,
) -> Intervention:
    def attn_patch(layer: int, T: int) -> tuple[torch.Tensor, torch.Tensor, int] | None:
        heads = heads_by_layer.get(layer)
        if offset is None or not heads or T - 2 <= 128:
            return None
        return torch.tensor(heads), torch.arange(128, T - 2, 2), offset

    iv.attn_patch = attn_patch
    return iv


@torch.no_grad()
def prev_attention(model: torch.nn.Module, seqs: torch.Tensor, batch: int = 32) -> torch.Tensor:
    dev = next(model.parameters()).device
    gen = torch.Generator().manual_seed(2)
    n_layers, n_heads = len(model.blocks), model.blocks[0].attn.n_heads
    total = torch.zeros(n_layers, n_heads, len(BUCKETS))
    for i in range(0, len(seqs), batch):
        hooks = Hooks()
        hooks.wants_qk = True
        model(seqs[i : i + batch].to(dev), hooks)
        T = seqs.shape[1]
        for b, (lo, hi) in enumerate(BUCKETS):
            cand = torch.arange(lo, min(hi, T - 2), 2)
            pos = cand[torch.randperm(len(cand), generator=gen)[:16]].to(dev)
            for layer, (q, k) in hooks.qk.items():
                s = q[:, :, pos] @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
                s = s.masked_fill(
                    torch.arange(T, device=dev)[None, :] > pos[:, None], float("-inf")
                )
                w = s.softmax(-1).gather(-1, (pos - 1).view(1, 1, -1, 1).expand(*s.shape[:3], 1))
                total[layer, :, b] += w.mean((0, 2, 3)).cpu() * len(q)
    return total / len(seqs)


@torch.no_grad()
def probe(
    model: torch.nn.Module,
    seqs: torch.Tensor,
    heads_by_layer: dict[int, list[int]],
    offset: int | None,
    n_train: int,
) -> dict[int, dict[str, list[float]]]:
    dev = next(model.parameters()).device
    rec = with_patch(StreamRecorder(), heads_by_layer, offset)
    for i in range(0, len(seqs), 64):
        model(seqs[i : i + 64].to(dev), rec)
    X = {layer: torch.cat(v) for layer, v in rec.rows.items()}
    return probe_accuracies(
        X, torch.cat(rec.labels), torch.cat(rec.buckets), torch.cat(rec.examples), n_train
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--runs", default="runs")
    p.add_argument("--out", default="results4")
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--n-cal", type=int, default=256)
    p.add_argument("--n-probe", type=int, default=1024)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    t0 = time.time()
    model = load(args.runs, args.seed, torch.device(args.device))

    cal = paired_set(args.n_cal, (1024,), "uniform", seed=20_000)[1024]["seq"]
    attn = prev_attention(model, cal)
    inside = attn[:, :, 0]
    chosen = [
        (l, h) for l in range(attn.shape[0]) for h in range(attn.shape[1]) if inside[l, h] >= 0.3
    ]
    if not chosen:
        flat = int(inside.flatten().argmax())
        chosen = [(flat // attn.shape[1], flat % attn.shape[1])]
    heads_by_layer: dict[int, list[int]] = {}
    for l, h in chosen:
        heads_by_layer.setdefault(l, []).append(h)
    print(
        f"seed {args.seed}: binding heads {chosen}; attention to previous by window",
        {f"{l}.{h}": [round(v, 3) for v in attn[l, h].tolist()] for l, h in chosen},
        flush=True,
    )

    res = {
        "seed": args.seed,
        "binding_heads": chosen,
        "prev_attention": [
            [[round(v, 4) for v in attn[l, h].tolist()] for h in range(attn.shape[1])]
            for l in range(attn.shape[0])
        ],
        "acc": {},
        "correct": {},
        "distance": {},
        "probe": {},
    }
    test = {
        pl: paired_set(args.n_test, (128, 1024), pl, seed=40_000 + i)
        for i, pl in enumerate(("uniform", "before", "after"))
    }
    for pl in test:
        res["distance"][f"{pl}/1024"] = test[pl][1024]["distance"].tolist()
    probe_seqs = paired_set(args.n_probe, (1024,), "uniform", seed=50_000)[1024]["seq"]
    for name, offset in OFFSETS.items():
        for pl, L in (("uniform", 128), ("uniform", 1024), ("before", 1024), ("after", 1024)):
            correct, _ = evaluate(
                model, test[pl][L], with_patch(Intervention(), heads_by_layer, offset)
            )
            res["acc"][f"{name}/{pl}/{L}"] = round(correct.float().mean().item(), 4)
            res["correct"][f"{name}/{pl}/{L}"] = "".join(
                "1" if c else "0" for c in correct.tolist()
            )
        res["probe"][name] = probe(
            model, probe_seqs, heads_by_layer, offset, n_train=args.n_probe * 3 // 4
        )
        best_far = max(res["probe"][name][layer]["refit"][3] for layer in (1, 2, 3, 4))
        print(
            name,
            {k: v for k, v in res["acc"].items() if k.startswith(name + "/")},
            f"probe far {best_far:.3f}",
            flush=True,
        )

    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / f"seed{args.seed}.json").write_text(json.dumps(res))
    print("wrote", Path(args.out) / f"seed{args.seed}.json", f"({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
