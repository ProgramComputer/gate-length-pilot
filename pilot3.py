from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

from gatepilot.correct import Intervention, evaluate
from gatepilot.model import Model, rope_freqs
from gatepilot.task import TRAIN_LEN, n_records, paired_set

HIGH = 6
N_DISTRACT = 16
CONDITIONS = {
    "baseline": None,
    "sem_a0.5": 0.5,
    "sem_a0.25": 0.25,
    "sem_a0": 0.0,
    "sem_freeze": "freeze",
    "pos_a2": 2.0,
    "pos_a4": 4.0,
    "PI": "PI",
}


class Hooks(Intervention):
    def __init__(
        self,
        edits: Mapping[int, tuple[torch.Tensor | None, float, torch.Tensor | None]] | None = None,
    ) -> None:
        super().__init__()
        self.edits, self.logits, self.qk = edits or {}, {}, {}

    def rope_edit(
        self,
        layer: int,
    ) -> tuple[torch.Tensor | None, float, torch.Tensor | None] | None:
        return self.edits.get(layer)

    def on_logits(self, layer: int, s: torch.Tensor) -> None:
        self.logits[layer] = s

    def on_qk(self, layer: int, q: torch.Tensor, k: torch.Tensor) -> None:
        self.qk[layer] = (q, k)


def load(runs: str, seed: int, device: torch.device) -> Model:
    ckpt = torch.load(Path(runs) / f"seed{seed}" / "model.pt", map_location="cpu")
    model = Model(pos=ckpt["args"].get("pos", "nope") if "args" in ckpt else "nope")
    model.load_state_dict(ckpt["state"])
    return model.eval().to(device)


def record_positions(
    data: dict[str, torch.Tensor],
    L: int,
    idx: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if idx is None:
        idx = n_records(L) - 1 - data["distance"]
    key = 1 + 2 * idx
    return key, key + 1


@torch.no_grad()
def pick_heads(
    model: Model,
    data: dict[str, torch.Tensor],
    n: int = 2,
    batch: int = 64,
) -> list[tuple[int, int, str, float]]:
    dev = next(model.parameters()).device
    key_pos, val_pos = record_positions(data, TRAIN_LEN)
    totals = torch.zeros(len(model.blocks), model.blocks[0].attn.n_heads, 2)
    for i in range(0, len(data["seq"]), batch):
        hooks = Hooks()
        hooks.wants_logits = True
        model(data["seq"][i : i + batch].to(dev), hooks)
        rows = torch.arange(len(data["seq"][i : i + batch]))
        for layer, s in hooks.logits.items():
            w = s[:, :, -1].softmax(-1).cpu()
            totals[layer, :, 0] += w[rows, :, key_pos[i : i + batch]].sum(0)
            totals[layer, :, 1] += w[rows, :, val_pos[i : i + batch]].sum(0)
    totals /= len(data["seq"])
    order = totals.flatten().argsort(descending=True)[:n]
    H = totals.shape[1]
    return [
        (
            int(o // (2 * H)),
            int(o // 2 % H),
            ("key", "value")[int(o % 2)],
            round(totals.flatten()[o].item(), 4),
        )
        for o in order
    ]


@torch.no_grad()
def diagnostics(
    model: Model,
    data: dict[str, torch.Tensor],
    L: int,
    heads: list[tuple[int, int, str, float]],
    batch: int = 64,
) -> dict[str, float]:
    dev = next(model.parameters()).device
    dh = model.blocks[0].attn.qkv.out_features // 3 // model.blocks[0].attn.n_heads
    phase = torch.exp(1j * torch.arange(L, dtype=torch.float64)[:, None] * rope_freqs(dh).double())
    gen = torch.Generator().manual_seed(1)
    n = n_records(L)
    gauss: list[torch.Tensor]
    truth: list[torch.Tensor]
    positional: list[torch.Tensor]
    gauss, truth, positional = [], [], []
    for i in range(0, len(data["seq"]), batch):
        seq = data["seq"][i : i + batch]
        hooks = Hooks()
        hooks.wants_qk = True
        model(seq.to(dev), hooks)
        target = n - 1 - data["distance"][i : i + batch]
        others = torch.randint(n - 1, (len(seq), N_DISTRACT), generator=gen)
        others = others + (others >= target[:, None]).long()
        rows = torch.arange(len(seq))
        for layer, head, token, _ in heads:
            q, k = (t[:, head].double().cpu() for t in hooks.qk[layer])
            shift = 0 if token == "key" else 1
            k_plus = k[rows, 1 + 2 * target + shift]
            k_minus = k[rows[:, None], 1 + 2 * others + shift]
            complex_ = lambda x: torch.complex(x[..., 0::2], x[..., 1::2])
            Q = complex_(q[:, -1])
            z_plus = Q * complex_(k_plus).conj() / math.sqrt(dh)
            z_minus = Q[:, None] * complex_(k_minus).conj() / math.sqrt(dh)
            D = ((z_plus[:, None] - z_minus) @ phase.T).real
            sign = torch.sign(D[..., :1])
            Do = D * sign
            p_rev = torch.special.ndtr(-Do.mean(-1) / Do.std(-1).clamp_min(1e-12))
            gauss.append(p_rev[sign[..., 0] != 0])
            truth.append((D < 0).double().mean(-1).flatten())
            S = (z_plus @ phase.T).real
            positional.append((S.diff(dim=-1).abs().mean(-1) / z_plus.abs().pow(2).sum(-1).sqrt()))
    return {
        "semantic_score": round(1 - torch.cat(gauss).mean().item(), 4),
        "reversal_true": round(torch.cat(truth).mean().item(), 4),
        "positional_score": round(torch.cat(positional).mean().item(), 5),
    }


def edits_for(
    name: str,
    heads: list[tuple[int, int, str, float]],
    L: int,
    n_layers: int,
    n_heads: int,
    dh: int,
) -> Mapping[int, tuple[torch.Tensor | None, float, torch.Tensor | None]]:
    value = CONDITIONS[name]
    if value is None:
        return {}
    if value == "PI":
        return {layer: (None, TRAIN_LEN / L, None) for layer in range(n_layers)}
    edits: dict[int, tuple[torch.Tensor, float, torch.Tensor]] = {}
    for layer, head, _, _ in heads:
        scale, _, frozen = edits.get(
            layer, (torch.ones(n_heads, dh), 1.0, torch.zeros(n_heads, dh // 2, dtype=torch.bool))
        )
        if value == "freeze":
            frozen[head, :HIGH] = True
        else:
            scale[head, : 2 * HIGH] = value**0.5
        edits[layer] = (scale, 1.0, frozen)
    return edits


def part_a(args: argparse.Namespace, dev: torch.device) -> tuple[str, dict[str, Any]]:
    model = load(args.runs, args.seed, dev)
    n_layers, n_heads = len(model.blocks), model.blocks[0].attn.n_heads
    dh = model.blocks[0].attn.qkv.out_features // 3 // n_heads
    cal = paired_set(256, (128, 1024), "uniform", seed=20_000)
    heads = pick_heads(model, cal[128])
    res = {"seed": args.seed, "heads": heads, "diagnostics": {}, "acc": {}, "correct": {}}
    for L in (128, 1024):
        res["diagnostics"][str(L)] = diagnostics(model, cal[L], L, heads)
    print("heads", heads, "diagnostics", res["diagnostics"], flush=True)
    test = {
        pl: paired_set(args.n_test, (128, 1024), pl, seed=40_000 + i)
        for i, pl in enumerate(("uniform", "before", "after"))
    }
    for name in CONDITIONS:
        for pl, L in (("uniform", 128), ("uniform", 1024), ("before", 1024), ("after", 1024)):
            correct, _ = evaluate(
                model, test[pl][L], Hooks(edits_for(name, heads, L, n_layers, n_heads, dh))
            )
            res["acc"][f"{name}/{pl}/{L}"] = round(correct.float().mean().item(), 4)
            res["correct"][f"{name}/{pl}/{L}"] = "".join(
                "1" if c else "0" for c in correct.tolist()
            )
        print(name, {k: v for k, v in res["acc"].items() if k.startswith(name + "/")}, flush=True)
    return f"a_seed{args.seed}.json", res


REGIONS = {"inside": (1, 128), "far": (512, 640)}


def backfit(
    W: torch.Tensor,
    d: torch.Tensor,
    ii: torch.Tensor,
    jj: torch.Tensor,
    a_fixed: torch.Tensor | None = None,
    iters: int = 30,
) -> tuple[torch.Tensor, float]:
    def group_mean(x: torch.Tensor, idx: torch.Tensor, size: int) -> torch.Tensor:
        return torch.zeros(size, dtype=x.dtype, device=x.device).index_add_(
            0, idx, x
        ) / torch.zeros(size, dtype=x.dtype, device=x.device).index_add_(
            0, idx, torch.ones_like(x)
        ).clamp_min(
            1
        )

    a = a_fixed if a_fixed is not None else torch.zeros(128, dtype=W.dtype, device=W.device)
    b = torch.zeros(int(ii.max()) + 1, dtype=W.dtype, device=W.device)
    c = torch.zeros(int(jj.max()) + 1, dtype=W.dtype, device=W.device)
    for _ in range(iters):
        if a_fixed is None:
            a = group_mean(W - b[ii] - c[jj], d, 128)
        b = group_mean(W - a[d] - c[jj], ii, len(b))
        c = group_mean(W - a[d] - b[ii], jj, len(c))
    fit = a[d] + b[ii] + c[jj]
    return a, torch.corrcoef(torch.stack([W, fit]))[0, 1].item()


def detrend(a: torch.Tensor) -> torch.Tensor:
    x = torch.arange(len(a), dtype=a.dtype, device=a.device)
    A = torch.stack([torch.ones_like(x), x], 1)
    return a - A @ torch.linalg.lstsq(A, a[:, None]).solution[:, 0]


class LogitFits(Hooks):
    wants_logits = True

    def __init__(self) -> None:
        super().__init__()
        self.out = {}

    def on_logits(self, layer: int, s: torch.Tensor) -> None:
        s = s[0].double()
        idx: dict[
            str, tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ] = {}
        for name, (lo, hi) in REGIONS.items():
            i = torch.arange(lo, hi, device=s.device)[:, None].expand(-1, 128)
            j = i - torch.arange(128, device=s.device)[None, :]
            keep = j >= 0
            i, j = i[keep], j[keep]
            idx[name] = (i, j, i - j, i - lo, j - j.min())
        for head in range(s.shape[0]):
            r: dict[str, float | torch.Tensor] = {}
            for name, (i, j, d, ii, jj) in idx.items():
                a, r[f"corr_{name}"] = backfit(s[head, i, j], d, ii, jj)
                a = detrend(a)
                r[f"a_{name}"] = a
                r[f"share_{name}"] = (a[d].var() / s[head, i, j].var()).item()
            i, j, d, ii, jj = idx["far"]
            _, r["corr_transfer"] = backfit(s[head, i, j], d, ii, jj, a_fixed=r["a_inside"])
            r["rho_a"] = torch.corrcoef(torch.stack([r["a_inside"], r["a_far"]]))[0, 1].item()
            self.out.setdefault((layer, head), []).append(r)


@torch.no_grad()
def part_b(args: argparse.Namespace, dev: torch.device) -> tuple[str, dict[str, Any]]:
    model = load(args.runs, args.seed, dev)
    pos = "rope" if model.blocks[0].attn.rope else "nope"
    seqs = paired_set(args.n_fit, (1024,), "uniform", seed=50_000)[1024]["seq"]
    fits = LogitFits()
    for s in seqs:
        model(s[None].to(dev), fits)
    res = {"seed": args.seed, "pos": pos, "heads": {}}
    for (layer, head), rs in fits.out.items():
        res["heads"][f"{layer}.{head}"] = {
            **{
                k: round(sum(r[k] for r in rs) / len(rs), 4)
                for k in (
                    "corr_inside",
                    "corr_far",
                    "corr_transfer",
                    "rho_a",
                    "share_inside",
                    "share_far",
                )
            },
            "a_inside": [
                round(v, 4) for v in torch.stack([r["a_inside"] for r in rs]).mean(0).tolist()
            ],
            "a_far": [round(v, 4) for v in torch.stack([r["a_far"] for r in rs]).mean(0).tolist()],
        }
    best = max(res["heads"], key=lambda h: res["heads"][h]["share_inside"])
    res["positional_head"] = best
    print(
        pos,
        args.seed,
        "positional head",
        best,
        {k: v for k, v in res["heads"][best].items() if not k.startswith("a_")},
        flush=True,
    )
    return f"b_{pos}_seed{args.seed}.json", res


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("part", choices=("a", "b"))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--runs", default="runs_rope")
    p.add_argument("--out", default="results3")
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--n-fit", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    t0 = time.time()
    name, res = (part_a if args.part == "a" else part_b)(args, torch.device(args.device))
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / name).write_text(json.dumps(res))
    print("wrote", Path(args.out) / name, f"({time.time() - t0:.0f}s)")
