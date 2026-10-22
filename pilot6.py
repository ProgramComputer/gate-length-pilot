from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from pilot5 import LENGTHS, N_VALS, PLACEMENTS, REPO, TRAIN_LEN, Hooks, NoPELlama, Task, bits

FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
)
PREAMBLE = (
    "There are some word pairs hidden inside a lot of irrelevant text. "
    "Find the pairs and remember them."
)
K_CHOICES = (16, 8, 4)


class FillerTask:
    def __init__(self, tok: Tokenizer, k: int) -> None:
        words = Task(tok)
        self.keys, self.values, self.nl, self.k = words.keys, words.values, words.nl, k
        self.filler = torch.tensor(tok.encode(FILLER * 600, add_special_tokens=False).ids)
        self.head = [1] + tok.encode(PREAMBLE, add_special_tokens=False).ids + [self.nl]
        self.fixed = len(self.head) + 1 + 3 * k + 2

    def build(self, a: int, b: int, keys: torch.Tensor, vals: torch.Tensor, q: int) -> torch.Tensor:
        recs = torch.stack([keys, vals, torch.full_like(keys, self.nl)], -1).flatten()
        parts = [
            torch.tensor(self.head),
            self.filler[:a],
            torch.tensor([self.nl]),
            recs,
            self.filler[a : a + b],
            torch.tensor([self.nl, int(keys[q])]),
        ]
        return torch.cat(parts)

    def value_positions(self, a: int) -> torch.Tensor:
        start = len(self.head) + a + 1
        return torch.tensor([start + 3 * i + 1 for i in range(self.k)])

    def paired_set(
        self,
        n: int,
        placement: str,
        seed: int,
        lengths: tuple[int, ...] = LENGTHS,
    ) -> dict[int, dict[str, torch.Tensor]]:
        g = torch.Generator().manual_seed(seed)
        budget = TRAIN_LEN - self.fixed
        out: dict[int, dict[str, list[Any]]] = {
            L: {"seq": [], "answer": [], "a": []} for L in lengths
        }
        for _ in range(n):
            keys = self.keys[torch.randperm(len(self.keys), generator=g)[: self.k]]
            vals = self.values[torch.randint(N_VALS, (self.k,), generator=g)]
            q = int(torch.randint(self.k, (1,), generator=g))
            a = int(budget * (0.1 + 0.8 * torch.rand(1, generator=g)))
            u = float(torch.rand(1, generator=g))
            for L in lengths:
                extra = L - TRAIN_LEN
                share = {"before": 1.0, "after": 0.0, "uniform": u}[placement]
                a2 = a + round(share * extra)
                b2 = budget - a + (extra - round(share * extra))
                seq = self.build(a2, b2, keys, vals, q)
                assert len(seq) == L
                out[L]["seq"].append(seq)
                out[L]["answer"].append(int(vals[q]))
                out[L]["a"].append(a2)
        return {
            L: {
                "seq": torch.stack(d["seq"]),
                "answer": torch.tensor(d["answer"]),
                "a": torch.tensor(d["a"]),
            }
            for L, d in out.items()
        }


@torch.no_grad()
def score(
    model: NoPELlama,
    task: FillerTask,
    data: dict[str, torch.Tensor],
    heads: dict[int, list[int]] | None = None,
    offset: int | None = None,
    batch: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = model.w["embed_tokens.weight"].device
    vals = task.values.to(dev)
    restricted: list[torch.Tensor]
    full: list[torch.Tensor]
    restricted, full = [], []
    step = 1 if offset is not None else batch
    for i in range(0, len(data["seq"]), step):
        seq = data["seq"][i : i + step].to(dev)
        hooks: Hooks | None = None
        if offset is not None:
            hooks = Hooks(patch_heads=heads, offset=offset)
            vp = task.value_positions(int(data["a"][i]))
            hooks.value_positions = vp[vp >= TRAIN_LEN]
        logits = model(seq, hooks)
        ans = data["answer"][i : i + step].to(dev)
        restricted.append((vals[logits[:, vals].argmax(-1)] == ans).cpu())
        full.append((logits.argmax(-1) == ans).cpu())
    return torch.cat(restricted), torch.cat(full)


@torch.no_grad()
def block_stats(
    model: NoPELlama,
    task: FillerTask,
    data: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = model.w["embed_tokens.weight"].device
    p = torch.zeros(model.n_layers, model.n_heads)
    m = torch.zeros_like(p)
    for i in range(len(data["seq"])):
        hooks = Hooks(positions=task.value_positions(int(data["a"][i])))
        model(data["seq"][i : i + 1].to(dev), hooks)
        for l in range(model.n_layers):
            p[l] += hooks.p_prev[l].mean(-1)
            m[l] += hooks.margin[l].mean(-1)
    return p / len(data["seq"]), m / len(data["seq"])


def stage_tune(model: NoPELlama, tok: Tokenizer, n: int) -> dict[str, Any]:
    res: dict[str, Any] = {}
    for k in K_CHOICES:
        task = FillerTask(tok, k)
        data = task.paired_set(n, "before", seed=60_000, lengths=(TRAIN_LEN,))[TRAIN_LEN]
        r, f = score(model, task, data)
        res[str(k)] = {"restricted": r.float().mean().item(), "full": f.float().mean().item()}
        print("k", k, res[str(k)], flush=True)
    passing = [k for k in K_CHOICES if res[str(k)]["restricted"] >= 0.70]
    res["chosen_k"] = max(passing) if passing else None
    return res


def stage_feasibility(model: NoPELlama, tok: Tokenizer, k: int, n: int) -> dict[str, Any]:
    task, res = FillerTask(tok, k), {"k": k}
    for i, pl in enumerate(PLACEMENTS):
        data = task.paired_set(n, pl, seed=40_000 + i)
        for L in LENGTHS:
            r, f = score(model, task, data[L])
            res[f"{pl}/{L}"] = {
                "restricted": r.float().mean().item(),
                "full": f.float().mean().item(),
                "correct": bits(r),
            }
            print(
                pl,
                L,
                round(r.float().mean().item(), 3),
                round(f.float().mean().item(), 3),
                flush=True,
            )
    return res


def stage_full(model: NoPELlama, tok: Tokenizer, k: int, n: int, n_cal: int) -> dict[str, Any]:
    task = FillerTask(tok, k)
    inside = task.paired_set(n_cal, "uniform", seed=20_000, lengths=(TRAIN_LEN,))[TRAIN_LEN]
    far = task.paired_set(n_cal, "before", seed=20_001, lengths=(8192,))[8192]
    p_in, m_in = block_stats(model, task, inside)
    p_far, m_far = block_stats(model, task, far)
    chosen = [
        (l, h) for l in range(model.n_layers) for h in range(model.n_heads) if p_in[l, h] >= 0.3
    ]
    if not chosen:
        flat = int(p_in.flatten().argmax())
        chosen = [(flat // model.n_heads, flat % model.n_heads)]
    heads: dict[int, list[int]] = {}
    for l, h in chosen:
        heads.setdefault(l, []).append(h)
    print("binding heads", chosen, flush=True)
    res: dict[str, Any] = {
        "k": k,
        "binding_heads": chosen,
        "p_inside": p_in.tolist(),
        "p_far": p_far.tolist(),
        "margin_inside": m_in.tolist(),
        "margin_far": m_far.tolist(),
        "acc": {},
        "correct": {},
    }
    for i, pl in enumerate(PLACEMENTS):
        data = task.paired_set(n, pl, seed=40_000 + i)
        for L in LENGTHS:
            for name, offset in (("none", None), ("patch", 1), ("sham", 4)):
                r, f = score(model, task, data[L], heads, offset)
                res["acc"][f"{name}/{pl}/{L}"] = {
                    "restricted": r.float().mean().item(),
                    "full": f.float().mean().item(),
                }
                res["correct"][f"{name}/{pl}/{L}"] = bits(r)
                print(name, pl, L, round(r.float().mean().item(), 3), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("tune", "feasibility", "full"), required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-tune", type=int, default=100)
    ap.add_argument("--n-cal", type=int, default=64)
    ap.add_argument("--out", default="results6")
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
    model = NoPELlama(hf_hub_download(REPO, "model.safetensors"), dev)
    out = Path(args.out)
    out.mkdir(exist_ok=True)
    if args.stage == "tune":
        res = stage_tune(model, tok, args.n_tune)
    else:
        k = json.loads((out / "tune.json").read_text())["chosen_k"]
        if k is None:
            raise SystemExit("no k reached 70% at 2048 on the tuning set: NOT TESTABLE")
        res = (
            stage_feasibility(model, tok, k, args.n)
            if args.stage == "feasibility"
            else stage_full(model, tok, k, args.n, args.n_cal)
        )
    (out / f"{args.stage}.json").write_text(json.dumps(res))
    print(f"wrote {out / (args.stage + '.json')} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
