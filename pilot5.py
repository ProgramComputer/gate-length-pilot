from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tokenizers import Tokenizer

REPO = "AntNLP/TinyLlama-NoPE-1.1B"
TRAIN_LEN = 2048
LENGTHS = (2048, 4096, 8192)
WINDOWS = ((512, 2048), (2048, 4096), (4096, 8192))
PLACEMENTS = ("before", "uniform", "after")
N_VALS = 512
PREAMBLE = "Each line below pairs a word with another word. Remember the pairs."


class NoPELlama(torch.nn.Module):
    def __init__(self, path: str, device: torch.device) -> None:
        super().__init__()
        w: dict[str, torch.Tensor] = {
            k.removeprefix("model."): v.to(device, torch.bfloat16)
            for k, v in load_file(path).items()
        }
        self.w, self.n_layers = w, sum(1 for k in w if k.endswith("self_attn.q_proj.weight"))
        self.n_heads, self.n_kv, self.dh = 32, 4, 64

    def rms(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + 1e-5)).to(x.dtype) * weight

    def forward(
        self,
        ids: torch.Tensor,
        hooks: Hooks | None = None,
        all_logits: bool = False,
    ) -> torch.Tensor:
        w, (B, T) = self.w, ids.shape
        x = F.embedding(ids, w["embed_tokens.weight"])
        for l in range(self.n_layers):
            p = f"layers.{l}."
            h = self.rms(x, w[p + "input_layernorm.weight"])
            q = (
                (h @ w[p + "self_attn.q_proj.weight"].T)
                .view(B, T, self.n_heads, self.dh)
                .transpose(1, 2)
            )
            k = (
                (h @ w[p + "self_attn.k_proj.weight"].T)
                .view(B, T, self.n_kv, self.dh)
                .transpose(1, 2)
            )
            v = (
                (h @ w[p + "self_attn.v_proj.weight"].T)
                .view(B, T, self.n_kv, self.dh)
                .transpose(1, 2)
            )
            rep = self.n_heads // self.n_kv
            k, v = k.repeat_interleave(rep, dim=1), v.repeat_interleave(rep, dim=1)
            if hooks is not None:
                hooks.measure(l, q, k)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            if hooks is not None:
                y = hooks.patch(l, y, v)
            x = x + y.transpose(1, 2).reshape(B, T, -1) @ w[p + "self_attn.o_proj.weight"].T
            h = self.rms(x, w[p + "post_attention_layernorm.weight"])
            g, u = h @ w[p + "mlp.gate_proj.weight"].T, h @ w[p + "mlp.up_proj.weight"].T
            x = x + (F.silu(g) * u) @ w[p + "mlp.down_proj.weight"].T
        x = self.rms(x, w["norm.weight"])
        return (x if all_logits else x[:, -1]) @ w["lm_head.weight"].T


class Hooks:
    def __init__(
        self,
        positions: torch.Tensor | None = None,
        patch_heads: dict[int, list[int]] | None = None,
        offset: int | None = None,
        patch_from: int = TRAIN_LEN,
    ) -> None:
        self.positions = positions
        self.patch_heads = patch_heads or {}
        self.offset, self.patch_from = offset, patch_from
        self.value_positions = None
        self.p_prev, self.margin = {}, {}

    def measure(self, layer: int, q: torch.Tensor, k: torch.Tensor) -> None:
        if self.positions is None:
            return
        pos = self.positions.to(q.device)
        s = (q[:, :, pos] @ k.transpose(-1, -2)).float() / math.sqrt(q.shape[-1])
        T = k.shape[2]
        future = torch.arange(T, device=q.device)[None, :] > pos[:, None]
        s = s.masked_fill(future, float("-inf"))
        prev = s.gather(-1, (pos - 1).view(1, 1, -1, 1).expand(*s.shape[:3], 1))[..., 0]
        self.p_prev[layer] = (
            (s.softmax(-1).gather(-1, (pos - 1).view(1, 1, -1, 1).expand(*s.shape[:3], 1))[..., 0])
            .mean(0)
            .cpu()
        )
        others = s.masked_fill(future, 0.0)
        n_other = pos.float()
        mean_other = (others.sum(-1) - prev) / n_other.view(1, 1, -1)
        self.margin[layer] = (prev - mean_other).mean(0).cpu()

    def patch(self, layer: int, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        heads = self.patch_heads.get(layer)
        if (
            self.offset is None
            or not heads
            or self.value_positions is None
            or len(self.value_positions) == 0
        ):
            return y
        h = torch.tensor(heads, device=y.device)[:, None]
        pos = self.value_positions.to(y.device)[None, :]
        y = y.clone()
        y[:, h, pos] = v[:, h, pos - self.offset]
        return y


class Task:
    def __init__(self, tok: Tokenizer) -> None:
        vocab = tok.get_vocab()
        words = sorted(i for t, i in vocab.items() if re.fullmatch("▁[a-z]{4,10}", t))
        g = torch.Generator().manual_seed(0)
        words = [words[i] for i in torch.randperm(len(words), generator=g)]
        self.values, self.keys = torch.tensor(words[:N_VALS]), torch.tensor(words[N_VALS:])
        self.nl = tok.token_to_id("<0x0A>")
        self.head = [1] + tok.encode(PREAMBLE, add_special_tokens=False).ids + [self.nl]
        assert (
            len(self.keys) > self.n_records(max(LENGTHS)) + 10
        ), "not enough single-token key words"

    def n_records(self, L: int) -> int:
        return (L - len(self.head) - 1) // 3

    def value_pos(self, i: int) -> int:
        return len(self.head) + 3 * i + 1

    def build(self, keys: torch.Tensor, vals: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        recs = torch.stack([keys, vals, torch.full_like(keys, self.nl)], -1).flatten(-2)
        head = torch.tensor(self.head).expand(len(keys), -1)
        return torch.cat([head, recs, keys.gather(1, target[:, None])], 1)

    def paired_set(self, n: int, placement: str, seed: int) -> dict[int, dict[str, torch.Tensor]]:
        g = torch.Generator().manual_seed(seed)
        n0 = self.n_records(TRAIN_LEN)
        order = torch.rand(n, len(self.keys), generator=g).argsort(1)
        base_k, spare_k = self.keys[order[:, :n0]], self.keys[order[:, n0:]]
        base_v = self.values[torch.randint(N_VALS, (n, n0), generator=g)]
        q = torch.randint(n0, (n,), generator=g)
        lo = {"uniform": torch.zeros_like(q), "before": torch.zeros_like(q), "after": q + 1}[
            placement
        ]
        hi = {"uniform": torch.full_like(q, n0), "before": q, "after": torch.full_like(q, n0)}[
            placement
        ]
        out: dict[int, dict[str, torch.Tensor]] = {}
        for L in LENGTHS:
            m = self.n_records(L) - n0
            gaps = lo[:, None] + (torch.rand(n, m, generator=g) * (hi - lo + 1)[:, None]).long()
            key = torch.cat(
                [torch.arange(n0).float().expand(n, n0), gaps.float() - 0.5], 1
            ).argsort(dim=1, stable=True)
            ks = torch.cat([base_k, spare_k[:, :m]], 1).gather(1, key)
            vs = torch.cat(
                [base_v, self.values[torch.randint(N_VALS, (n, m), generator=g)]], 1
            ).gather(1, key)
            target = (key == q[:, None]).long().argmax(1)
            answer = vs.gather(1, target[:, None])[:, 0]
            assert torch.equal(answer, base_v.gather(1, q[:, None])[:, 0])
            out[L] = {"seq": self.build(ks, vs, target), "answer": answer, "target": target}
        return out


@torch.no_grad()
def accuracy(
    model: NoPELlama,
    task: Task,
    data: dict[str, torch.Tensor],
    hooks_fn: Callable[[torch.Tensor], Hooks] | None = None,
    batch: int = 4,
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = model.w["embed_tokens.weight"].device
    restricted: list[torch.Tensor]
    full: list[torch.Tensor]
    restricted, full = [], []
    vals = task.values.to(dev)
    for i in range(0, len(data["seq"]), batch):
        seq = data["seq"][i : i + batch].to(dev)
        hooks = hooks_fn(seq) if hooks_fn else None
        logits = model(seq, hooks)
        ans = data["answer"][i : i + batch].to(dev)
        restricted.append((vals[logits[:, vals].argmax(-1)] == ans).cpu())
        full.append((logits.argmax(-1) == ans).cpu())
    return torch.cat(restricted), torch.cat(full)


def bits(t: torch.Tensor) -> str:
    return "".join("1" if b else "0" for b in t.tolist())


def stage_verify(model: NoPELlama, tok: Tokenizer, path: str) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    text = (
        "The quick brown fox jumps over the lazy dog. Machine learning models are trained on large "
        "amounts of text, and they learn to predict the next word in a sentence from the words "
        "that come before it."
    )
    ids = torch.tensor([[1] + tok.encode(text, add_special_tokens=False).ids])
    dev = model.w["embed_tokens.weight"].device
    logits = model(ids.to(dev), all_logits=True)[0].float()
    out["lm_loss_nats"] = F.cross_entropy(logits[:-1], ids[0, 1:].to(dev)).item()
    try:
        from transformers import LlamaForCausalLM
        from transformers.models.llama import modeling_llama

        modeling_llama.apply_rotary_pos_emb = lambda q, k, *a, **kw: (q, k)
        ref = LlamaForCausalLM.from_pretrained(REPO, torch_dtype=torch.float32).to(dev).eval()
        ref_logits = ref(ids.to(dev)).logits[0].float()
        out["max_abs_logit_diff_vs_reference"] = (ref_logits - logits).abs().max().item()
        out["argmax_agreement_vs_reference"] = (
            (ref_logits.argmax(-1) == logits.argmax(-1)).float().mean().item()
        )
        del ref
    except Exception as e:
        out["reference_error"] = repr(e)[:300]
    return out


def stage_feasibility(model: NoPELlama, task: Task, n: int) -> dict[str, dict[str, float | str]]:
    res: dict[str, dict[str, float | str]] = {}
    for i, pl in enumerate(PLACEMENTS):
        data = task.paired_set(n, pl, seed=40_000 + i)
        for L in LENGTHS:
            r, f = accuracy(model, task, data[L])
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


@torch.no_grad()
def window_stats(
    model: NoPELlama,
    task: Task,
    seqs: torch.Tensor,
    per_window: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(2)
    n_rec = task.n_records(seqs.shape[1] + 1)
    vpos = torch.tensor([task.value_pos(i) for i in range(n_rec)])
    p_sum = torch.zeros(model.n_layers, model.n_heads, len(WINDOWS))
    m_sum = torch.zeros_like(p_sum)
    dev = model.w["embed_tokens.weight"].device
    for s in seqs:
        pos, win = [], []
        for b, (lo, hi) in enumerate(WINDOWS):
            cand = vpos[(vpos >= lo) & (vpos < hi)]
            pick = cand[torch.randperm(len(cand), generator=gen)[:per_window]]
            pos.append(pick)
            win.append(torch.full_like(pick, b))
        pos, win = torch.cat(pos), torch.cat(win)
        hooks = Hooks(positions=pos)
        model(s[None].to(dev), hooks)
        for l in range(model.n_layers):
            for b in range(len(WINDOWS)):
                p_sum[l, :, b] += hooks.p_prev[l][:, win == b].mean(-1)
                m_sum[l, :, b] += hooks.margin[l][:, win == b].mean(-1)
    return p_sum / len(seqs), m_sum / len(seqs)


def stage_full(model: NoPELlama, task: Task, n: int, n_cal: int) -> dict[str, Any]:
    cal = task.paired_set(n_cal, "uniform", seed=20_000)[8192]["seq"]
    p, m = window_stats(model, task, cal)
    inside = p[:, :, 0]
    chosen = [(l, h) for l in range(p.shape[0]) for h in range(p.shape[1]) if inside[l, h] >= 0.3]
    if not chosen:
        flat = int(inside.flatten().argmax())
        chosen = [(flat // p.shape[1], flat % p.shape[1])]
    heads: dict[int, list[int]] = {}
    for l, h in chosen:
        heads.setdefault(l, []).append(h)
    print("binding heads", chosen, flush=True)
    res: dict[str, Any] = {
        "binding_heads": chosen,
        "p_prev": p.tolist(),
        "margin": m.tolist(),
        "acc": {},
        "correct": {},
    }
    vpos_all = {
        L: torch.tensor([task.value_pos(i) for i in range(task.n_records(L))]) for L in LENGTHS
    }
    for i, pl in enumerate(PLACEMENTS):
        data = task.paired_set(n, pl, seed=40_000 + i)
        for L in LENGTHS:
            for name, offset in (("none", None), ("patch", 1), ("sham", 4)):

                def hooks_fn(seq: torch.Tensor, offset: int | None = offset, L: int = L) -> Hooks:
                    hk = Hooks(patch_heads=heads, offset=offset)
                    vp = vpos_all[L]
                    hk.value_positions = vp[vp >= TRAIN_LEN]
                    return hk

                r, f = accuracy(model, task, data[L], hooks_fn)
                res["acc"][f"{name}/{pl}/{L}"] = {
                    "restricted": r.float().mean().item(),
                    "full": f.float().mean().item(),
                }
                res["correct"][f"{name}/{pl}/{L}"] = bits(r)
                print(name, pl, L, round(r.float().mean().item(), 3), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("verify", "feasibility", "full"), required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--n-cal", type=int, default=32)
    ap.add_argument("--out", default="results5")
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    path = hf_hub_download(REPO, "model.safetensors")
    tok = Tokenizer.from_file(hf_hub_download(REPO, "tokenizer.json"))
    model = NoPELlama(path, dev)
    task = Task(tok)
    print(
        f"{len(task.keys)} key words, {len(task.values)} value words, "
        f"preamble {len(task.head)} tokens",
        flush=True,
    )
    res = {
        "verify": lambda: stage_verify(model, tok, path),
        "feasibility": lambda: stage_feasibility(model, task, args.n),
        "full": lambda: stage_full(model, task, args.n, args.n_cal),
    }[args.stage]()
    Path(args.out).mkdir(exist_ok=True)
    (Path(args.out) / f"{args.stage}.json").write_text(json.dumps(res))
    print(
        json.dumps({k: v for k, v in res.items() if k not in ("correct", "p_prev", "margin")})[
            :2000
        ]
    )
    print(f"wrote {Path(args.out) / (args.stage + '.json')} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
