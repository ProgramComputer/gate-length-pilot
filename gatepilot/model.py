from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .task import N_VALS, VOCAB


class NoIntervention:
    observing = False
    wants_qk = wants_logits = False

    def begin(self, seq: torch.Tensor) -> None:
        pass

    def logit_scale(self, T: int) -> None:
        return None

    def rope_edit(self, layer: int) -> None:
        return None

    def __call__(self, site: str, layer: int, x: torch.Tensor) -> torch.Tensor:
        return x


def rope_freqs(dh: int, base: float = 10000.0, device: torch.device | None = None) -> torch.Tensor:
    return base ** (-torch.arange(0, dh, 2, device=device, dtype=torch.float32) / dh)


def rotary(
    x: torch.Tensor,
    base: float = 10000.0,
    pos_scale: float = 1.0,
    frozen: torch.Tensor | None = None,
) -> torch.Tensor:
    T, dh = x.shape[-2], x.shape[-1]
    angle = (torch.arange(T, device=x.device, dtype=torch.float32) * pos_scale)[
        :, None
    ] * rope_freqs(dh, base, x.device)
    if frozen is not None:
        angle = angle * (~frozen.to(x.device)).float()[:, None, :]
    cos, sin = angle.cos(), angle.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, d: int, n_heads: int, rope: bool = False) -> None:
        super().__init__()
        self.n_heads, self.rope = n_heads, rope
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        layer: int,
        iv: Any,
        logit_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_heads, D // self.n_heads).permute(2, 0, 3, 1, 4)
        edit = iv.rope_edit(layer)
        if edit is not None and edit[0] is not None:
            q, k = q * edit[0].to(q.device)[:, None, :], k * edit[0].to(k.device)[:, None, :]
        if iv.wants_qk:
            iv.on_qk(layer, q, k)
        if self.rope:
            pos_scale, frozen = (edit[1], edit[2]) if edit is not None else (1.0, None)
            q, k = rotary(q, pos_scale=pos_scale, frozen=frozen), rotary(
                k, pos_scale=pos_scale, frozen=frozen
            )
        if logit_scale is not None:
            q = q * logit_scale.to(q.device).view(1, 1, T, 1)
        if iv.observing:
            w = (q[:, :, -1:] @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)[:, :, 0]
            iv.observe("attn_entropy", layer, -(w * w.clamp_min(1e-12).log()).sum(-1))
        if iv.wants_logits:
            iv.on_logits(layer, q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1]))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, d_ff: int, rope: bool = False) -> None:
        super().__init__()
        self.norm1, self.attn = nn.RMSNorm(d), Attention(d, n_heads, rope)
        self.norm2 = nn.RMSNorm(d)
        self.w_gate = nn.Linear(d, d_ff, bias=False)
        self.w_up = nn.Linear(d, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        layer: int,
        iv: Any,
        logit_scale: torch.Tensor | None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), layer, iv, logit_scale)
        x = iv("resid", layer, x)
        h = self.norm2(x)
        a = iv("gate", layer, self.w_gate(h))
        g = iv("gate_post", layer, F.silu(a))
        u = iv("up", layer, self.w_up(h))
        return x + self.w_down(g * u)


class Model(nn.Module):
    def __init__(
        self,
        d: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        pos: str = "nope",
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, d)
        self.blocks = nn.ModuleList(
            Block(d, n_heads, d_ff, rope=pos == "rope") for _ in range(n_layers)
        )
        self.norm = nn.RMSNorm(d)
        self.head = nn.Linear(d, N_VALS, bias=False)

    def forward(self, seq: torch.Tensor, iv: Any = NoIntervention()) -> torch.Tensor:
        iv.begin(seq)
        scale = iv.logit_scale(seq.shape[1])
        x = iv("stream", 0, self.embed(seq))
        for layer, block in enumerate(self.blocks):
            x = iv("stream", layer + 1, block(x, layer, iv, scale))
        return self.head(self.norm(x[:, -1]))
