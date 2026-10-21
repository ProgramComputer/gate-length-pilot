from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .task import N_VALS, VOCAB


class NoIntervention:
    observing = False

    def begin(self, seq: torch.Tensor) -> None:
        pass

    def logit_scale(self, T: int) -> None:
        return None

    def __call__(self, site: str, layer: int, x: torch.Tensor) -> torch.Tensor:
        return x


def rotary(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    T, dh = x.shape[-2], x.shape[-1]
    freq = base ** (-torch.arange(0, dh, 2, device=x.device, dtype=torch.float32) / dh)
    angle = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * freq
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
        if self.rope:
            q, k = rotary(q), rotary(k)
        if logit_scale is not None:
            q = q * logit_scale.to(q.device).view(1, 1, T, 1)
        if iv.observing:
            w = (q[:, :, -1:] @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])).softmax(-1)[:, :, 0]
            iv.observe("attn_entropy", layer, -(w * w.clamp_min(1e-12).log()).sum(-1))
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
