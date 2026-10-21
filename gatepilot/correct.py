from __future__ import annotations

import math
from collections import defaultdict

import torch

from .task import BOS_ROLE, KEY, N_ROLES, QMARK, QUERY, TRAIN_LEN, VAL, roles

CONDITIONS = ("none", "gate", "attn", "gate+attn", "up", "resid", "sham")
PER_EXAMPLE = 16


def log_length_scale(T: int) -> torch.Tensor:
    t = torch.arange(T, dtype=torch.float32)
    return torch.clamp(torch.log(t + 1) / math.log(TRAIN_LEN), min=1.0)


class Intervention:
    wants_qk = wants_logits = False

    def rope_edit(
        self,
        layer: int,
    ) -> tuple[torch.Tensor | None, float, torch.Tensor | None] | None:
        return None

    def __init__(self, attn_temp: bool = False, uniform_temp: float | None = None) -> None:
        self.attn_temp = attn_temp
        self.uniform_temp = uniform_temp
        self.affine = {}
        self.record = set()
        self.record_roles = ()
        self.record_from = None
        self.samples = defaultdict(list)
        self.observing = False
        self.observed = defaultdict(list)
        self.size = {}
        self.gen = torch.Generator().manual_seed(0)

    def observe(self, name: str, layer: int, value: torch.Tensor) -> None:
        self.observed[name, layer].append(value.float().cpu())

    def logit_scale(self, T: int) -> torch.Tensor | None:
        if self.uniform_temp is not None:
            return torch.full((T,), float(self.uniform_temp))
        return log_length_scale(T) if self.attn_temp else None

    def begin(self, seq: torch.Tensor) -> None:
        r, pos = roles(seq), torch.arange(seq.shape[1], device=seq.device)
        self.fix_role = torch.where(pos >= TRAIN_LEN, r, BOS_ROLE)
        if self.record:
            self.rec_role = torch.where(pos >= self.record_from, r, BOS_ROLE)

    def __call__(self, site: str, layer: int, x: torch.Tensor) -> torch.Tensor:
        if (site, layer) in self.affine:
            scale, shift = self.affine[site, layer]
            x = x * scale[self.fix_role] + shift[self.fix_role]
        if (site, layer) in self.record:
            for role in self.record_roles:
                rows = x[self.rec_role == role]
                keep = PER_EXAMPLE * x.shape[0]
                if len(rows) > keep:
                    rows = rows[
                        torch.randperm(len(rows), generator=self.gen)[:keep].to(rows.device)
                    ]
                self.samples[site, layer, role].append(rows.float().cpu())
        return x


@torch.no_grad()
def collect(
    model: torch.nn.Module,
    seqs: torch.Tensor,
    iv: Intervention,
    pairs: list[tuple[str, int]],
    record_from: int,
    record_roles: tuple[int, ...] = (KEY, VAL, QMARK, QUERY),
    batch: int = 64,
) -> dict[tuple[str, int, int], torch.Tensor]:
    dev = next(model.parameters()).device
    iv.record, iv.record_roles, iv.record_from, iv.observing = (
        set(pairs),
        record_roles,
        record_from,
        True,
    )
    for i in range(0, len(seqs), batch):
        model(seqs[i : i + batch].to(dev), iv)
    out = {k: torch.cat(v) for k, v in iv.samples.items()}
    out.update({(name, layer, QUERY): torch.cat(v) for (name, layer), v in iv.observed.items()})
    iv.record, iv.samples, iv.observing, iv.observed = (
        set(),
        defaultdict(list),
        False,
        defaultdict(list),
    )
    return out


def moments(
    samples: dict[tuple[str, int, int], torch.Tensor],
    site: str,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    C = samples[site, layer, QUERY].shape[1]
    mu, sd = torch.zeros(N_ROLES, C), torch.ones(N_ROLES, C)
    for role in (KEY, VAL, QMARK, QUERY):
        x = samples[site, layer, role]
        mu[role], sd[role] = x.mean(0), x.std(0) + 1e-6
    return mu, sd


def fit_correction(
    model: torch.nn.Module,
    site: str,
    short: torch.Tensor,
    long: torch.Tensor,
    attn_temp: bool = False,
    sham_seed: int | None = None,
) -> Intervention:
    dev = next(model.parameters()).device
    n_layers = len(model.blocks)
    iv = Intervention(attn_temp)
    short_samples = collect(
        model, short, Intervention(), [(site, l) for l in range(n_layers)], TRAIN_LEN // 2
    )
    for layer in range(n_layers):
        long_samples = collect(model, long, iv, [(site, layer)], TRAIN_LEN)
        mu_s, sd_s = moments(short_samples, site, layer)
        mu_l, sd_l = moments(long_samples, site, layer)
        shift = (mu_s - mu_l) / sd_l
        ratio = sd_s / sd_l
        if sham_seed is not None:
            perm = torch.randperm(
                ratio.shape[1], generator=torch.Generator().manual_seed(sham_seed + layer)
            )
            shift, ratio = shift[:, perm], ratio[:, perm]
        iv.affine[site, layer] = (ratio.to(dev), (mu_l * (1 - ratio) + sd_l * shift).to(dev))
        iv.size[layer] = (shift[QUERY] ** 2 + (ratio[QUERY] - 1) ** 2).sqrt().mean().item()
    return iv


def make_condition(
    name: str,
    model: torch.nn.Module,
    short: torch.Tensor | None,
    long: torch.Tensor | None,
    length: int,
) -> Intervention:
    attn = name in ("attn", "gate+attn")
    if name in ("none", "attn") or length <= TRAIN_LEN:
        return Intervention(attn_temp=attn)
    site = {"gate": "gate", "gate+attn": "gate", "up": "up", "resid": "resid", "sham": "gate"}[name]
    return fit_correction(
        model, site, short, long, attn_temp=attn, sham_seed=1 if name == "sham" else None
    )


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    data: dict[str, torch.Tensor],
    iv: Intervention,
    batch: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = next(model.parameters()).device
    correct: list[torch.Tensor]
    logp: list[torch.Tensor]
    correct, logp = [], []
    for i in range(0, len(data["seq"]), batch):
        lp = model(data["seq"][i : i + batch].to(dev), iv).log_softmax(-1).cpu()
        ans = data["answer"][i : i + batch]
        correct.append(lp.argmax(-1) == ans)
        logp.append(lp.gather(1, ans[:, None])[:, 0])
    return torch.cat(correct), torch.cat(logp)


def discrepancy(
    short: dict[tuple[str, int, int], torch.Tensor],
    long: dict[tuple[str, int, int], torch.Tensor],
    site: str,
    layer: int,
    role: int = QUERY,
) -> torch.Tensor:
    xs, xl = short[site, layer, role], long[site, layer, role]
    sd_s = xs.std(0) + 1e-6
    return ((xl.mean(0) - xs.mean(0)) ** 2 + (xl.std(0) - xs.std(0)) ** 2).sqrt() / sd_s
