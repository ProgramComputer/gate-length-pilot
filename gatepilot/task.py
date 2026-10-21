from __future__ import annotations

from collections.abc import Iterable

import torch

N_KEYS, N_VALS = 512, 64
BOS, Q = 0, 1
KEY0 = 2
VAL0 = KEY0 + N_KEYS
VOCAB = VAL0 + N_VALS

TRAIN_LEN = 128
EVAL_LENS = (128, 256, 512, 1024)
PLACEMENTS = ("uniform", "before", "after")

BOS_ROLE, KEY, VAL, QMARK, QUERY = range(5)
N_ROLES = 5


def n_records(length: int) -> int:
    return (length - 3) // 2


def to_tokens(
    keys: torch.Tensor,
    vals: torch.Tensor,
    q: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, n = keys.shape
    rows = torch.arange(B)
    seq = torch.empty(B, 2 * n + 3, dtype=torch.long)
    seq[:, 0] = BOS
    seq[:, 1:-2:2] = KEY0 + keys
    seq[:, 2:-2:2] = VAL0 + vals
    seq[:, -2] = Q
    seq[:, -1] = KEY0 + keys[rows, q]
    return seq, vals[rows, q]


def roles(seq: torch.Tensor) -> torch.Tensor:
    r = torch.full_like(seq, VAL)
    r[:, 0] = BOS_ROLE
    r[:, 1:-2:2] = KEY
    r[:, -2] = QMARK
    r[:, -1] = QUERY
    return r


def train_batch(
    batch_size: int,
    gen: torch.Generator,
    max_records: int = n_records(TRAIN_LEN),
) -> tuple[torch.Tensor, torch.Tensor]:
    n = int(torch.randint(4, max_records + 1, (1,), generator=gen))
    keys = torch.rand(batch_size, N_KEYS, generator=gen).argsort(1)[:, :n]
    vals = torch.randint(N_VALS, (batch_size, n), generator=gen)
    q = torch.randint(n, (batch_size,), generator=gen)
    return to_tokens(keys, vals, q)


def paired_set(
    n_examples: int,
    lengths: Iterable[int],
    placement: str,
    seed: int,
) -> dict[int, dict[str, torch.Tensor]]:
    gen = torch.Generator().manual_seed(seed)
    n0 = n_records(TRAIN_LEN)
    key_order = torch.rand(n_examples, N_KEYS, generator=gen).argsort(1)
    base_keys, spare_keys = key_order[:, :n0], key_order[:, n0:]
    base_vals = torch.randint(N_VALS, (n_examples, n0), generator=gen)
    q = torch.randint(n0, (n_examples,), generator=gen)
    lo = {"uniform": torch.zeros_like(q), "before": torch.zeros_like(q), "after": q + 1}[placement]
    hi = {"uniform": torch.full_like(q, n0), "before": q, "after": torch.full_like(q, n0)}[
        placement
    ]
    out: dict[int, dict[str, torch.Tensor]] = {}
    for L in lengths:
        m = n_records(L) - n0
        gaps = (
            lo[:, None] + (torch.rand(n_examples, m, generator=gen) * (hi - lo + 1)[:, None]).long()
        )
        sort_key = torch.cat(
            [torch.arange(n0).float().expand(n_examples, n0), gaps.float() - 0.5], 1
        )
        order = sort_key.argsort(dim=1, stable=True)
        keys = torch.cat([base_keys, spare_keys[:, :m]], 1).gather(1, order)
        vals = torch.cat(
            [base_vals, torch.randint(N_VALS, (n_examples, m), generator=gen)], 1
        ).gather(1, order)
        target = (order == q[:, None]).long().argmax(1)
        seq, answer = to_tokens(keys, vals, target)
        assert torch.equal(answer, base_vals[torch.arange(n_examples), q])
        out[L] = {"seq": seq, "answer": answer, "distance": keys.shape[1] - 1 - target}
    return out
