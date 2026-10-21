from __future__ import annotations

import torch
import torch.nn.functional as F

from .correct import Intervention
from .task import KEY0, N_KEYS

BUCKETS = ((64, 128), (128, 256), (256, 512), (512, 1024))
PER_BUCKET = 16


class StreamRecorder(Intervention):
    def __init__(self, uniform_temp: float | None = None) -> None:
        super().__init__(uniform_temp=uniform_temp)
        self.pick = torch.Generator().manual_seed(0)
        self.rows, self.labels, self.buckets, self.examples = {}, [], [], []
        self.n_seen = 0

    def begin(self, seq: torch.Tensor) -> None:
        super().begin(seq)
        B, T = seq.shape
        pos: list[torch.Tensor]
        bucket: list[torch.Tensor]
        pos, bucket = [], []
        for b, (lo, hi) in enumerate(BUCKETS):
            candidates = torch.arange(lo, min(hi, T - 2), 2)
            choice = torch.rand(B, len(candidates), generator=self.pick).argsort(1)[:, :PER_BUCKET]
            pos.append(candidates[choice])
            bucket.append(torch.full_like(choice, b))
        self.pos = torch.cat(pos, 1)
        self.batch_rows = torch.arange(B)[:, None].expand_as(self.pos)
        self.labels.append((seq.cpu()[self.batch_rows, self.pos - 1] - KEY0).flatten())
        self.buckets.append(torch.cat(bucket, 1).flatten())
        self.examples.append((self.n_seen + self.batch_rows).flatten())
        self.n_seen += B

    def __call__(self, site: str, layer: int, x: torch.Tensor) -> torch.Tensor:
        x = super().__call__(site, layer, x)
        if site == "stream":
            rows = x[self.batch_rows.to(x.device), self.pos.to(x.device)]
            self.rows.setdefault(layer, []).append(rows.reshape(-1, x.shape[-1]).float().cpu())
        return x


@torch.no_grad()
def record_stream(
    model: torch.nn.Module,
    seqs: torch.Tensor,
    uniform_temp: float | None = None,
    batch: int = 64,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    dev = next(model.parameters()).device
    rec = StreamRecorder(uniform_temp)
    for i in range(0, len(seqs), batch):
        model(seqs[i : i + batch].to(dev), rec)
    X = {layer: torch.cat(v) for layer, v in rec.rows.items()}
    return X, torch.cat(rec.labels), torch.cat(rec.buckets), torch.cat(rec.examples)


def ridge_accuracy(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_test: torch.Tensor,
    y_test: torch.Tensor,
    alpha: float = 1.0,
) -> float:
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-6
    design = lambda X: torch.cat([(X - mu) / sd, torch.ones(len(X), 1)], 1).double()
    A = design(X_train)
    W = torch.linalg.solve(
        A.T @ A + alpha * torch.eye(A.shape[1], dtype=A.dtype),
        A.T @ F.one_hot(y_train, N_KEYS).double(),
    )
    return ((design(X_test) @ W).argmax(1) == y_test).float().mean().item()


def probe_accuracies(
    X: dict[int, torch.Tensor],
    labels: torch.Tensor,
    buckets: torch.Tensor,
    examples: torch.Tensor,
    n_train: int,
) -> dict[int, dict[str, list[float]]]:
    train, test = examples < n_train, examples >= n_train
    out: dict[int, dict[str, list[float]]] = {}
    for layer, x in X.items():
        fit_in = train & (buckets == 0)
        transfer = [
            ridge_accuracy(
                x[fit_in], labels[fit_in], x[test & (buckets == b)], labels[test & (buckets == b)]
            )
            for b in range(len(BUCKETS))
        ]
        refit = [
            ridge_accuracy(
                x[train & (buckets == b)],
                labels[train & (buckets == b)],
                x[test & (buckets == b)],
                labels[test & (buckets == b)],
            )
            for b in range(len(BUCKETS))
        ]
        out[layer] = {
            "transfer": [round(a, 4) for a in transfer],
            "refit": [round(a, 4) for a in refit],
        }
    return out
