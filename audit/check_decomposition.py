from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os

os.environ["MPLCONFIGDIR"] = str(ROOT / "audit" / "mplconfig")
import json
from typing import Any
import numpy as np
import torch
from pilot3 import LogitFits, backfit, detrend


def main() -> None:
    torch.set_num_threads(1)
    evidence: dict[str, Any] = {}
    role = (1 - 2 * (torch.arange(640) % 2)).double()
    logits = -(role[:, None] * role[None, :])
    fit = LogitFits()
    fit.on_logits(0, logits[None, None])
    row = fit.out[0, 0][0]
    evidence["role_only_counterexample"] = {k: v for k, v in row.items() if not torch.is_tensor(v)}
    evidence["role_only_counterexample"]["all_keys_have_identical_features"] = True
    assert row["rho_a"] > 0.9999 and row["share_inside"] > 0.99
    i = torch.arange(1, 128)[:, None].expand(-1, 128)
    j = i - torch.arange(128)[None, :]
    keep = j >= 0
    i, j = i[keep], j[keep]
    d = i - j
    ii, jj = i - 1, j
    known_a = detrend(torch.cos(torch.arange(128).double() / 40))
    W = known_a[d].clone()
    for _ in range(100):
        for group in (ii, jj):
            counts = torch.bincount(group).double()
            means = torch.zeros(len(counts), dtype=torch.double).index_add_(
                0, group, W
            ) / counts.clamp_min(1)
            W = W - means[group]
    recovered, corr = backfit(W, d, ii, jj, iters=300)
    share = float(detrend(recovered)[d].var() / W.var())
    evidence["non_partition_counterexample"] = {
        "perfect_additive_model_fit_corr": corr,
        "reported_a_share": share,
    }
    assert corr > 0.9999 and share > 1
    gen = torch.Generator().manual_seed(12)
    aa = torch.randn(128, generator=gen, dtype=torch.double)
    bb = torch.randn(640, generator=gen, dtype=torch.double)
    cc = torch.randn(640, generator=gen, dtype=torch.double)
    known_W = aa[d] + bb[i] + cc[j]
    a, corr = backfit(known_W, d, ii, jj, iters=30)
    evidence["known_model_30_rounds"] = {
        "fit_corr": corr,
        "detrended_a_corr_truth": float(
            torch.corrcoef(torch.stack([detrend(a), detrend(aa)]))[0, 1]
        ),
    }
    assert corr > 0.9999
    selected: list[dict[str, Any]]
    ranges: dict[str, dict[str, float]]
    selected, ranges = [], {}
    for pos in ("nope", "rope"):
        shares: list[float] = []
        for file in sorted((ROOT / "results3").glob(f"b_{pos}_seed*.json")):
            r = json.loads(file.read_text())
            h = r["heads"][r["positional_head"]]
            inside, far = np.array(h["a_inside"]), np.array(h["a_far"])
            parity = (-1.0) ** np.arange(128)
            item = {
                "pos": pos,
                "seed": r["seed"],
                "head": r["positional_head"],
                "registered_mean_per_example_rho": h["rho_a"],
                "rho_of_saved_mean_profiles": float(np.corrcoef(inside, far)[0, 1]),
                "inside_parity_r_squared": float(np.corrcoef(inside, parity)[0, 1] ** 2),
                "far_parity_r_squared": float(np.corrcoef(far, parity)[0, 1] ** 2),
                "share_inside": h["share_inside"],
                "share_far": h["share_far"],
            }
            if pos == "nope":
                binding = json.loads((ROOT / f"results4/seed{r['seed']}.json").read_text())
                layer, head = map(int, r["positional_head"].split("."))
                item["selected_head_is_binding_head"] = [layer, head] in binding["binding_heads"]
                item["selected_head_previous_attention_by_window"] = binding["prev_attention"][
                    layer
                ][head]
            selected.append(item)
            shares.extend(
                v[f"share_{region}"] for v in r["heads"].values() for region in ("inside", "far")
            )
        ranges[pos] = {
            "min_a_share": min(shares),
            "max_a_share": max(shares),
            "number_shares_above_one": sum(s > 1 for s in shares),
        }
    evidence["selected_heads"] = selected
    evidence["saved_share_ranges"] = ranges
    (ROOT / "audit/decomposition_checks.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
