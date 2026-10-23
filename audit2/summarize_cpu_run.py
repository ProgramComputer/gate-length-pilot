from __future__ import annotations

import itertools
import json
import math
from pathlib import Path
import numpy as np
from output_tag import tagged

ROOT = Path(__file__).resolve().parent


def exact_binomial(k: int, n: int, alpha: float = 0.05) -> list[float]:
    def tail(p: float, lower: bool) -> float:
        inds = range(k, n + 1) if lower else range(k + 1)
        return sum(math.comb(n, j) * p**j * (1 - p) ** (n - j) for j in inds)

    def root(lower: bool) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(80):
            mid = (lo + hi) / 2
            if (tail(mid, lower) < alpha / 2) == lower:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    return [0.0 if not k else root(True), 1.0 if k == n else root(False)]


def main() -> None:
    rows = [json.loads(s) for s in (ROOT / "cpu_run/raw.jsonl").read_text().splitlines()]
    summary = json.loads((ROOT / "cpu_run/summary.json").read_text())
    heads = summary["selected_heads_zero_based"]
    out = {
        "complete": summary["complete"],
        "elapsed_seconds": summary["elapsed_seconds"],
        "raw_rows": len(rows),
        "selection_n": sum(r["stage"] == "selection" for r in rows),
        "head_count": len(heads),
        "accuracy": {},
        "diagnostics": {},
        "paired_effects": {},
    }
    selection = np.array([r["metrics"]["p_previous"] for r in rows if r["stage"] == "selection"])
    rng = np.random.default_rng(882361)
    counts = [
        int((selection[rng.integers(len(selection), size=len(selection))].mean(0) >= 0.3).sum())
        for _ in range(5000)
    ]
    out["selection_head_count_bootstrap_95"] = np.quantile(counts, [0.025, 0.975]).tolist()
    for length in (2048, 4096, 8192):
        for cond in ("none", "patch", "sham"):
            cell = [
                r
                for r in rows
                if r["stage"] != "selection" and r["length"] == length and r["condition"] == cond
            ]
            if cell:
                k, n = sum(r["prediction"]["restricted_correct"] for r in cell), len(cell)
                full = sum(r["prediction"]["full_correct"] for r in cell)
                out["accuracy"][f"{length}_{cond}"] = {
                    "k": k,
                    "n": n,
                    "percent": 100 * k / n,
                    "exact_binomial_95_percent": [100 * x for x in exact_binomial(k, n)],
                    "full_k": full,
                }
    paired = {
        length: sorted(
            [r for r in rows if r["stage"] == "diagnostic" and r["length"] == length],
            key=lambda r: r["index"],
        )
        for length in (2048, 8192)
    }
    ix = np.array(list(itertools.product(range(4), repeat=4)))
    for metric in summary["diagnostics"]["2048"]:
        arr = {
            length: np.array(
                [np.mean([r["metrics"][metric][l][h] for l, h in heads]) for r in cell]
            )
            for length, cell in paired.items()
        }
        inside, far = arr[2048], arr[8192]
        ib, fb = inside[ix].mean(1), far[ix].mean(1)
        out["diagnostics"][metric] = {
            "inside_per_input": inside.tolist(),
            "far_per_input": far.tolist(),
            "inside_mean": float(inside.mean()),
            "far_mean": float(far.mean()),
            "inside_bootstrap_95": np.quantile(ib, [0.025, 0.975]).tolist(),
            "far_bootstrap_95": np.quantile(fb, [0.025, 0.975]).tolist(),
            "far_over_inside": float(far.mean() / inside.mean()),
            "ratio_paired_bootstrap_95": np.quantile(fb / ib, [0.025, 0.975]).tolist(),
        }
    for length in (4096, 8192):
        indexed = {
            (r["index"], r["condition"]): r
            for r in rows
            if r["stage"] != "selection" and r["length"] == length
        }
        for c in ("none", "sham"):
            inds = sorted(i for i, cond in indexed if cond == "patch" and (i, c) in indexed)
            differences = np.array(
                [
                    int(indexed[i, "patch"]["prediction"]["restricted_correct"])
                    - int(indexed[i, c]["prediction"]["restricted_correct"])
                    for i in inds
                ]
            )
            wins, losses = int((differences > 0).sum()), int((differences < 0).sum())
            discordant = wins + losses
            sign_p = (
                min(
                    1.0,
                    2
                    * sum(math.comb(discordant, j) for j in range(min(wins, losses) + 1))
                    / 2**discordant,
                )
                if discordant
                else 1.0
            )
            boots = differences[rng.integers(len(inds), size=(50000, len(inds)))].mean(1)
            out["paired_effects"][f"{length}_patch_minus_{c}"] = {
                "n": len(inds),
                "wins": wins,
                "losses": losses,
                "gain_pp": 100 * float(differences.mean()),
                "bootstrap_95_pp": (100 * np.quantile(boots, [0.025, 0.975])).tolist(),
                "exact_two_sided_sign_p": sign_p,
            }
    cal = [r for r in rows if r["stage"] == "selection"]
    out["selection_retrieval_descriptive"] = {
        "k": sum(r["prediction"]["restricted_correct"] for r in cal),
        "n": len(cal),
    }
    with (ROOT / tagged("cpu_run_independent_summary.json")).open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
