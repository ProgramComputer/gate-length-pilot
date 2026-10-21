from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from report import INK, MUTED, boot_ci, mean_ci, plt, save

RES = Path("results4")
NAMES = ("none", "patch", "sham")
COND_COLOR = {"none": "#52514e", "patch": "#2a78d6", "sham": "#008300"}
WINDOWS = ("64–127", "128–255", "256–511", "512–1023")


def load() -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(RES.glob("seed[0-9]*.json"))]


def correct(r: dict[str, Any], name: str, pl: str = "before", L: int = 1024) -> np.ndarray:
    return np.frombuffer(r["correct"][f"{name}/{pl}/{L}"].encode(), dtype=np.uint8) == ord("1")


def binding_attention(r: dict[str, Any]) -> np.ndarray:
    return np.mean([r["prev_attention"][l][h] for l, h in r["binding_heads"]], axis=0)


def best_refit(r: dict[str, Any], name: str, window: int) -> float:
    return max(r["probe"][name][layer]["refit"][window] for layer in ("1", "2", "3", "4"))


def compare(
    R: list[dict[str, Any]],
    a: str,
    b: str,
    pl: str = "before",
) -> tuple[list[float], list[np.ndarray]]:
    diffs: list[float]
    cis: list[np.ndarray]
    diffs, cis = [], []
    for r in R:
        x, y = correct(r, a, pl), correct(r, b, pl)
        diffs.append(100 * (x.mean() - y.mean()))
        cis.append(100 * boot_ci(x, y))
    return diffs, cis


def decision() -> None:
    R = load()
    out = [
        "# Pilot 4 (final): pre-registered decision\n",
        f"NoPE seeds {[r['seed'] for r in R]}\n",
        "## Binding heads and their attention to the previous token, by position window\n\n"
        "| seed | binding heads (layer, head) | "
        + " | ".join(WINDOWS)
        + " | far ≤ ½ inside |\n|---|---|---|---|---|---|---|",
    ]
    for r in R:
        a = binding_attention(r)
        out.append(
            f"| {r['seed']} | {r['binding_heads']} | "
            + " | ".join(f"{v:.3f}" for v in a)
            + f" | {'yes' if a[3] <= 0.5 * a[0] else 'no'} |"
        )
    D1 = all(binding_attention(r)[3] <= 0.5 * binding_attention(r)[0] for r in R)

    out.append(
        "\n## Retrieval with the relevant record far from the start "
        "but near the query ('before', 1024)\n\n"
        "| comparison | mean diff, pp (± seed CI) | per-seed [bootstrap CI] |\n|---|---|---|"
    )
    stats: dict[tuple[str, str], tuple[float, list[np.ndarray]]] = {}
    for a, b in (("patch", "none"), ("patch", "sham"), ("sham", "none")):
        diffs, cis = compare(R, a, b)
        m, h = mean_ci(diffs)
        stats[a, b] = (m, cis)
        out.append(
            f"| {a} − {b} | {m:+.1f} ± {h:.1f} | "
            + "; ".join(f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]" for d, (lo, hi) in zip(diffs, cis))
            + " |"
        )
    out.append(
        "\n| condition | before @1024 | uniform @1024 | after @1024 | uniform @128 | "
        "probe refit 512–1023 |\n|---|---|---|---|---|---|"
    )
    for name in NAMES:
        cells = [
            np.mean([100 * r["acc"][f"{name}/{pl}/{L}"] for r in R])
            for pl, L in (("before", 1024), ("uniform", 1024), ("after", 1024), ("uniform", 128))
        ]
        far = np.mean([100 * best_refit(r, name, 3) for r in R])
        out.append(f"| {name} | " + " | ".join(f"{c:.1f}" for c in cells) + f" | {far:.1f} |")
    D2 = (
        stats["patch", "none"][0] >= 20
        and all(lo > 0 for lo, _ in stats["patch", "none"][1])
        and stats["patch", "sham"][0] >= 20
    )
    D3 = all(best_refit(r, "patch", 3) >= 0.5 * best_refit(r, "none", 0) for r in R)
    verdict = (
        "BINDING ATTENTION IS THE BOTTLENECK"
        if D2
        else "QUERY SIDE IS THE BOTTLENECK" if D3 else "COPIED CONTENT IS CORRUPTED"
    )
    out.append(
        "\nD1 (binding heads' attention to the previous token far away ≤ ½ of inside, "
        f"every seed): {'yes' if D1 else 'no'}"
    )
    out.append(
        "D2 (patch raises 'before' @1024 by ≥ 20 pp, every seed's CI > 0, "
        f"and beats the sham by ≥ 20 pp): {'yes' if D2 else 'no'}"
    )
    out.append(
        "D3 (under the patch, the binding is decodable far away at ≥ ½ the inside level, "
        f"every seed): {'yes' if D3 else 'no'}"
    )
    out.append(f"\n## Verdict: **{verdict}**\n")
    RES.mkdir(exist_ok=True)
    (RES / "decision.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def fig11() -> None:
    R = load()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
    ax = axes[0]
    for r in R:
        ax.plot(range(4), binding_attention(r), marker="o", color="#2a78d6", alpha=0.7)
    ax.axvspan(-0.4, 0.4, color="#e6e5e0", zorder=0)
    ax.set_xticks(range(4), WINDOWS)
    ax.set(
        ylim=(0, 1),
        xlabel="token positions of the value",
        ylabel="attention to the key just before",
        title="a. Binding heads: do they still look back one token?",
    )
    for ax, (pl, title) in zip(
        axes[1:],
        (("before", "b. Record far from start, near query"), ("uniform", "c. Uniform placement")),
    ):
        for i, name in enumerate(NAMES):
            v = [100 * r["acc"][f"{name}/{pl}/1024"] for r in R]
            ax.bar(i, np.mean(v), width=0.7, color=COND_COLOR[name])
            ax.scatter([i] * len(v), v, s=8, color=INK, zorder=3)
        ax.set_xticks(range(3), ["no patch", "patch: copy t−1", "sham: copy t−3"])
        ax.set(
            ylim=(0, 102), ylabel="accuracy at 1024 (%)" if pl == "before" else None, title=title
        )
        ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Pilot 4: forcing NoPE's binding heads to copy the previous token past position 128 "
        "(6 seeds; dots = seeds)",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "fig11_pilot4_patching")


if __name__ == "__main__":
    items = {"fig11": fig11, "decision": decision}
    for name in sys.argv[1:] or items:
        items[name]()
