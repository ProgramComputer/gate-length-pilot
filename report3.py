from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from report import COLOR, INK, MUTED, boot_ci, mean_ci, plt, save

RES = Path("results3")
SEM = ("sem_a0.5", "sem_a0.25", "sem_a0", "sem_freeze")
POS = ("pos_a2", "pos_a4")
COND_COLOR = {
    "baseline": "#52514e",
    "sem_a0.5": "#9cc2f0",
    "sem_a0.25": "#5c97e2",
    "sem_a0": "#2a78d6",
    "sem_freeze": "#143f78",
    "pos_a2": "#f2a07b",
    "pos_a4": "#eb6834",
    "PI": "#1baf7a",
}


def load(pattern: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(RES.glob(pattern))]


def correct(r: dict[str, Any], name: str, pl: str, L: int) -> np.ndarray:
    return np.frombuffer(r["correct"][f"{name}/{pl}/{L}"].encode(), dtype=np.uint8) == ord("1")


def gain(
    rs: list[dict[str, Any]],
    name: str,
    base: str = "baseline",
    pl: str = "after",
    L: int = 1024,
) -> tuple[list[float], list[np.ndarray]]:
    diffs: list[float]
    cis: list[np.ndarray]
    diffs, cis = [], []
    for r in rs:
        x, y = correct(r, name, pl, L), correct(r, base, pl, L)
        diffs.append(100 * (x.mean() - y.mean()))
        cis.append(100 * boot_ci(x, y))
    return diffs, cis


def decision() -> None:
    A, NB, RB = (
        load("a_seed[0-9]*.json"),
        load("b_nope_seed[0-9]*.json"),
        load("b_rope_seed[0-9]*.json"),
    )
    out = ["# Pilot 3: pre-registered decisions\n"]

    out.append(
        "## A. RoPE Profiler (Wu, Du & Peng 2026) on the RoPE models\n\n"
        "| seed | heads (layer.head, token, attention) | semantic score 128 → 1024 | "
        "true reversal rate 128 → 1024 | positional score 128 → 1024 | worse failure |\n"
        "|---|---|---|---|---|---|"
    )
    directions: list[str] = []
    for r in A:
        d128, d1024 = r["diagnostics"]["128"], r["diagnostics"]["1024"]
        sem_worse = (1 - d1024["semantic_score"]) / max(1e-9, 1 - d128["semantic_score"])
        pos_worse = d128["positional_score"] / max(1e-9, d1024["positional_score"])
        directions.append("semantic" if sem_worse >= pos_worse else "positional")
        heads = "; ".join(f"{l}.{h} {t} {w:.2f}" for l, h, t, w in r["heads"])
        out.append(
            f"| {r['seed']} | {heads} | "
            f"{d128['semantic_score']:.3f} → {d1024['semantic_score']:.3f} | "
            f"{d128['reversal_true']:.3f} → {d1024['reversal_true']:.3f} | "
            f"{d128['positional_score']:.4f} → {d1024['positional_score']:.4f} | {directions[-1]} |"
        )
    A1 = all(
        r["diagnostics"]["1024"]["reversal_true"] >= 2 * r["diagnostics"]["128"]["reversal_true"]
        for r in A
    )
    direction = max(set(directions), key=directions.count)
    out.append(
        "\n| condition | after @1024 (per seed) | gain over baseline, pp (mean ± seed CI) | "
        "per-seed [bootstrap CI] | uniform @128 | before @1024 |\n|---|---|---|---|---|---|"
    )
    mean_gain: dict[str, tuple[float, list[float], list[np.ndarray]]] = {}
    for name in ("baseline",) + SEM + POS + ("PI",):
        accs = [100 * r["acc"][f"{name}/after/1024"] for r in A]
        diffs, cis = gain(A, name)
        m, h = mean_ci(diffs)
        mean_gain[name] = (m, diffs, cis)
        out.append(
            f"| {name} | "
            + ", ".join(f"{a:.1f}" for a in accs)
            + f" | {m:+.1f} ± {h:.1f} | "
            + "; ".join(f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]" for d, (lo, hi) in zip(diffs, cis))
            + f" | {np.mean([100 * r['acc'][f'{name}/uniform/128'] for r in A]):.1f}"
            + f" | {np.mean([100 * r['acc'][f'{name}/before/1024'] for r in A]):.1f} |"
        )
    chosen, other = (SEM, POS) if direction == "semantic" else (POS, SEM)
    best = max(chosen, key=lambda c: mean_gain[c][0])
    best_other = max(other, key=lambda c: mean_gain[c][0])
    A2 = mean_gain[best][0] >= 10 and all(lo > 0 for lo, _ in mean_gain[best][2])
    A3 = mean_gain[best][0] - mean_gain[best_other][0] >= 5
    verdict_a = (
        "TRANSFERS"
        if (A1 and A2 and A3)
        else ("DETECTS BUT DOES NOT FIX" if A1 else "DOES NOT DETECT")
    )
    out.append(f"\nDiagnosed direction (majority of seeds): **{direction}** ({directions})")
    out.append(f"A1 (true reversal rate at 1024 ≥ 2× at 128, every seed): {'yes' if A1 else 'no'}")
    out.append(
        f"A2 (best {direction}-direction condition, {best}, gains ≥ 10 pp "
        f"with every seed's CI > 0): {'yes' if A2 else 'no'}"
    )
    out.append(
        f"A3 ({best} beats the best opposite-direction condition, "
        f"{best_other}, by ≥ 5 pp): {'yes' if A3 else 'no'}"
    )
    out.append(f"\n### Verdict A: **{verdict_a}**\n")

    out.append(
        "## B. Han & Ji three-axis fit, a(i−j) + b(i) + c(j)\n\n"
        "| model | seed | positional head | fit corr inside | fit corr far | "
        "rho_a (shape inside vs far) | transfer corr | mean fit corr inside, all heads |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    for name, rs in (("NoPE", NB), ("RoPE", RB)):
        for r in rs:
            h = r["heads"][r["positional_head"]]
            allc = np.mean([v["corr_inside"] for v in r["heads"].values()])
            out.append(
                f"| {name} | {r['seed']} | {r['positional_head']} | "
                f"{h['corr_inside']:.3f} | {h['corr_far']:.3f} | "
                f"{h['rho_a']:.3f} | {h['corr_transfer']:.3f} | {allc:.3f} |"
            )
    B1 = all(r["heads"][r["positional_head"]]["rho_a"] <= 0.5 for r in NB)
    B2 = all(r["heads"][r["positional_head"]]["rho_a"] >= 0.8 for r in RB)
    verdict_b = (
        "NOPE-SPECIFIC BREAK" if (B1 and B2) else ("BREAKS IN BOTH" if B1 else "NO BREAK IN NOPE")
    )
    out.append(
        "\nB1 (NoPE positional profile changes far away: rho_a ≤ 0.5, every seed): "
        f"{'yes' if B1 else 'no'}"
    )
    out.append(
        "B2 (RoPE positional profile keeps its shape: rho_a ≥ 0.8, every seed): "
        f"{'yes' if B2 else 'no'}"
    )
    out.append(f"\n### Verdict B: **{verdict_b}**\n")
    RES.mkdir(exist_ok=True)
    (RES / "decision.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def fig9() -> None:
    A = load("a_seed[0-9]*.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    x = np.arange(len(A))
    for k, (L, c) in enumerate(((128, "#9cc2f0"), (1024, "#143f78"))):
        ax.bar(
            x + (k - 0.5) * 0.36,
            [100 * r["diagnostics"][str(L)]["reversal_true"] for r in A],
            width=0.34,
            color=c,
            label=f"window {L}",
        )
    ax.set_xticks(x, [f"seed {r['seed']}" for r in A])
    ax.set(
        ylabel="% of distances where a distractor\nkey outscores the relevant key",
        title="a. Reversal of the correct preference",
    )
    ax.legend(fontsize=7.5)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    names = ("baseline",) + SEM + POS + ("PI",)
    for i, name in enumerate(names):
        v = [100 * r["acc"][f"{name}/after/1024"] for r in A]
        ax.bar(i, np.mean(v), width=0.7, color=COND_COLOR[name])
        ax.scatter([i] * len(v), v, s=8, color=INK, zorder=3)
    ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
    ax.set(
        ylim=(0, 102),
        ylabel="accuracy (%)",
        title="b. Record far from the query ('after', 1024): interventions",
    )
    ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Pilot 3A: RoPE Profiler diagnosis and fixes on a failure we control "
        "(RoPE models, 3 seeds)",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "fig9_pilot3a_rope_profiler")


def fig10() -> None:
    NB, RB = load("b_nope_seed[0-9]*.json"), load("b_rope_seed[0-9]*.json")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.3))
    for ax, name, rs, color in ((axes[0], "NoPE", NB, "#2a78d6"), (axes[1], "RoPE", RB, "#eb6834")):
        for r in rs:
            h = r["heads"][r["positional_head"]]
            ax.plot(range(1, 33), h["a_inside"][1:33], color=color, lw=1.5)
            ax.plot(range(1, 33), h["a_far"][1:33], color=color, lw=1.5, ls=":")
        ax.set(
            xlabel="distance i − j (tokens)",
            ylabel="positional part a(i−j), detrended" if name == "NoPE" else None,
            title=f"{name}: positional head per seed (solid: positions < 128, dotted: 512–639)",
        )
    ax = axes[2]
    for k, (name, rs, color) in enumerate((("NoPE", NB, "#2a78d6"), ("RoPE", RB, "#eb6834"))):
        vals = [[v["rho_a"] for v in r["heads"].values()] for r in rs]
        star = [r["heads"][r["positional_head"]]["rho_a"] for r in rs]
        for v in vals:
            ax.scatter(
                np.full(len(v), k) + np.random.default_rng(0).uniform(-0.15, 0.15, len(v)),
                v,
                s=8,
                color=color,
                alpha=0.5,
            )
        ax.scatter(
            [k] * len(star),
            star,
            s=40,
            color=color,
            edgecolor=INK,
            zorder=3,
            label=f"{name} positional head",
        )
    ax.axhline(0.5, color=MUTED, ls=":", lw=1)
    ax.axhline(0.8, color=MUTED, ls=":", lw=1)
    ax.set_xticks([0, 1], ["NoPE", "RoPE"])
    ax.set(
        ylim=(-1.05, 1.05),
        ylabel="rho_a: same shape inside and far?",
        title="all heads (small), positional head (large)",
    )
    ax.legend(fontsize=7)
    fig.suptitle(
        "Pilot 3B: does the relative-position part of attention keep its shape "
        "past the trained length?",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "fig10_pilot3b_disentanglement")


if __name__ == "__main__":
    items = {"fig9": fig9, "fig10": fig10, "decision": decision}
    for name in sys.argv[1:] or items:
        items[name]()
