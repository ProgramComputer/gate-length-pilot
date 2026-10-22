from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from report import INK, MUTED, boot_ci, plt, save

RES = Path("results5")
WINDOWS = ("512–2047", "2048–4095", "4096–8191")
LENGTHS = (2048, 4096, 8192)
COND_COLOR = {"none": "#52514e", "patch": "#2a78d6", "sham": "#008300"}


def load(name: str) -> dict[str, Any] | None:
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def bits(s: str) -> np.ndarray:
    return np.frombuffer(s.encode(), dtype=np.uint8) == ord("1")


def binding(full: dict[str, Any], key: str) -> np.ndarray:
    return np.mean([full[key][l][h] for l, h in full["binding_heads"]], axis=0)


def decision() -> None:
    feas, full = load("feasibility"), load("full")
    out = [
        "# Pilot 5 (real model): pre-registered decision\n",
        "Model: AntNLP/TinyLlama-NoPE-1.1B (trained on 2048 tokens)\n",
        "## Feasibility (baseline, accuracy over the 512 value words)\n\n"
        "| placement | 2048 | 4096 | 8192 |\n|---|---|---|---|",
    ]
    for pl in ("before", "uniform", "after"):
        out.append(
            f"| {pl} | "
            + " | ".join(f"{100 * feas[f'{pl}/{L}']['restricted']:.1f}" for L in LENGTHS)
            + " |"
        )
    a2048, a8192 = feas["before/2048"]["restricted"], feas["before/8192"]["restricted"]
    F = a2048 >= 0.70 and a8192 <= a2048 - 0.20
    out.append(f"\nF (≥ 70% at 2048 and ≥ 20 pp lower at 8192, 'before'): {'yes' if F else 'NO'}")
    if not F or full is None:
        verdict = "NOT TESTABLE" if not F else "PENDING"
        out.append(f"\n## Verdict: **{verdict}**\n")
        (RES / "decision.md").write_text("\n".join(out), encoding="utf-8")
        print("\n".join(out))
        return

    p, m = binding(full, "p_prev"), binding(full, "margin")
    out.append(
        f"\n## Binding heads ({len(full['binding_heads'])}): {full['binding_heads']}\n\n"
        "| window (value positions) | " + " | ".join(WINDOWS) + " |\n|---|---|---|---|"
    )
    out.append("| attention to the previous token | " + " | ".join(f"{v:.3f}" for v in p) + " |")
    out.append("| score margin over other tokens | " + " | ".join(f"{v:.2f}" for v in m) + " |")
    D1 = p[2] <= 0.5 * p[0]
    ratio = m[2] / m[0]
    D1b = (
        "PREFERENCE KEPT (dilution)"
        if ratio >= 0.8
        else "PREFERENCE LOST (position)" if ratio <= 0.5 else "IN BETWEEN"
    )

    out.append(
        "\n## Retrieval with the relevant record far from the start "
        "but near the query ('before')\n\n"
        "| length | none | patch | sham | patch − none [bootstrap CI] | patch − sham |\n"
        "|---|---|---|---|---|---|"
    )
    for L in LENGTHS:
        acc = {c: full["acc"][f"{c}/before/{L}"]["restricted"] for c in COND_COLOR}
        x, y = bits(full["correct"][f"patch/before/{L}"]), bits(full["correct"][f"none/before/{L}"])
        lo, hi = 100 * boot_ci(x, y)
        out.append(
            f"| {L} | {100 * acc['none']:.1f} | {100 * acc['patch']:.1f} | "
            f"{100 * acc['sham']:.1f} | "
            f"{100 * (acc['patch'] - acc['none']):+.1f} [{lo:+.1f}, {hi:+.1f}] | "
            f"{100 * (acc['patch'] - acc['sham']):+.1f} |"
        )
    x, y = bits(full["correct"]["patch/before/8192"]), bits(full["correct"]["none/before/8192"])
    s = bits(full["correct"]["sham/before/8192"])
    D2 = x.mean() - y.mean() >= 0.20 and boot_ci(x, y)[0] > 0 and x.mean() - s.mean() >= 0.20
    verdict = (
        "TOY MECHANISM HOLDS IN THE REAL MODEL"
        if D1 and D2
        else (
            "ATTENTION FADES BUT THE PATCH DOES NOT RESCUE"
            if D1
            else "BINDING ATTENTION DOES NOT FADE"
        )
    )
    out.append(
        f"\nD1 (binding heads' attention at 4096–8191 ≤ ½ of inside): {'yes' if D1 else 'no'}"
    )
    out.append(f"D1b (margin far / inside = {ratio:.2f}): {D1b}")
    out.append(
        "D2 (patch raises 'before' @8192 by ≥ 20 pp, CI > 0, and beats the sham by ≥ 20 pp): "
        f"{'yes' if D2 else 'no'}"
    )
    out.append(f"\n## Verdict: **{verdict}**; why the attention changes: **{D1b}**\n")
    (RES / "decision.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def fig12() -> None:
    full = load("full")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
    p, m = binding(full, "p_prev"), binding(full, "margin")
    axes[0].plot(range(3), p, marker="o", color="#2a78d6")
    axes[0].set(
        xticks=range(3),
        xticklabels=WINDOWS,
        ylim=(0, 1),
        ylabel="attention to the previous token",
        title="a. Binding heads: look back one token?",
    )
    axes[1].plot(range(3), m, marker="o", color="#eb6834")
    axes[1].set(
        xticks=range(3),
        xticklabels=WINDOWS,
        ylabel="score of t−1 minus mean of others",
        title="b. Preference vs dilution",
    )
    for ax in axes[:2]:
        ax.axvspan(-0.4, 0.4, color="#e6e5e0", zorder=0)
        ax.set_xlabel("value-token positions (trained on ≤ 2048)")
    ax = axes[2]
    for c, color in COND_COLOR.items():
        ax.plot(
            LENGTHS,
            [100 * full["acc"][f"{c}/before/{L}"]["restricted"] for L in LENGTHS],
            marker="o",
            color=color,
            label={"none": "no patch", "patch": "patch: copy t−1", "sham": "sham: copy t−4"}[c],
        )
    ax.set(
        xscale="log",
        xticks=LENGTHS,
        xticklabels=[str(L) for L in LENGTHS],
        ylim=(0, 102),
        ylabel="accuracy (%)",
        xlabel="input length (tokens)",
        title="c. 'before' placement retrieval",
    )
    ax.minorticks_off()
    ax.legend(fontsize=7.5)
    fig.suptitle(
        "Pilot 5: TinyLlama-NoPE-1.1B (Wang et al., 2024), binding heads past the trained length",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "fig12_pilot5_real_model")


if __name__ == "__main__":
    items = {"fig12": fig12, "decision": decision}
    for name in sys.argv[1:] or items:
        items[name]()
