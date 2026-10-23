from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from report import INK, boot_ci, plt, save

RES = Path("results6")
LENGTHS = (2048, 4096, 8192)


def load(name: str) -> dict[str, Any] | None:
    p = RES / f"{name}.json"
    return json.loads(p.read_text()) if p.exists() else None


def bits(s: str) -> np.ndarray:
    return np.frombuffer(s.encode(), dtype=np.uint8) == ord("1")


def decision() -> None:
    tune, feas, full = load("tune"), load("feasibility"), load("full")
    out = [
        "# Pilot 6 (real model, redesigned): pre-registered decision\n",
        "## Tuning: block size k, on 2048-token inputs only (seed 60000)\n\n"
        "| k | accuracy (512 words) | full vocab |\n|---|---|---|",
    ]
    for k in ("16", "8", "4"):
        out.append(f"| {k} | {100 * tune[k]['restricted']:.1f} | {100 * tune[k]['full']:.1f} |")
    out.append(f"\nChosen k: {tune['chosen_k']}")
    verdict, F = None, False
    if tune["chosen_k"] is None:
        verdict = "NOT TESTABLE (no k reached 70% inside the trained length)"
    elif feas is not None:
        out.append(
            "\n## Feasibility (baseline accuracy over the 512 value words)\n\n"
            "| placement | 2048 | 4096 | 8192 |\n|---|---|---|---|"
        )
        for pl in ("before", "uniform", "after"):
            out.append(
                f"| {pl} | "
                + " | ".join(f"{100 * feas[f'{pl}/{L}']['restricted']:.1f}" for L in LENGTHS)
                + " |"
            )
        c2048, c8192 = (bits(feas[f"before/{L}"]["correct"]) for L in (2048, 8192))
        solved = 10 * c2048.sum() >= 7 * c2048.size
        F = solved and 5 * (c2048.sum() - c8192.sum()) >= c2048.size
        out.append(
            f"\nF (≥ 70% at 2048 and ≥ 20 pp lower at 8192, 'before'): {'yes' if F else 'NO'}"
        )
        if not F:
            verdict = (
                "NOT TESTABLE (no length failure to explain)"
                if solved
                else "NOT TESTABLE (the task is not solved inside the trained length)"
            )
    if F and full is None:
        verdict = "GATE PASSED; MAIN TEST NOT YET RUN (awaiting approval)"
    if F and full is not None:
        hs = full["binding_heads"]
        p_in = np.mean([full["p_inside"][l][h] for l, h in hs])
        p_far = np.mean([full["p_far"][l][h] for l, h in hs])
        m_in = np.mean([full["margin_inside"][l][h] for l, h in hs])
        m_far = np.mean([full["margin_far"][l][h] for l, h in hs])
        D1 = p_far <= 0.5 * p_in
        ratio = m_far / m_in
        D1b = (
            "PREFERENCE KEPT (dilution)"
            if ratio >= 0.8
            else "PREFERENCE LOST (position)" if ratio <= 0.5 else "IN BETWEEN"
        )
        x, y, s = (bits(full["correct"][f"{c}/before/8192"]) for c in ("patch", "none", "sham"))
        D2 = (
            5 * (x.sum() - y.sum()) >= x.size
            and boot_ci(x, y)[0] > 0
            and 5 * (x.sum() - s.sum()) >= x.size
        )
        out.append(
            f"\n## Binding heads ({len(hs)}): attention to previous token "
            f"{p_in:.3f} inside → {p_far:.3f} far; "
            f"margin {m_in:.2f} → {m_far:.2f} (ratio {ratio:.2f})"
        )
        out.append("\n| condition | before 2048 | before 4096 | before 8192 |\n|---|---|---|---|")
        for c in ("none", "patch", "sham"):
            out.append(
                f"| {c} | "
                + " | ".join(
                    f"{100 * full['acc'][f'{c}/before/{L}']['restricted']:.1f}" for L in LENGTHS
                )
                + " |"
            )
        out.append(f"\nD1: {'yes' if D1 else 'no'}; D1b: {D1b}; D2: {'yes' if D2 else 'no'}")
        verdict = (
            "TOY MECHANISM HOLDS IN THE REAL MODEL"
            if D1 and D2
            else (
                "ATTENTION FADES BUT THE PATCH DOES NOT RESCUE"
                if D1
                else "BINDING ATTENTION DOES NOT FADE"
            )
        ) + f"; {D1b}"
    out.append(f"\n## Verdict: **{verdict}**\n")
    (RES / "decision.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def fig13() -> None:
    full, b1 = load("full"), load("none_batch1")
    hs = full["binding_heads"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.3))
    for ax, (key_in, key_far, title, ylab) in zip(
        axes[:2],
        (
            (
                "p_inside",
                "p_far",
                "a. Binding heads: look back one token?",
                "attention to the previous token",
            ),
            (
                "margin_inside",
                "margin_far",
                "b. Preference vs dilution",
                "score of t−1 minus mean of others",
            ),
        ),
    ):
        inside = [full[key_in][l][h] for l, h in hs]
        far = [full[key_far][l][h] for l, h in hs]
        for a, b in zip(inside, far):
            ax.plot([0, 1], [a, b], color="#9cc2f0", lw=0.8, alpha=0.6)
        ax.plot(
            [0, 1],
            [np.mean(inside), np.mean(far)],
            color="#143f78",
            marker="o",
            lw=2.5,
            label="mean of heads",
        )
        ax.set(
            xticks=[0, 1],
            xticklabels=["block inside\n(< 2048)", "block far\n(> 6000, 8192 input)"],
            ylabel=ylab,
            title=title,
            xlim=(-0.3, 1.3),
        )
        ax.legend(fontsize=7.5)
    ax = axes[2]
    lengths = (2048, 4096, 8192)
    series = {
        "no patch": [full["acc"][f"none/before/{L}"]["restricted"] for L in lengths],
        "patch: copy t−1": [full["acc"][f"patch/before/{L}"]["restricted"] for L in lengths],
        "sham: copy t−4": [full["acc"][f"sham/before/{L}"]["restricted"] for L in lengths],
    }
    if b1:
        series["no patch (one at a time)"] = [b1[f"none/before/{L}"]["restricted"] for L in lengths]
    for (name, v), color, ls in zip(
        series.items(), ("#52514e", "#2a78d6", "#008300", "#52514e"), ("-", "-", "-", ":")
    ):
        ax.plot(lengths, [100 * x for x in v], marker="o", color=color, ls=ls, label=name)
    ax.set(
        xscale="log",
        xticks=lengths,
        xticklabels=[str(L) for L in lengths],
        ylim=(0, 102),
        ylabel="accuracy (%)",
        xlabel="input length (tokens; trained on ≤ 2048)",
        title="c. Pair block far from start, near query",
    )
    ax.minorticks_off()
    ax.legend(fontsize=7)
    fig.suptitle(
        f"Pilot 6: TinyLlama-NoPE-1.1B, {len(hs)} binding heads", x=0.01, ha="left", fontsize=9.5
    )
    fig.tight_layout()
    save(fig, "fig13_pilot6_real_model")


if __name__ == "__main__":
    if sys.argv[1:] == ["fig13"]:
        fig13()
    else:
        decision()
        if load("full") is not None:
            fig13()
