from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from report import (
    COLOR,
    LENS,
    MUTED,
    PLACE_COLOR,
    PLACEMENTS,
    T95,
    boot_ci,
    mean_ci,
    plt,
    save,
    set_length_axis,
    target_position,
)

RES = Path("results2")
TEMPS = ("1.0", "1.2", "1.5", "1.8")
LAYERS = ("1", "2", "3", "4")
WINDOWS = ("64–127", "128–255", "256–511", "512–1023")
POS_COLOR = {"nope": "#2a78d6", "rope": "#eb6834"}


def load2(pos: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(RES.glob(f"{pos}_seed[0-9]*.json"))]


def correct2(r: dict[str, Any], c: str, L: int, pl: str = "uniform") -> np.ndarray:
    return np.frombuffer(r["correct"][f"{c}/{pl}/{L}"].encode(), dtype=np.uint8) == ord("1")


def beyond(r: dict[str, Any], L: int) -> np.ndarray:
    return target_position(r, L) >= 128


def best_refit(r: dict[str, Any], c: str, window: int) -> float:
    return max(r["probe"][c][layer]["refit"][window] for layer in LAYERS)


def best_transfer(r: dict[str, Any], c: str, window: int) -> float:
    return max(r["probe"][c][layer]["transfer"][window] for layer in LAYERS)


def decision() -> None:
    N, R = load2("nope"), load2("rope")
    out = [
        "# Pilot 2: pre-registered decision\n",
        f"NoPE seeds {[r['seed'] for r in N]}, RoPE seeds {[r['seed'] for r in R]}\n",
    ]

    out.append(
        "## Binding probe: best refit accuracy over layers 1–4 (chance 0.2%)\n\n"
        "| model | seed | c | inside 64–127 (B_in) | 128–255 | 256–511 "
        "| 512–1023 (B_far) | B_far ≤ ½·B_in |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    for name, rs in (("NoPE", N), ("RoPE", R)):
        for r in rs:
            for c in TEMPS:
                w = [best_refit(r, c, b) for b in range(4)]
                out.append(
                    f"| {name} | {r['seed']} | {c} | "
                    + " | ".join(f"{100 * a:.1f}%" for a in w)
                    + f" | {'yes' if w[3] <= 0.5 * w[0] else 'no'} |"
                )
    V1 = all(best_refit(r, "1.0", 0) >= 0.5 for r in N)
    P1 = all(best_refit(r, c, 3) <= 0.5 * best_refit(r, c, 0) for r in N for c in TEMPS)
    P3 = all(
        best_refit(r, "1.0", 0) >= 0.5 and best_refit(r, "1.0", 3) >= 0.5 * best_refit(r, "1.0", 0)
        for r in R
    )
    V2 = all(r["acc"]["1.0/uniform/128"] >= 0.95 for r in R)

    out.append(
        "\n## Temperature rescue of beyond items (NoPE, L = 1024, uniform placement)\n\n"
        "| c | beyond-item accuracy (per seed) | gain over c = 1.0, pp: mean ± seed CI "
        "| per-seed [bootstrap CI] | rescue |\n|---|---|---|---|---|"
    )
    rescue: dict[str, bool] = {}
    for c in TEMPS[1:]:
        diffs: list[float]
        cis: list[np.ndarray]
        accs: list[float]
        diffs, cis, accs = [], [], []
        for r in N:
            m = beyond(r, 1024)
            x, y = correct2(r, c, 1024)[m], correct2(r, "1.0", 1024)[m]
            diffs.append(100 * (x.mean() - y.mean()))
            cis.append(100 * boot_ci(x, y))
            accs.append(100 * x.mean())
        mean, half = mean_ci(diffs)
        rescue[c] = bool(mean >= 20 and all(lo > 0 for lo, _ in cis))
        out.append(
            f"| {c} | "
            + ", ".join(f"{a:.1f}%" for a in accs)
            + f" | {mean:+.1f} ± {half:.1f} | "
            + "; ".join(f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]" for d, (lo, hi) in zip(diffs, cis))
            + f" | {'YES' if rescue[c] else 'no'} |"
        )
    Rx = any(rescue.values())

    out.append(
        "\n## Accuracy (uniform placement), mean over seeds\n\n"
        "| model | c | 128 | 256 | 512 | 1024 | inside items @1024 | beyond items @1024 |\n"
        "|---|---|---|---|---|---|---|---|"
    )
    for name, rs in (("NoPE", N), ("RoPE", R)):
        for c in TEMPS:
            acc = [np.mean([100 * r["acc"][f"{c}/uniform/{L}"] for r in rs]) for L in LENS]
            ins = np.mean([100 * correct2(r, c, 1024)[~beyond(r, 1024)].mean() for r in rs])
            bey = np.mean([100 * correct2(r, c, 1024)[beyond(r, 1024)].mean() for r in rs])
            out.append(
                f"| {name} | {c} | "
                + " | ".join(f"{a:.1f}" for a in acc)
                + f" | {ins:.1f} | {bey:.1f} |"
            )

    out.append(
        "\n## By placement (c = 1.0), mean over seeds\n\n"
        "| model | placement | 256 | 512 | 1024 |\n|---|---|---|---|---|"
    )
    for name, rs in (("NoPE", N), ("RoPE", R)):
        for pl in PLACEMENTS:
            out.append(
                f"| {name} | {pl} | "
                + " | ".join(
                    f"{np.mean([100 * r['acc'][f'1.0/{pl}/{L}'] for r in rs]):.1f}"
                    for L in LENS[1:]
                )
                + " |"
            )

    out.append(f"\nV1 (NoPE probe works inside the range, B_in ≥ 50%): {'yes' if V1 else 'NO'}")
    out.append(f"V2 (RoPE seeds ≥ 95% at length 128): {'yes' if V2 else 'NO'}")
    out.append(f"R (some temperature rescues beyond items by ≥ 20 pp): {'YES' if Rx else 'no'}")
    out.append(f"P1 (NoPE binding missing far away at every c): {'yes' if P1 else 'no'}")
    out.append(f"P3 (RoPE keeps the binding far away): {'yes' if P3 else 'no'}")
    if not (V1 and V2):
        verdict = "INCONCLUSIVE"
    elif Rx and not P1:
        verdict = "ATTENTION DISTRACTION"
    elif P1 and not Rx and P3:
        verdict = "POSITION LOSS"
    else:
        verdict = "MIXED"
    out.append(f"\n## Verdict: **{verdict}**\n")
    RES.mkdir(exist_ok=True)
    (RES / "decision.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def fig7() -> None:
    N, R = load2("nope"), load2("rope")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3))
    ax = axes[0]
    for name, rs, ls in (("NoPE", N, "-"), ("RoPE", R, "--")):
        for pl in PLACEMENTS:
            v = np.array([[100 * r["acc"][f"1.0/{pl}/{L}"] for L in LENS] for r in rs])
            ax.plot(
                LENS, v.mean(0), ls=ls, marker="o", color=PLACE_COLOR[pl], label=f"{name}, {pl}"
            )
    set_length_axis(ax)
    ax.set(
        ylim=(0, 102),
        ylabel="accuracy (%)",
        title="a. Accuracy by placement (solid NoPE, dashed RoPE)",
    )
    ax.legend(fontsize=7, ncol=2)
    ax = axes[1]
    x = np.arange(4)
    for name, rs in (("NoPE", N), ("RoPE", R)):
        pos = name.lower()
        refit = np.array([[100 * best_refit(r, "1.0", b) for b in range(4)] for r in rs])
        transfer = np.array([[100 * best_transfer(r, "1.0", b) for b in range(4)] for r in rs])
        ax.plot(
            x,
            refit.mean(0),
            marker="o",
            color=POS_COLOR[pos],
            label=f"{name}: probe refit per window",
        )
        ax.plot(
            x,
            transfer.mean(0),
            marker="s",
            ls=":",
            color=POS_COLOR[pos],
            label=f"{name}: in-range probe transferred",
        )
        for row in refit:
            ax.scatter(x, row, s=8, color=POS_COLOR[pos], alpha=0.5)
    ax.axvspan(-0.4, 0.4, color="#e6e5e0", zorder=0)
    ax.set_xticks(x, WINDOWS)
    ax.set(
        ylim=(0, 102),
        xlabel="token positions of the probed values (length-1024 inputs)",
        ylabel="key decoded at value position (%)",
        title="b. Is the key–value binding still there?",
    )
    ax.legend(fontsize=7)
    fig.suptitle(
        "Pilot 2: NoPE vs RoPE (temperature 1.0; mean over 3 seeds, dots = seeds)",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "fig7_pilot2_binding")


def fig8() -> None:
    N, R = load2("nope"), load2("rope")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.3), sharey=True)
    temps = [float(c) for c in TEMPS]
    for ax, name, rs in ((axes[0], "NoPE", N), (axes[1], "RoPE", R)):
        series = {
            "accuracy at 128": [
                np.mean([100 * r["acc"][f"{c}/uniform/128"] for r in rs]) for c in TEMPS
            ],
            "inside items @1024": [
                np.mean([100 * correct2(r, c, 1024)[~beyond(r, 1024)].mean() for r in rs])
                for c in TEMPS
            ],
            "beyond items @1024": [
                np.mean([100 * correct2(r, c, 1024)[beyond(r, 1024)].mean() for r in rs])
                for c in TEMPS
            ],
            "binding probe, 512–1023": [
                np.mean([100 * best_refit(r, c, 3) for r in rs]) for c in TEMPS
            ],
        }
        for (label, v), color, ls in zip(
            series.items(), ("#52514e", "#1baf7a", "#2a78d6", "#eb6834"), ("-", "-", "-", ":")
        ):
            ax.plot(temps, v, marker="o", color=color, ls=ls, label=label)
        ax.set(
            xticks=temps,
            xlabel="uniform attention temperature c (logits × c at every position)",
            title=f"{name}",
            ylim=(0, 102),
            ylabel="%" if name == "NoPE" else None,
        )
    axes[0].legend(fontsize=7.5)
    fig.suptitle(
        "Pilot 2: does sharper attention bring back records beyond the trained length? "
        "(mean over 3 seeds)",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "fig8_pilot2_temperature")


if __name__ == "__main__":
    items = {"fig7": fig7, "fig8": fig8, "decision": decision}
    for name in sys.argv[1:] or items:
        items[name]()
