from __future__ import annotations

import json
import math
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "readme"
BUILD = HERE / "build"
DPI = 300
WIDTH = 6.75

NONE, PATCH, SHAM, REBUILD = "#52514e", "#2a78d6", "#008300", "#eb6834"
COND = {
    "none": (NONE, "o", "no patch"),
    "patch": (PATCH, "s", "patch (copy the key)"),
    "sham": (SHAM, "^", "sham (copy a wrong key)"),
}
plt.rcParams.update(
    {
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "lines.linewidth": 1.4,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)

FIXES = (
    ("Gate-statistics correction", "p1", "gate"),
    ("Value-branch correction", "p1", "up"),
    ("Residual-stream correction", "p1", "resid"),
    ("Log-length attention temperature", "p1", "attn"),
    ("Gate correction + temperature", "p1", "gate+attn"),
    ("Uniform temperature 1.2", "p2", "1.2"),
    ("Uniform temperature 1.5", "p2", "1.5"),
    ("Uniform temperature 1.8", "p2", "1.8"),
)
T95 = {3: 4.303, 6: 2.571}


def load(path: str) -> Any:
    return json.loads((ROOT / path).read_text())


def bits(s: str) -> np.ndarray:
    return np.frombuffer(s.encode(), dtype=np.uint8) == ord("1")


def clopper_pearson(k: int, n: int) -> tuple[float, float]:
    def cdf(j: int, p: float) -> float:
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(j + 1))

    def solve(f: Callable[[float], bool]) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return lo

    low = 0.0 if k == 0 else solve(lambda p: 1 - cdf(k - 1, p) < 0.025)
    high = 1.0 if k == n else solve(lambda p: cdf(k, p) > 0.025)
    return 100 * low, 100 * high


def toy_acc(src: str, seed: int, key: str, pl: str, L: int = 1024) -> float:
    path = {
        "p1": f"results/seed{seed}.json",
        "p2": f"results2/nope_seed{seed}.json",
        "p4": f"results4/seed{seed}.json",
    }[src]
    return 100 * load(path)["acc"][f"{key}/{pl}/{L}"]


def gains(src: str, key: str, pl: str) -> np.ndarray:
    base = "1.0" if src == "p2" else "none"
    seeds = range(6) if src == "p4" else range(3)
    return np.array([toy_acc(src, s, key, pl) - toy_acc(src, s, base, pl) for s in seeds])


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(HERE / f"{name}.png")
    plt.close(fig)
    print("wrote", f"readme/{name}.png")


def shade_trained(ax: plt.Axes, right: float, label_y: float) -> None:
    left = ax.get_xlim()[0]
    ax.axvspan(left, right, color="0.9", zorder=0, lw=0)
    ax.text(
        math.sqrt(left * right),
        label_y,
        "trained",
        ha="center",
        va="top",
        fontsize=7.5,
        color="0.35",
    )


def fig2() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.45), sharey=True)
    lens = np.array([128, 256, 512, 1024])
    none = np.array([[toy_acc("p1", s, "none", "before", L) for L in lens] for s in range(3)])
    a.set_xscale("log", base=2)
    a.set_xlim(70, 1450)
    shade_trained(a, 128, 98)
    a.fill_between(lens, none.min(0), none.max(0), color=NONE, alpha=0.15, lw=0)
    c, m, _ = COND["none"]
    a.plot(lens, none.mean(0), color=c, marker=m)
    for cond in ("patch", "sham"):
        v = np.array([toy_acc("p4", s, cond, "before") for s in range(6)])
        c, m, _ = COND[cond]
        a.scatter(np.full(6, 1024), v, s=9, color=c, alpha=0.45, lw=0, zorder=3)
        a.plot(1024, v.mean(), color=c, marker=m, ms=5.5, zorder=4)
        if cond == "patch":
            a.annotate(
                f"{v.mean():.1f}",
                (1024, v.mean()),
                xytext=(-7, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=8,
                color=c,
            )
    a.text(
        1000,
        30,
        "patch and sham:\n1024 only, 6 seeds",
        ha="right",
        va="center",
        fontsize=7,
        color="0.35",
    )
    a.set_xticks(lens, [str(L) for L in lens])
    a.minorticks_off()
    a.set(
        xlabel="input length (tokens)",
        ylabel="accuracy (%)",
        ylim=(-3, 103),
        title="(a) Toy NoPE models, trained on ≤128",
    )

    full = load("results6/full.json")
    lens = np.array([2048, 4096, 8192])
    b.set_xscale("log", base=2)
    b.set_xlim(1150, 9800)
    shade_trained(b, 2048, 98)
    for cond, (c, m, label) in COND.items():
        k = np.array([bits(full["correct"][f"{cond}/before/{L}"]).sum() for L in lens])
        acc = 100 * k / 200
        lo, hi = np.array([clopper_pearson(int(x), 200) for x in k]).T
        b.fill_between(lens, lo, hi, color=c, alpha=0.13, lw=0)
        b.plot(lens, acc, color=c, marker=m, label=label)
        if cond != "sham":
            b.annotate(
                f"{acc[-1]:.1f}",
                (8192, acc[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=c,
            )
    b.set_xticks(lens, [str(L) for L in lens])
    b.minorticks_off()
    b.set(xlabel="input length (tokens)", title="(b) TinyLlama-NoPE-1.1B, trained on 2048")
    fig.legend(
        *b.get_legend_handles_labels(), loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0)
    )
    fig.tight_layout()
    save(fig, "fig2_length")


def fig3() -> None:
    full = load("results6/full.json")
    p_in, p_far = np.array(full["p_inside"]), np.array(full["p_far"])
    fig = plt.figure(figsize=(WIDTH, 2.55))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.045, 0.42, 0.95], wspace=0.12)
    axes = [fig.add_subplot(grid[0, i]) for i in (0, 1, 2, 4)]
    vmax = 0.9
    for ax, data, title in (
        (axes[0], p_in, "(a) TinyLlama: pairs inside 2048"),
        (axes[1], p_far, "(b) TinyLlama: pairs past 6000"),
    ):
        im = ax.imshow(
            data,
            origin="lower",
            cmap="Blues",
            vmin=0,
            vmax=vmax,
            aspect="auto",
            interpolation="nearest",
        )
        for l, h in full["binding_heads"]:
            ax.add_patch(Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False, ec="black", lw=0.45))
        ax.grid(False)
        ax.set(title=title, xlabel="head", xticks=[0, 8, 16, 24, 31], yticks=[0, 5, 10, 15, 21])
    axes[0].set_ylabel("layer")
    axes[1].set_yticklabels([])
    cb = fig.colorbar(im, cax=axes[2])
    cb.set_label("attention to the key ($t-1$)", fontsize=8)
    cb.ax.tick_params(labelsize=7, width=0.5)
    cb.outline.set_linewidth(0.5)

    ax = axes[3]
    rows = np.array(
        [
            np.mean(
                [
                    load(f"results4/seed{s}.json")["prev_attention"][l][h]
                    for l, h in load(f"results4/seed{s}.json")["binding_heads"]
                ],
                axis=0,
            )
            for s in range(6)
        ]
    )
    x = np.arange(4)
    ax.axvspan(-0.45, 0.45, color="0.9", zorder=0, lw=0)
    ax.text(0, 0.97, "trained", ha="center", va="top", fontsize=7.5, color="0.35")
    for r in rows:
        ax.plot(x, r, color=PATCH, alpha=0.3, lw=0.8)
    ax.plot(x, rows.mean(0), color=PATCH, marker="o", lw=1.6)
    ax.set_xlim(-0.45, 3.4)
    ax.set_xticks(x, ["64–127", "128–255", "256–511", "512–1023"], fontsize=7)
    ax.set(
        ylim=(0, 1),
        title="(c) Toy models: by position",
        xlabel="position of the value token",
        ylabel="attention to the key ($t-1$)",
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.2, top=0.88)
    save(fig, "fig3_heads")


def fig4() -> None:
    rows = [(label, src, key) for label, src, key in FIXES] + [
        ("Binding-head patch (oracle)", "p4", "patch")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.7), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, pl, title in (
        (axes[0], "uniform", "(a) pairs anywhere"),
        (axes[1], "before", "(b) pairs far from the start"),
    ):
        for yi, (label, src, key) in zip(y, rows):
            g = gains(src, key, pl)
            color = PATCH if key == "patch" else "#b9b8b2"
            ax.barh(yi, g.mean(), height=0.62, color=color, zorder=2)
            ax.scatter(g, np.full(len(g), yi), s=7, color="black", zorder=3, lw=0)
            if key == "patch":
                ax.text(g.max() + 3, yi, f"+{g.mean():.1f}", va="center", fontsize=8, color=PATCH)
        ax.axvline(0, color="black", lw=0.6)
        ax.set(
            xlim=(-12, 122),
            xticks=[0, 25, 50, 75, 100],
            title=title,
            xlabel="gain in accuracy (percentage points)",
        )
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [label for label, _, _ in rows])
    fig.tight_layout()
    save(fig, "fig4_gains")


def fig5() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.0), gridspec_kw={"width_ratios": [1.15, 1]})
    a.plot([0, 100], [0, 100], color="0.6", lw=0.8, zorder=1)
    for s in load("audit/pilot4_comparison.json")["seeds"]:
        for cond in ("none", "patch", "sham"):
            ox, oy = s["claimed_accuracy_percent"][cond], s["audit_accuracy_percent"][cond]
            xlo, xhi = clopper_pearson(round(ox * 10), 1000)
            ylo, yhi = clopper_pearson(round(oy * 2), 200)
            a.errorbar(
                ox,
                oy,
                xerr=[[ox - xlo], [xhi - ox]],
                yerr=[[oy - ylo], [yhi - oy]],
                color=COND[cond][0],
                marker="o",
                ms=4,
                lw=0.7,
                capsize=0,
                zorder=3,
            )
    full, ind = load("results6/full.json"), load("audit2/cpu_run_independent_summary.json")
    for cond in ("none", "patch", "sham"):
        k = int(bits(full["correct"][f"{cond}/before/4096"]).sum())
        ox, (xlo, xhi) = 100 * k / 200, clopper_pearson(k, 200)
        r = ind["accuracy"][f"4096_{cond}"]
        oy, (ylo, yhi) = r["percent"], r["exact_binomial_95_percent"]
        a.errorbar(
            ox,
            oy,
            xerr=[[ox - xlo], [xhi - ox]],
            yerr=[[oy - ylo], [yhi - oy]],
            color=COND[cond][0],
            marker="D",
            ms=5,
            mfc="white",
            mew=1.2,
            lw=0.7,
            capsize=0,
            zorder=4,
        )
    handles = [
        plt.Line2D([], [], color=c, marker="s", ls="", ms=5, label=label)
        for c, _, label in COND.values()
    ]
    handles += [
        plt.Line2D([], [], color="black", marker="o", ls="", ms=4, label="toy, 6 seeds × 3"),
        plt.Line2D(
            [], [], color="black", marker="D", ls="", ms=5, mfc="white", label="TinyLlama at 4096"
        ),
    ]
    a.legend(handles=handles, loc="upper left", fontsize=7, handletextpad=0.3, borderaxespad=0.2)
    a.set(
        xlim=(-3, 103),
        ylim=(-3, 103),
        aspect="equal",
        xlabel="original accuracy (%)",
        ylabel="independent rebuild (%)",
        title="(a) retrieval accuracy",
    )

    b.plot([0, 1.5], [0, 1.5], color="0.6", lw=0.8, zorder=1)
    b.axhspan(0.5, 0.8, color="0.92", zorder=0, lw=0)
    for s in load("audit/pilot4_comparison.json")["seeds"]:
        r = load(f"results4/seed{s['seed']}.json")
        att = np.mean([r["prev_attention"][l][h] for l, h in r["binding_heads"]], axis=0)
        b.scatter(
            att[3] / att[0],
            s["audit_far_inside_attention_ratio"],
            s=14,
            color=PATCH,
            zorder=3,
            lw=0,
        )
    hs = full["binding_heads"]
    mean = lambda key: np.mean([full[key][l][h] for l, h in hs])
    d = ind["diagnostics"]
    for name, orig, est, marker in (
        ("attention, TinyLlama", mean("p_far") / mean("p_inside"), d["p_previous"], "D"),
        (
            "score margin, TinyLlama",
            mean("margin_far") / mean("margin_inside"),
            d["margin_earlier"],
            "P",
        ),
    ):
        lo, hi = est["ratio_paired_bootstrap_95"]
        color = PATCH if name.startswith("attention") else REBUILD
        b.errorbar(
            orig,
            est["far_over_inside"],
            yerr=[[est["far_over_inside"] - lo], [hi - est["far_over_inside"]]],
            color=color,
            marker=marker,
            ms=5,
            lw=0.8,
            capsize=2,
            zorder=4,
        )
        b.annotate(
            name,
            (orig, est["far_over_inside"]),
            xytext=(-7, 0) if "margin" in name else (6, -2),
            textcoords="offset points",
            ha="right" if "margin" in name else "left",
            va="center",
            fontsize=7.5,
            color=color,
        )
    b.annotate(
        "attention, toy (6 seeds)",
        (0.03, 0.05),
        xytext=(0.12, 0.24),
        fontsize=7.5,
        color=PATCH,
        arrowprops=dict(arrowstyle="-", color=PATCH, lw=0.5),
    )
    b.text(1.47, 0.65, "registered\n“in between”", ha="right", va="center", fontsize=7, color="0.4")
    b.set(
        xlim=(0, 1.5),
        ylim=(0, 1.5),
        aspect="equal",
        xlabel="original (far ÷ inside)",
        ylabel="independent rebuild",
        title="(b) far ÷ inside ratios",
    )
    fig.tight_layout()
    save(fig, "fig5_reproduction")


def fig1() -> None:
    import pypdfium2 as pdfium

    BUILD.mkdir(exist_ok=True)
    run = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={BUILD}",
            "fig1_concept.tex",
        ],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    if run.returncode:
        sys.exit("fig1_concept.tex failed:\n" + "\n".join(run.stdout.splitlines()[-25:]))
    image = pdfium.PdfDocument(BUILD / "fig1_concept.pdf")[0].render(scale=DPI / 72).to_pil()
    image.save(HERE / "fig1_concept.png", optimize=True)
    print("wrote readme/fig1_concept.png", image.size)


FIGURES = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}

if __name__ == "__main__":
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
