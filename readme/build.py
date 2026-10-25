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

NONE, PATCH, SHAM, REBUILD, ORIGINAL = "#52514e", "#2a78d6", "#008300", "#eb6834", "#0b0b0b"
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
    ("Log-length logit scaling", "p1", "attn"),
    ("Gate correction + log-length scaling", "p1", "gate+attn"),
    ("Uniform logit scale ×1.2", "p2", "1.2"),
    ("Uniform logit scale ×1.5", "p2", "1.5"),
    ("Uniform logit scale ×1.8", "p2", "1.8"),
)


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


def toy_binding(seed: int) -> np.ndarray:
    r = load(f"results4/seed{seed}.json")
    return np.mean([r["prev_attention"][l][h] for l, h in r["binding_heads"]], axis=0)


def real_correct(cond: str, L: int) -> np.ndarray:
    if cond == "none":
        return bits(load("results6/none_batch1.json")[f"none/before/{L}"]["correct"])
    return bits(load("results6/full.json")["correct"][f"{cond}/before/{L}"])


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
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.75))
    lens = np.array([128, 256, 512, 1024])
    none = np.array([[toy_acc("p1", s, "none", "before", L) for L in lens] for s in range(3)])
    a.set_xscale("log", base=2)
    a.set_xlim(70, 1500)
    shade_trained(a, 128, 99)
    a.fill_between(lens, none.min(0), none.max(0), color=NONE, alpha=0.15, lw=0)
    a.plot(lens, none.mean(0), color=NONE, marker="o")
    jitter = 2 ** np.linspace(-0.07, 0.07, 6)
    for cond in ("patch", "sham"):
        v = np.array([toy_acc("p4", s, cond, "before") for s in range(6)])
        c, m, _ = COND[cond]
        a.scatter(1024 * jitter, v, s=11, color=c, alpha=0.55, lw=0, zorder=3)
        a.plot(1024, v.mean(), color=c, marker=m, ms=6, mec="white", mew=0.7, zorder=4)
        if cond == "patch":
            a.annotate(
                f"mean {v.mean():.1f}\n(seeds: {v.min():.0f}–{v.max():.0f}%)",
                (1024 / 1.09, v.mean()),
                xytext=(-4, 0),
                textcoords="offset points",
                ha="right",
                va="center",
                fontsize=7.5,
                color=c,
            )
    a.text(78, 4.5, "chance 1.6%", fontsize=7, color="0.35")
    a.text(
        1420,
        47,
        "band: range over 3 seeds\ndots: 6 seeds; marker: mean",
        ha="right",
        va="center",
        fontsize=7,
        color="0.35",
    )
    a.set_xticks(lens, ["128\n1×", "256\n2×", "512\n4×", "1024\n8×"])
    a.minorticks_off()
    a.set(
        xlabel="input length (tokens)",
        ylabel="accuracy (%), 64 possible values",
        ylim=(-3, 103),
        title="(a) Toy NoPE models (1.1M), trained on ≤128",
    )

    lens = np.array([2048, 4096, 8192])
    b.set_xscale("log", base=2)
    b.set_xlim(1150, 11500)
    shade_trained(b, 2048, 99)
    for cond, (c, m, label) in COND.items():
        k = np.array([real_correct(cond, L).sum() for L in lens])
        acc = 100 * k / 200
        lo, hi = np.array([clopper_pearson(int(x), 200) for x in k]).T
        b.fill_between(lens, lo, hi, color=c, alpha=0.13, lw=0)
        b.plot(lens, acc, color=c, marker=m, label=label)
        if cond != "sham":
            b.annotate(
                f"{acc[-1]:.1f}",
                (8192, acc[-1]),
                xytext=(7, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color=c,
            )
    b.text(1230, 4.5, "chance 0.2%", fontsize=7, color="0.35")
    b.text(
        11000,
        40,
        "bands: 95% CI,\n200 inputs per point",
        ha="right",
        va="center",
        fontsize=7,
        color="0.35",
    )
    b.set_xticks(lens, ["2048\n1×", "4096\n2×", "8192\n4×"])
    b.minorticks_off()
    b.set(
        xlabel="input length (tokens)",
        ylabel="accuracy (%), 512 candidate words",
        ylim=(-3, 103),
        title="(b) TinyLlama-NoPE-1.1B, trained on 2048",
    )
    handles, labels = b.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_length")


def fig3() -> None:
    full = load("results6/full.json")
    p_in, p_far = np.array(full["p_inside"]), np.array(full["p_far"])
    heads = full["binding_heads"]
    m_in, m_far = (np.mean([p[l][h] for l, h in heads]) for p in (p_in, p_far))
    fig = plt.figure(figsize=(WIDTH, 2.75))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.045, 0.42, 0.95], wspace=0.12)
    axes = [fig.add_subplot(grid[0, i]) for i in (0, 1, 2, 4)]
    for ax, data, title in (
        (
            axes[0],
            p_in,
            f"(a) TinyLlama: pairs inside 2048\n{len(heads)} outlined heads, mean {m_in:.2f}",
        ),
        (axes[1], p_far, f"(b) pairs past position 6000\nsame heads, mean {m_far:.2f}"),
    ):
        im = ax.imshow(
            data,
            origin="lower",
            cmap="Blues",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
        for l, h in heads:
            ax.add_patch(Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False, ec="black", lw=0.5))
        ax.grid(False)
        ax.set(title=title, xlabel="head", xticks=[0, 8, 16, 24, 31], yticks=[0, 5, 10, 15, 21])
    axes[0].set_ylabel("layer")
    axes[1].set_yticklabels([])
    cb = fig.colorbar(im, cax=axes[2])
    cb.set_label("attention share on the key ($t-1$)", fontsize=8)
    cb.ax.tick_params(labelsize=7, width=0.5)
    cb.outline.set_linewidth(0.5)

    ax = axes[3]
    rows = np.array([toy_binding(s) for s in range(6)])
    x = np.arange(4)
    ax.axvspan(-0.45, 0.45, color="0.9", zorder=0, lw=0)
    ax.text(0, 0.97, "trained", ha="center", va="top", fontsize=7.5, color="0.35")
    for r in rows:
        ax.plot(x, r, color=PATCH, alpha=0.3, lw=0.8)
    ax.plot(x, rows.mean(0), color=PATCH, marker="o", lw=1.6)
    for xi, v in ((0, rows.mean(0)[0]), (3, rows.mean(0)[3])):
        ax.annotate(
            f"{v:.2f}",
            (xi, v),
            xytext=(2, 7),
            textcoords="offset points",
            ha="center",
            fontsize=7.5,
            color=PATCH,
        )
    ax.set_xlim(-0.45, 3.4)
    ax.set_xticks(x, ["64–127", "128–255", "256–511", "512–1023"], fontsize=7)
    ax.set(
        ylim=(0, 1),
        title="(c) Toy models, by position\nthin: 6 seeds; thick: mean",
        xlabel="position of the value token",
        ylabel="attention share on the key ($t-1$)",
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.84)
    save(fig, "fig3_heads")


def fig4() -> None:
    rows = list(FIXES) + [("Binding-head patch (oracle)", "p4", "patch")]
    y = np.r_[np.arange(len(FIXES), 0, -1) + 0.6, 0.0]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.0), sharey=True)
    panels = (
        (axes[0], "uniform", "(a) pairs anywhere"),
        (axes[1], "before", "(b) pairs far from the start"),
    )
    for ax, pl, title in panels:
        base = np.mean([toy_acc("p1", s, "none", pl) for s in range(3)])
        for yi, (label, src, key) in zip(y, rows):
            g = gains(src, key, pl)
            oracle = key == "patch"
            color = PATCH if oracle else NONE
            ax.plot([g.min(), g.max()], [yi, yi], color=color, lw=1.0, alpha=0.6, zorder=2)
            dy = rng.uniform(-0.14, 0.14, len(g))
            if oracle:
                ax.scatter(g[:3], yi + dy[:3], s=13, color=color, lw=0, zorder=3)
                ax.scatter(
                    g[3:], yi + dy[3:], s=13, facecolor="white", edgecolor=color, lw=0.8, zorder=3
                )
            else:
                ax.scatter(g, yi + dy, s=10, color=color, alpha=0.7, lw=0, zorder=3)
            ax.plot(
                g.mean(),
                yi,
                marker="D",
                ms=5.5 if oracle else 4.5,
                color=color,
                mec="white",
                mew=0.6,
                zorder=4,
            )
            if oracle:
                ax.annotate(
                    f"+{g.mean():.1f}",
                    (g.mean(), yi),
                    xytext=(0, 7),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color=PATCH,
                )
        ax.axvline(0, color="black", lw=0.6)
        ax.axhline(0.8, color="0.6", lw=0.5)
        ax.set(
            xlim=(-12, 105),
            xticks=[0, 25, 50, 75, 100],
            title=f"{title}; baseline {base:.1f}%",
            xlabel="gain in accuracy at 1024 tokens (points)",
        )
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [label for label, _, _ in rows])
    axes[0].set_ylim(-0.6, y.max() + 0.6)
    axes[1].text(
        103,
        y[0] + 0.3,
        "tested corrections, pilots 1–2",
        ha="right",
        va="center",
        fontsize=7.5,
        color="0.35",
    )
    axes[1].text(2, 0.5, "oracle, pilot 4", ha="left", va="center", fontsize=7.5, color="0.35")
    handles = [
        plt.Line2D([], [], color=NONE, marker="o", ls="", ms=3.5, alpha=0.7, label="training seed"),
        plt.Line2D(
            [],
            [],
            color=PATCH,
            marker="o",
            ls="",
            ms=3.5,
            mfc="white",
            label="fresh seed (oracle only)",
        ),
        plt.Line2D([], [], color=NONE, marker="D", ls="", ms=4.5, label="mean"),
        plt.Line2D([], [], color=NONE, lw=1.0, alpha=0.6, label="range"),
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.62, 1.0), handletextpad=0.3
    )
    fig.tight_layout()
    save(fig, "fig4_gains")


def fig5() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 3.05), gridspec_kw={"width_ratios": [1.05, 1]})
    comp = load("audit/pilot4_comparison.json")["seeds"]
    full, ind = load("results6/full.json"), load("audit2/cpu_run_independent_summary.json")
    points: list[tuple[float, float, float, float, float, float, str, str]] = []
    for s in comp:
        for cond in ("none", "patch", "sham"):
            ox, oy = s["claimed_accuracy_percent"][cond], s["audit_accuracy_percent"][cond]
            points.append(
                (
                    ox,
                    *clopper_pearson(round(ox * 10), 1000),
                    oy,
                    *clopper_pearson(round(oy * 2), 200),
                    cond,
                    "toy",
                )
            )
    for cond in ("none", "patch", "sham"):
        k = int(real_correct(cond, 4096).sum())
        r = ind["accuracy"][f"4096_{cond}"]
        points.append(
            (
                100 * k / 200,
                *clopper_pearson(k, 200),
                r["percent"],
                *r["exact_binomial_95_percent"],
                cond,
                "real",
            )
        )

    inset = a.inset_axes([0.6, 0.07, 0.36, 0.36])
    for ax in (a, inset):
        ax.plot([0, 100], [0, 100], color="0.6", lw=0.8, zorder=1)
        for x, xlo, xhi, yv, ylo, yhi, cond, model in points:
            kw = (
                dict(marker="o", ms=4)
                if model == "toy"
                else dict(marker="D", ms=5, mfc="white", mew=1.2)
            )
            ax.errorbar(
                x,
                yv,
                xerr=[[x - xlo], [xhi - x]],
                yerr=[[yv - ylo], [yhi - yv]],
                color=COND[cond][0],
                lw=0.7,
                capsize=0,
                zorder=3,
                **kw,
            )
    inset.set(xlim=(0, 4.5), ylim=(-0.3, 4.5), xticks=[0, 2, 4], yticks=[0, 2, 4])
    inset.tick_params(labelsize=6.5, length=2)
    inset.set_title("near 0%", fontsize=7, pad=2)
    handles = [
        plt.Line2D([], [], color=c, marker="s", ls="", ms=5, label=label)
        for c, _, label in COND.values()
    ]
    handles += [
        plt.Line2D([], [], color="black", marker="o", ls="", ms=4, label="toy, 6 seeds"),
        plt.Line2D(
            [],
            [],
            color="black",
            marker="D",
            ls="",
            ms=5,
            mfc="white",
            label="TinyLlama, 4096 tokens",
        ),
    ]
    a.legend(
        handles=handles,
        loc="upper left",
        fontsize=7,
        handletextpad=0.3,
        borderaxespad=0.2,
        labelspacing=0.35,
    )
    a.set(
        xlim=(-3, 103),
        ylim=(-3, 103),
        aspect="equal",
        xlabel="original accuracy (%)",
        ylabel="re-implementation (%)",
        title="(a) retrieval accuracy (95% CIs)",
    )

    hs = full["binding_heads"]
    mean = lambda key: np.mean([full[key][l][h] for l, h in hs])
    d = ind["diagnostics"]
    toy_orig = np.array([toy_binding(s["seed"])[3] / toy_binding(s["seed"])[0] for s in comp])
    toy_re = np.array([s["audit_far_inside_attention_ratio"] for s in comp])
    rows = (
        ("attention, toy\n(6 seeds)", 3),
        ("attention,\nTinyLlama", 2),
        ("score margin,\nTinyLlama", 1),
    )
    b.scatter(toy_orig, np.full(6, 3.13), s=10, color=ORIGINAL, alpha=0.5, lw=0)
    b.scatter(toy_re, np.full(6, 2.87), s=10, color=REBUILD, alpha=0.6, lw=0)
    b.plot(toy_orig.mean(), 3.13, marker="o", color=ORIGINAL, ms=5)
    b.plot(toy_re.mean(), 2.87, marker="D", color=REBUILD, ms=5)
    for yv, orig, est in (
        (2, mean("p_far") / mean("p_inside"), d["p_previous"]),
        (1, mean("margin_far") / mean("margin_inside"), d["margin_earlier"]),
    ):
        lo, hi = est["ratio_paired_bootstrap_95"]
        v = est["far_over_inside"]
        b.plot(orig, yv + 0.13, marker="o", color=ORIGINAL, ms=5)
        b.errorbar(
            v,
            yv - 0.13,
            xerr=[[v - lo], [hi - v]],
            marker="D",
            color=REBUILD,
            ms=5,
            lw=1.0,
            capsize=2,
        )
    for yv in (3, 2):
        b.plot([0.5, 0.5], [yv - 0.35, yv + 0.35], color="0.4", lw=0.8)
    b.text(0.52, 3.38, "fades if ≤ 0.5", fontsize=7, color="0.35", va="bottom")
    b.add_patch(Rectangle((0.5, 0.62), 0.3, 0.76, color="0.88", lw=0, zorder=0))
    b.text(0.65, 0.6, "registered\n“in between”", ha="center", va="top", fontsize=7, color="0.35")
    b.set_yticks([r for _, r in rows], [label for label, _ in rows])
    b.set(xlim=(0, 1.5), ylim=(0.2, 3.6), xlabel="far ÷ inside", title="(b) ratios: far ÷ inside")
    b.grid(axis="y", visible=False)
    b.legend(
        handles=[
            plt.Line2D([], [], color=ORIGINAL, marker="o", ls="", ms=5, label="original"),
            plt.Line2D(
                [], [], color=REBUILD, marker="D", ls="", ms=5, label="re-implementation (95% CI)"
            ),
        ],
        loc="center right",
        bbox_to_anchor=(1.0, 0.58),
        fontsize=7,
        handletextpad=0.3,
    )
    fig.tight_layout(w_pad=1.5)
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
