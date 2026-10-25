from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

if shutil.which("latex") is None:
    os.environ["PATH"] = (
        str(Path.home() / "AppData/Local/Programs/MiKTeX/miktex/bin/x64")
        + os.pathsep
        + os.environ["PATH"]
    )

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "readme"
BUILD = HERE / "build"
DPI = 300
WIDTH = 6.75

NONE, PATCH, SHAM, REIMPL = "tab:gray", "tab:blue", "tab:orange", "tab:red"
COND = {
    "none": (NONE, "o", "no patch"),
    "patch": (PATCH, "s", "patch"),
    "sham": (SHAM, "^", "sham"),
}
plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.size": 9,
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "lines.markersize": 4.5,
        "legend.frameon": False,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
    }
)
TRAINED = Patch(color="0.9", label="training length")

FIXES = (
    ("gate-statistics correction", "p1", "gate"),
    ("value-branch correction", "p1", "up"),
    ("residual-stream correction", "p1", "resid"),
    ("log-length logit scaling", "p1", "attn"),
    ("gate correction + log-length scaling", "p1", "gate+attn"),
    (r"logit scale $\times$1.2", "p2", "1.2"),
    (r"logit scale $\times$1.5", "p2", "1.5"),
    (r"logit scale $\times$1.8", "p2", "1.8"),
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


def fig2() -> None:
    fig, (a, b) = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    lens = np.array([128, 256, 512, 1024])
    none = np.array([[toy_acc("p1", s, "none", "before", L) for L in lens] for s in range(3)])
    a.set_xscale("log", base=2)
    a.set_xlim(80, 1400)
    a.axvspan(80, 128, color="0.9", lw=0, zorder=0)
    a.fill_between(lens, none.min(0), none.max(0), color=NONE, alpha=0.2, lw=0)
    a.plot(lens, none.mean(0), color=NONE, marker="o")
    jitter = 2 ** np.linspace(-0.06, 0.06, 6)
    for cond in ("patch", "sham"):
        v = np.array([toy_acc("p4", s, cond, "before") for s in range(6)])
        c, m, _ = COND[cond]
        a.scatter(1024 * jitter, v, s=10, color=c, alpha=0.6, lw=0, zorder=3)
        a.plot(1024, v.mean(), color=c, marker=m, ms=6, mec="black", mew=0.5, zorder=4)
    a.set_xticks(lens, [str(L) for L in lens])
    a.minorticks_off()
    a.set(
        xlabel="input length (tokens)",
        ylabel=r"accuracy (\%)",
        ylim=(-3, 103),
        title="(a) toy models",
    )

    lens = np.array([2048, 4096, 8192])
    b.set_xscale("log", base=2)
    b.set_xlim(1300, 9000)
    b.axvspan(1300, 2048, color="0.9", lw=0, zorder=0)
    for cond, (c, m, label) in COND.items():
        k = np.array([real_correct(cond, L).sum() for L in lens])
        lo, hi = np.array([clopper_pearson(int(x), 200) for x in k]).T
        b.fill_between(lens, lo, hi, color=c, alpha=0.2, lw=0)
        b.plot(lens, 100 * k / 200, color=c, marker=m, label=label)
    b.set_xticks(lens, [str(L) for L in lens])
    b.minorticks_off()
    b.set(xlabel="input length (tokens)", ylim=(-3, 103), title="(b) TinyLlama-NoPE-1.1B")
    handles, labels = b.get_legend_handles_labels()
    fig.legend(
        handles + [TRAINED],
        labels + ["training length"],
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.tight_layout()
    save(fig, "fig2_length")


def fig3() -> None:
    full = load("results6/full.json")
    p_in, p_far = np.array(full["p_inside"]), np.array(full["p_far"])
    fig = plt.figure(figsize=(WIDTH, 2.5))
    grid = fig.add_gridspec(1, 5, width_ratios=[1, 1, 0.045, 0.5, 1.05], wspace=0.1)
    axes = [fig.add_subplot(grid[0, i]) for i in (0, 1, 2, 4)]
    for ax, data, title in (
        (axes[0], p_in, "(a) within 2048 tokens"),
        (axes[1], p_far, "(b) beyond position 6000"),
    ):
        im = ax.imshow(
            data,
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
        )
        for l, h in full["binding_heads"]:
            ax.add_patch(Rectangle((h - 0.5, l - 0.5), 1, 1, fill=False, ec="white", lw=0.6))
        ax.grid(False)
        ax.set(title=title, xlabel="head", xticks=[0, 8, 16, 24, 31], yticks=[0, 5, 10, 15, 21])
    axes[0].set_ylabel("layer")
    axes[1].set_yticklabels([])
    cb = fig.colorbar(im, cax=axes[2])
    cb.ax.tick_params(labelsize=7)
    cb.set_label(r"attention to key ($t-1$)", fontsize=8)

    ax = axes[3]
    rows = np.array([toy_binding(s) for s in range(6)])
    x = np.arange(4)
    ax.axvspan(-0.4, 0.4, color="0.9", lw=0, zorder=0)
    for r in rows:
        ax.plot(x, r, color=PATCH, alpha=0.3, lw=0.8)
    ax.plot(x, rows.mean(0), color=PATCH, marker="o")
    ax.set_xlim(-0.4, 3.3)
    ax.set_xticks(
        x, ["64--127", "128--255", "256--511", "512--1023"], fontsize=7, rotation=25, ha="right"
    )
    ax.set(
        ylim=(0, 1),
        title="(c) toy models",
        xlabel="position of the value token",
        ylabel=r"attention to key ($t-1$)",
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.19, top=0.9)
    save(fig, "fig3_heads")


def fig4() -> None:
    rows = list(FIXES) + [("binding-head patch", "p4", "patch")]
    y = np.r_[np.arange(len(FIXES), 0, -1) + 0.6, 0.0]
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.9), sharey=True)
    for ax, pl, title in (
        (axes[0], "uniform", "(a) pairs anywhere"),
        (axes[1], "before", "(b) pairs far from the start"),
    ):
        for yi, (label, src, key) in zip(y, rows):
            g = gains(src, key, pl)
            color = PATCH if key == "patch" else NONE
            ax.plot(
                g.mean(),
                yi,
                marker="D",
                ms=4.5,
                color=color,
                mec="black",
                mew=0.5,
                alpha=0.8,
                zorder=3,
            )
            ax.scatter(
                g,
                yi + rng.uniform(-0.22, 0.22, len(g)),
                s=10,
                color=color,
                alpha=0.8,
                lw=0,
                zorder=4,
            )
        ax.axvline(0, color="black", lw=0.6)
        ax.axhline(0.8, color="0.6", lw=0.5)
        ax.set(xlim=(-10, 102), title=title, xlabel="gain in accuracy (percentage points)")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [label for label, _, _ in rows])
    axes[0].set_ylim(-0.6, y.max() + 0.6)
    handles = [
        plt.Line2D([], [], color=NONE, marker="o", ls="", ms=3.5, alpha=0.6, label="seed"),
        plt.Line2D([], [], color=NONE, marker="D", ls="", ms=5, mec="black", mew=0.5, label="mean"),
    ]
    axes[1].legend(handles=handles, loc="center right")
    fig.tight_layout()
    save(fig, "fig4_gains")


def fig5() -> None:
    fig = plt.figure(figsize=(WIDTH, 5.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.35, 0.7], hspace=0.45, wspace=0.25)
    a, a2, b = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    comp = load("audit/pilot4_comparison.json")["seeds"]
    full, ind = load("results6/full.json"), load("audit2/cpu_run_independent_summary.json")

    def point(
        ax: plt.Axes,
        x: float,
        xlo: float,
        xhi: float,
        yv: float,
        ylo: float,
        yhi: float,
        cond: str,
        ms: float,
    ) -> None:
        ax.errorbar(
            x,
            yv,
            xerr=[[x - xlo], [xhi - x]],
            yerr=[[yv - ylo], [yhi - yv]],
            color=COND[cond][0],
            lw=0.7,
            capsize=0,
            marker=COND[cond][1],
            ms=ms,
            zorder=3,
        )

    inset = a.inset_axes([0.62, 0.07, 0.36, 0.38])
    for ax in (a, inset):
        ax.plot([0, 100], [0, 100], color="0.6", lw=0.8, zorder=1)
        for s in comp:
            for cond in ("none", "patch", "sham"):
                ox, oy = s["claimed_accuracy_percent"][cond], s["audit_accuracy_percent"][cond]
                point(
                    ax,
                    ox,
                    *clopper_pearson(round(ox * 10), 1000),
                    oy,
                    *clopper_pearson(round(oy * 2), 200),
                    cond,
                    4,
                )
    inset.set(xlim=(0, 4.5), ylim=(-0.3, 4.5), xticks=[0, 2, 4], yticks=[0, 2, 4])
    inset.tick_params(labelsize=7, length=2)
    a.set(
        xlim=(-3, 103),
        ylim=(-3, 103),
        aspect="equal",
        xlabel=r"original accuracy (\%)",
        ylabel=r"re-implementation (\%)",
        title="(a) toy models, 6 seeds",
    )

    a2.plot([0, 100], [0, 100], color="0.6", lw=0.8, zorder=1)
    for cond in ("none", "patch", "sham"):
        k = int(real_correct(cond, 4096).sum())
        r = ind["accuracy"][f"4096_{cond}"]
        point(
            a2,
            100 * k / 200,
            *clopper_pearson(k, 200),
            r["percent"],
            *r["exact_binomial_95_percent"],
            cond,
            5,
        )
    a2.set(
        xlim=(-3, 103),
        ylim=(-3, 103),
        aspect="equal",
        xlabel=r"original accuracy (\%)",
        title="(b) TinyLlama, 4096 tokens",
    )
    handles = [
        plt.Line2D([], [], color=c, marker=m, ls="", ms=5, label=label)
        for c, m, label in COND.values()
    ]
    a2.legend(handles=handles, loc="lower right")

    hs = full["binding_heads"]
    mean = lambda key: np.mean([full[key][l][h] for l, h in hs])
    d = ind["diagnostics"]
    toy_orig = np.array([toy_binding(s["seed"])[3] / toy_binding(s["seed"])[0] for s in comp])
    toy_re = np.array([s["audit_far_inside_attention_ratio"] for s in comp])
    b.scatter(toy_orig, np.full(6, 3.13), s=10, color="black", alpha=0.4, lw=0)
    b.scatter(toy_re, np.full(6, 2.87), s=10, color=REIMPL, alpha=0.5, lw=0)
    b.plot(toy_orig.mean(), 3.13, marker="o", color="black", ms=5)
    b.plot(toy_re.mean(), 2.87, marker="D", color=REIMPL, ms=5)
    for yv, orig, est in (
        (2, mean("p_far") / mean("p_inside"), d["p_previous"]),
        (1, mean("margin_far") / mean("margin_inside"), d["margin_earlier"]),
    ):
        lo, hi = est["ratio_paired_bootstrap_95"]
        v = est["far_over_inside"]
        b.plot(orig, yv + 0.13, marker="o", color="black", ms=5)
        b.errorbar(
            v,
            yv - 0.13,
            xerr=[[v - lo], [hi - v]],
            marker="D",
            color=REIMPL,
            ms=5,
            lw=1.0,
            capsize=2,
        )
    b.add_patch(Rectangle((0.5, 0.62), 0.3, 0.76, color="0.9", lw=0, zorder=0))
    for yv in (3, 2):
        b.plot([0.5, 0.5], [yv - 0.35, yv + 0.35], color="0.5", lw=0.8, ls=":")
    b.set_yticks([3, 2, 1], ["attention, toy", "attention, TinyLlama", "score margin, TinyLlama"])
    b.set(xlim=(0, 1.5), ylim=(0.45, 3.55), xlabel="ratio, beyond / within", title="(c) ratios")
    b.grid(axis="y", visible=False)
    b.legend(
        handles=[
            plt.Line2D([], [], color="black", marker="o", ls="", ms=5, label="original"),
            plt.Line2D([], [], color=REIMPL, marker="D", ls="", ms=5, label="re-implementation"),
        ],
        loc="upper right",
    )
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
