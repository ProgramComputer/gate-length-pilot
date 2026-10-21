from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES, FIG = Path("results"), Path("figures")
LENS, LONG, PLACEMENTS, N_LAYERS = (
    (128, 256, 512, 1024),
    (512, 1024),
    ("uniform", "before", "after"),
    4,
)
CONDS = ("none", "gate", "attn", "gate+attn", "up", "resid", "sham")
LABEL = {
    "none": "none",
    "gate": "gate corr.",
    "attn": "attn temp.",
    "gate+attn": "gate + attn",
    "up": "value-branch corr.",
    "resid": "residual corr.",
    "sham": "sham (shuffled gate)",
}
COLOR = {
    "none": "#52514e",
    "gate": "#2a78d6",
    "attn": "#eb6834",
    "gate+attn": "#1baf7a",
    "up": "#eda100",
    "resid": "#e87ba4",
    "sham": "#008300",
}
PLACE_COLOR = {"uniform": "#2a78d6", "before": "#eb6834", "after": "#1baf7a"}
LAYER_COLOR = ["#9cc2f0", "#5c97e2", "#2a78d6", "#143f78"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"
T95 = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}
plt.rcParams.update(
    {
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "lines.linewidth": 2,
        "lines.markersize": 5,
        "legend.frameon": False,
        "figure.dpi": 160,
        "savefig.bbox": "tight",
    }
)


def load(prefix: str = "seed") -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(RES.glob(f"{prefix}[0-9]*.json"))]


def correct(r: dict[str, Any], cond: str, L: int, pl: str = "uniform") -> np.ndarray:
    return np.frombuffer(r["correct"][f"{cond}/{pl}/{L}"].encode(), dtype=np.uint8) == ord("1")


def mean_ci(x: list[float] | np.ndarray) -> tuple[float, float]:
    x = np.asarray(x, float)
    return x.mean(), (T95[len(x)] * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0)


def boot_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000) -> np.ndarray:
    idx = np.random.default_rng(0).integers(0, len(a), (n_boot, len(a)))
    return np.percentile(a[idx].mean(1) - b[idx].mean(1), [2.5, 97.5])


def compare(R: list[dict[str, Any]], a: str, b: str, L: int, threshold_pp: float) -> dict[str, Any]:
    diffs: list[float]
    cis: list[np.ndarray]
    diffs, cis = [], []
    for r in R:
        x, y = correct(r, a, L), correct(r, b, L)
        diffs.append(100 * (x.mean() - y.mean()))
        cis.append(100 * boot_ci(x, y))
    m, h = mean_ci(diffs)
    return dict(
        mean=m,
        half=h,
        per_seed=diffs,
        boot=cis,
        passed=bool(m >= threshold_pp and all(lo > 0 for lo, _ in cis)),
    )


def layer_mean(r: dict[str, Any], cond: str, L: int, site: str) -> float:
    return np.mean([r["manip"][f"{cond}/{L}/{site}/{layer}"] for layer in range(N_LAYERS)])


def decision() -> None:
    R, B = load(), load("baseline_seed") or load()
    out = ["# Pre-registered decision\n", f"Seeds: {[r['seed'] for r in R]}\n"]

    out.append(
        "## Sanity gate (baseline, uniform placement)\n\n"
        "| seed | acc@128 | acc@1024 | passes |\n|---|---|---|---|"
    )
    sanity = True
    for r in B:
        a128, a1024 = r["acc"]["none/uniform/128"], r["acc"]["none/uniform/1024"]
        ok = a128 >= 0.95 and a1024 <= a128 - 0.20
        sanity &= ok
        out.append(f"| {r['seed']} | {a128:.3f} | {a1024:.3f} | {'yes' if ok else 'NO'} |")

    crit = {
        "G1": ("gate", "none", 10),
        "G2": ("gate", "sham", 5),
        "G3a": ("gate", "up", 5),
        "G3b": ("gate", "resid", 5),
        "G4": ("gate+attn", "attn", 5),
    }
    descriptive = {
        "attn vs none": ("attn", "none"),
        "up vs none": ("up", "none"),
        "resid vs none": ("resid", "none"),
        "sham vs none": ("sham", "none"),
    }
    G = {(k, L): compare(R, a, b, L, t) for k, (a, b, t) in crit.items() for L in LONG}
    out.append(
        "\n## Criteria (uniform placement; pp = percentage points)\n\n"
        "| criterion | L | mean diff (pp) ± seed 95% CI "
        "| per-seed diff [bootstrap 95% CI] | pass |\n|---|---|---|---|---|"
    )
    for (k, L), g in G.items():
        a, b, t = crit[k]
        seeds = "; ".join(
            f"{d:+.1f} [{lo:+.1f}, {hi:+.1f}]" for d, (lo, hi) in zip(g["per_seed"], g["boot"])
        )
        out.append(
            f"| {k}: {a} − {b} ≥ {t} | {L} | {g['mean']:+.1f} ± {g['half']:.1f} | {seeds} | "
            f"{'PASS' if g['passed'] else 'fail'} |"
        )
    for name, (a, b) in descriptive.items():
        for L in LONG:
            g = compare(R, a, b, L, 0)
            out.append(
                f"| (descriptive) {name} | {L} | {g['mean']:+.1f} ± {g['half']:.1f} | "
                + "; ".join(f"{d:+.1f}" for d in g["per_seed"])
                + " | – |"
            )

    out.append(
        "\n## Manipulation checks (held-out check set, query position, mean over layers)\n\n"
        "| correction | site | L | discrepancy none → corrected (per seed) | reduction ≥ 50% |\n"
        "|---|---|---|---|---|"
    )
    M1 = True
    for cond, site in (
        ("gate", "gate"),
        ("gate+attn", "gate"),
        ("up", "up"),
        ("resid", "resid"),
        ("sham", "gate"),
    ):
        for L in LONG:
            pairs = [(layer_mean(r, "none", L, site), layer_mean(r, cond, L, site)) for r in R]
            ok = all(after <= 0.5 * before for before, after in pairs)
            if cond == "gate":
                M1 &= ok
            out.append(
                f"| {cond} | {site} | {L} | "
                + "; ".join(f"{b:.3f} → {a:.3f}" for b, a in pairs)
                + f" | {'yes' if ok else 'no'} |"
            )
    for L in LONG:
        ent = [
            (
                layer_mean(r, "none", L, "attn_entropy"),
                layer_mean(r, "attn", L, "attn_entropy"),
                np.mean(
                    [
                        r["drift"][f"{L}/attn_entropy/{layer}"]["mean_short"]
                        for layer in range(N_LAYERS)
                    ]
                ),
            )
            for r in R
        ]
        out.append(
            f"| attn | query attention entropy (nats; short ref.) | {L} | "
            + "; ".join(f"{n:.2f} → {a:.2f} ({s:.2f})" for n, a, s in ent)
            + " | – |"
        )
    short_ok = all(
        abs(r["acc"][f"{c}/uniform/128"] - r["acc"]["none/uniform/128"]) <= 0.01
        for r in R
        for c in CONDS
    )
    out.append(
        "\nShort-length check (every condition within 1 pp of none at 128): "
        f"{'yes' if short_ok else 'NO'}"
    )

    strict = all(d >= 20 for L in LONG for d in G["G1", L]["per_seed"])
    ok = lambda *ks: all(G[k, L]["passed"] for k in ks for L in LONG)
    if not (sanity and M1 and short_ok):
        verdict = "INCONCLUSIVE"
    elif ok("G1", "G2", "G3a", "G3b", "G4"):
        verdict = "PURSUE"
    elif ok("G1", "G2", "G4"):
        verdict = "PIVOT"
    else:
        verdict = "ABANDON"
    out.append(
        "Secondary strict criterion (gate − none ≥ 20 pp at 512 and 1024 in every seed): "
        f"{'met' if strict else 'not met'}"
    )
    out.append(f"\n## Verdict: **{verdict}**\n")
    (RES / "decision.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))


def set_length_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(LENS, [str(L) for L in LENS])
    ax.minorticks_off()
    ax.set_xlabel("input length (tokens); trained on ≤128")
    ax.axvspan(100, 128 * 1.08, color=GRID, zorder=0)


def acc_curve(
    ax: plt.Axes,
    R: list[dict[str, Any]],
    cond: str,
    pl: str = "uniform",
    label: str | None = None,
    color: str | None = None,
    offset: float = 1.0,
) -> None:
    per_seed = np.array([[r["acc"][f"{cond}/{pl}/{L}"] for L in LENS] for r in R]) * 100
    stats = [mean_ci(per_seed[:, i]) for i in range(len(LENS))]
    xs = np.array(LENS) * offset
    for row in per_seed:
        ax.plot(xs, row, color=color or COLOR[cond], lw=0.8, alpha=0.35)
    ax.errorbar(
        xs,
        [m for m, _ in stats],
        yerr=[h for _, h in stats],
        color=color or COLOR[cond],
        marker="o",
        capsize=2,
        label=label or LABEL[cond],
    )


def target_position(r: dict[str, Any], L: int, pl: str = "uniform") -> np.ndarray:
    n = (L - 3) // 2
    return 1 + 2 * (n - 1 - np.array(r["distance"][f"{pl}/{L}"]))


def fig1() -> None:
    R = load("baseline_seed") or load()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.1))
    for pl in PLACEMENTS:
        acc_curve(axes[0], R, "none", pl, label=pl, color=PLACE_COLOR[pl])
    axes[0].axhline(100 / 64, color=MUTED, ls=":", lw=1)
    axes[0].text(1024, 100 / 64 + 2, "chance", color=MUTED, ha="right", fontsize=7.5)
    set_length_axis(axes[0])
    axes[0].set(ylabel="accuracy (%)", ylim=(0, 102), title="a. By distractor placement")
    axes[0].legend(title="inserted records go…", fontsize=7.5, title_fontsize=7.5)
    bins = [0, 16, 32, 64, 128, 256, 512]
    for L, c in zip(LENS, LAYER_COLOR):
        d = np.concatenate([r["distance"][f"uniform/{L}"] for r in R])
        k = np.concatenate([correct(r, "none", L) for r in R])
        mids: list[float]
        accs: list[float]
        mids, accs = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (d >= lo) & (d < hi)
            if sel.sum() >= 20:
                mids.append(np.sqrt(max(lo, 1) * hi)), accs.append(100 * k[sel].mean())
        axes[1].plot(mids, accs, marker="o", color=c, label=str(L))
    axes[1].set_xscale("log", base=2)
    axes[1].set(
        xlabel="records between relevant record and query",
        ylabel="accuracy (%)",
        ylim=(0, 102),
        title="b. Uniform placement, by distance",
    )
    axes[1].legend(title="length", fontsize=7.5, title_fontsize=7.5)
    edges = [0, 32, 64, 128, 256, 512, 1024]
    for L, c in zip(LENS, LAYER_COLOR):
        pos = np.concatenate([target_position(r, L) for r in R])
        k = np.concatenate([correct(r, "none", L) for r in R])
        mids, accs = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (pos >= lo) & (pos < hi)
            if sel.sum() >= 20:
                mids.append(np.sqrt(max(lo, 8) * hi)), accs.append(100 * k[sel].mean())
        axes[2].plot(mids, accs, marker="o", color=c, label=str(L))
    axes[2].axvline(128, color=MUTED, ls=":", lw=1)
    axes[2].text(124, 90, "trained\nlength", color=MUTED, fontsize=7.5, ha="right")
    axes[2].set_xscale("log", base=2)
    axes[2].set(
        xlabel="token position of the relevant record",
        ylim=(0, 102),
        title="c. Uniform placement, by position",
    )
    fig.suptitle("No intervention (b, c: seeds pooled)", x=0.01, ha="left", fontsize=9.5)
    fig.tight_layout()
    save(fig, "fig1_baseline")


def fig2() -> None:
    R = load()
    sites = [
        ("gate", "gate pre-activation"),
        ("gate_post", "gate post-activation (SiLU)"),
        ("up", "value branch (W_up h)"),
        ("resid", "residual stream"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(13, 5.2))
    for ax, (site, title) in zip(axes[0], sites):
        for layer in range(N_LAYERS):
            v = np.array([[r["drift"][f"{L}/{site}/{layer}"]["w2"] for L in LENS] for r in R])
            for row in v:
                ax.plot(LENS, row, color=LAYER_COLOR[layer], lw=0.7, alpha=0.4)
            ax.plot(
                LENS, v.mean(0), color=LAYER_COLOR[layer], marker="o", label=f"layer {layer + 1}"
            )
        set_length_axis(ax)
        ax.set(
            title=title, ylabel="drift at query (W2, short-SD units)" if site == "gate" else None
        )
    axes[0][0].legend(fontsize=7.5)
    top = max(ax.get_ylim()[1] for ax in axes[0][:4])
    for ax in axes[0][:4]:
        ax.set_ylim(0, top)
    ax = axes[0][4]
    for layer in range(N_LAYERS):
        v = np.array(
            [[r["drift"][f"{L}/attn_entropy/{layer}"]["mean_long"] for L in LENS] for r in R]
        )
        ax.plot(LENS, v.mean(0), marker="o", color=LAYER_COLOR[layer])
    ax.plot(LENS, np.log(np.array(LENS) - 1), color=MUTED, ls=":", lw=1)
    ax.text(1024, np.log(1023) - 0.4, "uniform (ln T)", color=MUTED, ha="right", fontsize=7.5)
    set_length_axis(ax)
    ax.set(title="query attention entropy", ylabel="nats (mean over heads)")
    r0 = R[0]
    for layer, ax in enumerate(axes[1][:4]):
        h = r0["hist"][f"gate/{layer}"]
        hp = r0["hist"][f"gate_post/{layer}"]
        bins = np.histogram_bin_edges(h["short"] + h["long"], 40)
        ax.hist(h["short"], bins, color="#9b9a95", alpha=0.75, label="length 128")
        ax.hist(h["long"], bins, color="#2a78d6", alpha=0.6, label="length 1024")
        ax.set(
            title=f"layer {layer + 1}: most-drifted channel ({h['channel']})",
            xlabel="gate pre-activation",
            ylabel="count" if layer == 0 else None,
        )
        ax.grid(False)
        ax.text(
            0.02,
            0.95,
            f"post-SiLU mean {np.mean(hp['short']):.2f} → {np.mean(hp['long']):.2f}",
            transform=ax.transAxes,
            fontsize=7,
            color=MUTED,
            va="top",
        )
    axes[1][0].legend(fontsize=7.5, loc="center left")
    ax = axes[1][4]
    for layer in range(N_LAYERS):
        smd = np.sort(np.abs(np.array(r0["drift"][f"1024/gate/{layer}"]["smd"])))[::-1]
        ax.plot(np.arange(1, len(smd) + 1), smd, color=LAYER_COLOR[layer], lw=1.5)
    ax.set(
        xscale="log",
        title="per-channel |mean shift| at 1024",
        xlabel="channel rank",
        ylabel="|Δmean| / short SD",
    )
    fig.suptitle(
        "Drift at the query position, length L vs length 128 "
        "(calibration set; top: mean over seeds, thin "
        f"lines = seeds; bottom: seed {r0['seed']})",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "fig2_gate_drift")


def fig3() -> None:
    R = load()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for ax, conds, title in (
        (axes[0], ("none", "gate", "attn", "gate+attn"), "a. Factorial: gate × attention"),
        (axes[1], ("none", "gate", "up", "resid", "sham"), "b. Gate vs. controls"),
    ):
        for i, c in enumerate(conds):
            acc_curve(ax, R, c, offset=1 + 0.025 * (i - len(conds) / 2))
        set_length_axis(ax)
        ax.axhline(100 / 64, color=MUTED, ls=":", lw=1)
        ax.set(ylim=(0, 102), ylabel="accuracy (%)", title=title)
        ax.legend(fontsize=7.5)
    ax = axes[2]
    conds = ("gate", "attn", "gate+attn", "up", "resid", "sham")
    for j, L in enumerate(LONG):
        for i, c in enumerate(conds):
            g = compare(R, c, "none", L, 0)
            x = i + (j - 0.5) * 0.32
            ax.scatter(
                [x] * len(g["per_seed"]), g["per_seed"], s=10, color=COLOR[c], alpha=0.45, zorder=3
            )
            ax.errorbar(
                x,
                g["mean"],
                yerr=g["half"],
                fmt="o" if L == 512 else "s",
                color=COLOR[c],
                capsize=2,
            )
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xticks(
        range(len(conds)),
        [LABEL[c].replace(" (shuffled gate)", "") for c in conds],
        rotation=25,
        ha="right",
    )
    ax.set(
        ylabel="accuracy − none (pp)",
        title="c. Gain over no intervention (○ 512, □ 1024; dots = seeds)",
    )
    ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Accuracy (uniform placement), mean over seeds ± 95% t-CI; thin lines = individual seeds",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "fig3_interventions")


def fig4() -> None:
    R = load()
    panels = (("gate", "gate pre-activation"), ("up", "value branch"), ("resid", "residual stream"))
    conds = ("none", "gate", "sham", "up", "resid", "attn", "gate+attn")
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2))
    for ax, (site, title) in zip(axes, panels):
        for j, L in enumerate(LONG):
            for i, c in enumerate(conds):
                v = [layer_mean(r, c, L, site) for r in R]
                x = i + (j - 0.5) * 0.36
                ax.bar(
                    x,
                    np.mean(v),
                    width=0.34,
                    color=COLOR[c],
                    alpha=1.0 if L == 1024 else 0.55,
                    edgecolor="white",
                )
                ax.scatter([x] * len(v), v, s=6, color=INK, zorder=3)
        ax.set_xticks(
            range(len(conds)), [LABEL[c].split(" (")[0] for c in conds], rotation=30, ha="right"
        )
        ax.set(
            title=f"{title} drift at query",
            ylabel="W2 vs length 128 (layer mean)" if site == "gate" else None,
        )
        ax.grid(axis="x", visible=False)
    ax = axes[3]
    for j, L in enumerate(LONG):
        for i, c in enumerate(("none", "attn", "gate+attn")):
            v = [layer_mean(r, c, L, "attn_entropy") for r in R]
            ax.bar(
                i + (j - 0.5) * 0.36,
                np.mean(v),
                width=0.34,
                color=COLOR[c],
                alpha=1.0 if L == 1024 else 0.55,
                edgecolor="white",
            )
    ref = np.mean(
        [
            [r["drift"][f"128/attn_entropy/{layer}"]["mean_long"] for layer in range(N_LAYERS)]
            for r in R
        ]
    )
    ax.axhline(ref, color=MUTED, ls=":", lw=1)
    ax.text(2.4, ref + 0.05, "length 128", color=MUTED, ha="right", fontsize=7.5)
    ax.set_xticks(
        range(3), [LABEL[c] for c in ("none", "attn", "gate+attn")], rotation=30, ha="right"
    )
    ax.set(title="query attention entropy", ylabel="nats (layer mean)")
    ax.grid(axis="x", visible=False)
    fig.suptitle(
        "Manipulation checks on the held-out check set "
        "(light bars 512, dark bars 1024; dots = seeds)",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "fig4_manipulation")


def fig5() -> None:
    R = load()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.1), sharey=True)
    for ax, pl in zip(axes, PLACEMENTS):
        for i, c in enumerate(("none", "gate", "attn", "gate+attn", "sham")):
            acc_curve(ax, R, c, pl, offset=1 + 0.025 * (i - 2.5))
        set_length_axis(ax)
        ax.set(
            title=f"inserted records: {pl}",
            ylim=(0, 102),
            ylabel="accuracy (%)" if pl == "uniform" else None,
        )
    axes[0].legend(fontsize=7.5)
    fig.suptitle(
        "Interventions by distractor placement "
        "(before = distance to query unchanged; after = distance grows)",
        x=0.01,
        ha="left",
        fontsize=9.5,
    )
    fig.tight_layout()
    save(fig, "fig5_placement")


def fig6() -> None:
    R = load()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for ax, inside, title in (
        (axes[0], True, "relevant record inside trained range (position < 128)"),
        (axes[1], False, "relevant record beyond it (position ≥ 128)"),
    ):
        for j, L in enumerate(LONG):
            for i, c in enumerate(CONDS):
                v: list[float] = []
                for r in R:
                    sel = (target_position(r, L) < 128) == inside
                    v.append(100 * correct(r, c, L)[sel].mean())
                x = i + (j - 0.5) * 0.36
                ax.bar(
                    x,
                    np.mean(v),
                    width=0.34,
                    color=COLOR[c],
                    alpha=1.0 if L == 1024 else 0.55,
                    edgecolor="white",
                )
                ax.scatter([x] * len(v), v, s=6, color=INK, zorder=3)
        ax.set_xticks(
            range(len(CONDS)), [LABEL[c].split(" (")[0] for c in CONDS], rotation=30, ha="right"
        )
        ax.set(title=title, ylim=(0, 102), ylabel="accuracy (%)" if inside else None)
        ax.grid(axis="x", visible=False)
    n_in = {L: np.mean([(target_position(R[0], L) < 128).mean()]) for L in LONG}
    fig.suptitle(
        "Exploratory: where the corrections help "
        "(uniform placement; light bars 512, dark bars 1024; "
        f"dots = seeds; {100 * n_in[512]:.0f}% / {100 * n_in[1024]:.0f}% of test items are inside "
        "at 512 / 1024)",
        x=0.01,
        ha="left",
        fontsize=9,
    )
    fig.tight_layout()
    save(fig, "fig6_exploratory_position")


def save(fig: plt.Figure, name: str) -> None:
    FIG.mkdir(exist_ok=True)
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("wrote", FIG / f"{name}.png")


if __name__ == "__main__":
    items = {
        "fig1": fig1,
        "fig2": fig2,
        "fig3": fig3,
        "fig4": fig4,
        "fig5": fig5,
        "fig6": fig6,
        "decision": decision,
    }
    for name in sys.argv[1:] or items:
        items[name]()
