from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parent.parent
HERE = ROOT / "readme"
DATA, BUILD = HERE / "data", HERE / "build"
DPI = 300

FIXES = (
    ("Gate-statistics correction", "p1", "gate"),
    ("Value-branch correction", "p1", "up"),
    ("Residual-stream correction", "p1", "resid"),
    ("Log-length attention temperature", "p1", "attn"),
    ("Gate correction + temperature", "p1", "gate+attn"),
    ("Uniform attention temperature 1.2", "p2", "1.2"),
    ("Uniform attention temperature 1.5", "p2", "1.5"),
    ("Uniform attention temperature 1.8", "p2", "1.8"),
)


def load(path: str) -> Any:
    return json.loads((ROOT / path).read_text())


def bits(s: str) -> np.ndarray:
    return np.frombuffer(s.encode(), dtype=np.uint8) == ord("1")


def toy_acc(src: str, seed: int, key: str, pl: str, L: int = 1024) -> float:
    path = {
        "p1": f"results/seed{seed}.json",
        "p2": f"results2/nope_seed{seed}.json",
        "p4": f"results4/seed{seed}.json",
    }[src]
    return 100 * load(path)["acc"][f"{key}/{pl}/{L}"]


def gain(src: str, key: str, pl: str) -> float:
    base = "1.0" if src == "p2" else "none"
    seeds = range(6) if src == "p4" else range(3)
    return float(np.mean([toy_acc(src, s, key, pl) - toy_acc(src, s, base, pl) for s in seeds]))


def export() -> None:
    v: dict[str, float] = {}
    for i, L in enumerate((128, 256, 512, 1024)):
        v[f"toyCliff{'ABCD'[i]}"] = np.mean(
            [toy_acc("p1", s, "none", "before", L) for s in range(3)]
        )
    full = load("results6/full.json")
    real = {
        c: [100 * bits(full["correct"][f"{c}/before/{L}"]).mean() for L in (2048, 4096, 8192)]
        for c in ("none", "patch", "sham")
    }
    for i in range(3):
        v[f"realCliff{'ABC'[i]}"] = real["none"][i]
    toy = np.mean(
        [
            np.mean(
                [
                    load(f"results4/seed{s}.json")["prev_attention"][l][h]
                    for l, h in load(f"results4/seed{s}.json")["binding_heads"]
                ],
                axis=0,
            )
            for s in range(6)
        ],
        axis=0,
    )
    for i in range(4):
        v[f"toyAtt{'ABCD'[i]}"] = toy[i]
    hs = full["binding_heads"]
    v["realAttIn"] = np.mean([full["p_inside"][l][h] for l, h in hs])
    v["realAttFar"] = np.mean([full["p_far"][l][h] for l, h in hs])
    v["realHeads"] = len(hs)
    for c in ("none", "patch", "sham"):
        v[f"toy{c.title()}"] = np.mean([toy_acc("p4", s, c, "before") for s in range(6)])
        v[f"realTwo{c.title()}"], v[f"realFour{c.title()}"] = real[c][1], real[c][2]
    v["realFourPatchFull"] = 100 * full["acc"]["patch/before/8192"]["full"]
    rows = [(label, gain(src, key, "before")) for label, src, key in FIXES]
    rows.append(("Binding-head patch (oracle)", gain("p4", "patch", "before")))
    DATA.mkdir(exist_ok=True)
    (DATA / "fig4.dat").write_text(
        "y gain hl\n"
        + "".join(
            f"{len(rows) - i} {g:.4f} {int(i == len(rows) - 1)}\n" for i, (_, g) in enumerate(rows)
        )
    )
    (DATA / "fig4_labels.tex").write_text(
        r"\pgfplotsset{fixlabels/.style={yticklabels={"
        + ",".join(label for label, _ in reversed(rows))
        + "}}}\n"
    )
    v["bestFixAnywhere"] = max(gain(src, key, "uniform") for _, src, key in FIXES)
    v["fixMaxFar"] = max(g for _, g in rows[:-1])
    comp = load("audit/pilot4_comparison.json")
    v["auditToyNone"] = comp["mean_audit_accuracy_percent"]["none"]
    v["auditToyPatch"] = comp["mean_audit_accuracy_percent"]["patch"]
    ind = load("audit2/cpu_run_independent_summary.json")
    v["auditTwoNone"] = ind["accuracy"]["4096_none"]["percent"]
    v["auditTwoPatch"] = ind["accuracy"]["4096_patch"]["percent"]
    v["auditTwoN"] = ind["accuracy"]["4096_patch"]["n"]
    v["marginOrig"] = np.mean([full["margin_far"][l][h] for l, h in hs]) / np.mean(
        [full["margin_inside"][l][h] for l, h in hs]
    )
    v["marginAudit"] = ind["diagnostics"]["margin_earlier"]["far_over_inside"]

    def places(k: str) -> int:
        return (
            0
            if k in ("realHeads", "auditTwoN")
            else 2 if k.startswith(("toyAtt", "realAtt", "margin")) else 1
        )

    lines = [
        rf"\newcommand{{\{k}}}{{{float(x):.4f}}}\newcommand{{\{k}L}}{{{float(x):.{places(k)}f}}}"
        for k, x in v.items()
    ]
    (DATA / "values.tex").write_text("\n".join(lines) + "\n")


def build(name: str) -> None:
    tex = next(HERE.glob(f"{name}_*.tex"))
    BUILD.mkdir(exist_ok=True)
    run = subprocess.run(
        [
            "pdflatex",
            "-interaction=nonstopmode",
            "-halt-on-error",
            f"-output-directory={BUILD}",
            tex.name,
        ],
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    if run.returncode:
        sys.exit(f"{tex.name} failed:\n" + "\n".join(run.stdout.splitlines()[-25:]))
    pdf = pdfium.PdfDocument(BUILD / f"{tex.stem}.pdf")
    image = pdf[0].render(scale=DPI / 72).to_pil()
    image.save(HERE / f"{tex.stem}.png", optimize=True)
    print("wrote", (HERE / f"{tex.stem}.png").relative_to(ROOT), image.size)


if __name__ == "__main__":
    export()
    for name in sys.argv[1:] or [f"fig{i}" for i in range(1, 6)]:
        build(name)
