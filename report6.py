from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from report import boot_ci

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
        a2048, a8192 = feas["before/2048"]["restricted"], feas["before/8192"]["restricted"]
        F = a2048 >= 0.70 and a8192 <= a2048 - 0.20
        out.append(
            f"\nF (≥ 70% at 2048 and ≥ 20 pp lower at 8192, 'before'): {'yes' if F else 'NO'}"
        )
        if not F:
            verdict = (
                "NOT TESTABLE (no length failure to explain)"
                if a2048 >= 0.70
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
        D2 = x.mean() - y.mean() >= 0.20 and boot_ci(x, y)[0] > 0 and x.mean() - s.mean() >= 0.20
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


if __name__ == "__main__":
    decision()
