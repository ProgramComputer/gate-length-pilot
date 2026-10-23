from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from output_tag import tagged

OUT = ROOT / tagged("phase_c_decision_edges")
OUT.mkdir(exist_ok=False)
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl_cache")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
import contextlib
import io
import json
import sys
from typing import Any
from unittest import mock

sys.path.insert(0, str(ROOT.parent))
import torch

torch.set_num_threads(4)
import report6
import pilot6


def records(
    patch_count: int = 140,
    none_count: int = 100,
    sham_count: int = 0,
    inside: float = 0.85,
    far: float = 0.05,
    ratio: float = 0.65,
) -> dict[str, Any]:
    tune = {str(k): {"restricted": 0.81, "full": 0.3} for k in (16, 8, 4)}
    tune["chosen_k"] = 16
    feas = {
        f"{pl}/{L}": {"restricted": inside if L == 2048 else far if L == 8192 else 0.5}
        for pl in ("before", "uniform", "after")
        for L in (2048, 4096, 8192)
    }
    full = {
        "binding_heads": [[0, 0]],
        "p_inside": [[0.6]],
        "p_far": [[0.1]],
        "margin_inside": [[10]],
        "margin_far": [[10 * ratio]],
        "correct": {},
        "acc": {},
    }
    for cond, n in (("patch", patch_count), ("none", none_count), ("sham", sham_count)):
        full["correct"][f"{cond}/before/8192"] = "1" * n + "0" * (200 - n)
        for L in (2048, 4096, 8192):
            full["acc"][f"{cond}/before/{L}"] = {"restricted": n / 200}
    return {"tune": tune, "feasibility": feas, "full": full}


def decision(name: str, data: dict[str, Any]) -> str:
    written: list[str] = []

    def write(path: Path, text: str, *args: str | None, **kwargs: str | None) -> int:
        with (OUT / (name + ".md")).open("x", encoding="utf-8") as f:
            f.write(text)
        written.append(text)
        return len(text)

    with (
        mock.patch.object(report6, "load", lambda name: data[name]),
        mock.patch.object(Path, "write_text", write),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        report6.decision()
    return next(line for line in written[0].splitlines() if "## Verdict" in line)


def main() -> None:
    result = {"SYNTHETIC_ONLY": True, "cases": {}}
    for label, k in (("D2_39_of_200", 139), ("D2_40_of_200", 140), ("D2_41_of_200", 141)):
        result["cases"][label] = {
            "exact_gain_pp": (k - 100) / 2,
            "verdict": decision(label, records(patch_count=k)),
        }
    assert "DOES NOT RESCUE" in result["cases"]["D2_40_of_200"]["verdict"]
    assert "HOLDS" in result["cases"]["D2_41_of_200"]["verdict"]
    inside = float(torch.tensor([True] * 140 + [False] * 60).float().mean())
    result["cases"]["F_exact_70_percent"] = {
        "stored_float32": inside,
        "verdict": decision("F_exact_70_percent", records(inside=inside, patch_count=160)),
    }
    assert "NOT TESTABLE" in result["cases"]["F_exact_70_percent"]["verdict"]

    class DummyTask:
        def __init__(self, tok: object, k: int) -> None:
            pass

        def paired_set(self, *args: object, **kwargs: object) -> dict[int, dict[str, torch.Tensor]]:
            return {2048: {}}

    def score(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor([True] * 70 + [False] * 30), torch.zeros(100, dtype=torch.bool)

    with (
        mock.patch.object(pilot6, "FillerTask", DummyTask),
        mock.patch.object(pilot6, "score", score),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        tune = pilot6.stage_tune(None, None, 100)
    result["cases"]["tune_exact_70_percent"] = tune
    assert tune["chosen_k"] is None
    for ratio, expected in (
        (0.5, "PREFERENCE LOST"),
        (0.65, "IN BETWEEN"),
        (0.8, "PREFERENCE KEPT"),
    ):
        label = f"ratio_{ratio}"
        result["cases"][label] = {"verdict": decision(label, records(patch_count=160, ratio=ratio))}
        assert expected in result["cases"][label]["verdict"]
    with (OUT / "checks.json").open("x", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
