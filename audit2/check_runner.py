from __future__ import annotations

import argparse
import json
import time

import torch
from tokenizers import Tokenizer

import independent_nope as a


class FakeModel:
    def __init__(self, fallback: bool = False) -> None:
        self.fallback = fallback

    def forward(
        self,
        ids: list[int],
        **kwargs: object,
    ) -> tuple[torch.Tensor, dict[str, list[list[float]]], None]:
        stats = {m: [[0.0] * a.HEADS for _ in range(a.LAYERS)] for m in a.METRICS}
        stats["p_previous"][2][11] = 0.2 if self.fallback else 0.4
        return torch.zeros((1, a.VOCAB)), stats, None


def main() -> None:
    a.configure(4)
    out = a.output_dir("phase_a_controlflow_smoke")
    tasks = a.Tasks(Tokenizer.from_file(str(a.CACHE / a.REVISION / "tokenizer.json")))
    results = {"SYNTHETIC_ONLY": True, "tests": {}}
    for fallback in (False, True):
        sub = out / ("fallback" if fallback else "threshold")
        sub.mkdir()
        a.new_json(sub / "manifest.json", {"SYNTHETIC_ONLY": True})
        args = argparse.Namespace(selection=2, diagnostics=1, n4096=1, n8192=1, max_hours=0.01)
        assert a.run(FakeModel(fallback), tasks, sub, args, time.perf_counter())
        summary = json.loads((sub / "summary.json").read_text())
        selection = json.loads((sub / "selection.json").read_text())
        assert summary["complete"]
        assert summary["selected_heads_zero_based"] == [[2, 11]]
        assert selection["fallback"] == fallback
        assert len(summary["cells"]) == 7
        assert all(cell["n"] == 1 for cell in summary["cells"].values())
        rows = [json.loads(line) for line in (sub / "raw.jsonl").read_text().splitlines()]
        assert len(rows) == 9
        assert all(len(r["input_ids"]) == r["length"] for r in rows)
        assert len(summary["paired_effects"]) == 4
        results["tests"][
            sub.name
        ] = "pass: selection, deduplication, raw IDs, paired summaries, all 7 cells"
    sub = out / "deadline"
    sub.mkdir()
    args = argparse.Namespace(selection=2, diagnostics=1, n4096=1, n8192=1, max_hours=0.001)
    assert not a.run(FakeModel(), tasks, sub, args, time.perf_counter() - 100)
    summary = json.loads((sub / "summary.json").read_text())
    assert not summary["complete"] and summary["selection_inputs"] == 0
    results["tests"]["deadline"] = "pass: partial summary and orderly stop"
    for i in range(4):
        for length in (2048, 4096, 8192):
            for placement in ("before", "uniform", "after"):
                task = tasks.make(a.TEST_SEED, i, length, placement)
                text = tasks.t.decode(task["ids"][1:], skip_special_tokens=False)
                assert tasks.t.encode(text, add_special_tokens=False).ids == task["ids"][1:]
    results["tests"]["full_prompt_roundtrip"] = "pass: 4 examples x 3 lengths x 3 placements"
    a.new_json(out / "checks.json", results)
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
