from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
from typing import Any
import numpy as np
import torch
from gatepilot.model import Model
from gatepilot.correct import Intervention, evaluate
from gatepilot.task import paired_set, EVAL_LENS
from pilot3 import Hooks, edits_for, CONDITIONS
from pilot4 import with_patch


def main() -> None:
    torch.set_num_threads(2)
    pick = np.sort(np.random.default_rng(83017).choice(1000, 12, replace=False))
    datasets: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    evidence = {"test_indices": pick.tolist(), "groups": {}}
    for name, lengths, placement, seed in [
        ("full_uniform", EVAL_LENS, "uniform", 40000),
        ("reduced_after", (128, 1024), "after", 40002),
        ("reduced_before", (128, 1024), "before", 40001),
    ]:
        data = paired_set(1000, lengths, placement, seed)
        datasets[name] = {L: {k: v[pick] for k, v in item.items()} for L, item in data.items()}

    def check(
        group: str,
        model: Model,
        data: dict[str, torch.Tensor],
        iv: Intervention,
        record: dict[str, Any],
        key: str,
    ) -> None:
        actual, _ = evaluate(model, data, iv, batch=4)
        expected = torch.tensor([record["correct"][key][i] == "1" for i in pick])
        item = evidence["groups"].setdefault(group, {"predictions_checked": 0, "mismatches": []})
        item["predictions_checked"] += len(pick)
        bad = (actual != expected).nonzero().flatten().tolist()
        item["mismatches"].extend([{"key": key, "test_index": int(pick[i])} for i in bad])

    for pos, runs in [("nope", "runs"), ("rope", "runs_rope")]:
        for seed in range(3):
            checkpoint = torch.load(
                ROOT / f"{runs}/seed{seed}/model.pt", map_location="cpu", weights_only=False
            )
            model = Model(pos=pos).eval()
            model.load_state_dict(checkpoint["state"])
            r = json.loads((ROOT / f"results2/{pos}_seed{seed}.json").read_text())
            for temp in (1.0, 1.2, 1.5, 1.8):
                for L in (128, 1024):
                    check(
                        f"pilot2_{pos}_seed{seed}",
                        model,
                        datasets["full_uniform"][L],
                        Intervention(uniform_temp=temp),
                        r,
                        f"{temp}/uniform/{L}",
                    )
            if pos == "nope":
                first = json.loads((ROOT / f"results/seed{seed}.json").read_text())
                assert all(
                    first["correct"][f"none/{pl}/{L}"] == r["correct"][f"1.0/{pl}/{L}"]
                    for pl in ("uniform", "before", "after")
                    for L in EVAL_LENS
                )
            else:
                third = json.loads((ROOT / f"results3/a_seed{seed}.json").read_text())
                for condition in CONDITIONS:
                    edit = edits_for(condition, third["heads"], 1024, 4, 4, 32)
                    check(
                        f"pilot3a_rope_seed{seed}",
                        model,
                        datasets["reduced_after"][1024],
                        Hooks(edit),
                        third,
                        f"{condition}/after/1024",
                    )
    for seed in range(6):
        checkpoint = torch.load(
            ROOT / f"runs/seed{seed}/model.pt", map_location="cpu", weights_only=False
        )
        model = Model().eval()
        model.load_state_dict(checkpoint["state"])
        fourth = json.loads((ROOT / f"results4/seed{seed}.json").read_text())
        heads: dict[int, list[int]] = {}
        for layer, head in fourth["binding_heads"]:
            heads.setdefault(layer, []).append(head)
        for name, offset in [("none", None), ("patch", 1), ("sham", 3)]:
            check(
                f"pilot4_nope_seed{seed}",
                model,
                datasets["reduced_before"][1024],
                with_patch(Intervention(), heads, offset),
                fourth,
                f"{name}/before/1024",
            )
    evidence["pilot1_vs_pilot2_all_baseline_bitstrings_identical"] = True
    evidence["total_predictions_checked"] = sum(
        x["predictions_checked"] for x in evidence["groups"].values()
    )
    evidence["total_mismatches"] = sum(len(x["mismatches"]) for x in evidence["groups"].values())
    (ROOT / "audit/saved_prediction_checks.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))
    assert evidence["total_mismatches"] == 0


if __name__ == "__main__":
    main()
