from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os

os.environ["MPLCONFIGDIR"] = str(ROOT / "audit" / "mplconfig")
import ast
import hashlib
import json
from typing import Any
import numpy as np
import torch
from gatepilot.probe import ridge_accuracy
from gatepilot.task import paired_set, EVAL_LENS


def main() -> None:
    torch.set_num_threads(1)
    evidence: dict[str, Any] = {}
    patterns = torch.tensor([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])
    X = patterns.repeat(64, 1)
    labels = (X[:, 0] * X[:, 1] > 0).long()
    linear = ridge_accuracy(X, labels, patterns, (patterns[:, 0] * patterns[:, 1] > 0).long())
    product = X[:, :1] * X[:, 1:]
    nonlinear = ridge_accuracy(
        product,
        labels,
        patterns[:, :1] * patterns[:, 1:],
        (patterns[:, 0] * patterns[:, 1] > 0).long(),
    )
    assert linear == 0.5 and nonlinear == 1.0
    evidence["probe_counterexample"] = {
        "linear_accuracy": linear,
        "product_feature_accuracy": nonlinear,
        "label_is_deterministic_function_of_representation": True,
    }
    positions = [96, 192, 384, 768]
    probabilities: list[float] = []
    for position in positions:
        logits = torch.zeros(position + 1, dtype=torch.double)
        logits[position - 1] = np.log(64.0)
        probabilities.append(float(logits.softmax(0)[position - 1]))
    evidence["fixed_predecessor_logit_counterexample"] = {
        "positions": positions,
        "unchanged_predecessor_logit_advantage": float(np.log(64.0)),
        "previous_token_attention": probabilities,
        "satisfies_far_less_than_half_inside": probabilities[-1] <= probabilities[0] / 2,
    }
    assert evidence["fixed_predecessor_logit_counterexample"]["satisfies_far_less_than_half_inside"]
    keys = torch.arange(64)
    answer = torch.arange(64)
    stored_value = (answer + 1) % 64
    decoded_key = keys.clone()
    perfect_reader_output = stored_value.clone()
    evidence["unlocalized_residual_error_counterexample"] = {
        "key_binding_probe_accuracy": float((decoded_key == keys).float().mean()),
        "answer_accuracy_of_perfect_reader_of_corrupted_stored_values": float(
            (perfect_reader_output == answer).float().mean()
        ),
    }
    tree = ast.parse((ROOT / "report3.py").read_text())
    direction = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.IfExp)
        and isinstance(n.body, ast.Constant)
        and n.body.value == "semantic"
    )
    sem_worse = (1 - 0.5) / (1 - 0.75)
    pos_worse = 0.5 / 0.25
    actual = eval(
        compile(ast.Expression(direction), "report3.py", "eval"),
        {"sem_worse": sem_worse, "pos_worse": pos_worse},
    )
    registered = "semantic" if sem_worse > pos_worse else "positional"
    assert actual == "semantic" and registered == "positional"
    evidence["direction_tie"] = {
        "semantic_factor": sem_worse,
        "positional_factor": pos_worse,
        "code_direction": actual,
        "registered_direction": registered,
        "actual_data_factors": [],
    }
    for seed in range(3):
        r = json.loads((ROOT / f"results3/a_seed{seed}.json").read_text())
        a, b = r["diagnostics"]["128"], r["diagnostics"]["1024"]
        evidence["direction_tie"]["actual_data_factors"].append(
            {
                "seed": seed,
                "semantic": (1 - b["semantic_score"]) / max(1e-9, 1 - a["semantic_score"]),
                "positional": a["positional_score"] / b["positional_score"],
                "diagnosed_token_types": [h[2] for h in r["heads"]],
            }
        )
    rows: list[dict[str, float]] = []
    for seed in range(6):
        r = json.loads((ROOT / f"results4/seed{seed}.json").read_text())
        attention = [r["prev_attention"][l][h][0] for l, h in r["binding_heads"]]
        rows.append(
            {
                "seed": seed,
                "none": r["acc"]["none/before/1024"],
                "patch": r["acc"]["patch/before/1024"],
                "sham": r["acc"]["sham/before/1024"],
                "patch_gain_pp": 100
                * (r["acc"]["patch/before/1024"] - r["acc"]["none/before/1024"]),
                "individual_selected_head_inside_attention_min": min(attention),
                "individual_selected_head_inside_attention_max": max(attention),
            }
        )
    evidence["saved_pilot4_claims"] = rows
    sets: dict[str, set[str]] = {}
    for name, size, seed in [("cal", 512, 20000), ("chk", 512, 30000), ("test", 1000, 40000)]:
        seq = paired_set(size, EVAL_LENS, "uniform", seed)[1024]["seq"]
        sets[name] = {hashlib.sha256(row.numpy().tobytes()).hexdigest() for row in seq}
    evidence["identical_rows_between_named_sets"] = {
        f"{a}/{b}": len(sets[a] & sets[b])
        for a, b in [("cal", "chk"), ("cal", "test"), ("chk", "test")]
    }
    assert not any(evidence["identical_rows_between_named_sets"].values())
    (ROOT / "audit/claim_checks.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
