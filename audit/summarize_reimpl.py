from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import numpy as np
from gatepilot.task import paired_set


def main() -> None:
    independent = json.loads((ROOT / "audit/reimpl_pilot4.json").read_text())
    n = independent["arguments"]["examples"]
    answers = paired_set(n, (128, 1024), "before", 40001)[1024]["answer"].numpy()
    draws = np.random.default_rng(0).integers(0, n, (2000, n))
    evidence = {"seeds": []}
    table = [
        "| Seed | None: claimed / audit (%) | Patch: claimed / audit (%) "
        "| Sham: claimed / audit (%) | Patch gain: claimed / audit (pp) | Assessment |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in independent["seeds"]:
        seed = row["seed"]
        saved = json.loads((ROOT / f"results4/seed{seed}.json").read_text())
        correct = {
            c: np.array(row["conditions"][c]["predictions"]) == answers
            for c in ("none", "patch", "sham")
        }
        old = {c: 100 * saved["acc"][f"{c}/before/1024"] for c in correct}
        new = {c: 100 * correct[c].mean() for c in correct}
        delta = correct["patch"].astype(float) - correct["none"].astype(float)
        interval = (100 * np.percentile(delta[draws].mean(1), [2.5, 97.5])).tolist()
        assert interval[0] > 0
        detail = {
            "seed": seed,
            "claimed_accuracy_percent": old,
            "audit_accuracy_percent": new,
            "claimed_gain_pp": old["patch"] - old["none"],
            "audit_gain_pp": new["patch"] - new["none"],
            "audit_paired_bootstrap_gain_ci_pp": interval,
            "audit_far_inside_attention_ratio": row["selected_mean_attention_by_window"][-1]
            / row["selected_mean_attention_by_window"][0],
        }
        evidence["seeds"].append(detail)
        table.append(
            f"| {seed} | {old['none']:.1f} / {new['none']:.1f} | "
            f"{old['patch']:.1f} / {new['patch']:.1f} | "
            f"{old['sham']:.1f} / {new['sham']:.1f} | "
            f"{detail['claimed_gain_pp']:+.1f} / {detail['audit_gain_pp']:+.1f} | Agrees |"
        )
    evidence["mean_claimed_accuracy_percent"] = {
        c: float(np.mean([r["claimed_accuracy_percent"][c] for r in evidence["seeds"]]))
        for c in correct
    }
    evidence["mean_audit_accuracy_percent"] = {
        c: float(np.mean([r["audit_accuracy_percent"][c] for r in evidence["seeds"]]))
        for c in correct
    }
    evidence["audit_D1"] = all(
        r["audit_far_inside_attention_ratio"] <= 0.5 for r in evidence["seeds"]
    )
    evidence["audit_D2"] = (
        np.mean([r["audit_gain_pp"] for r in evidence["seeds"]]) >= 20
        and all(r["audit_paired_bootstrap_gain_ci_pp"][0] > 0 for r in evidence["seeds"])
        and evidence["mean_audit_accuracy_percent"]["patch"]
        - evidence["mean_audit_accuracy_percent"]["sham"]
        >= 20
    )
    assert evidence["audit_D1"] and evidence["audit_D2"]
    (ROOT / "audit/pilot4_comparison.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "audit/pilot4_comparison.md").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("\n".join(table))
    print(json.dumps({k: v for k, v in evidence.items() if k != "seeds"}, indent=2))


if __name__ == "__main__":
    main()
