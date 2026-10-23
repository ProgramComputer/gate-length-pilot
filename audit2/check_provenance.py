from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
from pathlib import Path
import hashlib
import itertools
import json
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import torch

torch.set_num_threads(4)
from tokenizers import Tokenizer
import independent_nope as a
import pilot6 as p6
from output_tag import tagged


def main() -> None:
    manifest = json.loads((ROOT / "cpu_run/manifest.json").read_text())
    rows = [json.loads(s) for s in (ROOT / "cpu_run/raw.jsonl").read_text().splitlines()]
    script_hash = hashlib.file_digest(
        (ROOT / "independent_nope.py").open("rb"), "sha256"
    ).hexdigest()
    assert script_hash == manifest["script_sha256"]
    tok = Tokenizer.from_file(str(a.CACHE / a.REVISION / "tokenizer.json"))
    builder = a.Tasks(tok)
    assert builder.manifest() == manifest["tasks"]
    groups: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
    for r in rows:
        regenerated = builder.make(r["seed"], r["index"], r["length"], r["placement"])
        assert regenerated["ids"] == r["input_ids"]
        assert regenerated["answer_id"] == r["answer_id"]
        assert r["prediction"]["restricted_correct"] == (
            r["prediction"]["restricted_prediction"] == r["answer_id"]
        )
        assert r["prediction"]["full_correct"] == (
            r["prediction"]["full_prediction"] == r["answer_id"]
        )
        groups.setdefault((r["seed"], r["index"], r["length"]), []).append(r)
    for group in groups.values():
        assert all(r["input_ids"] == group[0]["input_ids"] for r in group)
    sets: dict[str, set[str]] = {}
    task = p6.FillerTask(tok, 16)

    def collect(name: str, seqs: torch.Tensor) -> None:
        hs = [hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in seqs]
        assert len(set(hs)) == len(hs)
        sets[name] = set(hs)

    for k in (16, 8, 4):
        collect(
            f"tune_k{k}",
            p6.FillerTask(tok, k).paired_set(100, "before", 60000, lengths=(2048,))[2048]["seq"],
        )
    collect("cal_inside", task.paired_set(64, "uniform", 20000, lengths=(2048,))[2048]["seq"])
    collect("cal_far", task.paired_set(64, "before", 20001, lengths=(8192,))[8192]["seq"])
    for i, pl in enumerate(("before", "uniform", "after")):
        data = task.paired_set(200, pl, 40000 + i)
        for L in (2048, 4096, 8192):
            collect(f"test_{pl}_{L}", data[L]["seq"])
    overlap = {
        f"{x}/{y}": len(sets[x] & sets[y]) for x, y in itertools.combinations(sorted(sets), 2)
    }
    assert max(overlap.values()) == 0
    original = json.loads((ROOT.parent / "results6/full.json").read_text())
    own = json.loads((ROOT / "cpu_run/selection.json").read_text())
    diff = sorted(
        set(map(tuple, original["binding_heads"]))
        - set(map(tuple, own["selected_heads_zero_based"]))
    )
    out = {
        "independent_main_script_hash_unchanged": True,
        "main_script_sha256": script_hash,
        "independent_rows_verified": len(rows),
        "all_input_ids_exactly_regenerated": True,
        "condition_input_identity": True,
        "all_original_split_pairs_checked": len(overlap),
        "original_exact_input_overlaps": overlap,
        "extra_original_heads": [
            {
                "layer": l,
                "head": h,
                "original_inside_p": original["p_inside"][l][h],
                "independent_selection_p": own["all_head_mean_p"][l][h],
            }
            for l, h in diff
        ],
        "snapshot_hash_matches_phase_A": manifest["model_sha256"]
        == json.loads((ROOT / "phase_a_validation_v3/manifest.json").read_text())["model_sha256"],
    }
    with (ROOT / tagged("phase_c_provenance.json")).open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(
        json.dumps({k: v for k, v in out.items() if k != "original_exact_input_overlaps"}, indent=2)
    )


if __name__ == "__main__":
    main()
