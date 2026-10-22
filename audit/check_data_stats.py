from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import os

os.environ["MPLCONFIGDIR"] = str(ROOT / "audit" / "mplconfig")
import hashlib
import json
import numpy as np
import torch
from gatepilot.task import paired_set, EVAL_LENS, KEY0, VAL0, N_KEYS, N_VALS, roles, VAL
from gatepilot.probe import StreamRecorder
from report import boot_ci, mean_ci, target_position


def main() -> None:
    torch.set_num_threads(1)
    evidence = {"data": {}, "saved_json_consistency": {}, "test_set_identity": {}}
    inputs: dict[str, dict[int, dict[str, torch.Tensor]]] = {}
    for index, placement in enumerate(("uniform", "before", "after")):
        data = paired_set(1000, EVAL_LENS, placement, seed=40000 + index)
        inputs[placement] = data
        positions: dict[int, torch.Tensor] = {}
        for L, item in data.items():
            seq = item["seq"]
            keys, values = seq[:, 1:-2:2], seq[:, 2:-2:2]
            matched = keys == seq[:, -1, None]
            assert (matched.sum(1) == 1).all()
            assert ((keys >= KEY0) & (keys < VAL0)).all()
            assert ((values >= VAL0) & (values < VAL0 + N_VALS)).all()
            assert (keys.sort(1).values.diff(dim=1) != 0).all()
            target = matched.long().argmax(1)
            actual = values[torch.arange(len(seq)), target] - VAL0
            assert torch.equal(actual, item["answer"])
            assert torch.equal(keys.shape[1] - 1 - target, item["distance"])
            positions[L] = 1 + 2 * target
            assert torch.equal(item["answer"], data[128]["answer"])
            patch_pos = (
                torch.arange(128, seq.shape[1] - 2, 2)
                if seq.shape[1] - 2 > 128
                else torch.empty(0, dtype=torch.long)
            )
            actual_pos = (
                ((roles(seq)[0] == VAL) & (torch.arange(seq.shape[1]) >= 128)).nonzero().flatten()
            )
            assert torch.equal(patch_pos, actual_pos)
        if placement == "before":
            assert all(torch.equal(data[L]["distance"], data[128]["distance"]) for L in EVAL_LENS)
        if placement == "after":
            assert all(torch.equal(positions[L], positions[128]) for L in EVAL_LENS)
        evidence["data"][
            placement
        ] = "1000 examples x four lengths: keys, labels, pairing, distances, patch positions PASS"

    for index, placement in enumerate(("uniform", "before", "after")):
        full = inputs[placement]
        reduced = paired_set(1000, (128, 1024), placement, seed=40000 + index)
        evidence["test_set_identity"][placement] = {
            "short_identical": bool(torch.equal(full[128]["seq"], reduced[128]["seq"])),
            "long_identical_rows": int((full[1024]["seq"] == reduced[1024]["seq"]).all(1).sum()),
            "answers_identical": bool(torch.equal(full[1024]["answer"], reduced[1024]["answer"])),
            "long_distance_equal_rows": int(
                (full[1024]["distance"] == reduced[1024]["distance"]).sum()
            ),
        }
    for directory in ("results", "results2", "results3", "results4"):
        errors: list[list[str | float]]
        mismatched_distances: list[list[str | int]]
        files, conditions, errors, mismatched_distances = 0, 0, [], []
        for file in sorted((ROOT / directory).glob("*seed*.json")):
            record = json.loads(file.read_text())
            files += 1
            for key, correct in record.get("correct", {}).items():
                conditions += 1
                assert len(correct) == 1000 and set(correct) <= {"0", "1"}, (file, key)
                actual = correct.count("1") / len(correct)
                if abs(actual - record["acc"][key]) > 5.1e-5:
                    errors.append([file.name, key, actual, record["acc"][key]])
            for key, distance in record.get("distance", {}).items():
                placement, L = key.split("/")
                reference = inputs[placement][int(L)]["distance"].tolist()
                mismatch = sum(a != b for a, b in zip(distance, reference))
                if mismatch:
                    reduced = paired_set(
                        1000,
                        (128, 1024),
                        placement,
                        seed=40000 + ("uniform", "before", "after").index(placement),
                    )
                    assert distance == reduced[int(L)]["distance"].tolist()
                    mismatched_distances.append(
                        [file.name, key, mismatch, "matches reduced-length generator"]
                    )
        evidence["saved_json_consistency"][directory] = {
            "files": files,
            "scored_conditions": conditions,
            "accuracy_vs_bitstring_errors": errors,
            "distance_mismatches_from_pilot1": mismatched_distances,
        }
    seq = paired_set(12, (1024,), "uniform", 50000)[1024]["seq"]
    recorder = StreamRecorder()
    for start in range(0, 12, 4):
        batch = seq[start : start + 4]
        recorder.begin(batch)
        assert torch.equal(
            recorder.labels[-1].reshape(4, -1), batch[recorder.batch_rows, recorder.pos - 1] - KEY0
        )
        assert all(int(recorder.examples[-1].reshape(4, -1)[b, 0]) == start + b for b in range(4))
    examples = torch.cat(recorder.examples)
    assert set(examples[examples < 9].tolist()).isdisjoint(set(examples[examples >= 9].tolist()))
    evidence["probe_split_and_labels"] = (
        "12 inputs, 768 labelled rows; preceding-key labels and sequence-level split PASS"
    )
    a = np.array([True, False, True, True, False, False, True])
    b = np.array([False, False, True, False, True, False, True])
    draws = np.random.default_rng(0).integers(0, len(a), (2000, len(a)))
    reference = np.percentile((a.astype(float) - b.astype(float))[draws].mean(1), [2.5, 97.5])
    assert np.allclose(boot_ci(a, b), reference)
    t_mean, t_half = mean_ci([1, 2, 4])
    assert np.isclose(t_half, 4.303 * np.std([1, 2, 4], ddof=1) / np.sqrt(3))
    evidence["statistics"] = {
        "paired_bootstrap_matches_signed_difference": reference.tolist(),
        "three_seed_t_mean": float(t_mean),
        "three_seed_t_halfwidth": float(t_half),
    }
    result = ROOT / "audit" / "data_stats_checks.json"
    result.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
