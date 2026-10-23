from __future__ import annotations

import os

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
from output_tag import tagged

OUT = ROOT / tagged("phase_c_checks")
OUT.mkdir(exist_ok=False)
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl_cache")
import contextlib
import hashlib
import io
import json
import math
import runpy
import subprocess
import sys
from collections.abc import Iterator
from typing import Any
from unittest import mock

sys.path.insert(0, str(REPO))
import numpy as np
import torch
from tokenizers import Tokenizer
import pilot5 as p5
import pilot6 as p6
import independent_nope as a
from summarize_cpu_run import exact_binomial


def dump(path: Path, value: dict[str, Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def hashes() -> dict[str, str]:
    paths = (
        subprocess.check_output(["git", "--no-optional-locks", "ls-files", "-z"], cwd=REPO)
        .decode()
        .split("\0")
    )
    return {p: hashlib.file_digest((REPO / p).open("rb"), "sha256").hexdigest() for p in paths if p}


def prevent_writes(event: str, args: tuple[Any, ...]) -> None:
    if event == "open":
        path, mode, flags = args
        write = (isinstance(mode, str) and any(c in mode for c in "wax+")) or bool(
            flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        )
        if write and not isinstance(path, int) and not Path(path).resolve().is_relative_to(ROOT):
            raise PermissionError(f"Audit prevented write outside audit2: {path}")
    elif event in ("os.remove", "os.rename", "os.rmdir"):
        raise PermissionError(f"Audit forbids {event}")
    elif event == "os.mkdir" and not Path(args[0]).resolve().is_relative_to(ROOT):
        raise PermissionError(f"Audit prevented mkdir outside audit2: {args[0]}")
    elif event in ("socket.connect", "socket.getaddrinfo"):
        raise PermissionError("Network forbidden during audit")


def data_checks(tok: Tokenizer) -> dict[str, Any]:
    out: dict[str, Any] = {}
    task = p6.FillerTask(tok, 16)
    by_placement: dict[str, dict[str, Any]] = {}
    fingerprints: dict[str, set[str]] = {}
    for pl_index, pl in enumerate(p5.PLACEMENTS):
        data = task.paired_set(200, pl, 40000 + pl_index)
        positions: dict[int, list[list[int]]] = {L: [] for L in p5.LENGTHS}
        changed_suffix = {L: 0 for L in (4096, 8192)}
        changed_count: dict[int, list[int]] = {L: [] for L in (4096, 8192)}
        for i in range(200):
            query_positions: list[int] = []
            target_value_positions: list[int] = []
            for L in p5.LENGTHS:
                seq = data[L]["seq"][i]
                vp = task.value_positions(int(data[L]["a"][i]))
                keys, vals = seq[vp - 1], seq[vp]
                assert len(seq) == L and len(keys.unique()) == 16
                assert set(keys.tolist()).isdisjoint(task.values.tolist())
                q = (keys == seq[-1]).nonzero().flatten()
                assert len(q) == 1 and int(vals[int(q[0])]) == int(data[L]["answer"][i])
                assert torch.equal(seq[vp[1:] - 4], keys[:-1])
                assert torch.equal(data[2048]["answer"], data[L]["answer"])
                query_positions.append(L - 1)
                target_value_positions.append(int(vp[int(q[0])]))
                positions[L].append([int(vp.min()), int(vp.max())])
                if L != 2048:
                    shortvp = task.value_positions(int(data[2048]["a"][i]))
                    shorttail = data[2048]["seq"][i, shortvp[0] - 1 :]
                    longtail = seq[vp[0] - 1 :]
                    if pl == "before":
                        assert len(shorttail) == len(longtail)
                        changed_count[L].append(int((shorttail != longtail).sum()))
                        changed_suffix[L] += int(not torch.equal(shorttail, longtail))
                    if pl == "after":
                        assert torch.equal(vp, shortvp)
            if pl == "before":
                assert len(set(q - t for q, t in zip(query_positions, target_value_positions))) == 1
        by_placement[pl] = {
            "n": 200,
            "lengths": {str(L): list(data[L]["seq"].shape) for L in p5.LENGTHS},
            "value_position_minmax": {
                str(L): [min(p[0] for p in ps), max(p[1] for p in ps)]
                for L, ps in positions.items()
            },
            "changed_block_to_query_token_suffix_inputs": changed_suffix,
            "changed_tokens_mean": {
                str(L): float(np.mean(c)) for L, c in changed_count.items() if c
            },
        }
        for L in p5.LENGTHS:
            fingerprints[f"test_{pl}_{L}"] = {
                hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in data[L]["seq"]
            }
        if pl == "before":
            b = data[2048]["seq"][0]
            long = data[8192]["seq"][0]
            bp = task.value_positions(int(data[2048]["a"][0]))
            lp = task.value_positions(int(data[8192]["a"][0]))
            out["before_suffix_example"] = {
                "short_last_24": tok.decode(b[-24:].tolist()),
                "long_last_24": tok.decode(long[-24:].tolist()),
                "suffix_tokens_short": b[bp[-1] + 2 :].tolist(),
                "suffix_tokens_long": long[lp[-1] + 2 :].tolist(),
            }
    out["pilot6_data"] = by_placement
    all_sets: dict[str, set[str]] = {}
    for k in p6.K_CHOICES:
        t = p6.FillerTask(tok, k)
        tune = t.paired_set(100, "before", 60000, lengths=(2048,))[2048]["seq"]
        assert tune.shape == (100, 2048)
        all_sets[f"tune_k{k}"] = {hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in tune}
    inside = task.paired_set(64, "uniform", 20000, lengths=(2048,))[2048]
    far = task.paired_set(64, "before", 20001, lengths=(8192,))[8192]
    assert min(int(task.value_positions(int(x)).min()) for x in far["a"]) > 6000
    all_sets["cal_inside"] = {
        hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in inside["seq"]
    }
    all_sets["cal_far"] = {hashlib.sha256(s.numpy().tobytes()).hexdigest() for s in far["seq"]}
    out["exact_sequence_overlap_counts"] = {
        f"{x}/{y}": len(xs & ys)
        for x, xs in all_sets.items()
        for y, ys in {**all_sets, **fingerprints}.items()
        if x < y
    }
    assert all(v == 0 for v in out["exact_sequence_overlap_counts"].values())
    full = task.paired_set(200, "before", 40000)
    narrow = task.paired_set(200, "before", 40000, lengths=(2048,))
    assert torch.equal(full[2048]["seq"], narrow[2048]["seq"])
    out["length_subset_identity"] = "pass"
    out["feasibility_main_inputs"] = (
        "same paired_set(200, placement, 40000+i); exact reuse by construction"
    )
    t5 = p5.Task(tok)
    d5 = t5.paired_set(200, "before", 40000)
    out["pilot5_actual_lengths"] = {str(L): int(d5[L]["seq"].shape[1]) for L in p5.LENGTHS}
    for L, d in d5.items():
        for i in range(200):
            p = t5.value_pos(int(d["target"][i]))
            assert d["seq"][i, p - 1] == d["seq"][i, -1] and d["seq"][i, p] == d["answer"][i]
    out["pilot5_answer_query_positions"] = "pass on 600 sequences"
    out["filler_token_period_candidates"] = [
        n for n in range(1, 100) if torch.equal(task.filler[: 1000 - n], task.filler[n:1000])
    ]
    return out


def hook_checks() -> dict[str, Any]:
    result: dict[str, Any] = {}
    g = torch.Generator().manual_seed(67223)
    q = torch.randn(1, 32, 37, 64, generator=g)
    k = torch.randn(1, 32, 37, 64, generator=g)
    positions = torch.tensor([5, 18, 30])
    hk = p5.Hooks(positions=positions)
    hk.measure(0, q, k)
    own = a.measured_attention(q, k, positions.tolist())
    torch.testing.assert_close(
        hk.p_prev[0].mean(-1), torch.tensor(own["p_previous"]), atol=1e-7, rtol=1e-6
    )
    manual: list[float] = []
    for h in range(32):
        m: list[float] = []
        for pos in positions.tolist():
            scores = (q[0, h, pos] @ k[0, h, : pos + 1].T) / 8
            m.append(float(scores[pos - 1] - (scores.sum() - scores[pos - 1]) / pos))
        manual.append(np.mean(m))
    np.testing.assert_allclose(hk.margin[0].mean(-1).numpy(), manual, atol=5e-7)
    result["p_matches_independent"] = True
    result["margin_matches_self_inclusive_definition"] = True
    result["registered_earlier_margin_max_difference_random"] = float(
        np.max(np.abs(np.array(own["margin_earlier"]) - np.array(manual)))
    )
    q = torch.zeros(1, 32, 6, 64)
    k = torch.zeros_like(q)
    q[:, :, 5, 0] = 8
    k[:, :, 4, 0] = 2
    k[:, :, 5, 0] = 100
    hk = p5.Hooks(positions=torch.tensor([5]))
    hk.measure(0, q, k)
    result["margin_definition_counterexample"] = {
        "registered_earlier_only": 2.0,
        "production": float(hk.margin[0][0, 0]),
    }
    assert result["margin_definition_counterexample"]["production"] == -18.0
    v4 = (
        (torch.arange(4)[:, None, None] * 10000 + torch.arange(2051)[None, :, None])
        .float()
        .expand(4, 2051, 64)[None]
    )
    vg = v4.repeat_interleave(8, 1)
    targets = torch.tensor([2047, 2048, 2050])
    for cond, offset in (("patch", 1), ("sham", 4)):
        hk = p5.Hooks(patch_heads={3: [0, 7, 8, 31]}, offset=offset)
        hk.value_positions = targets[targets >= 2048]
        base = torch.full((1, 32, 2051, 64), -777.0)
        got = hk.patch(3, base, vg)
        expected = base.clone()
        a.replace_outputs(expected, v4, targets.tolist(), [0, 7, 8, 31], cond)
        assert torch.equal(got, expected)
        assert torch.equal(base, torch.full_like(base, -777.0))
        assert torch.equal(hk.patch(2, base, vg), base)
    result["patch_sham_group_sources_and_boundary"] = (
        "exact match to independent replacement; original y unmodified"
    )
    prev = 4.0
    even = np.zeros(99)
    concentrated = np.full(99, -10 / 98)
    concentrated[0] = 10

    def prob(others: np.ndarray) -> float:
        return float(np.exp(prev) / (np.exp(prev) + np.exp(others).sum()))

    result["unchanged_margin_strong_competitor_counterexample"] = {
        "margin_even": prev - float(even.mean()),
        "margin_concentrated": prev - float(concentrated.mean()),
        "p_even": prob(even),
        "p_concentrated": prob(concentrated),
    }
    return result


def trace_stages(tok: Tokenizer) -> dict[str, list[dict[str, Any]]]:
    events: list[dict[str, Any]] = []
    original_set = p6.FillerTask.paired_set

    def traced(
        self: p6.FillerTask,
        n: int,
        placement: str,
        seed: int,
        lengths: tuple[int, ...] = p5.LENGTHS,
    ) -> dict[int, dict[str, torch.Tensor]]:
        events.append(
            {"k": self.k, "n": n, "placement": placement, "seed": seed, "lengths": list(lengths)}
        )
        return original_set(self, n, placement, seed, lengths)

    def fake_score(
        model: object,
        task: p6.FillerTask,
        data: dict[str, torch.Tensor],
        heads: dict[int, list[int]] | None = None,
        offset: int | None = None,
        batch: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ones(len(data["seq"]), dtype=torch.bool), torch.zeros(
            len(data["seq"]), dtype=torch.bool
        )

    with (
        mock.patch.object(p6.FillerTask, "paired_set", traced),
        mock.patch.object(p6, "score", fake_score),
    ):
        chosen = p6.stage_tune(None, tok, 100)
    assert chosen["chosen_k"] == 16 and all(
        e["lengths"] == [2048] and e["seed"] == 60000 for e in events
    )
    batch_calls: list[dict[str, int | list[int] | None]] = []

    class Fake:
        w = {"embed_tokens.weight": torch.zeros(1)}

        def __call__(self, seq: torch.Tensor, hooks: p5.Hooks | None = None) -> torch.Tensor:
            batch_calls.append(
                {
                    "batch": len(seq),
                    "offset": None if hooks is None else hooks.offset,
                    "positions": None if hooks is None else hooks.value_positions.tolist(),
                }
            )
            return torch.zeros(len(seq), 32000)

    task = p6.FillerTask(tok, 16)
    data = task.paired_set(5, "before", 40000, lengths=(4096,))[4096]
    for off in (None, 1, 4):
        p6.score(Fake(), task, data, {8: [7]}, off)
    assert [r["batch"] for r in batch_calls] == [4, 1] + [1] * 10
    batch1_calls: list[dict[str, int]] = []

    def check_score(
        model: object,
        task: p6.FillerTask,
        data: dict[str, torch.Tensor],
        heads: dict[int, list[int]] | None = None,
        offset: int | None = None,
        batch: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch1_calls.append({"batch": batch, "n": len(data["seq"]), "length": data["seq"].shape[1]})
        return fake_score(model, task, data)

    def local_download(repo: str, name: str, *args: object, **kwargs: object) -> str:
        return str(a.CACHE / a.REVISION / name)

    import huggingface_hub

    with (
        mock.patch.object(huggingface_hub, "hf_hub_download", local_download),
        mock.patch.object(p5, "NoPELlama", lambda *args: Fake()),
        mock.patch.object(p6, "score", check_score),
    ):
        with redirect_text({REPO / "results6/none_batch1.json": OUT / "batch1_synthetic.json"}):
            runpy.run_path(str(REPO / "pilot6_batch1.py"), run_name="__main__")
    assert len(batch1_calls) == 9 and all(c["batch"] == 1 and c["n"] == 200 for c in batch1_calls)
    return {
        "tune_calls": events,
        "score_batch_calls": batch_calls,
        "batch1_script_calls_SYNTHETIC": batch1_calls,
    }


@contextlib.contextmanager
def redirect_text(mapping: dict[Path, Path]) -> Iterator[None]:
    original = Path.write_text
    mapping = {p.resolve(): q for p, q in mapping.items()}

    def redirected(self: Path, data: str, *args: str | None, **kwargs: str | None) -> int:
        if self.resolve() in mapping:
            with mapping[self.resolve()].open("x", encoding=kwargs.get("encoding", "utf-8")) as f:
                return f.write(data)
        raise PermissionError(f"Unmapped original write: {self}")

    with mock.patch.object(Path, "write_text", redirected):
        yield


def saved_checks() -> dict[str, Any]:
    f = json.loads((REPO / "results6/full.json").read_text())
    feas = json.loads((REPO / "results6/feasibility.json").read_text())
    b1 = json.loads((REPO / "results6/none_batch1.json").read_text())
    p5feas = json.loads((REPO / "results5/feasibility.json").read_text())
    out = {"checked_cells": 0, "batch_differences": {}, "full_vocab_before": {}, "effects": {}}

    def checkcell(cell: dict[str, Any], bits: str) -> None:
        assert len(bits) == 200 and set(bits) <= set("01")
        assert abs(cell["restricted"] - bits.count("1") / 200) < 4e-8

    for key, cell in p5feas.items():
        checkcell(cell, cell["correct"])
        out["checked_cells"] += 1
    for key, cell in feas.items():
        if key != "k":
            checkcell(cell, cell["correct"])
            out["checked_cells"] += 1
    for key, cell in f["acc"].items():
        checkcell(cell, f["correct"][key])
        out["checked_cells"] += 1
    for key, cell in b1.items():
        checkcell(cell, cell["correct"])
        out["checked_cells"] += 1
    chosen = np.argwhere(np.array(f["p_inside"]) >= 0.3).tolist()
    assert chosen == f["binding_heads"]
    hs = chosen
    out["binding_means"] = {
        key: float(np.mean([f[key][l][h] for l, h in hs]))
        for key in ("p_inside", "p_far", "margin_inside", "margin_far")
    }
    out["margin_ratio"] = out["binding_means"]["margin_far"] / out["binding_means"]["margin_inside"]
    out["margin_per_head_ratio_range"] = [
        float(x)
        for x in (
            np.min([f["margin_far"][l][h] / f["margin_inside"][l][h] for l, h in hs]),
            np.max([f["margin_far"][l][h] / f["margin_inside"][l][h] for l, h in hs]),
        )
    ]
    own = json.loads((ROOT / "cpu_run/selection.json").read_text())["selected_heads_zero_based"]
    out["heads"] = {
        "original": len(hs),
        "independent": len(own),
        "shared": len(set(map(tuple, hs)) & set(map(tuple, own))),
        "original_only": sorted(set(map(tuple, hs)) - set(map(tuple, own))),
        "independent_only": sorted(set(map(tuple, own)) - set(map(tuple, hs))),
    }
    rows = [json.loads(s) for s in (ROOT / "cpu_run/raw.jsonl").read_text().splitlines()]
    out["independent_reweighted_original_heads"] = {}
    for metric in ("p_previous", "margin_earlier"):
        out["independent_reweighted_original_heads"][metric] = {
            str(L): float(
                np.mean(
                    [
                        r["metrics"][metric][l][h]
                        for r in rows
                        if r["stage"] == "diagnostic" and r["length"] == L
                        for l, h in hs
                    ]
                )
            )
            for L in (2048, 8192)
        }
    for pl in p5.PLACEMENTS:
        for L in p5.LENGTHS:
            key = f"none/{pl}/{L}"
            a_bits = f["correct"][key]
            b_bits = b1[key]["correct"]
            assert a_bits == feas[f"{pl}/{L}"]["correct"]
            out["batch_differences"][key] = {
                "net_pp": 100 * (b_bits.count("1") - a_bits.count("1")) / 200,
                "flips": sum(x != y for x, y in zip(a_bits, b_bits)),
            }
            if L == 2048 or pl == "after":
                assert f["correct"][f"patch/{pl}/{L}"] == f["correct"][f"sham/{pl}/{L}"] == b_bits
            if pl == "before":
                out["full_vocab_before"][str(L)] = {
                    c: f["acc"][f"{c}/before/{L}"]["full"] for c in ("none", "patch", "sham")
                }
    import report

    for L in p5.LENGTHS:

        def bits(s: str) -> np.ndarray:
            return np.array([x == "1" for x in s])

        patch = bits(f["correct"][f"patch/before/{L}"])
        sham = bits(f["correct"][f"sham/before/{L}"])
        for label, none in (
            ("registered", bits(f["correct"][f"none/before/{L}"])),
            ("batch1", bits(b1[f"none/before/{L}"]["correct"])),
        ):
            difference = patch.astype(int) - none.astype(int)
            ci = 100 * report.boot_ci(patch, none)
            rng = np.random.default_rng(0)
            ix = rng.integers(0, len(patch), (2000, len(patch)))
            independent_ci = 100 * np.percentile(difference[ix].mean(1), [2.5, 97.5])
            np.testing.assert_allclose(ci, independent_ci, atol=1e-12)
            out["effects"][f"{L}_{label}"] = {
                "gain_pp": 100 * float(difference.mean()),
                "bootstrap_95_pp": ci.tolist(),
                "patch_minus_sham_pp": 100 * float(patch.mean() - sham.mean()),
                "wins": int((difference > 0).sum()),
                "losses": int((difference < 0).sum()),
            }
    return out


def reports() -> dict[str, dict[str, bool | list[str]]]:
    import report

    report.FIG = OUT / "figures"
    out: dict[str, dict[str, bool | list[str]]] = {}
    for file, args, original, target in (
        ("report5.py", ["decision"], "results5/decision.md", "decision5.md"),
        ("report6.py", [], "results6/decision.md", "decision6.md"),
    ):
        with (
            (OUT / (file + ".stdout.txt")).open("x", encoding="utf-8") as log,
            contextlib.redirect_stdout(log),
            redirect_text({REPO / original: OUT / target}),
            mock.patch.object(sys, "argv", [file] + args),
        ):
            runpy.run_path(str(REPO / file), run_name="__main__")
        text = (OUT / target).read_text(encoding="utf-8")
        out[file] = {
            "matches_existing_decision_exact_text": text
            == (REPO / original).read_text(encoding="utf-8"),
            "verdict_line": [s for s in text.splitlines() if "## Verdict" in s],
        }
        assert out[file]["matches_existing_decision_exact_text"]
    assert (OUT / "figures/fig13_pilot6_real_model.png").exists()
    return out


def main() -> None:
    a.configure(4)
    before = hashes()
    dump(OUT / "tracked_before.json", before)
    sys.addaudithook(prevent_writes)
    tok = Tokenizer.from_file(str(a.CACHE / a.REVISION / "tokenizer.json"))
    result = {"data": data_checks(tok), "hooks": hook_checks()}
    with (
        (OUT / "stage_trace_SYNTHETIC.stdout.txt").open("x", encoding="utf-8") as log,
        contextlib.redirect_stdout(log),
    ):
        result["stage_trace_SYNTHETIC"] = trace_stages(tok)
    result["saved"] = saved_checks()
    result["reports"] = reports()
    after = hashes()
    dump(OUT / "tracked_after.json", after)
    result["integrity"] = {
        "tracked_files": len(before),
        "changed": [p for p in before if before[p] != after[p]],
    }
    assert not result["integrity"]["changed"]
    dump(OUT / "checks.json", result)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("data", "stage_trace_SYNTHETIC")},
            indent=2,
        )
    )
    print("Before suffix changes:", result["data"]["pilot6_data"]["before"])


if __name__ == "__main__":
    main()
