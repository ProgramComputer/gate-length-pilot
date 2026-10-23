from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
from pathlib import Path
import json
import math
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import pilot5 as p5
import pilot6 as p6
import independent_nope as a
from output_tag import tagged


def dump(p: Path, obj: dict[str, Any]) -> None:
    with Path(p).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


class CompareHooks(p5.Hooks):
    def __init__(self, positions: list[int]) -> None:
        super().__init__(positions=torch.tensor(positions))
        self.earlier = {}
        self.self_correction = {}

    def measure(self, layer: int, q: torch.Tensor, k: torch.Tensor) -> None:
        super().measure(layer, q, k)
        pos = self.positions
        s = (q[:, :, pos] @ k.transpose(-1, -2)).float() / 8
        cols = torch.arange(k.shape[-2])
        prev = s.gather(-1, (pos - 1)[None, None, :, None].expand(1, 32, -1, 1))[..., 0]
        mask = cols[None, :] < (pos - 1)[:, None]
        mean = s.masked_fill(~mask[None, None], 0).sum(-1) / (pos - 1)[None, None]
        self.earlier[layer] = (prev - mean).mean(0)
        self.self_correction[layer] = self.earlier[layer] - self.margin[layer]


@torch.inference_mode()
def main() -> None:
    a.configure(4)
    out = ROOT / tagged("phase_c_forward")
    out.mkdir(exist_ok=False)
    t0 = time.perf_counter()
    model = a.NoPE(a.CACHE / a.REVISION)
    original = p5.NoPELlama.__new__(p5.NoPELlama)
    torch.nn.Module.__init__(original)
    original.w = {k.removeprefix("model."): v for k, v in model.w.items()}
    original.n_layers, original.n_heads, original.n_kv, original.dh = 22, 32, 4, 64
    tok = Tokenizer.from_file(str(a.CACHE / a.REVISION / "tokenizer.json"))
    ids = [1] + tok.encode(
        "A child opened the window and looked out at the garden. "
        "The birds were singing in the trees.",
        add_special_tokens=False,
    ).ids
    got, _, _ = model.forward(ids, all_logits=True)
    want = original(torch.tensor([ids]), all_logits=True)[0]
    discrepancy = float((got - want).abs().max())
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    x = torch.randn(2, 5, 2048, generator=torch.Generator().manual_seed(72))
    n1 = a.rms(x, model.w["model.norm.weight"])
    n2 = original.rms(x, model.w["model.norm.weight"])
    assert torch.equal(n1, n2)
    flipped = ids.copy()
    flipped[-1] = tok.token_to_id("▁garden")
    changed = original(torch.tensor([flipped]), all_logits=True)[0]
    torch.testing.assert_close(changed[:-1], want[:-1], atol=1e-5, rtol=1e-5)
    heads = {8: [7], 11: [27], 14: [2, 28]}
    pos = torch.tensor([8, 15, 20])

    class IndependentHook:
        def measure(self, *args: object) -> None:
            pass

        def patch(self, l: int, y: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            y = y.clone()
            for h in heads.get(l, []):
                for p in pos.tolist():
                    y[0, h, p] = v[0, h, p - self.offset]
            return y

    hook_diffs: dict[str, float] = {}
    for condition, offset in (("patch", 1), ("sham", 4)):
        hk = p5.Hooks(patch_heads=heads, offset=offset)
        hk.value_positions = pos
        ih = IndependentHook()
        ih.offset = offset
        one = original(torch.tensor([ids]), hooks=hk, all_logits=True)
        two = original(torch.tensor([ids]), hooks=ih, all_logits=True)
        hook_diffs[condition] = float((one - two).abs().max())
        assert torch.equal(one, two)
    result = {
        "precision": "float32 shared checkpoint weights; original constructor bypassed",
        "tokens": len(ids),
        "original_vs_independent_max_abs_logit": discrepancy,
        "rms_exact_equal": True,
        "causality_max_abs_before_query": float((changed[:-1] - want[:-1]).abs().max()),
        "short_real_forward_hook_equivalence_max_abs": hook_diffs,
        "matched_inputs": [],
    }
    print("Short forward checks:", json.dumps(result), flush=True)
    dump(out / "short_checks.json", result)
    original_result = json.loads((ROOT.parent / "results6/full.json").read_text())
    hs = original_result["binding_heads"]
    task = p6.FillerTask(tok, 16)
    data = task.paired_set(64, "before", 20001, lengths=(2048, 8192))
    for L in (2048, 8192):
        seq = data[L]["seq"][0]
        positions = task.value_positions(int(data[L]["a"][0]))
        hook = CompareHooks(positions.tolist())
        start = time.perf_counter()
        logits = original(seq[None], hooks=hook)
        m_code = float(torch.stack([hook.margin[l][h].mean() for l, h in hs]).mean())
        m_reg = float(torch.stack([hook.earlier[l][h].mean() for l, h in hs]).mean())
        row = {
            "seed": 20001,
            "index": 0,
            "length": L,
            "positions": positions.tolist(),
            "mean_p": float(torch.stack([hook.p_prev[l][h].mean() for l, h in hs]).mean()),
            "margin_code": m_code,
            "margin_registered_earlier_only": m_reg,
            "margin_correction": m_reg - m_code,
            "seconds": time.perf_counter() - start,
            "prediction": a.classify(
                logits, {"answer_id": int(data[L]["answer"][0])}, task.values.tolist(), tok
            ),
        }
        result["matched_inputs"].append(row)
        dump(out / f"original_calibration_item_{L}.json", row)
        print("Matched calibration:", json.dumps(row), flush=True)
    data = task.paired_set(200, "before", 40000, lengths=(4096,))[4096]
    headmap: dict[int, list[int]] = {}
    for l, h in hs:
        headmap.setdefault(l, []).append(h)
    for condition, offset in (("none", None), ("patch", 1), ("sham", 4)):
        seq = data["seq"][0]
        start = time.perf_counter()
        logits, _, _ = model.forward(
            seq.tolist(),
            positions=task.value_positions(int(data["a"][0])).tolist(),
            selected=hs,
            condition=condition,
        )
        prediction = a.classify(
            logits, {"answer_id": int(data["answer"][0])}, task.values.tolist(), tok
        )
        row = {
            "seed": 40000,
            "index": 0,
            "length": 4096,
            "condition": condition,
            "seconds": time.perf_counter() - start,
            "prediction": prediction,
            "saved_gpu_restricted_correct": original_result["correct"][f"{condition}/before/4096"][
                0
            ]
            == "1",
        }
        result["matched_inputs"].append(row)
        dump(out / f"original_test_item_4096_{condition}.json", row)
        print("Matched test:", json.dumps(row), flush=True)
    result["elapsed_seconds"] = time.perf_counter() - t0
    dump(out / "checks.json", result)


if __name__ == "__main__":
    main()
