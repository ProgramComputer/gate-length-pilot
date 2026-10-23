from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
from output_tag import tagged

OUT = ROOT / tagged("phase_c_numeric_details")
OUT.mkdir(exist_ok=False)
os.environ["MPLCONFIGDIR"] = str(OUT / "mpl_cache")
os.environ["HF_HUB_OFFLINE"] = "1"
import contextlib
import io
import json
import sys
from unittest import mock

sys.path.insert(0, str(ROOT.parent))
import torch

torch.set_num_threads(4)
import pilot5
import report5


def main() -> None:
    gen = torch.Generator().manual_seed(999001)
    tensors = {
        "model.embed_tokens.weight": torch.randn(7, 8, generator=gen),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(8, 8, generator=gen),
        "lm_head.weight": torch.randn(7, 8, generator=gen),
    }
    with mock.patch.object(pilot5, "load_file", return_value=tensors):
        model = pilot5.NoPELlama("NOT_A_REAL_PATH", torch.device("cpu"))
    assert model.n_layers == 1
    for name, tensor in tensors.items():
        loaded = model.w[name.removeprefix("model.")]
        assert loaded.device.type == "cpu" and loaded.dtype == torch.bfloat16
        assert torch.equal(loaded, tensor.to(torch.bfloat16))
    x = torch.randn(2, 13, 2048, generator=gen).to(torch.bfloat16)
    weight = torch.randn(2048, generator=gen).to(torch.bfloat16)
    variance = (x.float() * x.float()).sum(-1, keepdim=True) / 2048
    expected = (x.float() / torch.sqrt(variance + 1e-5)).to(torch.bfloat16) * weight
    actual = model.rms(x, weight)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    feas = {
        f"{pl}/{L}": {"restricted": float(torch.tensor(140 / 200)) if L == 2048 else 0.05}
        for pl in ("before", "uniform", "after")
        for L in (2048, 4096, 8192)
    }
    text: list[str] = []

    def write(path: Path, body: str, *args: str | None, **kwargs: str | None) -> int:
        with (OUT / "report5_gate_exact70.md").open("x", encoding="utf-8") as f:
            f.write(body)
        text.append(body)
        return len(body)

    with (
        mock.patch.object(report5, "load", lambda name: feas if name == "feasibility" else None),
        mock.patch.object(Path, "write_text", write),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        report5.decision()
    assert "NOT TESTABLE" in text[0]
    out = {
        "constructor_SYNTHETIC_tensor_loading": (
            "pass: exact names, BF16 casts, CPU placement; "
            "constructor executed with a fake loader"
        ),
        "bf16_rmsnorm_scalar_formula_max_abs_difference": float((actual - expected).abs().max()),
        "report5_gate_boundary_SYNTHETIC": {
            "correct_count": 140,
            "n": 200,
            "stored_float32": float(torch.tensor(140 / 200)),
            "verdict_line": next(s for s in text[0].splitlines() if "## Verdict" in s),
        },
    }
    with (OUT / "checks.json").open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
