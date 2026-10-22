from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import json
import math
import torch
from gatepilot.correct import Intervention, CONDITIONS, make_condition, evaluate, fit_correction
from gatepilot.model import Model, rotary
from gatepilot.task import paired_set
from pilot3 import edits_for
from pilot4 import with_patch
from audit.reimpl_pilot4 import forward


def main() -> None:
    torch.set_num_threads(2)
    evidence = {"patch_comparison": [], "sham_saved_standardized_size_ratio": []}
    independent = json.loads((ROOT / "audit/reimpl_pilot4.json").read_text())
    test = paired_set(2, (128, 1024), "before", 40001)
    for row in independent["seeds"]:
        seed = row["seed"]
        saved = json.loads((ROOT / f"results4/seed{seed}.json").read_text())
        assert saved["binding_heads"] == row["binding_heads_zero_based"]
        checkpoint = torch.load(
            ROOT / f"runs/seed{seed}/model.pt", map_location="cpu", weights_only=False
        )
        state = checkpoint["state"]
        model = Model().eval()
        model.load_state_dict(state)
        heads: dict[int, list[int]] = {}
        for layer, head in saved["binding_heads"]:
            heads.setdefault(layer, []).append(head)
        errors: dict[str, float] = {}
        with torch.inference_mode():
            for name, offset in [("none", None), ("patch", 1), ("sham", 3)]:
                reference, _ = forward(test[1024]["seq"], state, saved["binding_heads"], offset)
                original = model(test[1024]["seq"], with_patch(Intervention(), heads, offset))
                errors[name] = float((reference - original).abs().max())
                assert torch.allclose(reference, original, atol=2e-5, rtol=2e-5)
            short_reference = model(test[128]["seq"])
            short_errors: dict[str, float] = {}
            for condition in CONDITIONS:
                iv = make_condition(condition, model, None, None, 128)
                short_errors[condition] = float(
                    (model(test[128]["seq"], iv) - short_reference).abs().max()
                )
                assert short_errors[condition] == 0
        evidence["patch_comparison"].append(
            {
                "seed": seed,
                "same_heads": True,
                "max_abs_logit_error": errors,
                "short_condition_max_error": max(short_errors.values()),
            }
        )

    class Capture(torch.nn.Module):
        def __init__(self, model: Model) -> None:
            super().__init__()
            self.model, self.logits = model, []

        def forward(self, seq: torch.Tensor, iv: Intervention) -> torch.Tensor:
            logits = self.model(seq, iv)
            self.logits.append(logits.detach().clone())
            return logits

    capture = Capture(model)
    evaluate(capture, test[1024], Intervention())
    changed = dict(test[1024], answer=(test[1024]["answer"] + 1) % 64)
    evaluate(capture, changed, Intervention())
    error = float((capture.logits[0] - capture.logits[1]).abs().max())
    assert error == 0
    evidence["scoring_label_canary_max_logit_error"] = error

    class Prefix(Intervention):
        def __init__(self) -> None:
            super().__init__()
            self.prefix = {}

        def __call__(self, site: str, layer: int, x: torch.Tensor) -> torch.Tensor:
            if site == "stream":
                self.prefix[layer] = x[:, :-2].detach().clone()
            return x

    altered = test[1024]["seq"].clone()
    altered[:, -1] = 2 + (altered[:, -1] - 2 + 1) % 512
    a, b = Prefix(), Prefix()
    with torch.inference_mode():
        model(test[1024]["seq"], a)
        model(altered, b)
    evidence["future_query_to_prefix_max_error"] = max(
        float((a.prefix[l] - b.prefix[l]).abs().max()) for l in a.prefix
    )
    assert evidence["future_query_to_prefix_max_error"] == 0

    gen = torch.Generator().manual_seed(78)
    q = torch.randn(1, 4, 20, 32, generator=gen)
    k = torch.randn(1, 4, 20, 32, generator=gen)
    band_errors: dict[str, float] = {}
    for name, alpha in [
        ("sem_a0.5", 0.5),
        ("sem_a0.25", 0.25),
        ("sem_a0", 0.0),
        ("pos_a2", 2.0),
        ("pos_a4", 4.0),
    ]:
        scale, _, _ = edits_for(name, [(1, 2, "value", 1)], 1024, 4, 4, 32)[1]
        qr, kr = rotary(q), rotary(k)
        actual = rotary(q * scale[:, None, :]) @ rotary(k * scale[:, None, :]).transpose(-1, -2)
        qr[:, 2, :, :12] *= alpha
        expected = qr @ kr.transpose(-1, -2)
        band_errors[name] = float((actual - expected).abs().max())
        assert torch.allclose(actual, expected, atol=2e-5, rtol=2e-5)
    evidence["rope_alpha_score_max_errors"] = band_errors
    for seed in range(3):
        record = json.loads((ROOT / f"results/seed{seed}.json").read_text())
        for L in (512, 1024):
            gate = sum(record["size"][f"gate/{L}"].values()) / 4
            sham = sum(record["size"][f"sham/{L}"].values()) / 4
            evidence["sham_saved_standardized_size_ratio"].append(
                {"seed": seed, "length": L, "sham_over_gate": sham / gate}
            )
    small = paired_set(16, (128, 256), "uniform", 20000)
    real = fit_correction(model, "gate", small[128]["seq"], small[256]["seq"])
    sham = fit_correction(model, "gate", small[128]["seq"], small[256]["seq"], sham_seed=1)
    evidence["small_fit_per_layer_size_real"] = real.size
    evidence["small_fit_per_layer_size_sham"] = sham.size
    assert abs(real.size[0] - sham.size[0]) < 1e-6
    assert any(abs(real.size[l] - sham.size[l]) > 1e-4 for l in range(1, 4))
    (ROOT / "audit/intervention_checks.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
