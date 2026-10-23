from __future__ import annotations

import os

os.environ["HF_HUB_OFFLINE"] = "1"
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
import pilot5
from output_tag import tagged


def main() -> None:
    torch.set_num_threads(4)
    q = torch.zeros(1, 32, 4, 64, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    q[:, :, 3, :2] = 8
    k[:, :, :, 0] = 128
    k[:, :, 2, 1] = 0.2578125
    for j in range(4):
        v[:, :, j, j] = 1
    hook = pilot5.Hooks(positions=torch.tensor([3]))
    hook.measure(0, q, k)
    with sdpa_kernel(SDPBackend.MATH):
        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        fp32 = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
    p_hook = float(hook.p_prev[0][0, 0])
    p_sdpa = float(output[0, 0, 3, 2])
    p_fp32 = float(fp32[0, 0, 3, 2])
    assert p_hook == 0.25 and p_sdpa >= 0.3 and abs(p_sdpa - p_fp32) < 0.002
    out = {
        "SYNTHETIC_ONLY": True,
        "head_dimension": 64,
        "qk_dtype": "bfloat16",
        "device": "cpu",
        "hook_predecessor_probability": p_hook,
        "sdpa_bf16_output_probability": p_sdpa,
        "fp32_accumulation_reference_probability": p_fp32,
        "threshold_0_3_hook_qualifies": p_hook >= 0.3,
        "threshold_0_3_sdpa_qualifies": p_sdpa >= 0.3,
        "interpretation": (
            "QK is rounded to BF16 before Hooks.measure casts to float32. "
            "This can change head selection. No claim that this canary describes the original "
            "39 heads or changes their actual verdict."
        ),
    }
    with (ROOT / tagged("phase_c_attention_precision.json")).open("x", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
