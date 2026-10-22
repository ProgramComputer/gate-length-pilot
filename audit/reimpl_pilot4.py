from __future__ import annotations

import sys

sys.dont_write_bytecode = True
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import hashlib
import json
import math
import time
from collections.abc import Sequence
import torch
import torch.nn.functional as F
from gatepilot.task import paired_set
from gatepilot.model import Model

WINDOWS = [(64, 128), (128, 256), (256, 512), (512, 1024)]


def norm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, (x.shape[-1],), weight)


@torch.inference_mode()
def forward(
    seq: torch.Tensor,
    state: dict[str, torch.Tensor],
    heads: Sequence[Sequence[int]] = (),
    offset: int | None = None,
    measure: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    x = F.embedding(seq, state["embed.weight"])
    B, T, D = x.shape
    measurements: list[torch.Tensor] = []
    for layer in range(4):
        prefix = f"blocks.{layer}."
        h = norm(x, state[prefix + "norm1.weight"])
        qkv = F.linear(h, state[prefix + "attn.qkv.weight"])
        q, k, v = qkv.reshape(B, T, 3, 4, D // 4).permute(2, 0, 3, 1, 4).unbind(0)
        if measure:
            values: list[torch.Tensor] = []
            for lo, hi in WINDOWS:
                positions = torch.arange(lo + lo % 2, min(hi, T - 2), 2)
                positions = positions[torch.linspace(0, len(positions) - 1, 16).round().long()]
                scores = q[:, :, positions] @ k.transpose(-1, -2) / math.sqrt(D // 4)
                mask = torch.arange(T)[None, :] > positions[:, None]
                scores.masked_fill_(mask, -torch.inf)
                weights = scores.softmax(-1)
                prev = weights.gather(-1, (positions - 1)[None, None, :, None].expand(B, 4, -1, 1))
                values.append(prev.mean((0, 2, 3)))
            measurements.append(torch.stack(values, -1))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        if offset is not None:
            positions = torch.arange(128, T - 2, 2)
            for selected_layer, selected_head in heads:
                if selected_layer == layer:
                    y[:, selected_head, positions, :] = v[:, selected_head, positions - offset, :]
        x = x + F.linear(y.transpose(1, 2).reshape(B, T, D), state[prefix + "attn.out.weight"])
        h = norm(x, state[prefix + "norm2.weight"])
        gate = F.silu(F.linear(h, state[prefix + "w_gate.weight"]))
        up = F.linear(h, state[prefix + "w_up.weight"])
        x = x + F.linear(gate * up, state[prefix + "w_down.weight"])
    logits = F.linear(norm(x[:, -1], state["norm.weight"]), state["head.weight"])
    return logits, torch.stack(measurements) if measure else None


def digest(t: torch.Tensor) -> str:
    return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--examples", type=int, default=200)
    parser.add_argument("--calibration", type=int, default=256)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(6)))
    parser.add_argument("--output", default="audit/reimpl_pilot4.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(93671)
    cal = paired_set(args.calibration, (128, 1024), "uniform", seed=20000)[1024]["seq"]
    test = paired_set(args.examples, (128, 1024), "before", seed=40001)[1024]
    result = {
        "arguments": vars(args),
        "windows_half_open": WINDOWS,
        "calibration_sampling": "16 evenly spaced value positions per input and window",
        "test_seq_sha256": digest(test["seq"]),
        "test_answer_sha256": digest(test["answer"]),
        "seeds": [],
    }
    start = time.monotonic()
    for seed in args.seeds:
        checkpoint = Path(f"runs/seed{seed}/model.pt")
        raw = checkpoint.read_bytes()
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state"]
        model = Model().eval()
        model.load_state_dict(state)
        with torch.inference_mode():
            reference = model(test["seq"][:2])
            independent, _ = forward(test["seq"][:2], state)
        max_error = float((reference - independent).abs().max())
        assert torch.allclose(reference, independent, atol=2e-5, rtol=2e-5), max_error
        sums = torch.zeros(4, 4, 4)
        for pos in range(0, len(cal), args.batch):
            batch = cal[pos : pos + args.batch]
            _, attention = forward(batch, state, measure=True)
            sums += attention * len(batch)
        attention = sums / len(cal)
        heads = (attention[:, :, 0] >= 0.3).nonzero().tolist()
        if not heads:
            best = int(attention[:, :, 0].flatten().argmax())
            heads = [[best // 4, best % 4]]
        row = {
            "seed": seed,
            "checkpoint_sha256": hashlib.sha256(raw).hexdigest(),
            "unpatched_forward_max_abs_error": max_error,
            "binding_heads_zero_based": heads,
            "attention_all_heads": attention.tolist(),
            "selected_mean_attention_by_window": torch.stack([attention[l, h] for l, h in heads])
            .mean(0)
            .tolist(),
            "conditions": {},
        }
        for condition, offset in [("none", None), ("patch", 1), ("sham", 3)]:
            predictions: list[int] = []
            for pos in range(0, args.examples, args.batch):
                logits, _ = forward(test["seq"][pos : pos + args.batch], state, heads, offset)
                predictions.extend(logits.argmax(-1).tolist())
            correct = torch.tensor(predictions) == test["answer"]
            row["conditions"][condition] = {
                "accuracy": float(correct.float().mean()),
                "correct": int(correct.sum()),
                "predictions": predictions,
            }
        row["patch_gain_pp"] = 100 * (
            row["conditions"]["patch"]["accuracy"] - row["conditions"]["none"]["accuracy"]
        )
        row["elapsed_seconds_cumulative"] = time.monotonic() - start
        result["seeds"].append(row)
        Path(args.output).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(
            json.dumps(
                {k: v for k, v in row.items() if k not in ("attention_all_heads", "conditions")}
                | {"accuracy": {k: v["accuracy"] for k, v in row["conditions"].items()}}
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
