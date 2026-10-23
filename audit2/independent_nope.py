from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

import argparse
import hashlib
import json
import math
import platform
import re
import sys
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parent
CACHE = Path(r"C:\Users\pkchi\.cache\huggingface\hub\models--AntNLP--TinyLlama-NoPE-1.1B\snapshots")
REVISION = "80d3e2f9c6aa04a649c0eedfbc3195c90f4854cd"
PREAMBLE = "Remember the following key and value pairs.\n"
FILLER = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
POOL_SEED = 261022871
SELECTION_SEED = 261022872
TEST_SEED = 261022873
LAYERS, HEADS, KV_HEADS, DIM, HIDDEN, INTERMEDIATE, VOCAB = 22, 32, 4, 64, 2048, 5632, 32000
METRICS = (
    "p_previous",
    "margin_earlier",
    "p_bos",
    "p_self",
    "previous_minus_max_other",
    "logsumexp_other_minus_previous",
    "previous_rank",
)


def new_json(path: Path, obj: dict[str, Any]) -> None:
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def output_dir(name: str) -> Path:
    p = (ROOT / name).resolve()
    if not p.is_relative_to(ROOT) or p == ROOT:
        raise ValueError("Output must be a new subdirectory under audit2")
    p.mkdir(parents=True, exist_ok=False)
    return p


def emit(f: TextIO, row: dict[str, Any]) -> None:
    f.write(json.dumps(row, allow_nan=False) + "\n")
    f.flush()


def rms(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-5) * weight


def manual_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    offset: int = 0,
) -> torch.Tensor:
    scores = q @ k.transpose(-1, -2) / math.sqrt(DIM)
    qi = torch.arange(q.shape[-2]) + offset
    kj = torch.arange(k.shape[-2])
    scores.masked_fill_(kj[None, :] > qi[:, None], -torch.inf)
    return scores.softmax(dim=-1) @ v


def measured_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: Sequence[int],
) -> dict[str, list[float]]:
    pos = torch.as_tensor(positions, dtype=torch.long)
    s = q[0, :, pos, :] @ k[0].transpose(-1, -2) / math.sqrt(DIM)
    cols = torch.arange(k.shape[-2])
    causal = cols[None, :] <= pos[:, None]
    s.masked_fill_(~causal[None, :, :], -torch.inf)
    gather_index = (pos - 1)[None, :, None].expand(HEADS, -1, 1)
    previous = s.gather(-1, gather_index).squeeze(-1)
    probs = s.softmax(-1)
    p_previous = probs.gather(-1, gather_index).squeeze(-1)
    earlier_other = cols[None, :] < (pos - 1)[:, None]
    other_sum = s.masked_fill(~earlier_other[None, :, :], 0).sum(-1)
    margin = previous - other_sum / (pos - 1)[None, :]
    competitors = s.clone()
    competitors.scatter_(-1, gather_index, -torch.inf)
    p_self = probs.gather(-1, pos[None, :, None].expand(HEADS, -1, 1)).squeeze(-1)
    values = (
        p_previous,
        margin,
        probs[:, :, 0],
        p_self,
        previous - competitors.max(-1).values,
        competitors.logsumexp(-1) - previous,
        1 + (competitors > previous[:, :, None]).sum(-1),
    )
    return {name: val.float().mean(-1).tolist() for name, val in zip(METRICS, values)}


def replace_outputs(
    a: torch.Tensor,
    v: torch.Tensor,
    positions: Sequence[int],
    heads: list[int],
    condition: str,
) -> None:
    targets = [int(p) for p in positions if p >= 2048]
    back = 1 if condition == "patch" else 4
    for head in heads:
        if targets:
            a[0, head, targets, :] = v[
                0, head // (HEADS // KV_HEADS), [p - back for p in targets], :
            ]


class NoPE:
    def __init__(self, snapshot: Path) -> None:
        self.snapshot = Path(snapshot)
        self.path = self.snapshot / "model.safetensors"
        self.reader = safe_open(str(self.path), framework="pt", device="cpu")
        self.w = {name: self.reader.get_tensor(name) for name in self.reader.keys()}
        expected = {"model.embed_tokens.weight", "lm_head.weight", "model.norm.weight"}
        layer_shapes = {
            "input_layernorm.weight": (HIDDEN,),
            "post_attention_layernorm.weight": (HIDDEN,),
            "self_attn.q_proj.weight": (HEADS * DIM, HIDDEN),
            "self_attn.k_proj.weight": (KV_HEADS * DIM, HIDDEN),
            "self_attn.v_proj.weight": (KV_HEADS * DIM, HIDDEN),
            "self_attn.o_proj.weight": (HIDDEN, HIDDEN),
            "mlp.gate_proj.weight": (INTERMEDIATE, HIDDEN),
            "mlp.up_proj.weight": (INTERMEDIATE, HIDDEN),
            "mlp.down_proj.weight": (HIDDEN, INTERMEDIATE),
        }
        for i in range(LAYERS):
            for suffix, shape in layer_shapes.items():
                name = f"model.layers.{i}.{suffix}"
                expected.add(name)
                assert tuple(self.w[name].shape) == shape, (name, self.w[name].shape)
        assert set(self.w) == expected, set(self.w).symmetric_difference(expected)
        assert self.w["model.embed_tokens.weight"].shape == (VOCAB, HIDDEN)
        assert self.w["lm_head.weight"].shape == (VOCAB, HIDDEN)
        assert self.w["model.norm.weight"].shape == (HIDDEN,)
        assert all(w.dtype == torch.float32 and w.device.type == "cpu" for w in self.w.values())
        assert self.w["lm_head.weight"].data_ptr() != self.w["model.embed_tokens.weight"].data_ptr()

    @torch.inference_mode()
    def forward(
        self,
        ids: list[int],
        *,
        positions: Sequence[int] = (),
        selected: Sequence[Sequence[int]] = (),
        condition: str = "none",
        all_logits: bool = False,
        cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        return_cache: bool = False,
        reference: bool = False,
    ) -> tuple[
        torch.Tensor,
        dict[str, list[list[float]]] | None,
        list[tuple[torch.Tensor, torch.Tensor]] | None,
    ]:
        assert condition in ("none", "patch", "sham")
        tok = torch.as_tensor(ids, dtype=torch.long, device="cpu").reshape(1, -1)
        length = tok.shape[1]
        x = F.embedding(tok, self.w["model.embed_tokens.weight"])
        stats: dict[str, list[list[float]]] | None = (
            {m: [] for m in METRICS} if len(positions) else None
        )
        new_cache: list[tuple[torch.Tensor, torch.Tensor]] | None = [] if return_cache else None
        by_layer: dict[int, list[int]] = {}
        for layer, head in selected:
            assert 0 <= layer < LAYERS and 0 <= head < HEADS
            by_layer.setdefault(layer, []).append(head)
        for layer in range(LAYERS):
            prefix = f"model.layers.{layer}."

            def w(suffix: str) -> torch.Tensor:
                return self.w[prefix + suffix]

            n = rms(x, w("input_layernorm.weight"))
            q = (
                F.linear(n, w("self_attn.q_proj.weight"))
                .view(1, length, HEADS, DIM)
                .transpose(1, 2)
            )
            k = (
                F.linear(n, w("self_attn.k_proj.weight"))
                .view(1, length, KV_HEADS, DIM)
                .transpose(1, 2)
            )
            v = (
                F.linear(n, w("self_attn.v_proj.weight"))
                .view(1, length, KV_HEADS, DIM)
                .transpose(1, 2)
            )
            offset = 0
            if cache is not None:
                assert not len(positions) and condition == "none"
                offset = cache[layer][0].shape[-2]
                k = torch.cat((cache[layer][0], k), dim=-2)
                v = torch.cat((cache[layer][1], v), dim=-2)
            if return_cache:
                new_cache.append((k, v))
            kg = k.repeat_interleave(HEADS // KV_HEADS, dim=1)
            vg = v.repeat_interleave(HEADS // KV_HEADS, dim=1)
            if reference:
                a = manual_attention(q, kg, vg, offset)
            elif offset:
                assert length == 1
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    a = F.scaled_dot_product_attention(q, kg, vg, is_causal=False)
            else:
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    a = F.scaled_dot_product_attention(q, kg, vg, is_causal=True)
            if stats is not None:
                row = measured_attention(q, kg, positions)
                for name in METRICS:
                    stats[name].append(row[name])
            if condition != "none" and layer in by_layer:
                replace_outputs(a, v, positions, by_layer[layer], condition)
            joined = a.transpose(1, 2).reshape(1, length, HIDDEN)
            x = x + F.linear(joined, w("self_attn.o_proj.weight"))
            del n, q, k, v, kg, vg, a, joined
            n = rms(x, w("post_attention_layernorm.weight"))
            gate = F.silu(F.linear(n, w("mlp.gate_proj.weight")))
            gate.mul_(F.linear(n, w("mlp.up_proj.weight")))
            x = x + F.linear(gate, w("mlp.down_proj.weight"))
            del n, gate
        x = rms(x if all_logits else x[:, -1:, :], self.w["model.norm.weight"])
        logits = F.linear(x, self.w["lm_head.weight"])[0]
        assert torch.isfinite(logits).all(), "Non-finite forward output"
        return logits, stats, new_cache


class Tasks:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.t = tokenizer
        self.preamble = tokenizer.encode(PREAMBLE, add_special_tokens=False).ids
        self.filler = tokenizer.encode(FILLER, add_special_tokens=False).ids
        self.newline = tokenizer.token_to_id("<0x0A>")
        assert self.newline == 13
        eligible = sorted(
            i for word, i in tokenizer.get_vocab().items() if re.fullmatch(r"▁[a-z]{4,10}", word)
        )
        rng = np.random.default_rng(POOL_SEED)
        pool = rng.choice(eligible, 1024, replace=False).tolist()
        self.keys, self.values = pool[:512], pool[512:]
        assert not set(self.keys) & set(self.values)
        for i in pool:
            word = tokenizer.id_to_token(i)[1:]
            encoded = tokenizer.encode("X\n " + word + "\n", add_special_tokens=False).ids
            assert encoded[-2:] == [i, self.newline], (word, encoded)
        self.eligible_count = len(eligible)

    def manifest(self) -> dict[str, Any]:
        return {
            "pool_seed": POOL_SEED,
            "selection_seed": SELECTION_SEED,
            "test_seed": TEST_SEED,
            "preamble": PREAMBLE,
            "filler": FILLER,
            "eligible_count": self.eligible_count,
            "key_pool": self.keys,
            "value_pool": self.values,
            "filler_key_pool_overlap": sorted(set(self.filler) & set(self.keys)),
            "filler_value_pool_overlap": sorted(set(self.filler) & set(self.values)),
        }

    def make(
        self,
        seed: int,
        index: int,
        total: int = 2048,
        placement: str = "before",
    ) -> dict[str, Any]:
        assert total in (2048, 4096, 8192) and placement in ("before", "uniform", "after")
        rng = np.random.default_rng(np.random.SeedSequence([seed, index]))
        keys = rng.choice(self.keys, 16, replace=False).tolist()
        values = rng.choice(self.values, 16, replace=False).tolist()
        query_line = int(rng.integers(16))
        fixed = 1 + len(self.preamble) + 1 + 3 * 16 + 2
        filler_count = 2048 - fixed
        base_before = int(
            rng.integers(math.ceil(0.1 * filler_count), math.floor(0.9 * filler_count) + 1)
        )
        base_after = filler_count - base_before
        extra = total - 2048
        split = int(rng.integers(extra + 1)) if extra else 0
        before = base_before + (
            extra if placement == "before" else split if placement == "uniform" else 0
        )
        after = base_after + (
            extra if placement == "after" else extra - split if placement == "uniform" else 0
        )

        def fill(n: int) -> list[int]:
            return (self.filler * ((n + len(self.filler) - 1) // len(self.filler)))[:n]

        ids = [1] + self.preamble + fill(before) + [self.newline]
        block_start = len(ids)
        positions: list[int] = []
        for key, value in zip(keys, values):
            ids.extend((key, value, self.newline))
            positions.append(len(ids) - 2)
        ids.extend(fill(after))
        ids.extend((self.newline, keys[query_line]))
        assert len(ids) == total
        assert ids[-1] == keys[query_line]
        for j, p in enumerate(positions):
            assert ids[p - 1] == keys[j] and ids[p] == values[j]
            if j:
                assert ids[p - 4] == keys[j - 1]
        return {
            "seed": seed,
            "index": index,
            "length": total,
            "placement": placement,
            "ids": ids,
            "keys": keys,
            "values": values,
            "positions": positions,
            "query_line": query_line,
            "answer_id": values[query_line],
            "block_start": block_start,
            "block_query_distance": total - 1 - positions[query_line],
            "base_filler_before": base_before,
            "base_filler_after": base_after,
            "sham_source_ids": [ids[p - 4] for p in positions],
        }


def classify(
    logits: torch.Tensor,
    task: dict[str, Any],
    value_pool: list[int],
    tokenizer: Tokenizer,
) -> dict[str, Any]:
    last = logits[-1]
    restricted = last[value_pool]
    guess = int(value_pool[int(restricted.argmax())])
    full = int(last.argmax())
    target = int(task["answer_id"])
    competitors = restricted.clone()
    competitors[value_pool.index(target)] = -torch.inf
    top = torch.topk(last, 5)
    return {
        "restricted_prediction": guess,
        "restricted_correct": guess == target,
        "full_prediction": full,
        "full_correct": full == target,
        "answer_logit": float(last[target]),
        "answer_minus_best_other_value": float(last[target] - competitors.max()),
        "restricted_prediction_word": tokenizer.id_to_token(guess),
        "full_prediction_token": tokenizer.id_to_token(full),
        "top5_ids": top.indices.tolist(),
        "top5_logits": top.values.tolist(),
    }


def task_metadata(task: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in task.items() if k != "ids"}


def configure(threads: int) -> None:
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)


def provenance(model: NoPE, tasks: Tasks, args: argparse.Namespace) -> dict[str, Any]:
    return {
        "utc": datetime.now(timezone.utc).isoformat(),
        "command": sys.argv,
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "dtype": "float32",
        "device": "cpu",
        "batch_size": 1,
        "rope": False,
        "snapshot": str(model.snapshot),
        "model_size": model.path.stat().st_size,
        "model_sha256": sha256(model.path),
        "tokenizer_sha256": sha256(model.snapshot / "tokenizer.json"),
        "script_sha256": sha256(__file__),
        "tasks": tasks.manifest(),
        "metric_definitions": {
            "margin_earlier": "s[t,t-1] minus mean s[t,j], 0 <= j < t-1",
            "p_previous": "causal softmax over j <= t, including self",
        },
    }


def validate(model: NoPE, tasks: Tasks, out: Path) -> None:
    t = tasks.t
    checks: dict[str, Any] = {}
    for i in range(32):
        base = tasks.make(TEST_SEED, i)
        for length in (4096, 8192):
            for placement in ("before", "uniform", "after"):
                long = tasks.make(TEST_SEED, i, length, placement)
                assert base["keys"] == long["keys"] and base["values"] == long["values"]
                assert base["answer_id"] == long["answer_id"]
                if placement == "before":
                    assert base["block_query_distance"] == long["block_query_distance"]
                    assert long["positions"][0] - base["positions"][0] == length - 2048
                elif placement == "after":
                    assert base["positions"] == long["positions"]
                if length == 8192 and placement == "before":
                    assert min(long["positions"]) > 6000
    checks["builder_checks"] = (
        "32 inputs x all lengths/placements: "
        "lengths, answer, adjacency, before distance, after position"
    )
    sample = tasks.make(TEST_SEED, 0)
    new_json(
        out / "task_example.json",
        {
            **task_metadata(sample),
            "ids": sample["ids"],
            "decoded": t.decode(sample["ids"], skip_special_tokens=False),
        },
    )

    paragraph = (
        "In the early morning, the town was quiet. A few people were walking to work, and the "
        "baker had already opened his shop. The smell of fresh bread drifted through the street. "
        "Anna stopped to buy a loaf before taking the bus to the library. She had promised to help "
        "her friend find a book about the history of the city. When she arrived, the doors were "
        "still closed, so she sat on a bench and watched the birds in the trees. After a few "
        "minutes, the librarian came outside and welcomed her in. They spent the rest of the "
        "morning looking through old maps and photographs, trying to understand how the "
        "neighborhood had changed over the years."
    )
    ids = [1] + t.encode(paragraph, add_special_tokens=False).ids
    start = time.perf_counter()
    logits, _, _ = model.forward(ids, all_logits=True)
    loss = F.cross_entropy(logits[:-1], torch.tensor(ids[1:]))
    checks["ordinary_english"] = {
        "text": paragraph,
        "tokens": len(ids),
        "loss_nats": float(loss),
        "seconds": time.perf_counter() - start,
    }
    if not 0.5 < float(loss) < 4:
        raise AssertionError(f"Implausible English LM loss: {float(loss)}")
    print("English LM:", checks["ordinary_english"], flush=True)
    short = ids[:48]
    fast, stats, cache = model.forward(
        short, positions=[10, 20, 35], all_logits=True, return_cache=True
    )
    ref, ref_stats, _ = model.forward(
        short, positions=[10, 20, 35], all_logits=True, reference=True
    )
    checks["flash_vs_manual_max_abs_logit"] = float((fast - ref).abs().max())
    checks["flash_vs_manual_max_abs_stat"] = max(
        float(np.max(np.abs(np.array(stats[m]) - np.array(ref_stats[m])))) for m in METRICS
    )
    torch.testing.assert_close(fast, ref, atol=3e-4, rtol=3e-4)
    inc, _, _ = model.forward(ids[48:49], cache=cache)
    full, _, _ = model.forward(ids[:49])
    checks["cached_vs_full_max_abs_logit"] = float((inc - full).abs().max())
    torch.testing.assert_close(inc, full, atol=4e-4, rtol=4e-4)
    prefix, _, _ = model.forward(short[:24], all_logits=True)
    checks["causality_prefix_max_abs_logit"] = float((prefix - fast[:24]).abs().max())
    torch.testing.assert_close(prefix, fast[:24], atol=3e-4, rtol=3e-4)
    for condition in ("patch", "sham"):
        identity, _, _ = model.forward(
            short,
            positions=[10, 20, 35],
            selected=[(0, 0), (7, 23), (21, 31)],
            condition=condition,
            all_logits=True,
        )
        checks[condition + "_short_identity_max_abs"] = float((identity - fast).abs().max())
        assert torch.equal(identity, fast)

    gen = torch.Generator().manual_seed(750203)
    q = torch.randn(1, HEADS, 17, DIM, generator=gen)
    k4 = torch.randn(1, KV_HEADS, 17, DIM, generator=gen)
    v4 = torch.randn(1, KV_HEADS, 17, DIM, generator=gen)
    kg, vg = k4.repeat_interleave(8, 1), v4.repeat_interleave(8, 1)
    computed = measured_attention(q, kg, [5, 12])
    loop_p: list[float]
    loop_margin: list[float]
    loop_p, loop_margin = [], []
    for h in range(HEADS):
        ps: list[float]
        ms: list[float]
        ps, ms = [], []
        for pos in (5, 12):
            scores = torch.tensor(
                [float(torch.dot(q[0, h, pos], k4[0, h // 8, j])) / 8 for j in range(pos + 1)]
            )
            ps.append(float(scores.softmax(-1)[pos - 1]))
            ms.append(float(scores[pos - 1] - scores[: pos - 1].mean()))
        loop_p.append(float(np.mean(ps)))
        loop_margin.append(float(np.mean(ms)))
    np.testing.assert_allclose(loop_p, computed["p_previous"], atol=2e-7)
    np.testing.assert_allclose(loop_margin, computed["margin_earlier"], atol=8e-7)
    checks["selected_row_scalar_reference"] = "pass"
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        grouped = F.scaled_dot_product_attention(q, kg, vg, is_causal=True)
    for h in range(HEADS):
        one = manual_attention(
            q[:, h : h + 1], k4[:, h // 8 : h // 8 + 1], v4[:, h // 8 : h // 8 + 1]
        )
        torch.testing.assert_close(grouped[:, h : h + 1], one, atol=1e-6, rtol=1e-5)
    checks["gqa_head_by_head_reference"] = "pass"

    sources = (
        (torch.arange(KV_HEADS)[:, None, None] * 10000 + torch.arange(2051)[None, :, None])
        .float()
        .expand(KV_HEADS, 2051, DIM)[None]
    )
    for condition, back in (("patch", 1), ("sham", 4)):
        dest = torch.full((1, HEADS, 2051, DIM), -777.0)
        replace_outputs(dest, sources, [2047, 2048, 2050], [0, 7, 8, 31], condition)
        expected = torch.full_like(dest, -777.0)
        for h in (0, 7, 8, 31):
            for pos in (2048, 2050):
                expected[0, h, pos] = (h // 8) * 10000 + pos - back
        assert torch.equal(dest, expected)
    checks["patch_sham_boundary_and_sources"] = "pass"

    continuations: list[dict[str, Any]] = []
    for prompt in ("The capital of France is", "Once upon a time, in a small village,"):
        prompt_ids = [1] + t.encode(prompt, add_special_tokens=False).ids
        logits, _, kv = model.forward(prompt_ids, return_cache=True)
        continuation: list[int] = []
        for _ in range(24):
            nxt = int(logits[-1].argmax())
            continuation.append(nxt)
            if nxt == 2:
                break
            logits, _, kv = model.forward([nxt], cache=kv, return_cache=True)
        row = {
            "prompt": prompt,
            "generated_ids": continuation,
            "generated_text": t.decode(continuation, skip_special_tokens=False),
        }
        continuations.append(row)
        print("Greedy:", row, flush=True)
    checks["greedy_continuations"] = continuations
    new_json(out / "validation.json", checks)
    print(
        "Validation passed:",
        json.dumps(
            {
                k: v
                for k, v in checks.items()
                if k not in ("ordinary_english", "greedy_continuations")
            }
        ),
        flush=True,
    )


def benchmark(model: NoPE, tasks: Tasks, out: Path, lengths: list[int]) -> None:
    with (out / "timings.jsonl").open("x", encoding="utf-8") as f:
        for length in lengths:
            task = tasks.make(SELECTION_SEED, 999, length)
            start_wall, start_cpu = time.perf_counter(), time.process_time()
            _, stats, _ = model.forward(task["ids"], positions=task["positions"])
            assert np.isfinite(np.array(stats["p_previous"])).all()
            row = {
                "length": length,
                "seconds_wall": time.perf_counter() - start_wall,
                "seconds_cpu": time.process_time() - start_cpu,
                "batch": 1,
                "all_head_diagnostics": True,
                "precision": "float32",
            }
            emit(f, row)
            print("Timing:", row, flush=True)


def summarize(
    out: Path,
    selection: list[dict[str, Any]],
    records: list[dict[str, Any]],
    selected: list[list[int]],
) -> dict[str, Any]:
    results: dict[str, Any] = {
        "selected_heads_zero_based": selected,
        "selection_inputs": len(selection),
        "cells": {},
    }
    for length in (2048, 4096, 8192):
        for condition in ("none", "patch", "sham"):
            rows = [r for r in records if r["length"] == length and r["condition"] == condition]
            if rows:
                results["cells"][f"{length}_{condition}"] = {
                    "n": len(rows),
                    "restricted_accuracy": float(
                        np.mean([r["prediction"]["restricted_correct"] for r in rows])
                    ),
                    "full_accuracy": float(
                        np.mean([r["prediction"]["full_correct"] for r in rows])
                    ),
                }
    results["diagnostics"] = {}
    for length in (2048, 4096, 8192):
        rows = [r for r in records if r["length"] == length and r["condition"] == "none"]
        if rows:
            results["diagnostics"][str(length)] = {
                metric: float(
                    np.mean([r["metrics"][metric][l][h] for r in rows for l, h in selected])
                )
                for metric in METRICS
            }
    results["paired_effects"] = {}
    for length in (4096, 8192):
        indexed = {
            (r["index"], r["condition"]): r["prediction"]["restricted_correct"]
            for r in records
            if r["length"] == length
        }
        for a, b in (("patch", "none"), ("patch", "sham")):
            inds = sorted(i for i, c in indexed if c == a and (i, b) in indexed)
            if inds:
                diffs = np.array([int(indexed[i, a]) - int(indexed[i, b]) for i in inds])
                rng = np.random.default_rng(261022874 + length)
                boots = diffs[rng.integers(len(diffs), size=(20000, len(diffs)))].mean(1)
                results["paired_effects"][f"{length}_{a}_minus_{b}"] = {
                    "n": len(inds),
                    "gain_pp": 100 * float(diffs.mean()),
                    "paired_bootstrap_95_pp": (100 * np.quantile(boots, [0.025, 0.975])).tolist(),
                }
    return results


def run(
    model: NoPE,
    tasks: Tasks,
    out: Path,
    args: argparse.Namespace,
    invocation_start: float,
) -> bool:
    deadline = invocation_start + args.max_hours * 3600
    selection: list[dict[str, Any]]
    records: list[dict[str, Any]]
    selected: list[list[int]]
    selection, records, selected = [], [], []
    complete = False
    with (out / "raw.jsonl").open("x", encoding="utf-8") as raw:

        def timed(
            task: dict[str, Any],
            stage: str,
            condition: str = "none",
            heads: Sequence[Sequence[int]] = (),
        ) -> dict[str, Any]:
            if time.perf_counter() >= deadline:
                raise TimeoutError("Wall budget reached between forward passes")
            start = time.perf_counter()
            logits, stats, _ = model.forward(
                task["ids"], positions=task["positions"], selected=heads, condition=condition
            )
            row = {
                "stage": stage,
                **task_metadata(task),
                "condition": condition,
                "seconds": time.perf_counter() - start,
                "elapsed_seconds": time.perf_counter() - invocation_start,
                "input_ids": task["ids"],
                "prediction": classify(logits, task, tasks.values, tasks.t),
                "metrics": stats,
            }
            emit(raw, row)
            print(
                f"{stage} i={task['index']} T={task['length']} {condition} {row['seconds']:.1f}s",
                flush=True,
            )
            return row

        try:
            for i in range(args.selection):
                selection.append(timed(tasks.make(SELECTION_SEED, i), "selection"))
            mean_p = np.mean([r["metrics"]["p_previous"] for r in selection], axis=0)
            selected = np.argwhere(mean_p >= 0.3).tolist()
            fallback = not bool(selected)
            if fallback:
                selected = [list(np.unravel_index(int(mean_p.argmax()), mean_p.shape))]
            selected = [[int(l), int(h)] for l, h in selected]
            new_json(
                out / "selection.json",
                {
                    "n": len(selection),
                    "threshold": 0.3,
                    "fallback": fallback,
                    "selected_heads_zero_based": selected,
                    "all_head_mean_p": mean_p.tolist(),
                },
            )
            print("Selected heads:", selected, "fallback:", fallback, flush=True)
            for i in range(args.diagnostics):
                for length in (2048, 8192):
                    records.append(timed(tasks.make(TEST_SEED, i, length), "diagnostic"))
            for length, n in ((4096, args.n4096), (8192, args.n8192)):
                for i in range(n):
                    task = tasks.make(TEST_SEED, i, length)
                    for condition in ("none", "patch", "sham"):
                        if any(
                            r["index"] == i
                            and r["length"] == length
                            and r["condition"] == condition
                            for r in records
                        ):
                            continue
                        records.append(timed(task, "retrieval", condition, selected))
            complete = True
        except TimeoutError as e:
            print(str(e), flush=True)
        finally:
            summary: dict[str, Any] = (
                summarize(out, selection, records, selected)
                if selected
                else {"selection_inputs": len(selection)}
            )
            summary.update(
                {
                    "complete": complete,
                    "elapsed_seconds": time.perf_counter() - invocation_start,
                    "requested": {
                        "selection": args.selection,
                        "diagnostics": args.diagnostics,
                        "n4096": args.n4096,
                        "n8192": args.n8192,
                    },
                }
            )
            new_json(out / "summary.json", summary)
    return complete


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("validate", "benchmark", "run"))
    p.add_argument("--snapshot", type=Path, default=CACHE / REVISION)
    p.add_argument("--output", required=True, help="NEW subdirectory name below audit2")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument(
        "--lengths", nargs="+", type=int, choices=(2048, 4096, 8192), default=[2048, 4096, 8192]
    )
    p.add_argument("--selection", type=int, default=16)
    p.add_argument("--diagnostics", type=int, default=8)
    p.add_argument("--n4096", type=int, default=32)
    p.add_argument("--n8192", type=int, default=8)
    p.add_argument("--max-hours", type=float, default=3.7)
    args = p.parse_args()
    assert 1 <= args.threads <= 4
    assert args.selection > 0 and args.diagnostics > 0
    assert args.n4096 >= 0 and args.n8192 >= 0 and 0 < args.max_hours <= 4
    invocation_start = time.perf_counter()
    configure(args.threads)
    out = output_dir(args.output)
    model = NoPE(args.snapshot)
    tasks = Tasks(Tokenizer.from_file(str(args.snapshot / "tokenizer.json")))
    new_json(out / "manifest.json", provenance(model, tasks, args))
    if args.mode == "validate":
        validate(model, tasks, out)
        benchmark(model, tasks, out, args.lengths)
    elif args.mode == "benchmark":
        benchmark(model, tasks, out, args.lengths)
    else:
        if not run(model, tasks, out, args, invocation_start):
            sys.exit(2)


if __name__ == "__main__":
    main()
