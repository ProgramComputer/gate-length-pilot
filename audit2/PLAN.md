# Phase A: independent CPU experiment, frozen before the main run

No main experiment has been run. The command below is ready for the owner to run unmodified. It uses no network, GPU, Hub API, transformers, or original pilot implementation. It creates new files only under `audit2/` and performs no git writes.

## What was read

Research sources: `PREREGISTRATION-5.md`, `PREREGISTRATION-6.md`, and `AUDIT.md` only. No pilot-5/6 source, reports, results directories, README, or logs were opened. Local checkpoint tensors and tokenizer were read as authorized. Registration 6 itself contains post-result numerical disclosures; this is therefore independent of the implementation and full results, but not completely blind to the claimed outcome.

Checkpoint snapshot found on disk:

`C:\Users\pkchi\.cache\huggingface\hub\models--AntNLP--TinyLlama-NoPE-1.1B\snapshots\80d3e2f9c6aa04a649c0eedfbc3195c90f4854cd`

Its `model.safetensors` is **4,400,216,536 bytes**, with float32 weights. The implementation asserts the supplied architecture against every tensor name and shape, including the untied readout. Full checkpoint and tokenizer SHA-256 hashes are in the validation manifest and will also be recorded in the main-run manifest.

## Implementation and fixed choices

`independent_nope.py` implements embedding, pre-attention RMSNorm (eps 1e-5), Q/K/V projections, scaled causal attention, output projection and residual, post-attention RMSNorm, SwiGLU and residual, final RMSNorm, and the language-model readout. No positional transformation is applied. Query heads 0–7 share KV head 0, 8–15 share KV head 1, etc. Everything uses float32, CPU, batch size 1, four Torch threads, and one interop thread. CPU flash attention avoids a full 32-head square attention matrix; requested diagnostic rows are computed explicitly from Q and K.

The independently drawn task uses:

- Exactly **16 distinct keys and 16 distinct values per block**. Fixed pools contain 512 keys and 512 candidate values, disjoint, sampled from the 7,427 tokenizer entries matching `▁[a-z]{4,10}`. Pool seed: **261022871**. No tuning of k is performed.
- Preamble: `Remember the following key and value pairs.` followed by a newline. Filler: the exact grass/sky/sun/Here/There passage in registration 6, repeated as a token stream and trimmed to length.
- A base block location drawn uniformly over integer filler offsets between 10% and 90%. Each record is `[key, value, newline]`; decoding produces the requested space-prefixed words. After suffix filler comes `[newline, query_key]`. Total length **includes BOS id 1 and the query**.
- Extra filler goes before, after, or at a random integer split for uniform placement. The builder supports all three; the budgeted inference below uses **before**. Keys, values, answer, and block-to-query distance are preserved across before lengths. Every 8192 before block lies beyond position 6000.
- Pool/selection/test RNGs are separate. Each example uses `SeedSequence([seed, index])`, so changing sample counts cannot change earlier examples. Selection seed: **261022872**; evaluation seed: **261022873**. Complete pools and input IDs are saved. No task accuracy was inspected during the timing passes.

Head selection uses the mean predecessor attention across the 16 block values and all selection inputs. Select every head with mean >= 0.3. If none qualifies, use the single highest head as the registration specifies, with an explicit `fallback: true` flag. **A fallback is not evidence that a qualifying binding head exists.**

Diagnostics use a separate, fresh sample, paired between 2048 and 8192. At each value position t:

- `p_previous` is causal softmax probability on t-1, with self included in the softmax denominator.
- `margin_earlier` is `score(t,t-1) - mean(score(t,j), 0 <= j < t-1)`, with scores divided by sqrt(64). Both self and predecessor are excluded from the comparison mean.
- Additional recorded measures are BOS attention, self attention, predecessor minus maximum competitor score, competitor log-sum-exp minus predecessor score, and predecessor rank. These help distinguish an average-score change from competition by a few tokens. All-head, per-input means over the 16 values are retained, not just selected-head averages.

The patch replaces each selected head's attention output, **before the output projection**, with its KV group's projected V at t-1, only at block value positions t >= 2048. The sham uses V at t-4. For rows 2–16, t-4 is the previous row's key; for the first row it is filler. This boundary detail is recorded, not described as a previous key for all 16 rows.

Every condition records argmax over the same 512 candidate values and over the full vocabulary, answer logits, answer-minus-best-other-value logits, and top-five predictions. Answer labels are used for scoring only. All three conditions use the same single-input computation and precision.

## Completed validation and timing

Evidence: `phase_a_validation_v3/validation.json`, `task_example.json`, `manifest.json`, and `timings.jsonl`.

- Ordinary English, 144 tokens including BOS: **2.186209 nats/token**.
- Greedy continuation of `The capital of France is`: `Paris.` followed by another capital-city sentence. The story prompt continues `there lived a man named Joseph. He was a good man, and had a wife and children. One day, Joseph`.
- Short explicit-matrix attention versus CPU flash: maximum absolute logit discrepancy **8.59e-6**. Cached next-token versus full-prefix discrepancy **7.16e-6**; causal-prefix discrepancy **7.63e-6**.
- Patch and sham are exactly the identity below 2048 on checked logits. Independent scalar calculations check the predecessor probability and score margin. A head-by-head reference checks GQA grouping. Sentinel-valued V tensors check patch/sham source positions, head mapping, the 2048 boundary, and untouched outputs.
- Builder invariants pass on 32 examples at all lengths/placements. Whole decoded prompts re-encode exactly on another 4 examples x 3 lengths x 3 placements.
- The final runner passes synthetic checks of threshold selection, fallback selection, condition pairing/reuse, raw input-ID serialization, summaries, and the deadline path. These outputs are explicitly marked **SYNTHETIC_ONLY** in `phase_a_controlflow_smoke/`; they are not model results.

One full forward at each length, including all-head block diagnostics:

| Tokens | Wall seconds | CPU seconds |
|---:|---:|---:|
| 2048 | 48.558 | 178.641 |
| 4096 | 162.864 | 434.469 |
| 8192 | 358.855 | 997.750 |

These are one-pass measurements on this laptop, not reliable upper bounds. A process snapshot during the long timing showed about 4.43 GB working set and 5.55 GB private memory; it was not a peak-memory measurement. No out-of-memory failure occurred.

The directories `phase_a_validation/` and `phase_a_validation_v2/` are retained aborted setup attempts: Path serialization and integer-rank averaging failed, respectively. Both defects were fixed. Use the completed **v3** evidence. Subsequent changes only added raw input IDs and elapsed-time logging, checked by the synthetic runner test.

## One command for the owner

Run this PowerShell command once. The script resolves outputs relative to its own location, so the current shell directory does not matter. `cpu_run` must not already exist; an existing output directory is refused rather than overwritten. There are no prompts or downloads.

```powershell
& 'C:\Users\pkchi\Documents\gate-length-pilot\.venv\Scripts\python.exe' -B 'C:\Users\pkchi\Documents\gate-length-pilot-audit2\audit2\independent_nope.py' run --output cpu_run --threads 4 --selection 16 --diagnostics 4 --n4096 18 --n8192 2 --max-hours 3.7
```

The execution order deliberately protects priorities (a) and (b):

| Order | Work | Inputs / model forwards |
|---:|---|---:|
| 1 | Select heads at 2048, seed 261022872, indices 0–15 | 16 / 16 |
| 2 | Fresh paired inside/far diagnostics, seed 261022873, indices 0–3, lengths 2048 and 8192 | 4 paired / 8 |
| 3 | Before @4096, indices 0–17, none/patch/sham | 18 paired triples / 54 |
| 4 | Before @8192, indices 0–1, patch/sham; reuse their already saved none forwards | 2 paired triples / 4 additional |

Totals: **20 forwards at 2048, 54 at 4096, 8 at 8192: 82 forwards**. Diagnostic means use four independent inputs (64 value positions per head per length); head selection uses sixteen inputs (256 value positions per head). Do not treat individual value positions as independent examples.

Measured-time estimate: `20*48.558 + 54*162.864 + 8*358.855 = 12,637 seconds`, or **3.51 hours**, plus loading, hashing, and output overhead. Allow about **3.6 hours**. The **3.7-hour guard** includes setup and checks the budget between forwards. If the laptop slows, it stops after the current forward, writes an explicitly incomplete summary, and exits with status 2. It does not interrupt a forward; at the measured 8192 cost this puts the guarded end around 3.8 hours. Successful completion returns status 0. Earlier priorities and raw completed rows remain usable if the optional tail is cut short.

## Outputs and interpretation limits

Under `audit2/cpu_run/`:

- `manifest.json`: command, package versions, architecture settings, checkpoint/tokenizer/script hashes, complete pools and seeds.
- `selection.json`: all-head selection means, selected zero-based layer/head pairs, threshold and fallback status.
- `raw.jsonl`: flushed after every forward; full input IDs, key/value/answer metadata, block and sham positions/sources, predictions, all-head diagnostic means, per-pass and cumulative timing.
- `summary.json`: sample counts, completion status, restricted/full accuracies, selected-head diagnostic means, and paired bootstrap effect intervals (20,000 resamples). The raw rows are the primary evidence.

This is a constrained independent audit sample, **not** the registered 64-input selection/measurement or 200-example feasibility/main experiment. There is no retuning or automatic registered verdict. The independent inside accuracy must be examined before interpreting retrieval failure as a mechanism test. Four inside examples cannot reliably certify the 70% gate. Eighteen 4096 trials have coarse uncertainty; the **two 8192 intervention examples are descriptive only**, and their bootstrap intervals must not be used as confirmatory evidence for D2. Original D2 is at 8192, not 4096.

Stable predecessor-minus-average margins do not by themselves establish pure dilution: strong competitors can change while that mean remains stable. Ratios also require checking the sign and size of the inside margin. A successful patch is an oracle intervention at known value positions and does not alone establish lost implicit position as the cause. The later phase must compare source, saved results, and these independent observations before deciding whether the owner's interpretation is justified.

**Stop point:** Phase A ends with this plan. Do not start the main run as part of Phase A. No `AUDIT-2.md` has been written.
