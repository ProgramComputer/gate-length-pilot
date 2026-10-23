# Pilot 6 results: the toy finding holds in a real 1.1B NoPE model

**Pre-registered verdict: TOY MECHANISM HOLDS IN THE REAL MODEL.** On whether the attention loss comes from a
weaker preference or from dilution, the registered reading is **IN BETWEEN** (margin ratio 0.68).

In Wang et al.'s TinyLlama-NoPE-1.1B, trained on 2048-token contexts, the heads that make each value token look
back at its key lose most of that look-back once the pairs sit past the trained length. Restoring it, with an
oracle patch, brings retrieval back from 6.5% to 65% at 4× the trained length, and from 51% to 85.5% (the in-range
level) at 2×. Patching the same heads to copy the wrong token instead destroys retrieval.

Plan: [PREREGISTRATION-6.md](PREREGISTRATION-6.md) (signed commit `01c0d03`, before any data). The feasibility
gate passed (`546760a`), and the user approved the main test before it ran. Decision:
[results6/decision.md](results6/decision.md). Compute: about 1 GPU-hour on one RTX 3090 (≈ $0.30).

![real model](figures/fig13_pilot6_real_model.png)

## Setup in one paragraph

The model is the published `AntNLP/TinyLlama-NoPE-1.1B`, run with a small forward pass verified against the
Hugging Face reference (100% agreement on next-token predictions; pilot 5). Each input hides a block of 16
` key value` word pairs inside the passkey-style filler that Wang et al. used, and ends with one of the keys; the
model must produce its value. Inputs are 2048 tokens, or 4096 / 8192 with extra filler. The main condition puts
the extra filler *before* the block, so the block ends up far from the start but stays as close to the query as in
training.

## What was found

| "before" placement (block far from start, near query) | 2048 (trained) | 4096 | 8192 |
|---|---|---|---|
| no patch | 84.5% | 51.0% | 6.5% |
| **patch**: binding heads copy the token before (the key) | 86.0% | **85.5%** | **65.0%** |
| sham: same heads copy the token 4 back | 86.0% | 1.0% | 1.0% |

Accuracy is over the 512 value words, 200 inputs each.

1. **Binding heads exist in the real model.** 39 of its 704 heads, mostly in layers 11–14, send ≥ 0.3 of a value
   token's attention to the key just before it (mean 0.46) when the block is inside the trained range.
2. **Their look-back collapses far away.** With the block past position 6000, that attention falls to 0.11 on
   average (range 0.005–0.19). D1 holds.
3. **Restoring it restores retrieval.** At 8192 the patch adds +58.5 pp (paired bootstrap 95% CI +52 to +65.5).
   At 4096 it adds +34.5 pp (+27.5 to +41.5), back to the in-range level. The wrong-token sham collapses accuracy,
   so the gain comes from restoring the *right* information. D2 holds.
4. **Preference or dilution: in between.** The score margin of the previous token over the average other token
   falls from 36.1 to 24.5 (ratio 0.68; per head, a median of 0.64, and none below 0.5). The preference weakens
   but is far from erased. An advantage that large over the *average* token would still dominate if the other
   tokens scored evenly, so the lost attention most likely goes to a few strongly scoring tokens. The
   beginning-of-sequence "attention sink" is one possibility. That is a hypothesis; it was not measured.
5. **A second failure the patch does not touch.** With the block early and far from the query ("after"
   placement), accuracy also collapses (2% at 8192), and the patch changes nothing there: the block lies inside
   2048, so nothing was patched. This failure lies elsewhere, plausibly in the query's long-distance look-up. The
   toy models showed the same split.
6. Uniform placement sits in between: the patch adds +13.0 pp at 4096 and +35.5 pp at 8192.

## How this changes the overall story

The toy result from pilot 4, that binding heads are a bottleneck past the trained length, now has a counterpart in
a real, published NoPE language model, under a pre-registered test with a sham control. Wang et al. (2024) explain
NoPE's long-context failure as attention distraction and fix it with attention scaling. These results point to a
specific set of heads, and show that restoring their one-token look-back recovers most of the lost retrieval.
Whether Wang et al.'s scaling works *through* these heads was not tested; it is the natural next question.

## Limitations

- **Oracle intervention.** The patch tells 39 heads exactly where to look. It locates a bottleneck; it is not a
  method.
- **One model and a synthetic task** with repetitive filler. Natural documents may behave differently.
- **The margin measure** compares the previous token with the *average* token. It cannot show where the lost
  attention goes. A max-competitor or attention-sink analysis would answer that.
- **Numerics.** The registered code scored the unpatched model in batches of 4 and the patched ones one at a time.
  bfloat16 rounding differs slightly by batch size: at 2048, where the patch is the identity, 86.0% vs 84.5%. A
  post-hoc rerun one input at a time (`pilot6_batch1.py`, [results6/none_batch1.json](results6/none_batch1.json))
  gives 86.0% at 2048 and 6.0% at 8192, so the like-for-like gain at 8192 is +59.0 pp (CI +52 to +65.5). Nothing
  changes.
- Block size (16 pairs) was chosen on in-range inputs only. This is a local signed pre-registration, not an
  independent timestamp. It has not yet been independently audited; the earlier audit covered pilots 1–4.

## Reproduce

```bash
python pilot6.py --stage tune          # chooses k on 2048-token inputs only
python pilot6.py --stage feasibility   # the gate
python pilot6.py --stage full          # binding heads, margins, patch and sham
python pilot6_batch1.py                # post-hoc: unpatched baseline one input at a time
python report6.py                      # results6/decision.md and fig13
```
