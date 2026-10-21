# Pilot 4 results (final): the binding heads are the bottleneck

**Pre-registered verdict: BINDING ATTENTION IS THE BOTTLENECK.** All three criteria held in all six NoPE seeds,
including three fresh seeds that played no part in suggesting the hypothesis.

In the NoPE model, the heads that tie each value to the key just before it stop looking at that key once
positions pass the trained length. Forcing them to look there restores retrieval.

Plan, decision rule and the hard stop: [PREREGISTRATION-4.md](PREREGISTRATION-4.md) (signed commit `571a707`,
before any data). Full tables: [results4/decision.md](results4/decision.md). Compute: about 45 minutes on one
rented RTX 3090 (≈ $0.20). This is the last experiment in the series.

![patching](figures/fig11_pilot4_patching.png)

## What was found

| | none | patch (copy the token before) | sham (copy a wrong key) |
|---|---|---|---|
| attention of binding heads to the token before, positions 64–127 → 512–1023 | 0.61–0.94 → 0.01–0.04 | – | – |
| "before" @1024 (record far from start, near query) | 1.7% | **76.1%** | 1.3% |
| uniform @1024 | 11.0% | **77.0%** | 10.1% |
| binding probe, positions 512–1023 | 3.0% | **99.0%** | 0.5% |
| uniform @128 (identity check) | 99.9% | 99.9% | 99.9% |

The table shows means over six seeds. The patch's gain on "before" @1024 was +96, +82, +19, +97, +97 and +56 pp
by seed. Every seed's bootstrap CI lies above 0, and the patch beats the sham by +75 pp on average.

1. **Binding heads exist and are easy to find.** Every seed has 4–6 heads, in layers 2–3, that put 0.6–0.9 of a
   value token's attention on the key just before it. That is how the model records "this value belongs to that
   key."
2. **That attention fades with absolute position** (fig. 11a): 0.6–0.9 inside the trained range, about 0.35–0.55
   at 128–255, about 0.1 at 256–511, and 0.01–0.04 at 512–1023. These are *different* heads from the "most
   positional" head in pilot 3B, whose profile stayed intact.
3. **Restoring it restores the computation** (fig. 11b–c). Copying the right token brings retrieval of far
   records from about 2% to 76% on average, and the binding probe from 3% to 99%. Copying a wrong key does
   nothing, so the effect comes from restoring the *right* information, not from any change at all.
4. **Not everything is restored.** In two seeds the patch recovers only +19 and +56 pp. Some of the failure sits
   on the query side too, which matches pilot 1's finding that the read-out also degrades at long lengths.
5. *Descriptive:* under the "after" placement, the patch and the sham both raise accuracy (45% → 75%). Blurred
   bindings far away seem to act as false matches that pull the query away from the real record. Clean bindings
   of any kind remove that interference.

## How this ties the four pilots together

| pilot | question | verdict |
|---|---|---|
| 1 | Does correcting SwiGLU gate statistics fix long inputs? | **ABANDON**: small, not gate-specific |
| 2 | Lost positional information or attention distraction? | **MIXED**: no temperature rescue; binding decays (RoPE partly too) |
| 3A | Does RoPE Profiler (Hao Peng's group) diagnose and fix a failure we control? | **DETECTS BUT DOES NOT FIX** |
| 3B | Does Han & Ji's positional part keep its shape far away in NoPE? | **NO BREAK** (for the most positional head) |
| 4 | Is the lost binding caused by the previous-token heads' attention? | **BINDING ATTENTION IS THE BOTTLENECK** |

The story that survives all four: in this NoPE model, failure past the trained length is mainly a failure of the
heads that bind each value to its preceding key. Their look-back-one-token attention depends on an implicit
position signal that fades with absolute position, which matches the theory that NoPE's positional information
comes from the causal mask and weakens with length (Kazemnejad et al., 2023; Dawes et al., 2026). The feed-forward
statistics (pilot 1) and a global attention temperature (pilot 2) do not reach that mechanism. Forcing the right
attention does.

## Limitations and novelty

- The patch is an **oracle intervention**. It locates the bottleneck; it is not a method. A practical fix would
  need to keep previous-token heads sharp past the trained length (for example local attention or recency
  biases, as in the hybrid models of Dawes et al. 2026 and Liu et al. 2026).
- One tiny model family, one synthetic task, NoPE only. The binding-head threshold and the 20 pp bars were
  judgment calls fixed in advance.
- **Novelty is modest.** Probing predecessor information ("carrying"; Cheng et al., 2026) and explaining NoPE's
  fading positions (Dawes et al., 2026) are recent work. What this pilot adds is a pre-registered *causal*
  demonstration on a controlled task, replicated on fresh seeds, that these heads are where the failure sits.

## Reproduce

```bash
python train.py --seed 3 --steps 60000 --curriculum 0.5   # fresh seeds 3, 4, 5 (0-2 come from pilot 1)
python pilot4.py --seed 0                                  # seeds 0-5
python report4.py                                          # fig11, results4/decision.md
```
