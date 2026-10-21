# Pre-registration 2: why are records beyond the trained length lost?

Written 2026-10-21, after pilot 1 and before any pilot-2 data existed. The git commit that adds this file is the
timestamp. The user approved the plan and the GPU spend before this was written.

## 1. Background and question

Pilot 1 (verdict ABANDON for gate alignment) found an exploratory pattern (RESULTS.md §5, fig. 6). In the NoPE
model, items whose relevant record lies **beyond** token position 128 are lost, and no statistics correction
recovers them. Items whose relevant record lies inside the range are mostly recoverable. Two explanations compete:

- **H-pos, lost positional signal.** A NoPE model gets position only from the causal mask, and that signal weakens
  as sequences get longer (Kazemnejad et al., 2023; Dawes et al., 2026, "How Local Mixing Encodes Relative
  Position in Global NoPE Attention"). Past the trained length, the model can no longer tie a value to the key
  just before it, so this *binding* is missing from the representation.
- **H-dist, attention distraction** (Wang et al., 2024, "Length Generalization of Causal Transformers without
  Position Encoding"). Attention spreads too thin at long lengths. Sharpening it with a temperature should restore
  binding and retrieval.

## 2. Models

- **NoPE:** the three pilot-1 checkpoints (`runs/seed0..2`), unchanged.
- **RoPE control:** three new seeds (0, 1, 2) with the same architecture and the same recipe (amendment A1:
  60,000 steps, curriculum 0.5, batch 64, lr 1e-3). Rotary embeddings (base 10000) are applied to queries and keys
  in every layer. Output goes to `runs_rope/seed*`. Sanity check: length-128 validation accuracy ≥ 95%. A seed
  that fails may get a recipe change chosen on length-128 data only, logged in §6.

## 3. Measurements (`pilot2.py`)

1. **Uniform attention temperature** c ∈ {1.0, 1.2, 1.5, 1.8}: attention logits at *every* position, in every
   head and layer, are multiplied by c. This follows Wang et al.'s uniform scale; their values for 2× / 4× / 8×
   length were about 1.2 / 1.5 / 1.8. All four values are reported and none is chosen by test accuracy. Scoring
   uses the pilot-1 test sets (seeds 40000–40002) at every length and placement.
2. **Binding probe.** Read the residual stream after the embedding (layer 0) and after each block (layers 1–4)
   at value positions. The label is the key token just before each position (512 classes, chance 0.2%).
   - Probe set: 1024 uniform-placement inputs of length 1024 (seed 50000). From each, 16 value positions are
     sampled in each window: [64,128) (inside the trained range), [128,256), [256,512) and [512,1024).
   - Examples 0–767 train the probes and 768–1023 test them.
   - Method: ridge regression onto one-hot labels, with standardized features and α = 1, scored by argmax
     accuracy. *Transfer* probe: fit on [64,128), tested on every window. *Refit* probe: fit and tested within
     each window. Both are rerun for each c.
   - B_in(c) = the best refit accuracy over layers 1–4 in [64,128). B_far(c) = the same in [512,1024).
3. **Beyond items:** the L = 1024, uniform-placement test items whose relevant record's key token sits at
   position ≥ 128 (about 89% of items).

## 4. Decision rule (`python report2.py decision`)

Per seed unless stated:

- **V1, the probe works (NoPE):** B_in(1.0) ≥ 50% in every NoPE seed. If not, all probes are refit once with a
  fallback MLP probe: one hidden layer of 512 ReLU units, Adam, lr 1e-3, 2000 steps, batch 512, standardized
  inputs. V1 is then checked again.
- **V2, the RoPE control learned the task:** length-128 test accuracy ≥ 95% in every RoPE seed.
- **R, temperature rescue (NoPE, L = 1024):** for some c ∈ {1.2, 1.5, 1.8}, the beyond-item accuracy at c minus
  that at c = 1.0 is ≥ 20 pp on average over seeds, *and* every seed's paired bootstrap 95% CI (2000 resamples)
  lies above 0.
- **P1, binding missing far away (NoPE):** B_far(c) ≤ ½ · B_in(c) for every c and every seed.
- **P3, RoPE keeps the binding:** B_in(1.0) ≥ 50% and B_far(1.0) ≥ ½ · B_in(1.0) in every RoPE seed.

Verdict:
- **INCONCLUSIVE** if V1 (after the fallback) or V2 fails.
- **ATTENTION DISTRACTION** if R holds and P1 fails. Wang et al.'s explanation fits, so the lost-position
  direction is not pursued (low novelty).
- **POSITION LOSS** if P1 holds, R fails and P3 holds. Pursue "lost positional signal beyond the trained length"
  as the thesis direction.
- **MIXED** otherwise. The report says which pieces held.

Reported descriptively, without affecting the verdict:
- all of the above at L = 512;
- accuracy at 128 under each c, since sharpening may cost in-range accuracy;
- RoPE accuracy by placement and length, and RoPE under each c;
- transfer-probe accuracy by window, which separates a changed format (refit works, transfer doesn't) from
  missing information (neither works);
- the layer-0 probe as a control, which should sit at chance.

## 5. Limitations known in advance

A linear probe can miss non-linearly stored information. That is why there is a fallback, and why B_far is
compared with B_in from the same probe type. The temperature grid has four values. It tests whether *this*
standard remedy works, not every possible attention fix.

## 6. Deviations and amendments

- *(none yet)*
