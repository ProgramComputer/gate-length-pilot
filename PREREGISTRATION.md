# Pre-registration: do SwiGLU gate statistics causally contribute to length-generalization failure?

Written 2026-10-21, before any trained model was evaluated at a length above 128. The git commit that adds this
file is the timestamp. Every analysis choice below is fixed; the only data seen so far is training-length (128)
validation accuracy of a throwaway learnability probe (seed 99, never used in the analysis).

## 1. Question and hypothesis

When a decoder-only Transformer trained on inputs of length ≤ 128 reads longer inputs, its SwiGLU feed-forward
gate pre-activations at the query position shift in distribution. **Does that shift causally contribute to the
accuracy loss, beyond what an attention-side failure explains?**

- **H1 (gate hypothesis):** correcting the gate statistics of long inputs back to their short-input values, with no
  retraining, recovers long-length accuracy. The recovery is larger than an equally sized sham correction and
  larger than the same correction applied to the FFN value branch or the residual stream, and it adds to a
  length-based attention-temperature fix.
- **H0:** it does not. In that case the gate shift is a symptom, or is redundant with attention, or any comparable
  perturbation does as well.

This is a diagnostic, favorable test of H1. The corrections are estimated on unlabeled long inputs, so a positive
result would show that the gate pathway matters. It would not show that a training-only method generalizes.

## 2. Model

Causal decoder-only Transformer (`gatepilot/model.py`): 4 layers, d_model 128, 4 heads, pre-norm RMSNorm, SwiGLU FFN
`W_down(SiLU(W_gate h) ⊙ W_up h)` with d_ff 512, **no positional embeddings** (NoPE), 1,131,904 parameters. The
readout is a linear map to the 64 value tokens at the final (query) position.

## 3. Task and data (`gatepilot/task.py`)

- Sequence `BOS k1 v1 … kn vn Q kq`. The answer is the value paired with `kq`. There are 512 key tokens and 64
  value tokens. Keys are unique within a sequence, and every key→value mapping is freshly random, so the answer
  cannot be memorized.
- Nominal length L holds n = ⌊(L−3)/2⌋ records (128 → 62, 256 → 126, 512 → 254, 1024 → 510).
- **Training:** each batch draws n uniformly from 4..62 (≤ 127 tokens), with the queried record uniform among them.
  The loss is cross-entropy on the answer only.
- **Paired evaluation sets:** each base example has 62 records (length 128). Its long versions (256, 512, 1024)
  insert irrelevant records (fresh keys, random values) while the answer stays the same. Each inserted record goes
  into a random gap allowed by the placement:
  - `uniform`: anywhere (**primary**);
  - `before`: only before the relevant record, so its distance to the query is unchanged and only its absolute
    position grows;
  - `after`: only between the relevant record and the query, so the distance grows.
  The distance (records between the relevant record and the query) is stored for every example.
- **Disjoint sets, fixed seeds:** training stream = model seed; validation (length 128 only) seed 10000;
  calibration set `cal` seed 20000 (512 examples per length, labels never used); check set `chk` seed 30000 (512
  per length, labels never used); test sets seeds 40000/40001/40002 for uniform/before/after (1000 base examples
  each, scored only). The same test examples are used for every model seed and every condition, so all
  comparisons are paired.

## 4. Training (`train.py`)

Three model seeds: **0, 1, 2**. AdamW (lr 1e-3, betas 0.9/0.98, weight decay 0.01), batch 64, 500 warmup steps,
cosine decay to 10% of peak, grad-norm clip 1.0. The default is 20,000 steps (changed by amendment A1 in §11:
60,000 steps with a length curriculum). Training logs only length-128
validation accuracy, loss and prediction spread. If a seed's final length-128 validation accuracy is below 95%,
training may be lengthened or adjusted using **length-128 data only**, and the change is logged in §11. The same
recipe is then used for all three seeds.

## 5. Sanity gate (before any intervention is run)

`python evaluate.py --seed S --baseline-only` scores the unmodified model at every length and placement. The gate
passes for a seed if, under uniform placement, **acc@128 ≥ 95%** and **acc@1024 ≤ acc@128 − 20 pp**. If any seed
fails, the setup is fixed first (for example more training, or different training lengths) and logged in §11. No
intervention result is computed until all three seeds pass. This step looks at baseline long-length accuracy by
design. It cannot look at any intervention.

## 6. Drift measurement (`evaluate.py`, calibration set)

At the aligned query position (the final token), for every layer and channel, compare inputs of length L with
length-128 inputs from the same paired calibration set:
- standardized mean shift (μ_L − μ_128)/σ_128 and log SD ratio, per channel;
- per-channel Gaussian W2 distance √((μ_L−μ_128)² + (σ_L−σ_128)²)/σ_128, averaged over channels, which is the
  summary "drift".

This is done for the gate pre-activation (the hypothesis), the gate post-activation SiLU(a), the value branch
W_up h, and the residual stream entering the FFN, plus the query position's attention entropy. Samples are equal
in size at every length (one query position per example).

## 7. Conditions (inference only, no retraining; `gatepilot/correct.py`)

| name | what it does |
|---|---|
| `none` | unmodified model |
| `gate` | gate correction |
| `attn` | attention-temperature intervention |
| `gate+attn` | both (gate correction fitted with the attention intervention on) |
| `up` | the same correction on the FFN value branch (non-gate control) |
| `resid` | the same correction on the residual stream entering the FFN (non-gate control) |
| `sham` | the gate correction's own adjustments, applied to randomly permuted channels (fixed permutation per layer) |

**Correction (used by gate, up, resid; sham permutes it).** Per layer, channel and token role (key, value,
Q marker, query), an affine map `a → μ_s + (σ_s/σ_L)(a − μ_L)`. μ_L and σ_L come from positions ≥ 128 of length-L
calibration inputs. μ_s and σ_s come from the same role at positions 64–127 of length-128 calibration inputs (for
the Q marker and the query, from their own positions). Each role contributes on average ≤ 16 sampled positions
per example, so the short and long samples are the same size. The maps are fitted **one layer at a time**: layer
l is fitted with the maps of layers < l already applied. They are applied only at positions ≥ 128. In a causal
model, earlier positions cannot see the inserted tokens, so they are already in distribution. A separate map is
fitted for each evaluation length. Labels are never used.

**Sham.** The correction is written as a mean change in SD units, m = (μ_s − μ_L)/σ_L, and an SD ratio
r = σ_s/σ_L. The sham gives channel i the (m, r) of channel π(i), under a fixed random permutation π per layer
(seed 1 + layer), sequentially fitted like the others. Its perturbation size has the same distribution as the
real correction's, but it is aimed at the wrong channels. The perturbation sizes are reported.

**Attention temperature (policy fixed now, never tuned).** The logits of the query at position t (0-based) are
multiplied by `max(1, ln(t+1)/ln 128)` in every head of every layer. This is the standard log-length scaling. It
is the identity inside the training range and ×1.43 at t = 1023.

At length 128, every condition is the identity by construction. This is verified numerically.

## 8. Manipulation checks (held-out check set `chk`)

For every condition and long length, recompute the query-position drift (W2, mean over layers) at the gate, value
branch and residual sites, plus the attention entropy. The decision-relevant check is **M1**: in every seed, at
L = 512 and 1024, the `gate` correction reduces the gate-pre-activation drift by **≥ 50%** relative to `none`. The
targeted reductions for up, resid and sham, and the entropy change for attn, are reported. The short-length check
requires every condition's accuracy at 128 to be within 1 pp of `none`.

## 9. Outcomes and statistics

- **Primary outcome:** answer accuracy (argmax over the 64 values) on the uniform-placement test set at
  L = 512 (4×) and L = 1024 (8×). Secondary: mean answer log-probability, all lengths, before/after placements,
  and accuracy by distance.
- For each comparison A − B and each seed: the paired accuracy difference on the same 1000 test examples, with a
  paired bootstrap 95% CI (2000 resamples, fixed RNG seed 0). Across seeds: the mean of the three per-seed
  differences, with a 95% t interval (df = 2).
- **pass(A, B, t)** ⇔ the mean over seeds of acc(A) − acc(B) is ≥ t pp **and** every seed's bootstrap CI lower
  bound is > 0.

## 10. Decision rule (`python report.py decision`)

Criteria, each required at **both** L = 512 and L = 1024:

| | criterion | meaning |
|---|---|---|
| G1 | pass(gate, none, 10) | gate correction rescues a meaningful amount |
| G2 | pass(gate, sham, 5) | more than an equally sized untargeted perturbation |
| G3 | pass(gate, up, 5) and pass(gate, resid, 5) | specific to the gate, not to any FFN-input / residual rescaling |
| G4 | pass(gate+attn, attn, 5) | adds beyond the attention intervention |

Verdict:
- **INCONCLUSIVE** if the sanity gate (§5), M1, or the short-length check fails. A correction that does not
  change what it targets cannot test H1.
- **PURSUE** if G1–G4 all pass. The gate-alignment idea continues to the training-only confirmation study.
- **PIVOT** if G1, G2 and G4 pass but G3 fails. FFN-side statistics matter beyond attention, but not specifically
  through the gate. The gate-alignment proposal is dropped and the thesis pivots to that broader mechanism.
- **ABANDON** otherwise: abandon the gate-alignment proposal and report a clean negative result. This covers the
  gate correction not rescuing, doing no better than the sham, or adding nothing beyond attention.

All criteria must hold in all seeds and at both lengths (an intersection–union test), so no multiplicity
correction is applied to the verdict. Descriptive comparisons (attn − none, up − none, resid − none,
sham − none) are reported uncorrected and do not affect the verdict. A secondary, stricter criterion (gate − none
≥ 20 pp at 512 and 1024 in every seed, from the original design review) is reported but does not change the
verdict.

## 11. Deviations and amendments

Any change after this commit is listed here with its date and reason, and with whether it was made before or
after any intervention result was seen.

- **A1 (2026-10-21, before any model was evaluated above length 128 and before any intervention was run). Training
  recipe.** With the §4 recipe (20k steps, n uniform in 4..62 from step 1), the learnability probe (seed 98)
  stayed at 5% length-128 accuracy. It learned only to guess a value present in the context, not to bind keys to
  values. A 100k-step run without changes (seed 96) was also at chance at 16k steps. Probes on length-128
  validation only (seeds 94, 95, 97) showed that a **length curriculum** solves this. The maximum number of
  records per batch ramps linearly from 8 to 62 over the first 50% of training, and n is uniform in 4..max. The
  adopted recipe changes only that and the step count: `--steps 60000 --curriculum 0.5`, with batch 64,
  lr 1e-3 and everything else as in §4. The probe with exactly this recipe (seed 97) reached 99.3% length-128
  validation accuracy at 40k steps. Probe seeds 94–99 are not used in any analysis. Training also logs
  accuracy on short inputs (≤ 8 records) as a learning monitor. Nothing else in §5–§10 changes.
- **Correction (2026-10-22, after the results, from the independent audit in `AUDIT.md`).** §7 says the sham's
  perturbation size "has the same distribution as the real correction's". That holds only for the first layer.
  Because the sham is refit layer by layer under its own upstream changes, its later-layer adjustments came out
  1.18–1.62× the real correction's size. RESULTS.md already reported this. G1 and G3 fail independently, so the
  ABANDON verdict does not depend on it.

## 12. Out of scope for this pilot

A second positional scheme, activation patching / localization, quantile-map corrections, and any training-time
regularizer. These are follow-ups, conditional on the verdict.
