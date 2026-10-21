# Pre-registration 3: two targeted tests on the trained models

Written 2026-10-21, after pilot 2 and before any pilot-3 data existed. The git commit that adds this file is the
timestamp. The user approved both tests and the GPU spend. Neither test retrains anything. Both use the three
NoPE checkpoints (`runs/`) and the three RoPE checkpoints (`runs_rope/`) as they are. Code: `pilot3.py`;
decisions: `report3.py`.

Why these tests: pilot 2's ideas (binding probes, hybrid fixes) overlap heavily with work from Sep–Oct 2026, so
these target open questions in two recent papers.

## A. Does RoPE Profiler diagnose and fix a failure we control? (Hao Peng's group)

Wu, Du & Peng (2026), "RoPE at the End of Its Rope?", score two RoPE failure modes from cached queries and keys:
*semantic reversal* and *positional insensitivity*. They then rescale high-frequency rotary components in the
indicated direction. They note that linking a measured weakness to an output error "requires evidence about its
downstream effect." The RoPE models here fail in a known way: when the relevant record lies farther from the
query than any distance seen in training (the "after" placement), accuracy at 1024 is 2.4%. The theory targets
long contexts *within* a model's trained window; this test is *beyond* it, so whether it transfers is the point.

- **Heads:** the two (layer, head, token) choices with the highest mean attention from the query position to the
  relevant record's key or value token. Measured on 256 length-128 calibration inputs (seed 20000).
- **Diagnostics,** re-implemented from their eqs. 173–174 using pre-rotation queries and keys of the chosen heads.
  The window M equals the input length (128 or 1024), on 256 calibration inputs (seed 20000) per length.
  - Triplets pair the query with the relevant key/value token (k+) and 16 random other records' tokens of the
    same type (k−).
  - *Semantic score* = 1 − mean Φ(−μ_D/σ_D), with D oriented so that D(0) > 0.
  - *True reversal rate* = the fraction of distances m ∈ [0, M) where k− outscores k+.
  - *Positional score* = the mean of |S(m+1) − S(m)| / ‖z‖.
- **High-frequency band:** rotary pairs 0–5, the ones that turn at least once within 128 tokens.
- **Interventions,** on the chosen heads only:
  - semantic direction: α ∈ {0.5, 0.25, 0} on the band, and freezing the band's rotations;
  - positional direction: α ∈ {2, 4} on the band.
  - α scales the band's contribution to the score (q and k are each scaled by √α).
  - Reference, not their method: position interpolation (Chen et al., 2023), positions × 128/L in all heads.
  - All conditions are scored on the pilot-1 test sets: after@1024 (primary), before@1024, and uniform@128
    (in-range cost).
- **Diagnosed direction:** per seed, *semantic* if (1 − semantic score) worsens from M = 128 to M = 1024 by a
  larger factor than the positional score does; otherwise *positional*. The majority over seeds decides.

**Decision:**
- A1, the diagnostic registers the failure: true reversal rate at 1024 ≥ 2× that at 128, in every seed.
- A2, the fix works: the best condition in the diagnosed direction raises after@1024 accuracy by ≥ 10 pp on
  average, and every seed's paired bootstrap CI lies above 0. Picking the best candidate follows their protocol
  and is optimistic.
- A3, the direction matters: that best condition beats the best opposite-direction condition by ≥ 5 pp.

Verdict: **TRANSFERS** (A1, A2 and A3 all hold), **DETECTS BUT DOES NOT FIX** (A1 holds but not both A2 and A3),
or **DOES NOT DETECT** (A1 fails).

## B. Does the relative-position part of attention keep its shape past the trained length? (Heng Ji)

Han & Ji (ACL 2025) approximate attention scores as W_ij ≈ a(i−j) + b(i) + c(j), a relative-position part plus
query and key parts. They studied RoPE and ALiBi LLMs; NoPE was not tested.

- **Data:** 32 probe inputs of length 1024 (seed 50000), one at a time. Attention logits for every head.
- **Fit regions:**
  - *inside:* query rows 1–127, all keys;
  - *far:* query rows 512–639, keys within 127 tokens back.
  Both regions cover the same distances 0–127.
- **Fit:** the three-axis model by alternating least squares (30 rounds). a(d) is detrended (a line in i−j is not
  identifiable from b and c).
- **Reported:** fit correlation inside and far; rho_a, the correlation of the inside and far a(d) profiles;
  transfer correlation (far region fitted with a fixed to the inside profile); a's share of variance.
- **Positional head:** per model, the head with the largest a-share inside.

**Decision:**
- B1: rho_a(positional head) ≤ 0.5 in every NoPE seed.
- B2: rho_a(positional head) ≥ 0.8 in every RoPE seed.

Verdict: **NOPE-SPECIFIC BREAK** (B1 and B2), **BREAKS IN BOTH** (B1 only), or **NO BREAK IN NOPE** (B1 fails).

## Limitations known in advance

- Both are small-model, single-task tests. RoPE Profiler is re-implemented from the paper, not run from their
  code, and its rank-based susceptibility score (which needs 49 reference tasks) is replaced by the raw
  128-vs-1024 comparison above.
- The band cut (pairs 0–5), the α grid, the two-head choice and the thresholds are judgment calls fixed here.
- Han & Ji's two-axis "fake distance" analysis needs explicit positions, so only their three-axis model is used.

## Deviations

- *(none yet)*
