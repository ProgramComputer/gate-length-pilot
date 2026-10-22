# Pilot 3 results: two targeted tests on the trained models

Plan and decision rules: [PREREGISTRATION-3.md](PREREGISTRATION-3.md) (signed commit `1ac4b35`, before any data).
Full tables: [results3/decision.md](results3/decision.md). No retraining. Compute: about 15 minutes on one rented
RTX 3090 (≈ $0.05).

| test | paper author | pre-registered verdict |
|---|---|---|
| A. Does RoPE Profiler diagnose and fix a failure we control? | Hao Peng | **DETECTS BUT DOES NOT FIX** |
| B. Does attention's relative-position part keep its shape past the trained length? | Heng Ji | **NO BREAK IN NOPE** |

## A. RoPE Profiler on a known failure (Wu, Du & Peng, 2026)

![A](figures/fig9_pilot3a_rope_profiler.png)

The RoPE models fail when the relevant record is farther from the query than any distance seen in training
(the "after" placement: 1–3% at 1024). The retrieval heads were picked automatically. In every seed they were
two layer-3 heads putting about 98% of the query's attention on the relevant record's value token.

- **The diagnostic detects it.** The chosen heads attend to *value* tokens, so the comparison is between the
  relevant record's value token and other records' value tokens. Inside the trained window, the relevant token
  never loses to a distractor (0% of distances). Over a 1024 window it loses at 10–18% of distances. Their semantic score drops from 1.00 to
  0.85–0.91, and all three seeds point to *semantic reversal* as the worse failure.
- **Their fix doesn't reach it.** Attenuating, removing or freezing the high-frequency band, or amplifying it,
  changes accuracy by at most +0.3 pp. Standard position interpolation, a reference fix that is not theirs, adds
  only +2.8 pp, and it collapses the "before" placement from 52% to 3%.
- **Reading:** removing the high-frequency band entirely changes nothing, so the reversals here likely come from
  the slow-rotating components, which never complete a turn within 128 tokens. Their theory and their fix are
  about high-frequency share within a trained window. Beyond the window, the diagnosis still works but the lever
  is in a different band. That is a hypothesis; which band carries the reversals was not measured directly.
- **Open question it raises:** is there a counterpart intervention when the reversal comes from components that
  never completed a rotation in training?

## B. Han & Ji's decomposition in a NoPE model (ACL 2025)

![B](figures/fig10_pilot3b_disentanglement.png)

- **The decomposition holds in NoPE too.** The three-axis fit a(i−j) + b(i) + c(j) explains NoPE attention scores
  well inside the trained range: mean correlation over heads 0.76–0.85, higher than RoPE's 0.45–0.51 on this task.
- **The distance-indexed part keeps its shape.** The most "positional" head keeps the same a(i−j) profile far
  away (rho 0.86–1.00 in NoPE, 0.97–0.99 in RoPE), so the pre-registered verdict is NO BREAK IN NOPE.
- **But this measure is confounded (found by the independent audit, `AUDIT.md` issue 1).** In this task, keys and
  values alternate, so anything that depends only on token *role* (key vs value) shows up as a function of i−j.
  A head that cannot tell which key comes just before a value, but treats keys and values differently, produces
  the same profile with rho = 1. In the real NoPE heads, that odd/even alternation explains 79–100% of the
  profile. So this result says little about whether the model can still *find* the token just before a value.
  Pilot 4 measured that directly. In seed 1, the head chosen here is itself a binding head whose look-back
  attention falls from 0.84 to 0.03, despite rho = 0.87.
- **Exploratory, not pre-registered:** far away, the distance-indexed part grows in NoPE. Its amplitude roughly
  doubles (fig. 10a), and its variance *ratio* rises from about 0.03–0.15 inside to 0.3–0.8 far away in many NoPE
  heads, while RoPE's stays flat. This ratio is not a share of explained variance (`AUDIT.md` issue 6), and given
  the role confound, "position takes over" would be an overreading. Treat it as a pattern to explain, not a
  result.

## Limitations

- Small models, one synthetic task, three seeds per model.
- RoPE Profiler was re-implemented from the paper's equations, not run from its code. Its 49-task rank comparison
  was replaced by a raw 128-vs-1024 comparison.
- The band cut, the α grid and the two-head choice were fixed in advance but are judgment calls.
- In B, the three-axis fit cannot separate relative position from key/value role alternation in this task
  (see above). The all-head pattern is exploratory.
- **Deviation:** the test inputs for part A are a different random realization from pilot 1's at length 1024
  (see `PREREGISTRATION-3.md`, Deviations).
- **Independent audit:** `AUDIT.md` reproduces both verdicts from the saved results and checks the RoPE edits
  numerically.

## Reproduce

```bash
python pilot3.py a --seed 0                    # RoPE models; seeds 0, 1, 2
python pilot3.py b --seed 0 --runs runs        # NoPE
python pilot3.py b --seed 0 --runs runs_rope   # RoPE
python report3.py                              # fig9, fig10, results3/decision.md
```
