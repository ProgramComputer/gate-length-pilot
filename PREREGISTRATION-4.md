# Pre-registration 4 (final): is NoPE's lost binding caused by its previous-token heads?

Written 2026-10-21, after pilot 3 and before any pilot-4 data existed. The git commit that adds this file is the
timestamp. The user approved this test and the GPU spend.

**Hard stop.** This is the last experiment in this series. Whatever the verdict, no further experiments follow it.
The next step is consolidating and understanding the four pilots.

## Question

Pilot 2 found that in the NoPE model, the key–value *binding* (which key precedes a value) cannot be decoded at
positions far past the trained length. Pilot 3B found that the most positional attention head keeps its
distance profile there. Exploratory analysis suggested that position swamps content far away. This pilot asks
which component causes the lost binding, by intervening on it directly.

## Models

NoPE, six seeds: the three pilot-1 checkpoints (`runs/seed0..2`), plus three new seeds (3, 4, 5) trained with
the identical recipe (amendment A1: 60,000 steps, curriculum 0.5) into `runs/seed3..5`. The new seeds make the
test out-of-sample with respect to the models that suggested the hypothesis.

## Measurements (`pilot4.py`)

1. **Binding heads.** For every layer and head: mean attention from value tokens to the token just before them
   (their key). Sampled at 16 value positions per window per input, on 256 length-1024 calibration inputs
   (seed 20000), in windows 64–127 (inside), 128–255, 256–511 and 512–1023. Binding heads are all heads with
   inside attention ≥ 0.3. If none qualifies, the single highest head is used.
2. **Patch.** At every value position ≥ 128, the binding heads output exactly the value vector of the token
   before (offset 1), as if they attended only to it. **Sham:** the same, but copying the key of the previous
   record (offset 3), a wrong key with the same form of intervention. **None:** no change.
3. **Outcomes.**
   - Retrieval accuracy on the pilot-1 test sets at 1024. The primary outcome is the *before* placement: the
     relevant record is far from the start but within trained distance of the query, which isolates
     absolute-position failure. Also reported: uniform and after at 1024, and uniform at 128 (identity check).
   - The pilot-2 binding probe (ridge, refit per window, best of layers 1–4) under each condition.

## Decision (`python report4.py decision`)

- **D1, attention to the previous token is lost far away:** binding heads' attention in 512–1023 ≤ ½ of their
  inside attention, in every seed. Reported with the verdict.
- **D2, the patch restores retrieval:** "before"@1024 accuracy under the patch minus none ≥ 20 pp on average,
  every seed's paired bootstrap 95% CI above 0, and the patch beats the sham by ≥ 20 pp on average.
- **D3, the patch restores the binding:** probe refit accuracy in 512–1023 under the patch ≥ ½ of the inside
  accuracy without the patch, in every seed.

Verdict:
- **BINDING ATTENTION IS THE BOTTLENECK:** D2 holds. Forcing the right attention restores retrieval.
- **QUERY SIDE IS THE BOTTLENECK:** D2 fails, D3 holds. The binding is restored, but retrieval still fails, so
  the failure is in the query's read-out.
- **COPIED CONTENT IS CORRUPTED:** D2 and D3 fail. Even the forced copy does not carry a decodable binding.

## Limitations known in advance

- The patch is an idealized intervention, an oracle pattern. It shows what *would* fix the failure, not a
  usable method.
- The 0.3 head threshold and the 20 pp bars are judgment calls fixed here.
- If the query-side read-out is also broken (pilot 1 saw in-range items drop at long lengths), D2 can fail even
  when the binding is restored. That is exactly what D3 separates.

## Deviations

Added 2026-10-22, after the results, from the independent audit in `AUDIT.md`. None changes a number or verdict.

- **Test inputs.** The registration promised the pilot-1 test sets. `pilot4.py` called the generator with lengths
  (128, 1024) instead of all four. That changes the random draws, so the length-1024 inputs are a different
  realization from pilot 1's (same seeds, answers and short inputs). All comparisons within this pilot remain
  paired.
- **Statistics helper.** `report.py`'s table of t quantiles stopped at five seeds. The df = 5 value (2.571) was
  added after the run so the six-seed confidence interval could be printed. It is only used for the reported
  ±; the D1–D3 criteria do not use it.
- **What D3 shows.** D3 checks that the *key* is decodable after the patch. It does not localize the remaining
  error to the query side (`AUDIT.md`, issue 7).
