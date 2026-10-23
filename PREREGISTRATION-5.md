# Pre-registration 5: does pilot 4's finding hold in a real NoPE language model?

Written 2026-10-22, before any pilot-5 data existed. The git commit that adds this file is the timestamp. This is
a *new* study. The four-pilot series stopped as registered. This study asks whether the confirmed toy result
generalizes; it is not a further search within that series. The user approved it and the GPU spend.

## Question

Pilot 4 found, in 1.1M-parameter NoPE models trained on 128 tokens, that "binding" heads (value tokens attending
to the key just before them) lose that attention past the trained length, and that forcing it back restores
retrieval. The independent audit (`AUDIT.md`, issue 2) noted that falling attention could reflect either a weaker
*preference* for the previous token (lost position) or an unchanged preference *diluted* by more tokens. Does the
same hold in a real 1.1B NoPE language model, and which of the two is it?

## Model and task

- **Model:** `AntNLP/TinyLlama-NoPE-1.1B` (Wang et al., 2024), 22 layers × 32 heads, trained without positional
  encoding on 2048-token contexts. It runs through `pilot5.py`'s own forward pass (bfloat16). Before any task
  data, `--stage verify` reports the language-model loss on a fixed paragraph and the agreement with the Hugging
  Face reference implementation, which uses the model card's NoPE patch.
- **Task:** the pilot-1 task written with single-token words. A preamble line, then one record per line,
  ` key value\n`, then the query key; the next token should be the value.
  - Keys and values are disjoint random sets of single-token lowercase words (`▁[a-z]{4,10}`), with 512 values.
  - Accuracy = argmax over the 512 value words at the final position (chance 0.2%). Full-vocabulary argmax is also
    reported.
  - Base inputs are 2048 tokens, inside the trained length. Versions of 4096 and 8192 tokens insert irrelevant
    records in the *before*, *uniform* or *after* placement, and the answer never changes. 200 examples per
    placement (seeds 40000–40002).

## Measurements (`pilot5.py`)

1. **Feasibility gate F** (`--stage feasibility`, baseline only). F passes if "before" accuracy at 2048 is ≥ 70%
   and accuracy at 8192 is ≥ 20 pp lower. If F fails, the task does not fit this model, the verdict is NOT
   TESTABLE, and nothing further runs.
2. **Binding heads.** On 32 uniform 8192-token inputs (seed 20000), at 32 sampled value positions per window
   (512–2047 inside; 2048–4095; 4096–8191), measure:
   - attention to the previous token, p;
   - the *score margin*: the previous token's attention score minus the mean score of all other earlier tokens.

   Binding heads are those with inside p ≥ 0.3. If none qualifies, the single highest head is used.
3. **Patch.** At every value position ≥ 2048, binding heads output the value vector of the previous token (the
   key). **Sham:** the value vector 4 tokens back, the previous record's key. **None:** no change. All are scored
   on every length and placement.

## Decision (`python report5.py decision`)

- **F:** as above.
- **D1:** binding heads' mean p at 4096–8191 ≤ ½ of inside.
- **D1b, preference or dilution:** the margin ratio far / inside. ≥ 0.8 means PREFERENCE KEPT (the drop in p is
  dilution). ≤ 0.5 means PREFERENCE LOST (fits fading position). In between is reported as such.
- **D2:** on "before" @8192, the patch minus none is ≥ 20 pp, its paired bootstrap 95% CI is above 0, and the
  patch beats the sham by ≥ 20 pp.

Verdict:
- **TOY MECHANISM HOLDS IN THE REAL MODEL** (D1 and D2);
- **ATTENTION FADES BUT THE PATCH DOES NOT RESCUE** (D1 only);
- **BINDING ATTENTION DOES NOT FADE** (not D1);
- **NOT TESTABLE** (F fails).

D1b is reported alongside the verdict as the answer to the audit's question.

## Limitations known in advance

- One pretrained model, so there are no training seeds. Uncertainty is over test examples only.
- The task is synthetic, though in natural tokens.
- A real model at 4× its trained length may fail in many places at once, so a negative D2 would not rule out
  binding heads as *a* cause.
- Wang et al.'s fine-tuned 8K model stores extra per-head scales that standard code does not load. Comparing with
  it is left for later.
- This registration is a local signed commit, not an independent timestamp. The user can push the repo to
  GitHub for one.

## Deviations

Added 2026-10-23, after the second audit ([AUDIT-2.md](AUDIT-2.md)). None of these changes the verdict.

- **A diagnostic ran after the failed gate**, although this plan says nothing further runs. It used shorter lists,
  only to rule out a pipeline bug, and RESULTS-5.md reports it as not pre-registered
  ([results5/diagnostic.txt](results5/diagnostic.txt), `logs/diag5.py`).
- **The 4096-token inputs are 4094 tokens**, because records are whole 3-token lines. The 2048 and 8192 inputs are
  exact, and the gate uses only those.
- **Decision thresholds.** `report5.py` compared rounded float accuracies with the 70% and 20 pp thresholds, which
  could fail exactly at a boundary (140/200 is stored as 0.699999988). On 2026-10-23 it was changed to compare
  integer counts. The saved results are far from the boundaries, and results5/decision.md is unchanged.
