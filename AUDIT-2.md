**Overall judgment: partly sound.** The large oracle retrieval rescue is genuine on this synthetic task: an independently written float32 CPU implementation reproduces falling predecessor attention and a substantial patch advantage over both baseline and sham. Pilot 5's NOT TESTABLE verdict and pilot 6's registered operational verdict reproduce from their saved data. I found no demonstrated error that flips either current verdict. However, the 4096 task changes supposedly preserved context, the attention instrumentation introduces extra bfloat16 rounding, and the margin includes a token excluded by its registered definition. My independent sample does not reproduce the reported margin ratio. The evidence supports a useful intervention site; it does not establish disappearing implicit position, an exclusive binding bottleneck, or a general retrieval method. A professor-facing account should state the oracle intervention, restricted answer vocabulary, numerical qualifications, and limited independent 8192 sample alongside the rescue.

Audit date: 2026-10-23. Branch: `audit-codex-2`. No network, GPU, installations, training, or git writes were used. Only new audit artifacts under `audit2/` and this document were written. All **190 original tracked files** are unchanged; hashes are recorded in `audit2/phase_c_checks/tracked_before.json` and `audit2/final_integrity.json`.

**Independent results, before consulting the original results or implementation**

In Phase A I read only registrations 5–6 and `AUDIT.md`, then implemented the model, task, selection, diagnostics and interventions independently. Registration 6 already disclosed some outcomes in its Deviations section, so this was implementation-independent, not completely outcome-blind. In Phase C I summarized the independent run before opening the original claims and source.

The owner's unmodified command completed **all 82 forwards in 8,254.36 seconds (2 h 17 min 34 s)**: 16 selection inputs, four fresh paired 2048/8192 diagnostic inputs, 18 none/patch/sham triples at 4096, and two triples at 8192, reusing their diagnostic baselines. There were 20/54/8 forwards at 2048/4096/8192. All 82 stored input sequences regenerate exactly, all treatment inputs match their paired baselines, and the implementation's hash still matches the run manifest. The independent forward's English loss was **2.186 nats/token**, with sensible greedy continuations.

Raw evidence: [cpu_run/raw.jsonl](audit2/cpu_run/raw.jsonl), [run manifest](audit2/cpu_run/manifest.json), [independent summary and intervals](audit2/cpu_run_independent_summary.json), [provenance checks](audit2/phase_c_provenance.json), and [Phase A plan](audit2/PLAN.md). The implementation uses the supplied local checkpoint revision `80d3e2f9c6aa04a649c0eedfbc3195c90f4854cd`; its checkpoint and tokenizer hashes are saved in the manifest.

The independent task uses its own word pools, preamble and random draws. Its 16 values per block are distinct; production samples values with replacement. Production's key pool has 6,915 words; the independent pool has 512. These differences, float32 versus bfloat16, and small samples prevent treating the comparison as an exact replication of every percentage.

| Measurement | Original claim / saved calculation | Independent CPU result | Assessment |
|---|---:|---:|---|
| Binding heads | 39 | 34 | Count differs; **all 34 are among the original 39**. The five additional original heads have original inside means only 0.3002–0.3156. |
| Predecessor attention, inside | 0.463185 | 0.471869 [0.427, 0.532] | Agrees approximately. |
| Predecessor attention, far | 0.107661 | 0.171167 [0.149, 0.191] | Drop agrees, exact far level does not reproduce. |
| Attention far / inside | 0.2324 | 0.3627 [0.325, 0.401] | Both satisfy D1, far <= half inside. |
| Score margin, inside | 36.070 | 30.872 [24.630, 36.124] | Lower independent mean, imprecise. Definitions differ as explained below. |
| Score margin, far | 24.456 | 34.700 [34.197, 35.248] | Does not reproduce quantitatively. |
| Margin far / inside | 0.678, IN BETWEEN | **1.124 [0.949, 1.436]** | Does not agree; the registered ratio rule would label the independent sample PREFERENCE KEPT. That label does not prove dilution. |

Diagnostic intervals enumerate all 4^4 paired input-bootstrap resamples, holding the selected heads fixed. They describe only these four inputs and omit selection and word-pool uncertainty. They are not a comprehensive cross-implementation significance test. Original diagnostics save aggregate head means, not per-input observations, so equivalent original input-level intervals cannot be reconstructed.

Reweighting my saved diagnostics to the **original 39 heads** still gives attention 0.450 → 0.171 and margin 30.18 → 33.75. Thus different head membership alone does not explain the disagreement. A later spot check using one paired input from the **original generator**, seed 20001/index 0, gives float32 margins **38.772 → 26.820**, ratio **0.692**. This shows the original pattern on one matched task realization; it neither validates the full 64-input aggregate nor resolves the variation across task realizations.

| Before placement | Claimed accuracy (%) | Independent correct / n | Independent accuracy and exact binomial 95% interval (%) |
|---|---:|---:|---:|
| 2048 none | 84.5 | 4/4 | 100 [39.8, 100] |
| 2048 patch / sham | 86.0 / 86.0 | Not run as separate main cells | Identity below 2048 verified separately; the claimed difference from none is numerical. |
| 4096 none | 51.0 | 10/18 | 55.6 [30.8, 78.5] |
| 4096 patch | 85.5 | 18/18 | 100 [81.5, 100] |
| 4096 sham | 1.0 | 0/18 | 0 [0, 18.5] |
| 8192 none | 6.5 | 0/4 | 0 [0, 60.2] |
| 8192 patch | 65.0 | 2/2 | 100 [15.8, 100] |
| 8192 sham | 1.0 | 0/2 | 0 [0, 84.2] |

All these accuracy claims are compatible with the small independent samples. At 4096, patch minus none is **+44.4 pp**, paired-bootstrap 95% CI **[+22.2, +66.7]**: eight improvements, no harms, exact two-sided paired sign-test p=0.0078125. The claimed gain is +34.5 pp [27.5, 41.5]. Patch beats sham on all 18 independent inputs. At 8192, the paired baseline is 0/2 and patch is 2/2: this is descriptive support only, with sign-test p=0.5. The raw summary's degenerate bootstrap [100,100] at n=2 must **not** be interpreted as precision or confirmation of D2. The audit did not independently reproduce the registered 200-example 8192 criterion.

For additional context, independent full-vocabulary accuracy at 4096 was none **3/18**, patch **9/18**, sham **0/18**. Production's corresponding figures are **21.5%, 52.0%, 0%**. Production's headline 8192 patch accuracy is **65% over 512 candidates**, but only **25.5% over the full vocabulary**.

**Issues, ranked by scientific impact**

No severity below means that a current saved verdict has been demonstrated to flip. Synthetic canaries establish a defect or inferential limitation, not that their particular scores occur in this checkpoint.

| Rank | File:line | Problem and consequence | Runnable evidence and observed output | Severity |
|---:|---|---|---|---|
| 1 | `RESULTS-6.md:56`, `RESULTS-6.md:64`; `report6.py:57` | The oracle rescue identifies a useful intervention site, not an exclusive natural bottleneck or the cause of attention loss. The average-score margin cannot separate pure dilution from strengthened competitors. The independent margin-ratio result also fails to reproduce the claimed 0.68. | `summarize_cpu_run.py`: margin **30.87 → 34.70**, while attention **0.472 → 0.171** and patch improves retrieval. Predecessor-minus-best-competitor falls **+1.057 → -1.354**, and mean predecessor rank rises **1.39 → 27.28**. `check_phase_c.py`: with the same 99 alternatives and unchanged mean margin **4**, concentrating their scores changes predecessor probability **0.35546 → 0.002463**. No extra tokens or weaker average margin are needed. | weakens a claim |
| 2 | `pilot6.py:44`, `pilot6.py:64` | Longer before inputs slice suffix filler at the new prefix offset. They preserve distance, but at 4096 they change existing post-block content instead of just inserting filler before it. That adds a content confound to the 2x-length comparison. | `check_phase_c.py`: all **200/200** before inputs at 4096 change their block-to-query token suffix, by **986.64 tokens on average**. At 8192, **0/200** change: the filler period is 24 tokens and 6144 is divisible by 24. This defect does **not** explain away D2 at 8192. My builder preserves the suffix and still reproduces a 4096 rescue. | weakens a claim (4096 contrast) |
| 3 | `pilot5.py:88` | The hook materializes QK in bfloat16, then casts its rounded result to float32. This adds rounding that differs from SDPA's attention computation. Reported probabilities and near-cutoff head membership therefore need not equal the model's actual attention. | `check_attention_precision.py`: a CPU BF16 canary gives hook p **0.250000**, BF16 SDPA output p **0.300781**, float32-accumulation reference **0.301368**. The hook rejects the head at 0.3 while SDPA qualifies it. Actual effects on the 39 selected heads and their aggregate D1 are unmeasured. | weakens a claim (diagnostic precision) |
| 4 | `pilot5.py:95`–`98` | The registered comparison is with other **earlier** tokens. Code includes the current value token itself among the alternatives and divides by t instead of t-1. This is an undisclosed definition deviation. | `check_phase_c.py`: isolated example gives registered margin **2**, code **-18**. On the matched real input, `check_forward.py` gives code/registered **38.7717/38.8148** inside and **26.8198/26.8234** far. The observed correction is small and does not explain the independent ratio discrepancy. The complete corrected 64-input D1b was not rerun. | weakens a claim (D1b fidelity); no demonstrated current label change |
| 5 | `pilot6.py:81`; `RESULTS-6.md:69` | The original baseline uses batch 4, interventions batch 1. Thus condition contrasts initially include a numerical computation change, even where the intervention is identity. This is disclosed and substantially addressed by the saved batch-one rerun. | `check_phase_c.py`: before @2048 changes **5/200 correctness bits**, net **+1.5 pp**, from batch 4 to 1. Batch-one none matches patch and sham **bit for bit** in every 2048 cell and every after cell. At 8192, gain is **+58.5 [52,65.5]** pp originally and **+59.0 [52,65.5]** pp with batch-one none. | weakens a claim (original contrast); saved correction supports the effect |
| 6 | `pilot6.py:119`; `report5.py:40`; `report6.py:42`, `report6.py:59` | Closed registered thresholds can fail exactly at their boundaries because comparisons use rounded floating-point accuracies/differences. These are real, currently inactive decision errors. | `check_decision_edges.py`, `check_numeric_details.py`: **70/100** tuning yields `chosen_k: null`; **140/200** inside is stored as **0.699999988** and fails both reports' >=70% gate. A patch/none pair **140/200 vs 100/200**, exactly +20 pp with positive bootstrap lower bound, fails D2; +20.5 pp passes. Use integer counts/rational comparisons for these boundaries. | cosmetic for current results; can change boundary-case verdicts |
| 7 | `PREREGISTRATION-5.md:75`; `RESULTS-5.md:30`; `pilot5.py:123` | Registration 5 still says no deviations, although post-gate diagnostics were run despite “nothing further runs.” Results disclose the diagnostics, but the registration does not. Nominal 4096-token pilot-5 sequences also contain 4094 actual tokens because records are rounded down. | `check_phase_c.py`: pilot-5 actual lengths **2048 / 4094 / 8192**; `results5/diagnostic.txt` and `logs/diag5.py` document the post-gate diagnostic. Neither affects the failed 2048 feasibility gate. | cosmetic / protocol documentation |

Evidence for rows 1–5 and 7 is in [phase_c_checks/checks.json](audit2/phase_c_checks/checks.json), [forward checks](audit2/phase_c_forward/checks.json), and [attention precision canary](audit2/phase_c_attention_precision.json). Boundary evidence is in [decision-edge checks](audit2/phase_c_decision_edges/checks.json) and [numeric checks](audit2/phase_c_numeric_details/checks.json).

**Code review: suspicions resolved and limits of the checks**

| Suspicion | CPU check and result |
|---|---|
| Broken NoPE forward, GQA or RMSNorm | Original forward versus independent forward on a 21-token paragraph, sharing float32 checkpoint weights: maximum logit difference **0**. Original RMSNorm matches independently calculated BF16 normalization exactly. Constructor name mapping, BF16 conversion and CPU placement pass with a synthetic loader. Source applies no RoPE. Group/position sentinel vectors verify query heads share the correct KV head. |
| Wrong value/query/answer positions | Reconstructed **1,800** pilot-6 sequences across placements/lengths and **600** pilot-5 before sequences. Query key uniquely identifies its block record, preceding token is the key, stored answer is that value, and paired answers remain unchanged. Pilot 6's actual lengths are exact. Its 8192 before value positions range **6378–7987**. |
| Wrong patch heads, vector, offset or boundary | Production hook exactly matches independent projected-V replacement on sentinel tensors across positions 2047/2048/2050 and KV groups. Short real-network comparisons with the intervention threshold deliberately lowered for testing give max logit difference **0** for patch and sham. Correct source offsets are 1 and 4. For the first block row, sham copies filler; rows 2–16 copy the previous key. |
| Future/query leakage | Changing the final query leaves all earlier short-input logits unchanged, maximum difference **0**. Static scoring review confines answer labels to comparisons against logits. Patch positions are oracle task metadata, explicitly part of the design. This does not exhaust every conceivable leakage route. |
| Long inputs used in tuning, or split overlap | A traced tuning stage constructs exactly **100 inputs at 2048** for each k, seed 60000, then chooses the largest passing k. Saved accuracies 81/85/93% select 16. Seeds 20000/20001 calibrate; 40000–40002 test. All **91 pairwise comparisons** among reconstructed tuning/calibration/test sequence sets have zero identical inputs. Fixed word pools are shared by design. Feasibility and main test intentionally reuse the same test inputs; they are not independent replications. |
| Wrong decision or bootstrap logic | Gates, D1, D1b direction and D2 components match the registration away from boundary cases. Independent signed paired-difference bootstrap reproduces the original report intervals, including harmed examples. Saved original 8192 patch has **118 wins and 1 loss** against none; batch-one comparison has **119 wins and 1 loss**. |
| Batch-one check changes the dataset | The post-hoc script was executed with inference and output replaced by test doubles, preserving its task generation and loops: all nine cells use **200 inputs, batch 1**, the same seeds, lengths and selected k. Saved bitstrings confirm identity-condition equality. This tests the pipeline and saved data, not the original GPU rounding itself. |

The float32 comparison bypasses the production constructor's full-checkpoint BF16 conversion to isolate architecture from precision; it is not a Hugging Face or GPU bitwise comparison. One reconstructed original test input, seed 40000/index 0 at 4096, was also run through the independent float32 model for none/patch/sham: all **three correctness bits match** the saved GPU bits. That input is wrong in all three conditions; it is a narrow indexing/inference spot check, not an additional estimate of rescue accuracy.

**Decision reproduction and original-file integrity**

I executed the actual `report5.py decision` and default `report6.py` entry points through `runpy`. Reads remain on the original saved results; decision-file writes are redirected, and `report.save` writes to an audit figure directory. An audit hook rejects writes outside `audit2/`, deletion, and network access. Matplotlib's attempt to remove its new font-cache lock was rejected; it still produced the figure, which was visually inspected.

| Entry point | Reproduced verdict | Comparison |
|---|---|---|
| `report5.py decision` | **NOT TESTABLE** | Decision text exactly matches existing `results5/decision.md` and agrees with `RESULTS-5.md`. |
| `report6.py` | **TOY MECHANISM HOLDS IN THE REAL MODEL; IN BETWEEN** | Decision text exactly matches existing `results6/decision.md` and agrees with `RESULTS-6.md`. |

Both generated decision documents, stdout, and `figures/fig13_pilot6_real_model.png` are under [audit2/phase_c_checks/](audit2/phase_c_checks/). All **54 saved restricted-accuracy cells** agree with their 200-bit correctness strings. The pilot-6 feasibility none strings equal the main none strings. Reports correctly retain pilot 5's failed gate rather than interpreting that task as a negative test of the mechanism. The saved batch-one correction leaves F, D1, D1b and D2's outcomes unchanged.

**Changes needed in the write-ups before citation**

- **`RESULTS-5.md:14`–`16`:** replace the unrestricted “forward pass is correct” with what was checked: a healthy paragraph loss and agreement on that paragraph's next-token argmax. The 0.605 maximum logit discrepancy and a single paragraph do not prove equivalence on all inputs. The independent architecture check now adds support, but the Hugging Face comparison was not independently repeated. Keep NOT TESTABLE scoped to this long random-pair format, not a general inability of the model. Add the diagnostic and length-rounding deviations to registration 5.
- **`RESULTS-6.md:3`–`8`, `40`–`46`, `56`–`65`:** distinguish the registered operational label from a demonstrated positional mechanism. State that the average-margin decline is the original task's observation and did not reproduce on this independent realization. Preserve the already acknowledged strong-competitor limitation; do not read “PREFERENCE KEPT” as proof of pure dilution. Replace “locates a bottleneck” with “identifies heads at which an oracle overwrite improves this task.” Disclose the 4096 suffix-content change and the two instrumentation issues.
- **`RESULTS-6.md:48`–`50`:** the after condition's preserved prefix and causality support a failure in later computation, but an identity intervention does not uniquely diagnose a second mechanism or localize it specifically to query lookup. “Plausibly” is appropriate; retain it as a hypothesis.
- **`RESULTS-6.md:69`–`73`:** the numerical deviation is real and the correction is substantiated. Prefer the matched batch-one baseline for scientific effect sizes: 86.0% inside, 51.0% at 4096, 6.0% at 8192; +59.0 pp patch gain at 8192. “Nothing changes” should mean no registered criterion changes, not no predictions change.
- **`README.md:24`–`31`:** keep pilot 5's task-specific negative result. Qualify pilot 6's bold label as the registered verdict on an oracle, 512-candidate synthetic retrieval test. The 65% figure is not unrestricted next-token accuracy; that is 25.5%. Add that the independent audit reproduces the rescue direction and 4096 advantage, while its 8192 intervention sample is only two examples and its margin ratio differs. Do not imply that the entire 200-input 8192 result has now been independently rerun.

A defensible citation would say: **In one published NoPE checkpoint on a synthetic word-pair task, selected heads' predecessor attention declines on long inputs, and an oracle overwrite with the preceding key's projected value substantially improves retrieval over 512 candidate answers. An independent CPU implementation reproduces the 4096 rescue; the cause of the attention change remains unresolved.**

**What remains unverified**

The complete GPU bfloat16 run, all 200 original 8192 predictions, original full-vocabulary correctness counts, and the Hugging Face reference were not independently rerun. The 64-input head statistics were not recomputed with corrected margin definition or attention precision. The independent four-input diagnostic intervals are conditional and narrow in scope; the two 8192 treatment examples cannot confirm D2. Original selection and inside measurement use the same 64 inputs, so the reported inside means are selection-conditioned; my fresh reporting sample avoids that reuse but is small.

Original downloads use an unpinned Hub name and save no checkpoint revision/hash with the result JSONs. Local checkpoint identity is recorded for this audit, but historical GPU checkpoint bytes cannot be authenticated from those outputs. Local commit claims, GPU logs and scripts do not independently establish preregistration timing or human approval. No other models, natural documents, non-oracle method, attention-scaling explanation, or exclusive causal localization were tested. No internet verification of the literature or model provenance was attempted under the offline rule.

**Runnable evidence**

From the repository root, use `C:\Users\pkchi\Documents\gate-length-pilot\.venv\Scripts\python.exe -B` with the scripts below. Outputs use exclusive creation. To repeat checks without overwriting existing evidence, first set `$env:AUDIT2_CHECK_TAG='_recheck1'` in PowerShell (choose a new tag for another repeat). This suffix affects only new check-output names; inputs remain unchanged. No further long experiment is needed to read this audit.

| Script | Purpose |
|---|---|
| `audit2/summarize_cpu_run.py` | Independent run counts, exact binomial intervals, paired effects and diagnostic bootstrap. |
| `audit2/check_phase_c.py` | Reconstruct task inputs, verify hooks and saved numbers, trace stages with synthetic inference, execute and redirect both report entry points, compare tracked-file hashes. |
| `audit2/check_provenance.py` | Regenerate all 82 audit inputs, check script hash and pairing, compare all original split identities. |
| `audit2/check_forward.py` | Real float32 forward/hook equivalence and the limited matched original inputs; about ten minutes on this CPU. |
| `audit2/check_attention_precision.py` | Small BF16 attention-rounding canary. |
| `audit2/check_decision_edges.py` | Synthetic tuning/F/D2 and margin-ratio boundary cases. |
| `audit2/check_numeric_details.py` | Synthetic constructor, actual BF16 RMSNorm formula, and pilot-5 gate boundary. |
| `audit2/verify_final_integrity.py` | Read-only final comparison against all original tracked-file hashes. |

The main independent script and raw outputs have not been altered since the owner's run. Later evidence-script edits only add fresh output suffixes for repeatability.
