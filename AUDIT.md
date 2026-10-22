**Overall judgment: partly sound.** The pilot-4 retrieval rescue is genuine: an independently written CPU forward pass reproduces the large effect, including the weak response of seed 2 and intermediate response of seed 5. The saved decisions reproduce, and I found no demonstrated error that changes a registered verdict on these files. The scientific account is less secure. Pilot 3B's supposedly positional profiles are strongly confounded with alternating key/value roles; several write-up statements turn a limited measurement into a stronger mechanistic conclusion. Pilots 3 and 4 also change the long test inputs without recording the deviation, and pilot 1's sham is not size-matched. The defensible finding is an oracle rescue at selected attention heads in these checkpoints, with substantial seed variation. The specific explanation of fading implicit position information needs to remain a hypothesis.

Audit date: 2026-10-22. Worktree branch: `audit-codex`. Line numbers refer to the original files. All numerical experiments used the supplied Python, CPU only, with no installations or training. Only this file and files under `audit/` were written.

**Independent pilot-4 reproduction**

Before writing [audit/reimpl_pilot4.py](audit/reimpl_pilot4.py), I read the four permitted sources: registration 4, registration 1 section 3, `task.py`, and `model.py`, plus checkpoint structure needed for loading. I opened `pilot4.py`, `RESULTS-4.md`, and `results4/` only after all six independent seeds finished. The script implements normalization, Q/K/V projections, causal attention, attention replacement, FFNs, and readout directly from checkpoint tensors; it does not use the production attention-patch hook.

Calibration used 256 seed-20000 length-1024 inputs and 16 evenly spaced value positions per window. Production samples random positions; this independent sampling choice found exactly the same selected heads in every seed. Evaluation used exactly `paired_set(200, (128, 1024), "before", seed=40001)`, as requested. These are a new 200-example realization, rather than a literal prefix of the 1000-example run. Predictions use argmax over the 64 answer logits at the final token.

“Claimed” below means the per-seed saved accuracies underlying `results4/decision.md`; production used 1000 examples. Each cell gives **claimed / independent audit**.

| Seed | None (%) | Patch (%) | Sham (%) | Patch gain (pp) | Assessment |
|---|---:|---:|---:|---:|---|
| 0 | 1.5 / 3.5 | 97.9 / 97.5 | 1.1 / 1.5 | +96.4 / +94.0 | Agrees |
| 1 | 1.9 / 1.0 | 84.0 / 84.5 | 1.2 / 0.5 | +82.1 / +83.5 | Agrees |
| 2 | 1.4 / 0.5 | 20.0 / 19.0 | 1.5 / 1.5 | +18.6 / +18.5 | Agrees |
| 3 | 1.6 / 1.0 | 98.7 / 99.0 | 1.2 / 1.0 | +97.1 / +98.0 | Agrees |
| 4 | 1.8 / 1.5 | 98.4 / 99.0 | 1.2 / 1.0 | +96.6 / +97.5 | Agrees |
| 5 | 1.8 / 2.0 | 57.9 / 55.0 | 1.5 / 0.5 | +56.1 / +53.0 | Agrees |
| Mean | 1.67 / 1.58 | 76.15 / 75.67 | 1.28 / 1.00 | +74.48 / +74.08 | Agrees |

Independent mean attention to the preceding key, averaged over the selected heads:

| Seed | 64–127 | 128–255 | 256–511 | 512–1023 |
|---|---:|---:|---:|---:|
| 0 | .689 | .363 | .115 | .028 |
| 1 | .661 | .361 | .109 | .025 |
| 2 | .936 | .550 | .112 | .009 |
| 3 | .735 | .456 | .166 | .038 |
| 4 | .642 | .343 | .097 | .019 |
| 5 | .612 | .339 | .118 | .031 |

D1 and D2 also pass on this independent sample. Paired-bootstrap 95% intervals for patch-minus-none, in seed order, are `[90.5,97.0]`, `[78.0,88.5]`, `[13.5,24.0]`, `[96.0,99.5]`, `[95.0,99.5]`, and `[46.0,60.0]` pp. The independent patch-minus-sham mean is +74.67 pp. D3's full probe experiment was not independently rerun; its saved-data decision was reproduced.

The unpatched independent forward matched `Model` exactly on the validation inputs. After opening the implementation, additional comparisons of none/patch/sham logits on two long inputs per seed also had maximum absolute difference **0** in all 18 comparisons. This directly checks that the production intervention copies the projected V vector at `t−1` or `t−3`, before the attention output projection, at the intended value positions.

Raw predictions, checkpoint hashes, all-head measurements, and timing are in [audit/reimpl_pilot4.json](audit/reimpl_pilot4.json). Bootstrap comparisons are in [audit/pilot4_comparison.json](audit/pilot4_comparison.json). The main independent run took 408.8 seconds wall time with four CPU threads; the remaining checks were small inference, algebra, or saved-data checks.

**Issues, ranked by scientific impact**

No row below demonstrates a changed verdict on the current saved results. “Cosmetic” includes an inactive decision-rule edge case; it does not imply that the source matches the registration exactly. Counterexamples establish limitations of an inference; they are not claims that the real checkpoints use those counterexample mechanisms.

| Rank | File:line | Problem and consequence | Runnable evidence and observed output | Severity |
|---|---|---|---|---|
| 1 | `pilot3.py:211`; `RESULTS-3.md:39` | The fitted `a(i−j)` does not distinguish position computation from key/value role interactions. Alternating roles make a purely content-based role interaction look like a relative-distance profile. High profile correlation therefore does not establish that the ability to identify the preceding key survives. | `audit/check_decomposition.py`: role-only logits, with all keys having identical features, produce fit correlations **1.0**, `rho_a=1.0`, and inside a-share **.99988**. In the real selected NoPE heads, parity explains **87.47%, 79.41%, 99.84%** of inside mean-profile variance and **99.92%, 99.96%, 99.98%** far away. The original registered rho calculation is still reproduced; these parity calculations concern the saved mean profiles. | weakens a claim |
| 2 | `RESULTS-4.md:53`; `RESULTS-4.md:62` | The rescue localizes a useful intervention site. It does not identify fading implicit positional signal as the cause of the attention decline, establish an exclusive bottleneck, or show that the forced circuit exactly restores the natural one. Attention probability depends on competing tokens as well as predecessor preference. | `audit/check_claims.py`, `fixed_predecessor_logit_counterexample`: keep the predecessor logit advantage fixed at **ln(64)**. Attention at positions 96/192/384/768 is **.400/.250/.143/.077**, satisfying the registered attention-drop inequality without weakening the relative score preference. The real independent rescue remains large; the cause of its baseline attention decay was not isolated. | weakens a claim |
| 3 | `gatepilot/probe.py:3`; `RESULTS-2.md:52`; `RESULTS-2.md:91` | Failure of a refitted linear probe does not establish missing information rather than a nonlinear representation. The write-up acknowledges this limitation later, but its main explanatory language is stronger. The fallback is triggered by failure inside the range, so successful inside probes do not test nonlinear information far away. | `audit/check_claims.py`, `probe_counterexample`: the repository's ridge probe scores **50%** on a balanced XOR representation, while adding its deterministic product feature yields **100%**. The representation already contains the label. This is a logical counterexample, not evidence of nonlinear recovery in these models. | weakens a claim |
| 4 | `pilot3.py:153`; `pilot4.py:100`; `gatepilot/task.py:90` | Registrations 3 and 4 promise pilot-1 test sets. Requesting `(128,1024)` instead of all four lengths changes RNG consumption and therefore the long inputs. The deviation sections remain empty. Within-pilot comparisons stay paired, but cross-pilot long-input comparisons are not the promised identical examples. | `audit/check_data_stats.py`: for each placement, **0/1000** long sequences match between the two generation calls; short sequences and answer arrays match. Uniform target distances differ in **961/1000** rows. All six saved pilot-4 distance arrays match the reduced-length generation. Pilot 3 uses that same reduced-length call. | weakens a claim |
| 5 | `gatepilot/correct.py:118`; `gatepilot/correct.py:123` | Refitting the sham under its own upstream changes gives later layers different adjustment sets from the real correction. It is not an equally sized shuffled version of the real intervention. This is already disclosed in `RESULTS.md:125`, but weakens G2's control interpretation. G1 and G3 fail independently, so this does not overturn ABANDON. | `audit/check_interventions.py`: saved mean standardized sham/gate size ratios range from **1.184 to 1.615** at 512/1024. A small actual fit gives layer-0 sizes **.407/.407**, then layer-1 **.454/.810** for real/sham, confirming divergence after upstream interventions. This metric is standardized adjustment size, not a measurement of raw activation displacement. | weakens a claim |
| 6 | `pilot3.py:214`; `RESULTS-3.md:42` | `var(a[d])/var(logits)` is a component variance ratio, not an orthogonal fraction of variance explained by position. Covariance with row/key effects matters. Regions also weight distances differently. “Position takes over” is stronger than this statistic warrants, especially given issue 1. | `audit/check_decomposition.py`, `non_partition_counterexample`: an exactly additive score matrix has fit correlation **1.0** but reported a-share **1.20076**. Actual saved ratios stay below one; the counterexample shows that their interpretation as a variance partition is not justified. | weakens a claim |
| 7 | `RESULTS-4.md:37`; `report4.py:74` | Residual retrieval error after the patch is not specifically localized to the query side. D3 probes key identity, not correct value content or every intermediate binding operation. A key can be perfectly decodable while another unpatched part of the value pathway remains damaged. | `audit/check_claims.py`, `unlocalized_residual_error_counterexample`: key-binding accuracy **100%**, answer accuracy **0%**, with a perfect reader of deliberately corrupted stored values. The real saved seed-2 result is also incomplete: patch retrieval **20%**. Additional query-side failure is compatible with the data, but not established by D3. | weakens a claim |
| 8 | `RESULTS-4.md:32` | “These are different heads from the most positional head” is false for seed 1. Different head selection cannot by itself reconcile pilots 3B and 4 in every seed. | `audit/check_decomposition.py`: seed 1's selected positional head is **(2,0)**, and it is in pilot 4's binding-head list. Its previous-key attention falls **.8446 → .0295** despite registered profile correlation **.871**. Indices here are zero-based. | weakens a claim |
| 9 | `report3.py:50` | A tie in diagnostic worsening factors selects semantic via `>=`; the registration says semantic only for a larger factor, otherwise positional. This is a real inactive edge-case deviation. | `audit/check_claims.py`, `direction_tie`: equal factors **2.0/2.0** give code **semantic**, registered **positional**. Current saved factors are nowhere near a tie, so neither pilot-3 verdict changes. | cosmetic |
| 10 | `RESULTS-4.md:3`; `RESULTS-4.md:28`; `RESULTS-3.md:20` | Some wording conflates aggregate and individual measurements. D2's 20-pp bar is across seeds, not separately cleared by all six; the .6–.9 attention range describes head averages rather than every selected head. Pilot 3A calls its diagnosed tokens “keys,” although the selected tokens are values. | `audit/check_claims.py`: seed 2's saved gain is **+18.6 pp**; selected individual-head inside attention reaches **.3037**; all three pilot-3A seeds select **value/value**. The underlying registered aggregate calculations are unaffected. | cosmetic |

Machine-readable evidence is in [audit/decomposition_checks.json](audit/decomposition_checks.json), [audit/claim_checks.json](audit/claim_checks.json), [audit/data_stats_checks.json](audit/data_stats_checks.json), and [audit/intervention_checks.json](audit/intervention_checks.json).

The mechanism behind issue 1 is simple: let `r_i` denote the token's key/value role, encoded as −1/+1. Role-only logits `W_ij = −r_i r_j` become `−(−1)^(i−j)` in this alternating sequence. The fit assigns that interaction to `a(distance)`, although every key has identical features and the head cannot identify which key immediately precedes a value.

**Verdict reproduction and checks that did not reveal a defect**

I executed each original script's `decision` entry point through [audit/run_decisions.py](audit/run_decisions.py), using `runpy` with the original `decision` argument. These entry points write existing `results*/decision.md`; the harness redirects only those writes into `audit/decisions/`. Inputs and decision computations are unchanged. Each regenerated decision document matches its existing counterpart exactly as text.

| Entry point | Reproduced verdict | Matches write-up | Decision-relevant checks |
|---|---|---|---|
| `report.py decision` | ABANDON | `RESULTS.md`: yes | Sanity, M1, and short checks pass; G1/G2/G3 fail overall; G4 passes. |
| `report2.py decision` | MIXED | `RESULTS-2.md`: yes | V1/V2 pass; R/P1/P3 fail. The MLP fallback is not triggered by these results. |
| `report3.py decision` | DETECTS BUT DOES NOT FIX; NO BREAK IN NOPE | `RESULTS-3.md`: yes | A1 passes, A2/A3 fail; B1 fails and B2 passes. The scientific interpretation of B is limited by issues 1 and 6. |
| `report4.py decision` | BINDING ATTENTION IS THE BOTTLENECK | `RESULTS-4.md`: yes | Saved D1/D2/D3 pass. Independent retrieval/attention also pass D1/D2. |

The files are archived under [audit/decisions/](audit/decisions/); [summary.json](audit/decisions/summary.json) records the matches.

The review covered every requested source and registration, plus `train.py`. The runnable checks specifically addressed the following suspicions:

- **Wrong answer, distance, or role indexing:** 1000 examples × four lengths × three placements pass independent reconstruction of the queried key/value, uniqueness, answer preservation, placement invariants, and patch positions. Nominal length 1024 contains 1023 actual tokens; that is registered, not an off-by-one bug.
- **Answer leakage or a future-query leak:** changing only scoring answers leaves logits unchanged; changing the final query leaves all earlier recorded residual states unchanged, both with maximum difference **0**. No identical length-1024 rows occur between the tested named calibration/check/test sets. Static review finds answer labels confined to training, declared probe fitting/scoring, and answer scoring; patch selection does not read test answers. These checks do not exhaustively compare against every historical training example.
- **Probe-token leakage across its train/test split:** sampled labels match the preceding key; all sampled rows from an input stay on one side of the example-level split. A 12-input, 768-row check passes. Probe standardization is fit on training features.
- **Patch/sham implementation or short identity:** all six seeds' selected heads match the independent selection, long-condition logits match the independent replacement implementation, and every pilot-1 short condition is exactly identical in the checked logits.
- **Bootstrap/t-interval mistake:** the paired bootstrap matches an independently computed bootstrap of signed paired differences, including negative differences; the three-seed t half-width matches `4.303 × sample_SD / sqrt(3)`. There is no demonstrated pairing or degrees-of-freedom error.
- **Wrong RoPE score intervention:** independently checked band contributions scale by the intended alpha; maximum score discrepancy is below **4e-6** in the tested cases.
- **Saved accuracy inconsistent with saved examples:** all **744** accuracy entries in the **27** result JSONs match their 1000-bit correctness strings. Pilot-1 and pilot-2 baseline bitstrings are checked separately in the saved-prediction audit.
- **Broken alternating fit itself:** on a separately generated known additive model, 30 rounds recover a detrended profile with correlation **1.0**. This check does not validate the positional interpretation of the fitted component.

Finally, [audit/check_saved_predictions.py](audit/check_saved_predictions.py) reruns checkpoint inference at 12 independently selected original test indices. Across pilot-2 temperatures and positional schemes, pilot-3A conditions, and pilot-4 conditions/seeds, **1,080 correctness entries matched with zero mismatches**. Pilot-1 and pilot-2 baseline correctness strings also match in full. These spot-check results are saved in [audit/saved_prediction_checks.json](audit/saved_prediction_checks.json).

**Write-up changes needed before describing these results externally**

Keep the numerical oracle-rescue result and its seed-by-seed variation. Describe pilot 3B as stability of an additive distance-indexed component in an alternating-role task, with the parity confound explicit. Replace “information is missing/disappears” with “not linearly decodable by the tested probes.” Treat fading implicit position, the residual query-side localization, and false-match removal in the after-placement condition as hypotheses. Disclose the changed long test realizations and the sham mismatch. Correct the head-overlap and aggregate/individual wording in issue 10. The formal verdict labels can be reported as the outputs of the registered rules, with these interpretation limits attached.

Pilot 1 supports abandoning the tested gate-specific affine-alignment proposal. Its negative result does not establish that all gate mechanisms or richer corrections are ineffective; the write-up already states important limits. Pilot 3A supports failure of the registered intervention grid here. This audit checks that implementation against this repo's registration; it does not independently certify fidelity to external papers, novelty, or the causal connection between the diagnostic score and individual answer failures.

**Re-running the audit**

From the repository root, set `PYTHONDONTWRITEBYTECODE=1` and use `C:\Users\pkchi\Documents\gate-length-pilot\.venv\Scripts\python.exe -B` followed by a script below. All outputs remain under `audit/`.

| Script | Purpose |
|---|---|
| `audit/reimpl_pilot4.py` | Full independent six-seed, 200-example reproduction; CPU threads/batch/sample counts are CLI options. |
| `audit/summarize_reimpl.py` | Generate per-seed comparison and independent bootstrap intervals from that run. |
| `audit/run_decisions.py` | Run all four original decision entry points with writes redirected; verify tracked-file hashes. |
| `audit/check_data_stats.py` | Data, position, split, saved-accuracy, bootstrap, and t-interval checks. |
| `audit/check_interventions.py` | Compare patch math, short identity, scoring-label/causality canaries, RoPE edits, and sham sizes. |
| `audit/check_decomposition.py` | Role/parity confound, component-variance counterexample, known-fit check, and head overlap. |
| `audit/check_claims.py` | Negative-probe, attention-normalization, residual-localization counterexamples, and wording/tie checks. |
| `audit/check_saved_predictions.py` | Spot-check saved correctness against checkpoints at independently selected test indices. |

Training was not repeated, and the full saved probe accuracies and pilot-1 correction accuracies were not independently regenerated. Signed local commit order is consistent with the registration sequence, but this audit cannot independently establish what was observed before each commit. Final integrity verification found **0 changed files among all 133 original tracked files**. The hash manifest is `audit/decisions/tracked-file-sha256.json`; git status shows only the new `AUDIT.md` and `audit/` paths. No commits or pushes were made.
