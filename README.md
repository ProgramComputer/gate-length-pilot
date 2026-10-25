# gate-length-pilot

![Concept: binding heads look from each value back to its key; the look-back fades past the trained length; the oracle patch copies the key back in, the sham copies a wrong key](readme/fig1_concept.png)

**Figure 1.** Binding heads tie each value to the key just before it; past the trained length that look-back fades, and the oracle patch writes it back in.

![Accuracy vs input length for toy NoPE models and TinyLlama-NoPE-1.1B, with no patch, patch and sham](readme/fig2_length.png)

**Figure 2.** Past the trained length retrieval collapses in both models; patching the binding heads recovers most of it, and the sham does not (TinyLlama scored over 512 candidate words).

![Layer-by-head attention to the previous token in TinyLlama-NoPE-1.1B inside 2048 and past 6000, and toy models by position](readme/fig3_heads.png)

**Figure 3.** The binding heads stop looking back past the trained length: 0.46 → 0.11 on TinyLlama's 39 outlined heads, 0.71 → 0.02 in the toy models.

![Gain in accuracy at 8 times the trained length for eight earlier fixes and the binding-head patch](readme/fig4_gains.png)

**Figure 4.** None of eight earlier fixes recovers pairs far from the start; the binding-head patch, an oracle, adds 74.5 points.

![Original results against independent re-implementations: accuracies and far-to-inside ratios](readme/fig5_reproduction.png)

**Figure 5.** Rebuilt independently from the pre-registrations, the accuracies and the attention fade reproduce; TinyLlama's score-margin ratio does not.

[Overview, verdicts and how to reproduce](OVERVIEW.md) · figures: `python readme/build.py`
