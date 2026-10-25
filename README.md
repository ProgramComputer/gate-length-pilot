# gate-length-pilot

Pre-registered pilot experiments on one failure of transformers without positional encoding (NoPE): key-value retrieval when the input is longer than the training length. Models: 1.1M-parameter toy models trained on at most 128 tokens, and TinyLlama-NoPE-1.1B, trained on 2048. The patch is an oracle diagnostic: it is told which heads and positions to change, so it shows where an intervention helps, not a usable method. Write-ups and code: [OVERVIEW.md](OVERVIEW.md).

![Concept figure](readme/fig1_concept.png)

*Figure 1.* Toy token layout. (a) Within the training length, binding heads attend from each value token to its key. (b) Beyond it, that attention is weaker. (c) The patch replaces these heads' output at the value token with the head's value vector at position t-1 (the key token); the sham uses t-3 (t-4 in TinyLlama, not shown).

![Accuracy vs. input length](readme/fig2_length.png)

*Figure 2.* Accuracy vs. input length, with all added tokens placed before the key-value pairs. Gray background: within the training length. (a) Toy models: no patch is the mean of 3 seeds (gray band: range); patch and sham were run at 1024 tokens only, 6 seeds (dots, spread sideways) and their mean (large marker). (b) TinyLlama: 200 inputs per point, bands are 95% CIs, 512 candidate answers. The sham is a wrong-source control.

![Attention to the key by layer and head](readme/fig3_heads.png)

*Figure 3.* Attention from each value token to its key (t-1). (a, b) TinyLlama, every layer and head, for pairs within the first 2048 tokens and beyond position 6000 of 8192. Outlined: the 39 binding heads (attention of at least 0.3 in (a), selected on the same inputs); their mean is 0.46 in (a) and 0.11 in (b). (c) Toy models, mean over each seed's binding heads; thin lines: 6 seeds, thick: their mean, gray: training positions.

![Gain in accuracy for each intervention](readme/fig4_gains.png)

*Figure 4.* Toy models at 1024 tokens (8x the training length): gain over each seed's own unpatched run, with pairs anywhere (a) or after all added tokens (b). Dots: seeds (3; 6 for the patch). Diamonds: means. In (b), the largest mean gain among the other interventions is 1.4 percentage points; the patch's is 74.5. Logit scale multiplies every attention logit.

![Original results vs. re-implementation](readme/fig5_reproduction.png)

*Figure 5.* Re-implementation with separate model and intervention code on the same checkpoints (toy inputs from this repo's generator, TinyLlama inputs drawn anew). (a) Toy, 1024 tokens, one point per seed and condition; inset: low-accuracy corner. (b) TinyLlama, 4096 tokens, 200 original vs. 18 new inputs. Bars: 95% CIs, horizontal for the original, vertical for the re-implementation; diagonal: equality; every pair of intervals overlaps. (c) Beyond-to-within ratios, windows as in Figure 3; small dots: toy seeds; bars: bootstrap 95% CI. Score margin: logit of the key minus the mean logit of earlier tokens. Both attention ratios fall below the pre-registered 0.5 threshold (dotted); the margin ratio falls outside its pre-registered range (gray) and does not reproduce.
