# Pilot 5 results: real-model test, NOT TESTABLE

**Pre-registered verdict: NOT TESTABLE.** The feasibility gate failed. Wang et al.'s TinyLlama-NoPE-1.1B cannot
do the key–value retrieval task reliably even *inside* its 2048-token training length, so there is no
length-generalization failure to explain with this task. As registered, the binding-head measurements and the
patch were not run. Nothing here says whether pilot 4's mechanism holds in a real model; that is still open.

Plan: [PREREGISTRATION-5.md](PREREGISTRATION-5.md) (signed commit `b75c9b7`, before any data). Decision:
[results5/decision.md](results5/decision.md). Compute: 0.91 hours on one rented RTX 3090 (≈ $0.21), most of it
spent downloading.

## What ran

1. **The forward pass is correct.** `pilot5.py`'s own implementation scores 2.09 nats per token on an ordinary
   English paragraph, a sensible value for a 1.1B model. It agrees with the Hugging Face reference on 100% of
   next-token predictions; the maximum logit gap of 0.6 comes from bfloat16 versus float32.
   ([results5/verify.json](results5/verify.json))
2. **Feasibility gate: failed.** Accuracy over the 512 value words, 200 inputs per cell (chance 0.2%):

   | placement | 2048 (trained length) | 4096 | 8192 |
   |---|---|---|---|
   | before | 9.0% | 1.0% | 0.0% |
   | uniform | 6.0% | 0.0% | 0.0% |
   | after | 6.0% | 0.5% | 0.0% |

   The gate needed ≥ 70% at 2048.
3. **Diagnostic, after the gate and not pre-registered, run only to rule out a pipeline bug.** Shorter lists (50
   inputs each; [results5/diagnostic.txt](results5/diagnostic.txt)):

   | word pairs | tokens | accuracy (512-way) | top-1 over the whole vocabulary |
   |---|---|---|---|
   | 10 | 47 | 70% | 26% |
   | 50 | 167 | 40% | 16% |
   | 200 | 617 | 40% | 2% |
   | 400 | 1217 | 22% | 0% |

   The model does retrieve in context; on short lists the correct word is often its first or second guess. But
   with hundreds of random word pairs its accuracy collapses well before 2048 tokens. The pipeline works; the task
   is too hard for this base model.

## What this does and does not mean

- It does **not** count against pilot 4. The real-model question simply was not reached.
- A feasible real-model test needs a task the model solves inside 2048 tokens. One option is a short list of pairs
  (about 10–20, where the model gets 70%) inside long natural filler text, placed far from the start. That would
  be a *new* design with its own pre-registration, not a change to this one.

## Limitations

- The words were random single-token pieces of the vocabulary, many not English. Natural words might be easier.
- One model, base (not instruction-tuned), so the format was learned only from the in-context examples.
