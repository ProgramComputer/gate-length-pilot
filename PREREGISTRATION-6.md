# Pre-registration 6: real-model test, redesigned (a few pairs inside filler text)

Written 2026-10-22, after pilot 5 and before any pilot-6 data existed. The git commit that adds this file is the
timestamp. The user approved this redesign, and asked that only the tuning and feasibility stages run before they
are asked again.

## Why a redesign

Pilot 5's task, hundreds of random word pairs, was too hard for TinyLlama-NoPE-1.1B even inside its trained length
(6–9%; verdict NOT TESTABLE). The question is unchanged: does pilot 4's finding hold in a real NoPE model? That
finding is that binding heads lose their look-back past the trained length, and restoring it restores retrieval.
The audit's open question about *why* the look-back fades also carries over: weaker preference, or dilution.

## Model and task (`pilot6.py`)

- **Model:** `AntNLP/TinyLlama-NoPE-1.1B`, with pilot 5's verified forward pass (bfloat16).
- **Input:** a preamble line, then passkey-style filler ("The grass is green. The sky is blue. The sun is yellow.
  Here we go. There and back again.", repeated, as used by Wang et al.). Inside the filler sits a block of k lines,
  ` key value`, using pilot 5's single-token word pools. After more filler comes a newline and one of the block's
  keys, and the next token should be its value.
- **Accuracy:** argmax over the 512 value words (chance 0.2%). Full-vocabulary argmax is also reported.
- **Base inputs (2048 tokens):** the block starts at a uniformly random point between 10% and 90% of the filler.
  Longer versions (4096, 8192) add filler in one of three placements, and the answer never changes:
  - *before* the block: the block is far from the start but at the same distance from the query (primary);
  - *after* the block: the block stays early and the distance grows;
  - *uniform*: the extra filler is split at random.

## Stages

1. **Tune k** (`--stage tune`), on 2048-token inputs only (seed 60000, 100 inputs, "before" placement). Measure
   accuracy for k ∈ {16, 8, 4} and choose the largest k with ≥ 70%. If none qualifies, the verdict is NOT TESTABLE
   and nothing further runs. This stage never sees an input longer than the trained length.
2. **Feasibility gate F** (`--stage feasibility`), with the chosen k: 200 inputs per placement (seeds
   40000–40002) at 2048 / 4096 / 8192. F passes if "before" accuracy at 2048 is ≥ 70% and accuracy at 8192 is
   ≥ 20 pp lower. If F fails, the verdict is NOT TESTABLE.
3. **Main test** (`--stage full`). It runs **only after the user approves it**, which is not decided now.
   - **Binding heads:** on 64 inputs with the block inside (2048 tokens, uniform, seed 20000), the heads where the
     block's value tokens put ≥ 0.3 attention on the token just before them. If none qualifies, the single highest
     head is used.
   - Measured for those heads: attention to the previous token, *p*, and the score margin (the previous token's
     score minus the mean of all other earlier tokens). Both are taken inside, and far: 64 "before" inputs at
     8192 tokens (seed 20001), where the block sits past position 6000.
   - **Patch:** at the block's value positions ≥ 2048, the binding heads output the previous token's value vector.
     **Sham:** the token 4 back. All conditions are scored on all lengths and placements.

## Decision (`python report6.py`)

- **F:** as above.
- **D1:** binding heads' mean *p* far ≤ ½ of *p* inside.
- **D1b:** margin ratio far / inside. ≥ 0.8 means PREFERENCE KEPT (dilution); ≤ 0.5 means PREFERENCE LOST
  (position); in between is reported as such.
- **D2:** on "before" @8192, the patch minus none is ≥ 20 pp, its paired bootstrap 95% CI is above 0, and the
  patch beats the sham by ≥ 20 pp.

Verdict, with D1b reported alongside:
- **TOY MECHANISM HOLDS IN THE REAL MODEL** (D1 and D2);
- **ATTENTION FADES BUT THE PATCH DOES NOT RESCUE** (D1 only);
- **BINDING ATTENTION DOES NOT FADE** (not D1);
- **NOT TESTABLE** (tuning or F fails).

## Limitations known in advance

- One pretrained model, a synthetic task and repetitive filler text.
- k is chosen on in-range data. That makes the task easier, but cannot bias the long-length questions.
- The real model may fail past its trained length for many reasons at once, so a failed D2 would not rule out
  binding heads as one cause.
- This is a local signed commit, not an independent timestamp.

## Deviations

- *(none yet)*
