# gate-length-pilot

A small, pre-registered pilot experiment. Question: **does a length-dependent shift in SwiGLU feed-forward gate
statistics causally contribute to Transformers failing on inputs longer than they were trained on, beyond what
attention-side failure explains?**

- Design and the fixed decision rule: [PREREGISTRATION.md](PREREGISTRATION.md). It was committed before any
  long-length result existed; see `git log`.
- Results and verdict: [RESULTS.md](RESULTS.md).
- **Pilot 2** (follow-up: why records beyond the trained length are lost; NoPE vs a RoPE control, binding probes
  and attention temperature): [PREREGISTRATION-2.md](PREREGISTRATION-2.md) and [RESULTS-2.md](RESULTS-2.md).
  Code: `gatepilot/probe.py`, `pilot2.py`, `report2.py`. Verdict: MIXED.
- **Pilot 3** (two tests aimed at open questions in two recent papers: RoPE Profiler, Wu, Du & Peng
  2026; attention decomposition, Han & Ji 2025): [PREREGISTRATION-3.md](PREREGISTRATION-3.md) and
  [RESULTS-3.md](RESULTS-3.md). Code: `pilot3.py`, `report3.py`. Verdicts: DETECTS BUT DOES NOT FIX; NO BREAK IN NOPE.
- **Pilot 4** (final; is the lost binding caused by the previous-token heads? Six NoPE seeds, three of them
  fresh): [PREREGISTRATION-4.md](PREREGISTRATION-4.md) and [RESULTS-4.md](RESULTS-4.md). Code: `pilot4.py`,
  `report4.py`. Verdict: BINDING ATTENTION IS THE BOTTLENECK.

**Where things stand:** RESULTS-4.md has a one-table summary of all four pilots. The series stopped after pilot 4,
as pre-registered.

**Pilot 5** (new study: does pilot 4 hold in Wang et al.'s real 1.1B NoPE model?): [PREREGISTRATION-5.md](PREREGISTRATION-5.md)
and [RESULTS-5.md](RESULTS-5.md). Code: `pilot5.py`, `report5.py`. Verdict: NOT TESTABLE. The model cannot do the
task reliably even inside its training length, so the real-model question remains open.

**Pilot 6** (pilot 5 redesigned so the model can do the task: 16 pairs hidden in filler text):
[PREREGISTRATION-6.md](PREREGISTRATION-6.md) and [RESULTS-6.md](RESULTS-6.md). Code: `pilot6.py`, `report6.py`,
`pilot6_batch1.py`. Verdict: **TOY MECHANISM HOLDS IN THE REAL MODEL.** In TinyLlama-NoPE-1.1B, 39 binding heads
lose their look-back past 2048 tokens, and restoring it raises retrieval at 8192 from 6.5% to 65% (sham: 1%).

**Independent audit:** [AUDIT.md](AUDIT.md), with its checks in `audit/`. A separate AI (Codex, OpenAI)
re-implemented pilot 4 from its registration without seeing this code or results first, and reproduced its
numbers seed by seed. It also reproduced all four verdicts, found no error that changes one, and listed where the
write-ups overreached. The write-ups and the registrations' deviation sections were corrected on 2026-10-22 to
match the audit. No number or verdict changed.

## Layout

| file | what it is |
|---|---|
| `gatepilot/task.py` | randomized key-value retrieval; paired short/long inputs; distractor placement |
| `gatepilot/model.py` | 1.1M-parameter causal Transformer, SwiGLU FFN, no positional embeddings |
| `gatepilot/correct.py` | **the core**: gate correction, controls, sham, attention temperature, evaluation loop |
| `train.py` | trains one seed on lengths ≤ 128 |
| `evaluate.py` | fits every intervention on calibration data, scores the test sets, runs manipulation checks |
| `report.py` | figures, tables and the pre-registered decision |
| `runs/seed*/` | trained checkpoints and training logs |
| `results/` | per-seed results (`seed*.json`), sanity check (`baseline_seed*.json`), `decision.md` |
| `figures/` | the figures |
| `logs/` | training/evaluation logs and launch scripts from the GPU run, including the learnability probes |

## Reproduce

```bash
python -m venv .venv && .venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
```

Training and evaluation run on a CPU too, but a laptop CPU needs about 1–2 days in total.

```bash
python train.py --seed 0 --steps 60000 --curriculum 0.5      # repeat for seeds 1 and 2
python evaluate.py --seed 0 --baseline-only                  # sanity check: solves 128, fails longer
python evaluate.py --seed 0                                  # all conditions + manipulation checks
```

Each figure and the decision come from the saved results in `results/` (seconds, no GPU):

```bash
python report.py fig1       # baseline accuracy vs length, by distractor placement and distance
python report.py fig2       # gate drift per layer (plus value branch, residual, attention entropy)
python report.py fig3       # accuracy vs length for every condition, seed-level CIs
python report.py fig4       # manipulation checks
python report.py fig5       # interventions by distractor placement
python report.py fig6       # exploratory: relevant record inside vs beyond the trained range
python report.py decision   # writes results/decision.md with the verdict
```

On the GPU, training took about 12 minutes for the three seeds together, and the full evaluation about 2 minutes
per seed. `logs/` has the exact shell scripts and logs from that run.

```bash
python report.py            # everything
```
