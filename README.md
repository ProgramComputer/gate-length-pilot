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
