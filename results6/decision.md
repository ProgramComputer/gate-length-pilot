# Pilot 6 (real model, redesigned): pre-registered decision

## Tuning: block size k, on 2048-token inputs only (seed 60000)

| k | accuracy (512 words) | full vocab |
|---|---|---|
| 16 | 81.0 | 36.0 |
| 8 | 85.0 | 48.0 |
| 4 | 93.0 | 60.0 |

Chosen k: 16

## Feasibility (baseline accuracy over the 512 value words)

| placement | 2048 | 4096 | 8192 |
|---|---|---|---|
| before | 84.5 | 51.0 | 6.5 |
| uniform | 88.0 | 68.0 | 15.5 |
| after | 90.0 | 71.0 | 2.0 |

F (≥ 70% at 2048 and ≥ 20 pp lower at 8192, 'before'): yes

## Verdict: **GATE PASSED; MAIN TEST NOT YET RUN (awaiting approval)**
