# Pilot 6 (real model, redesigned): pre-registered decision

## Tuning: block size k, on 2048-token inputs only (seed 60000)

| k | accuracy (512 words) | full vocab |
|---|---|---|
| 16 | 81.0 | 30.0 |
| 8 | 81.0 | 30.0 |
| 4 | 81.0 | 30.0 |

Chosen k: 16

## Feasibility (baseline accuracy over the 512 value words)

| placement | 2048 | 4096 | 8192 |
|---|---|---|---|
| before | 70.0 | 50.0 | 5.0 |
| uniform | 70.0 | 50.0 | 5.0 |
| after | 70.0 | 50.0 | 5.0 |

F (≥ 70% at 2048 and ≥ 20 pp lower at 8192, 'before'): NO

## Verdict: **NOT TESTABLE (the task is not solved inside the trained length)**
