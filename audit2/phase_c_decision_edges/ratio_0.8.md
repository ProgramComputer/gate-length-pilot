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
| before | 85.0 | 50.0 | 5.0 |
| uniform | 85.0 | 50.0 | 5.0 |
| after | 85.0 | 50.0 | 5.0 |

F (≥ 70% at 2048 and ≥ 20 pp lower at 8192, 'before'): yes

## Binding heads (1): attention to previous token 0.600 inside → 0.100 far; margin 10.00 → 8.00 (ratio 0.80)

| condition | before 2048 | before 4096 | before 8192 |
|---|---|---|---|
| none | 50.0 | 50.0 | 50.0 |
| patch | 80.0 | 80.0 | 80.0 |
| sham | 0.0 | 0.0 | 0.0 |

D1: yes; D1b: PREFERENCE KEPT (dilution); D2: yes

## Verdict: **TOY MECHANISM HOLDS IN THE REAL MODEL; PREFERENCE KEPT (dilution)**
