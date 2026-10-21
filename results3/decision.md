# Pilot 3: pre-registered decisions

## A. RoPE Profiler (Wu, Du & Peng 2026) on the RoPE models

| seed | heads (layer.head, token, attention) | semantic score 128 → 1024 | true reversal rate 128 → 1024 | positional score 128 → 1024 | worse failure |
|---|---|---|---|---|---|
| 0 | 2.1 value 0.98; 2.3 value 0.98 | 1.000 → 0.854 | 0.000 → 0.178 | 0.0168 → 0.0216 | semantic |
| 1 | 2.2 value 0.97; 2.0 value 0.97 | 1.000 → 0.887 | 0.000 → 0.132 | 0.0146 → 0.0160 | semantic |
| 2 | 2.2 value 0.98; 2.1 value 0.98 | 1.000 → 0.911 | 0.000 → 0.101 | 0.0140 → 0.0150 | semantic |

| condition | after @1024 (per seed) | gain over baseline, pp (mean ± seed CI) | per-seed [bootstrap CI] | uniform @128 | before @1024 |
|---|---|---|---|---|---|
| baseline | 3.2, 1.5, 0.9 | +0.0 ± 0.0 | +0.0 [+0.0, +0.0]; +0.0 [+0.0, +0.0]; +0.0 [+0.0, +0.0] | 100.0 | 51.7 |
| sem_a0.5 | 3.3, 1.6, 1.0 | +0.1 ± 0.0 | +0.1 [-0.3, +0.6]; +0.1 [-0.2, +0.4]; +0.1 [+0.0, +0.3] | 100.0 | 51.6 |
| sem_a0.25 | 3.4, 1.6, 1.0 | +0.1 ± 0.1 | +0.2 [-0.3, +0.7]; +0.1 [-0.2, +0.4]; +0.1 [+0.0, +0.3] | 100.0 | 51.7 |
| sem_a0 | 3.4, 1.6, 1.0 | +0.1 ± 0.1 | +0.2 [-0.4, +0.7]; +0.1 [-0.2, +0.4]; +0.1 [+0.0, +0.3] | 100.0 | 51.8 |
| sem_freeze | 3.1, 1.8, 1.0 | +0.1 ± 0.5 | -0.1 [-0.8, +0.6]; +0.3 [-0.2, +0.8]; +0.1 [+0.0, +0.3] | 100.0 | 52.0 |
| pos_a2 | 2.8, 1.6, 1.1 | -0.0 ± 0.8 | -0.4 [-1.0, +0.1]; +0.1 [-0.3, +0.5]; +0.2 [-0.2, +0.6] | 100.0 | 51.5 |
| pos_a4 | 2.8, 2.2, 1.4 | +0.3 ± 1.5 | -0.4 [-1.3, +0.4]; +0.7 [+0.0, +1.5]; +0.5 [-0.1, +1.2] | 100.0 | 50.8 |
| PI | 5.8, 4.5, 3.8 | +2.8 ± 0.5 | +2.6 [+0.8, +4.4]; +3.0 [+1.5, +4.4]; +2.9 [+1.5, +4.3] | 100.0 | 3.1 |

Diagnosed direction (majority of seeds): **semantic** (['semantic', 'semantic', 'semantic'])
A1 (true reversal rate at 1024 ≥ 2× at 128, every seed): yes
A2 (best semantic-direction condition, sem_a0.25, gains ≥ 10 pp with every seed's CI > 0): no
A3 (sem_a0.25 beats the best opposite-direction condition, pos_a4, by ≥ 5 pp): no

### Verdict A: **DETECTS BUT DOES NOT FIX**

## B. Han & Ji three-axis fit, a(i−j) + b(i) + c(j)

| model | seed | positional head | fit corr inside | fit corr far | rho_a (shape inside vs far) | transfer corr | mean fit corr inside, all heads |
|---|---|---|---|---|---|---|---|
| NoPE | 0 | 3.3 | 0.699 | 0.915 | 0.862 | 0.878 | 0.832 |
| NoPE | 1 | 2.0 | 0.818 | 0.990 | 0.871 | 0.794 | 0.849 |
| NoPE | 2 | 1.2 | 0.984 | 0.991 | 0.999 | 0.986 | 0.760 |
| RoPE | 0 | 1.2 | 0.769 | 0.793 | 0.972 | 0.779 | 0.451 |
| RoPE | 1 | 1.1 | 0.765 | 0.759 | 0.987 | 0.752 | 0.508 |
| RoPE | 2 | 1.2 | 0.743 | 0.749 | 0.981 | 0.742 | 0.451 |

B1 (NoPE positional profile changes far away: rho_a ≤ 0.5, every seed): no
B2 (RoPE positional profile keeps its shape: rho_a ≥ 0.8, every seed): yes

### Verdict B: **NO BREAK IN NOPE**
