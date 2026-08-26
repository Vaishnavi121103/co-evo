# Retraining-policy comparison (3 seeds)

Source: `results\ember_multiseed\multiseed_raw.csv`  ·  27 runs (9 policies x 3 seeds)


## Per-policy results (mean ± std over seeds)

| Cadence | Data selection | Settled evasion ↓ | Attack success ↓ | Oscillation ↓ | Retrains | Cost (s) |
|---|---|---|---|---|---|---|
| every_n | bounded_buffer | 0.294 ± 0.032 | 0.274 ± 0.034 | 0.013 ± 0.012 | 3.0 | 48.0 ± 9.6 |
| every_n | full_replay | 0.050 ± 0.005 | 0.050 ± 0.005 | 0.015 ± 0.005 | 3.0 | 53.9 ± 11.4 |
| every_n | hard_mining | 0.350 ± 0.003 | 0.324 ± 0.004 | 0.071 ± 0.031 | 3.0 | 49.7 ± 16.3 |
| every_round | bounded_buffer | 0.293 ± 0.004 | 0.276 ± 0.004 | 0.026 ± 0.015 | 8.0 | 140.8 ± 52.1 |
| every_round | full_replay | 0.041 ± 0.002 | 0.041 ± 0.002 | 0.010 ± 0.006 | 8.0 | 147.8 ± 45.1 |
| every_round | hard_mining | 0.351 ± 0.066 | 0.331 ± 0.066 | 0.033 ± 0.006 | 8.0 | 139.3 ± 45.5 |
| threshold | bounded_buffer | 0.293 ± 0.004 | 0.276 ± 0.004 | 0.026 ± 0.015 | 8.0 | 134.7 ± 45.1 |
| threshold | full_replay | 0.076 ± 0.006 | 0.076 ± 0.006 | 0.008 ± 0.004 | 3.0 | 55.0 ± 20.5 |
| threshold | hard_mining | 0.351 ± 0.066 | 0.331 ± 0.066 | 0.033 ± 0.006 | 8.0 | 135.0 ± 46.8 |

## Marginal effect of each policy axis

Averaging over the other axis isolates one variable at a time -- the point of running a factorial design rather than one-off comparisons.


**cadence**

| Level | Settled evasion ↓ | Oscillation ↓ | Cost (s) |
|---|---|---|---|
| every_n | 0.231 ± 0.139 | 0.033 ± 0.033 | 50.5 ± 11.3 |
| every_round | 0.229 ± 0.147 | 0.023 ± 0.013 | 142.6 ± 41.5 |
| threshold | 0.240 ± 0.130 | 0.022 ± 0.014 | 108.2 ± 52.5 |

**data_selection**

| Level | Settled evasion ↓ | Oscillation ↓ | Cost (s) |
|---|---|---|---|
| bounded_buffer | 0.294 ± 0.016 | 0.021 ± 0.013 | 107.8 ± 56.9 |
| full_replay | 0.055 ± 0.016 | 0.011 ± 0.005 | 85.6 ± 53.1 |
| hard_mining | 0.351 ± 0.047 | 0.046 ± 0.025 | 108.0 ± 55.2 |

## Significance of the headline gap

- Most robust: **every_round / full_replay** (settled evasion 0.041)
- Least robust: **threshold / hard_mining** (settled evasion 0.351)
- Welch t-test: t = -8.165, p = 0.01457 (significant at α=0.05)

## Robustness-per-cost

| Cadence | Data selection | Settled evasion | Cost (s) | Evasion avoided per retrain-second |
|---|---|---|---|---|
| every_round | full_replay | 0.041 | 147.8 | 0.0065 |
| every_n | full_replay | 0.050 | 53.9 | 0.0176 |
| threshold | full_replay | 0.076 | 55.0 | 0.0168 |
| every_round | bounded_buffer | 0.293 | 140.8 | 0.0050 |
| threshold | bounded_buffer | 0.293 | 134.7 | 0.0052 |
| every_n | bounded_buffer | 0.294 | 48.0 | 0.0147 |
| every_n | hard_mining | 0.350 | 49.7 | 0.0131 |
| every_round | hard_mining | 0.351 | 139.3 | 0.0047 |
| threshold | hard_mining | 0.351 | 135.0 | 0.0048 |