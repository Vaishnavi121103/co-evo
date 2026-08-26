# Retraining-policy comparison (5 seeds)

Source: `results\ember_multiseed\multiseed_raw.csv`  ·  45 runs (9 policies x 5 seeds)


## Per-policy results (mean ± std over seeds)

| Cadence | Data selection | Settled evasion ↓ | Attack success ↓ | Oscillation ↓ | Retrains | Cost (s) |
|---|---|---|---|---|---|---|
| every_n | bounded_buffer | 0.300 ± 0.033 | 0.281 ± 0.035 | 0.015 ± 0.010 | 3.0 | 45.7 ± 7.5 |
| every_n | full_replay | 0.049 ± 0.004 | 0.049 ± 0.004 | 0.014 ± 0.004 | 3.0 | 51.7 ± 8.6 |
| every_n | hard_mining | 0.368 ± 0.034 | 0.342 ± 0.034 | 0.056 ± 0.030 | 3.0 | 47.1 ± 12.1 |
| every_round | bounded_buffer | 0.311 ± 0.024 | 0.294 ± 0.024 | 0.025 ± 0.010 | 8.0 | 151.2 ± 40.9 |
| every_round | full_replay | 0.042 ± 0.003 | 0.042 ± 0.003 | 0.011 ± 0.005 | 8.0 | 137.5 ± 34.9 |
| every_round | hard_mining | 0.376 ± 0.058 | 0.356 ± 0.058 | 0.041 ± 0.012 | 8.0 | 128.6 ± 35.4 |
| threshold | bounded_buffer | 0.311 ± 0.024 | 0.294 ± 0.024 | 0.025 ± 0.010 | 8.0 | 125.5 ± 34.3 |
| threshold | full_replay | 0.075 ± 0.005 | 0.075 ± 0.005 | 0.008 ± 0.003 | 3.0 | 50.4 ± 15.8 |
| threshold | hard_mining | 0.376 ± 0.058 | 0.356 ± 0.058 | 0.041 ± 0.012 | 8.0 | 125.1 ± 35.8 |

## Marginal effect of each policy axis

Averaging over the other axis isolates one variable at a time -- the point of running a factorial design rather than one-off comparisons.


**cadence**

| Level | Settled evasion ↓ | Oscillation ↓ | Cost (s) |
|---|---|---|---|
| every_n | 0.239 ± 0.144 | 0.028 ± 0.026 | 48.2 ± 9.3 |
| every_round | 0.243 ± 0.153 | 0.025 ± 0.015 | 139.1 ± 35.7 |
| threshold | 0.254 ± 0.138 | 0.024 ± 0.016 | 100.3 ± 45.9 |

**data_selection**

| Level | Settled evasion ↓ | Oscillation ↓ | Cost (s) |
|---|---|---|---|
| bounded_buffer | 0.307 ± 0.026 | 0.021 ± 0.011 | 107.5 ± 54.7 |
| full_replay | 0.055 ± 0.015 | 0.011 ± 0.005 | 79.9 ± 47.1 |
| hard_mining | 0.373 ± 0.047 | 0.046 ± 0.020 | 100.3 ± 47.8 |

## Significance of the headline gap

- Most robust: **every_round / full_replay** (settled evasion 0.042)
- Least robust: **threshold / hard_mining** (settled evasion 0.376)
- Welch t-test: t = -12.930, p = 0.0002011 (significant at α=0.05)

## Robustness-per-cost

| Cadence | Data selection | Settled evasion | Cost (s) | Evasion avoided per retrain-second |
|---|---|---|---|---|
| every_round | full_replay | 0.042 | 137.5 | 0.0070 |
| every_n | full_replay | 0.049 | 51.7 | 0.0184 |
| threshold | full_replay | 0.075 | 50.4 | 0.0184 |
| every_n | bounded_buffer | 0.300 | 45.7 | 0.0153 |
| threshold | bounded_buffer | 0.311 | 125.5 | 0.0055 |
| every_round | bounded_buffer | 0.311 | 151.2 | 0.0046 |
| every_n | hard_mining | 0.368 | 47.1 | 0.0134 |
| every_round | hard_mining | 0.376 | 128.6 | 0.0049 |
| threshold | hard_mining | 0.376 | 125.1 | 0.0050 |