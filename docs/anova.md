# Two-way ANOVA — mean_evasion_tail

Balanced 3×3 factorial, n=5 seeds per cell (45 runs). Source: `C:\Users\Vaishnavi\Desktop\rp\Vaishnavi\results\ember_multiseed\multiseed_raw.csv`.


## Cell means

| data_selection | every_n | every_round | threshold |
|---|---|---|---|
| bounded_buffer | 0.3002 | 0.3110 | 0.3110 |
| full_replay | 0.0494 | 0.0421 | 0.0747 |
| hard_mining | 0.3683 | 0.3761 | 0.3761 |

## ANOVA table

| Source | SS | df | MS | F | p | partial η² | ω² | |
|---|---|---|---|---|---|---|---|---|
| cadence | 0.00173 | 2 | 0.00086 | 0.77 | 0.472 | 0.041 (small) | -0.001 | ns |
| data_selection | 0.84535 | 2 | 0.42267 | 375.49 | 7.69e-25 | 0.954 (large) | 0.947 | *** |
| cadence x data_selection | 0.00179 | 4 | 0.00045 | 0.40 | 0.81 | 0.042 (small) | -0.003 | ns |
| Residual | 0.04052 | 36 | 0.00113 | | | | | |
| Total | 0.88938 | 44 | | | | | | |

`***` p<.001 · `**` p<.01 · `*` p<.05 · `ns` not significant. Effect-size bands are Cohen's conventions for η² (.01 small, .06 medium, .14 large).

## Interpretation

- The interaction is not significant (p = 0.81), so the two main effects can be read independently.
- **cadence**: F(2,36) = 0.77, p = 0.472, partial η² = 0.041 (small), ω² = -0.001.
- **data_selection**: F(2,36) = 375.49, p = 7.69e-25, partial η² = 0.954 (large), ω² = 0.947.

## Simple main effects — cadence within each data-selection level

| Data selection | F | p | | Spread across cadences |
|---|---|---|---|---|
| bounded_buffer | 0.17 | 0.841 | ns | 0.0108 |
| full_replay | 1.30 | 0.286 | ns | 0.0326 |
| hard_mining | 0.09 | 0.915 | ns | 0.0077 |

## Tukey HSD (95% family-wise)


**data_selection**

| A | B | mean diff | 95% CI | p | |
|---|---|---|---|---|---|
| bounded_buffer | full_replay | +0.2520 | [+0.2233, +0.2807] | 0 | *** |
| bounded_buffer | hard_mining | -0.0661 | [-0.0948, -0.0374] | 4.53e-06 | *** |
| full_replay | hard_mining | -0.3181 | [-0.3468, -0.2894] | 0 | *** |

**cadence**

| A | B | mean diff | 95% CI | p | |
|---|---|---|---|---|---|
| every_n | every_round | -0.0037 | [-0.1327, +0.1252] | 0.997 | ns |
| every_n | threshold | -0.0146 | [-0.1436, +0.1144] | 0.959 | ns |
| every_round | threshold | -0.0109 | [-0.1398, +0.1181] | 0.977 | ns |

## Assumption checks

- **Levene** (equal variances across cells): W = 1.334, p = 0.259 → not violated. Largest/smallest cell variance ratio 467.7.
- **Shapiro-Wilk** (normality of residuals): W = 0.912, p = 0.00228 → **violated**.

The two robustness checks below address **different** assumptions, so the one to quote is whichever matches the assumption that actually failed.

### Kruskal-Wallis — addresses non-normality

Rank-based and distribution-free, so it speaks directly to the Shapiro-Wilk result above. This is the robustness check to quote for this data.

| Factor | H | df | p | ε² | η²_H | |
|---|---|---|---|---|---|---|
| data_selection | 35.154 | 2 | 2.32e-08 | 0.799 | 0.789 | *** |
| cadence | 0.469 | 2 | 0.791 | 0.011 | -0.036 | ns |

### Dunn's test — rank-based pairwise post-hoc


**data_selection**

| A | B | mean rank A | mean rank B | z | p (raw) | p (Holm) | p (Bonf.) | |
|---|---|---|---|---|---|---|---|---|
| bounded_buffer | full_replay | 24.7 | 8.0 | +3.490 | 0.000482 | 0.000965 | 0.00145 | *** |
| bounded_buffer | hard_mining | 24.7 | 36.3 | -2.406 | 0.0161 | 0.0161 | 0.0484 | * |
| full_replay | hard_mining | 8.0 | 36.3 | -5.896 | 3.73e-09 | 1.12e-08 | 1.12e-08 | *** |

**cadence**

| A | B | mean rank A | mean rank B | z | p (raw) | p (Holm) | p (Bonf.) | |
|---|---|---|---|---|---|---|---|---|
| every_n | every_round | 22.8 | 21.5 | +0.278 | 0.781 | 1 | 1 | ns |
| every_n | threshold | 22.8 | 24.7 | -0.403 | 0.687 | 1 | 1 | ns |
| every_round | threshold | 21.5 | 24.7 | -0.681 | 0.496 | 1 | 1 | ns |

Holm is the primary correction — uniformly more powerful than Bonferroni at the same family-wise error rate — with Bonferroni shown alongside for reference.

### Welch one-way — addresses unequal variances

Reported for completeness only. Welch relaxes the equal-variance assumption, which Levene did **not** flag here, so it is not the relevant fallback for this data: it does nothing about non-normality.

- cadence: Welch F = 0.04, p = 0.958 ns
- data_selection: Welch F = 712.82, p = 3.79e-22 ***

### Verdict

- **data_selection**: ANOVA p = 7.69e-25, Kruskal-Wallis p = 2.32e-08 — the two agree.
- **cadence**: ANOVA p = 0.472, Kruskal-Wallis p = 0.791 — the two agree.

Both factors reach the same verdict under the rank-based test as under the F test, so the non-normal residuals do not change the conclusion. The ANOVA stays the primary analysis — it is what supplies the interaction term and the variance decomposition — with Kruskal-Wallis reported as the assumption-free confirmation.

## Caveat specific to this design

Under `bounded_buffer` and `hard_mining` the `threshold` and `every_round` cells are near-identical, because evasion never falls below the retraining trigger and the threshold policy fires every round — it *is* `every_round` in those conditions. That degeneracy is a genuine finding rather than a nuisance, but it inflates the interaction term, and it means the cadence factor has fewer effectively distinct levels than the design nominally provides.