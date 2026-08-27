# Is fine-tuning less stable, or just worse?

Both arms over seeds [0, 1, 2], 9 policy cells each. Tests run on **cell-centred residuals**, so the arms' very different means cannot leak into the dispersion estimate; what is left is seed-to-seed variation within an otherwise identical configuration.


## Dispersion

| Arm | Mean | SD of residuals | Variance | CV | Range |
|---|---|---|---|---|---|
| refit from scratch | 0.2333 | 0.0274 | 0.000753 | 0.118 | 0.0383–0.4167 |
| fine-tune | 0.7239 | 0.2135 | 0.045600 | 0.295 | 0.2521–0.9992 |

## Tests of equal variance

| Test | Statistic | p | Verdict |
|---|---|---|---|
| Levene (Brown-Forsythe, median-centred) | W = 20.472 | 3.54e-05 | variances differ |
| Bartlett (assumes normality) | K² = 70.168 | 5.45e-17 | variances differ |
| F-test of variances | F(18,18) = 60.60 | 3.39e-12 | variances differ |

Fine-tuning's run-to-run variance is **61x** that of refitting (95% CI 23–157x). Levene is the test to quote: the residuals are not normal, and Bartlett is sensitive to that.

## Why this matters for the claim

The two arms differ in mean *and* in dispersion, and the second is a separate finding. A policy whose settled evasion lands anywhere between 0.252 and 0.999 depending only on the seed is not simply a worse policy with a known cost -- it is unpredictable, and a defender cannot budget against it. Reporting a mean alone would hide exactly that. Refitting from scratch, by contrast, lands within 0.038–0.417 across the same seeds.