# Paper build

IEEE conference format (`IEEEtran`). Every number in the Results section comes
from the committed run artifacts under `results/`; every figure is regenerated
from those same files. Nothing is hand-entered.

## Figures

```bash
python paper/make_figures.py            # PDF (for LaTeX) + PNG (for slides)
python paper/make_figures.py --formats pdf
```

A figure whose source study has not finished is **skipped with a warning**
rather than drawn from partial data, so the paper can never quietly contain a
plot averaged over an uneven number of seeds. Re-run after any sweep extends.

| Figure | Source |
|---|---|
| `fig1_dynamics` | `results/ember_pilot_v6/result.json` |
| `fig2_policies` | `results/ember_multiseed/multiseed_raw.csv` |
| `fig3_marginals` | `results/ember_multiseed/multiseed_raw.csv` |
| `fig4_capacity` | `results/ember_cap800/` + `results/ember_multiseed/` |
| `fig5_frontier` | main sweep + `ember_minimax/` + `ember_baseline_frozen/` |
| `fig6_mode` | `results/ember_mode_axis/` + `results/ember_multiseed/` |

## Statistics

```bash
python experiments/analyze_results.py --raw results/ember_multiseed/multiseed_raw.csv --out docs/results.md
python experiments/anova.py --out docs/anova.md      # two-way ANOVA + Kruskal-Wallis + Dunn
```

## Compiling

`pdflatex` is not installed in this environment, so the repository ships
`main.tex` plus the figures rather than a built PDF.

```bash
pdflatex main && pdflatex main          # from paper/, twice for references
```

Or upload the `paper/` directory to Overleaf and select the IEEEtran class.
