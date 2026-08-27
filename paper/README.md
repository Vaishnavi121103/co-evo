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

## Word (.docx)

```bash
python paper/make_docx.py                  # two-column, IEEE page setup
python paper/make_docx.py --single-column  # easier to comment on
```

Uses pandoc (`pip install pypandoc_binary`) with the official IEEE
`conference-template-a4.docx` as the reference document, so A4 geometry, fonts
and the IEEE paragraph styles come from IEEE rather than from us. The script
then handles what pandoc cannot:

- **Remaps style names.** Pandoc emits its own identifiers (`Title`,
  `ImageCaption`, ...) which are not the ones the template defines
  (`papertitle`, `figurecaption`, ...). Unmapped styles fall back to Normal and
  the document silently looks nothing like the template.
- **Rebuilds the sectioning.** Pandoc copies one section layout from the
  reference document and picks the template's final single-column one. IEEE
  layout is a single-column title and abstract followed by a two-column body,
  which is two sections separated by a continuous break.
- **Rewrites figure paths** to the PNG twins, since Word cannot render the
  vector PDFs the LaTeX build uses, and **expands the custom metric macros**
  pandoc would otherwise drop silently.

**The `.tex` is the source of record.** Regenerate the `.docx` after any edit
rather than editing it directly, or the two will drift apart.

### JavaScript route

```bash
cd paper && npm install          # docx + jszip, first time only
node paper/make_docx.js          # -> main-js.docx
node paper/make_docx.js --single-column
```

A LaTeX to OOXML converter written directly against the document model, with
no pandoc dependency. It lifts `styles.xml` out of the IEEE template and passes
it as `externalStyles`, so paragraphs reference IEEE's own style ids and
inherit their real formatting; floats are numbered in source order so `ef`
resolves to `Fig. 4` / `Table II` the way LaTeX would.

**Choose the Python route if the equations matter.** Pandoc emits native Word
equation objects (OMML) that stay editable in Word; the JavaScript route
renders maths as Unicode text. For this paper the maths is superscripts,
subscripts, Greek letters and inequalities, all of which Unicode represents
faithfully, but the distinction matters if anyone needs to edit a formula.

| | Python (pandoc) | JavaScript |
|---|---|---|
| IEEE template styles | yes | yes |
| Equations | native, editable | Unicode text |
| Dependency | pandoc binary (~180 MB) | two npm packages |
| Output | `main.docx` | `main-js.docx` |

## Compiling

`pdflatex` is not installed in this environment, so the repository ships
`main.tex` plus the figures rather than a built PDF.

```bash
pdflatex main && pdflatex main          # from paper/, twice for references
```

Or upload the `paper/` directory to Overleaf and select the IEEEtran class.
