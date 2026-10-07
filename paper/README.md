# Paper

`main.tex` (LaTeX, for submission) and `paper.md` (readable version) describe the
project: the lot-grouped evaluation protocol, the imbalance handling, and the
portable model bundle.

## Regenerate tables and figures

Everything numeric is produced from run outputs, never typed by hand:

```bash
python paper/make_results.py --tag synthetic      # or --tag wm811k after real runs
```

It reads every `outputs/<run>/test_metrics.json` and writes:

- `tables/<tag>.tex` — the table `main.tex` includes
- `tables/<tag>.md`, `tables/<tag>_per_class.md` — for `paper.md`
- `tables/<tag>.csv` — raw numbers
- `figures/training_curves.png`, `figures/class_balance.png`,
  `figures/samples.png`, `figures/confusion_<run>.png`

After real WM-811K runs, re-run it and paste the regenerated Markdown table between
the `<!-- BEGIN:results -->` / `<!-- END:results -->` markers in `paper.md`;
`main.tex` picks up its table automatically via `\input`.

## Build the PDF

No LaTeX toolchain is installed in this repo's dev container, so the PDF is not
committed. Locally:

```bash
cd paper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

Needs `natbib`, `booktabs`, `graphicx`, `subcaption`, `hyperref`, `xcolor` — all in
TeX Live / MiKTeX. For an IEEE submission, replace the `\documentclass` line with
`\documentclass[conference]{IEEEtran}` and drop the `geometry` package, as noted in
the file.

## Honesty note

The committed results come from the **synthetic** pipeline-validation run. They show
the pipeline works end to end; they say nothing about real wafer-map performance. The
abstract, the results section and the threats-to-validity section all say so
explicitly. Replace them with real WM-811K numbers before circulating the paper.
