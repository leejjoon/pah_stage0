# Newcomer report: goal, architecture, containers

A report for a junior physics graduate student: the project goal, the T0/T0b architecture, and every container
package by package. One source, three outputs.

| Output | Command | Figures |
| --- | --- | --- |
| `build/report.html` (one self-contained file) | `docs/report/build.sh` | interactive widgets + static SVGs |
| `build/report.epub` | same | static SVGs |
| `build/report.pdf` | `docs/report/build.sh pdf` (headless Chromium prints the HTML) | static SVGs |

`build/` is not committed; run the script to regenerate it. The tools come from `requirements-report.txt`
through `uv` (pandoc 3.9 ships inside `pypandoc_binary`), so no system packages are needed.

## The intermediate representation

- **`report.md`** is the source: Pandoc Markdown with fenced divs for figures, pipe tables and TeX math.
  Edit the text here. Every figure is a div like this:
  `::: {.figure #fig-x data-widget="spectra"}` followed by `![caption](figures/x.svg)` and a closing `:::`.
  The `data-widget` attribute is optional. EPUB and PDF ignore it and show the image; the HTML build attaches
  the interactive version from `web/widgets.js`. Any converter that reads Pandoc Markdown (pandoc to docx,
  LaTeX or Typst, for example) gets the static figures.
- **`figures/*.svg`** are generated. Don't edit them by hand.
  - `scripts/make_diagrams.py` writes the explanatory diagrams as plain SVG.
  - `scripts/make_figures.py` writes the data plots with matplotlib, plus `web/report-data.js` for the widgets.
- **`generated/appendix.md`** lists every installed package. `scripts/appendix.py` writes it.
- **`data/`** holds the inputs, extracted once from the containers:
  - `pahdb.json` comes from `scripts/extract_pahdb.py`, run in container C;
  - `pipshow_*.txt` are `pip show` dumps from each image (the command is in `scripts/deps.py`);
  - `deps.json` is derived from those dumps.

  Rebuilding the report does not need the containers. The MD velocity figure reads the smoke-run trajectories
  in `$MD_DIR` (default `/var/tmp/jjlee-pah/md`). If they are absent, the existing SVG is kept.
- **`web/`** holds the HTML-only layer: `report.css` (light and dark tokens) and `widgets.js` (vanilla JS, no
  external libraries).

The timing figure and the widget read `reports/t0_timing_{calib,full}.csv`. The static figure uses the full
grid once it has every configuration. The numbers quoted in the prose of §4.4 come from
`reports/t0_timing_summary.md` (written by `t0/summarize.py`) and are updated by hand.
