#!/bin/bash
# Build the report from its intermediate representation (report.md + figures/*.svg + generated/appendix.md).
#   docs/report/build.sh            # HTML (self-contained, interactive) and EPUB
#   docs/report/build.sh pdf        # also a PDF, printed from the HTML by headless Chromium (static figures)
# Tools come from docs/report/requirements-report.txt through uv (no system packages needed):
# pandoc 3.9 (pypandoc_binary), matplotlib, numpy. The data in data/ were extracted from the containers once
# (scripts/extract_pahdb.py in C, pip show dumps; see scripts/deps.py); rebuilding does not need the containers.
set -euo pipefail
cd "$(dirname "$0")"
UV=${UV:-$(command -v uv || echo ~/.local/bin/uv)}
RUN=("$UV" run --no-project --with-requirements requirements-report.txt python)
OUT=${OUT:-build}
mkdir -p "$OUT"

"${RUN[@]}" scripts/deps.py
"${RUN[@]}" scripts/make_figures.py
"${RUN[@]}" scripts/make_diagrams.py
"${RUN[@]}" scripts/appendix.py
PANDOC=$("${RUN[@]}" -c "import pypandoc; print(pypandoc.get_pandoc_path())")
"$PANDOC" --version | head -1

SRC=(report.md generated/appendix.md)
COMMON=(--from markdown+fenced_divs+pipe_tables+tex_math_dollars --resource-path=. --number-sections=false)

# HTML: one self-contained file (figures, CSS, data and scripts embedded)
"$PANDOC" "${SRC[@]}" "${COMMON[@]}" --standalone --embed-resources --toc --toc-depth=2 --mathml \
  --css web/report.css --include-after-body=<(printf '<script>%s</script>\n<script>%s</script>\n' "$(cat web/report-data.js)" "$(cat web/widgets.js)") \
  --metadata pagetitle="PAH populations from IR bands" -o "$OUT/report.html"

# The same page for the claude.ai Artifact viewer (no document wrapper of its own)
"${RUN[@]}" scripts/to_artifact.py "$OUT/report.html" "$OUT/report-artifact.html"

# EPUB 3: static figures, MathML
"$PANDOC" "${SRC[@]}" "${COMMON[@]}" --toc --toc-depth=2 --mathml -t epub3 -o "$OUT/report.epub"

if [ "${1:-}" = pdf ]; then
  CHROME=${CHROME:-$(command -v chromium || command -v google-chrome || echo /snap/bin/chromium)}
  # snap Chromium cannot write to /tmp; print beside the HTML
  "$CHROME" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/$OUT/report.pdf" "file://$PWD/$OUT/report.html" 2>/dev/null
fi
ls -la "$OUT"
