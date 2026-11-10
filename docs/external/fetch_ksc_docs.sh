#!/bin/bash
# Mirror KISTI's GitBook guides as Markdown, using GitBook's llms.txt index and per-page .md export.
# Usage: docs/external/fetch_ksc_docs.sh   (run from anywhere; writes next to this script)
# Each guide gets llms.txt (index), llms-full.txt (whole guide in one file) and pages/<path>.md.
set -euo pipefail
cd "$(dirname "$0")"
mirror() {
  local space=$1 B=https://docs-ksc.gitbook.io/$1 fail=0
  rm -rf "$space/pages"; mkdir -p "$space/pages"
  curl -sfL "$B/llms.txt" -o "$space/llms.txt"
  curl -sfL "$B/llms-full.txt" -o "$space/llms-full.txt"
  grep -oE "\($B/[^)]+\.md\)" "$space/llms.txt" | tr -d '()' | sort -u > "$space/.urls"
  while read -r u; do
    rel=${u#"$B"/}; mkdir -p "$space/pages/$(dirname "$rel")"
    curl -sfL "$u" -o "$space/pages/$rel" || { echo "FAIL $u"; fail=$((fail + 1)); }
  done < "$space/.urls"
  echo "$space: $(wc -l < "$space/.urls") pages listed, $fail failed"
}
mirror hangang-user-guide      # https://docs-ksc.gitbook.io/hangang is the same space
mirror myksc-hangang
# The myhub (Harbor) instructions the MyKSC guide links to live in the Neuron guide; fetch only that page.
mkdir -p neuron-user-guide/pages/appendix
curl -sfL https://docs-ksc.gitbook.io/neuron-user-guide/appendix/appendix-12-how-to-use-containers.md \
  -o neuron-user-guide/pages/appendix/appendix-12-how-to-use-containers.md
date -u +"fetched %Y-%m-%dT%H:%MZ" > FETCHED
