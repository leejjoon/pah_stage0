#!/bin/bash
# Ingest an active-learning DFT batch (docs/plan/hangang_al1_handoff.md, "Hand back") on the project node and build the
# next dataset (docs/plan/charged_mlff_plan.md, Phase 3):
#   [BASE=a.extxyz,b.extxyz,...] d1/mlff_al_ingest.sh <al dft.extxyz> <gjf_<tag>/manifest.json> <out dataset dir> [mlff_qa options]
# 1. mlff_qa.py on the batch (xTB env, GFN2-xTB per frame, $PROCS processes on cores $CORES, default 64 on 80-143; flags as for the seed set) -> <out>_work/al_qa.extxyz
# 2. the batch's min frames that repeat a minimum already in the base data (same source and charge; AL round 1: they
#    are only there as mlff_qa.py's per-run energy reference) are removed; minima of new species (round 2) are kept.
#    The repeats' centred positions differ from the base ones by up to 3.5e-5 A (centring masses), beyond
#    mlff_dataset.py's 1e-6 A duplicate check. Their DFT energies are compared with the base minima instead
#    (a Hangang-vs-project-node check; expected to agree to ~1e-5 eV, the geometry difference).
# 3. mlff_dataset.py with the base inputs (default: dataset v2's, in v2's order) + the AL batch without the repeats;
#    species tables $SPECIES (default v2's; a round with new species appends its table), --exclude $EXCLUDE (default
#    none) --drop-flags $DROP (default unstable), as v2. Held-out files are compared with $REF (default dataset v2).
# 4. prints the training and evaluation commands (not run: the user decides when the GPUs are used).
# A --manifest 'none' passes through to mlff_qa.py (skips the missing-job check, for dry runs on a few frames).
set -euo pipefail
AL=$(realpath "${1:?al dft.extxyz}"); MAN=${2:?manifest.json or none}; OUT=${3:?out dataset dir}; shift 3
D=/var/tmp/jjlee-pah/d1; REPO=$(cd "$(dirname "$0")/.." && pwd)
BASE=${BASE:-$D/dataset_v2_work/seed_qa_v2.extxyz,$D/mlff_seed/transfer2/transfer2_qa.extxyz,$D/dataset_v2_work/hot_qa_v2.extxyz,$D/dataset_v2_work/t2md_qa_v2.extxyz}
SPECIES=${SPECIES:-$D/mlff_seed/species_v2.csv,$D/mlff_seed/transfer2/species_transfer2.csv}   # round 2: + species_r2.csv
EXCLUDE=${EXCLUDE:-/dev/null}   # e.g. d1/exclusions_nonkekule_q0.csv (per-charge exclusions)
REF=${REF:-$D/dataset_v2}       # dataset whose held-out files must stay the same (expected to differ when EXCLUDE or SPECIES touch held-out pairs)
DROP=${DROP:-unstable}          # --drop-flags (v4 on: unstable,short_ch)
CORES=${CORES:-80-143}; PROCS=${PROCS:-64}   # QA cores (taskset list) and processes, e.g. to stay off cores busy with MD
W=${OUT%/}_work; mkdir -p "$W"
XPY=/var/tmp/jjlee-pah/mamba/xtb/bin/python
echo "== 1. QA ($(grep -c Properties "$AL") frames)"
OMP_NUM_THREADS=1 taskset -c "$CORES" $XPY -W ignore "$REPO/d1/mlff_qa.py" "$AL" "$W/al_qa" --manifest "$MAN" --procs "$PROCS" "$@" | tail -25
echo "== 2. drop the batch's minima that repeat base minima, compare their energies"
$XPY - "$W/al_qa.extxyz" "$W/al_qa_nomin.extxyz" "$BASE" <<'PY'
import re, sys
KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
def frames(p):
    with open(p) as f:
        while (l := f.readline()).strip():
            n = int(l); h = f.readline(); rows = [f.readline() for _ in range(n)]
            yield {k: v.strip('"') for k, v in KV.findall(h)}, l + h + "".join(rows)
base = {}
for p in sys.argv[3].split(","):
    for i, _ in frames(p):
        if i.get("config_type") == "min":
            base.setdefault((i["source"], i["charge"]), float(i["dft_energy"]))
n = 0; dE = []; new_min = 0
with open(sys.argv[2], "w") as out:
    for i, txt in frames(sys.argv[1]):
        if i.get("config_type") == "min":
            k = (i["source"], i["charge"])
            if k in base:
                dE.append((abs(float(i["dft_energy"]) - base[k]), i["source"]))
                continue
            new_min += 1
        out.write(txt); n += 1
dE.sort(reverse=True)
print(f"{n} frames kept ({new_min} minima of new species); {len(dE)} repeated minima dropped"
      + (f"; max |E - E_base| = {dE[0][0]:.2e} eV ({dE[0][1]})" if dE else ""))
PY
echo "== 3. dataset"
$XPY "$REPO/d1/mlff_dataset.py" "$BASE,$W/al_qa_nomin.extxyz" "$SPECIES" "$OUT" --exclude "$EXCLUDE" --drop-flags "$DROP" | tail -5
sed -n '/after each step/,/^## Frames dropped/p' "$OUT/report.md"
for f in validation final_test transfer_test; do for q in 0 1; do
  cmp -s "$OUT/${f}_q$q.extxyz" "$REF/${f}_q$q.extxyz" || echo "WARNING: ${f}_q$q differs from $(basename "$REF")"
done; done
echo "== 4. next (run when the GPUs are free):"
echo "GPU=1 CONFIG=d1/mace_torch_v2.yaml nohup d1/mace_torch_baseline.sh $OUT $D/mace_torch/$(basename "$OUT" | sed 's/^dataset_//') > $D/mace_torch/$(basename "$OUT" | sed 's/^dataset_//').log 2>&1 &"
echo "then: d1/mlff_eval.py $OUT <out.json> --models v2=$D/mace_torch/v2/baseline.model,<new>=<run>/baseline.model --files $AL"
