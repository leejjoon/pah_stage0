#!/bin/bash
# After polar_v7 (coordinator 2026-11-24 steps 2-3): batched gate (c), gate (a) decision -> polar_v7_totmu, G5 (b).
# Copy of the wasp33a pipeline ~/pah/dipole/runs/polar_v7_after.sh (paths are wasp33a's; the PID is that run's).
set -uo pipefail
R=~/pah/dipole/runs; G=~/pah/g5; REPO=~/work/pah_stage0; P=~/pah/mace-venv/bin/python; F=~/pah/dipole/feasibility_v4
while kill -0 1742329 2>/dev/null; do sleep 120; done   # PID of ./polar_v7.sh (pgrep -f would match this script)
M=$R/polar_v7/polar.model
echo "== gate (a) $(date)"; sed -n '/by atom count/,/^$/p' $R/polar_v7_eval.log
echo "== gate (c) batched $(date)"
for n in C24H12_18 C100H26_7494; do for B in 64 256; do
  CUDA_VISIBLE_DEVICES=0 $P -W ignore $REPO/d1/dipole_md_check.py mace-batch $G/md_$n --dipole polar_b$B:PolarMACE:$M --batch $B 2>&1 | grep mace-batch
done
  $P $REPO/d1/dipole_md_check.py report $G/md_$n --dipole-names polar_b64,polar_b256 > $G/md_${n}_polar_report.txt 2>&1
  $P - <<PY
import json; r=json.load(open("$G/md_$n/report.json"))
print("$n", {k: r[k] for k in r if k.startswith("mace_polar")})
PY
done
fail=$($P - <<PY
import json, numpy as np
d = json.load(open("$R/polar_v7.json"))["frames"]; bad = []
for q, lim in (("q0", 0.06), ("q1", 0.10)):
    e = [np.abs(np.subtract(x["ref"], x["pred"])).mean() for k, v in d.items() if k.startswith(q) and "train" not in k for x in v if x["n"] >= 80]
    if np.mean(e) > lim: bad.append(f"{q}>=80 {np.mean(e):.3f}")
print(" ".join(bad))
PY
)
echo "== decision $(date): ${fail:-PASS gate (a)}"
if [ -n "$fail" ]; then
  echo "== launching polar_v7_totmu $(date)"
  ( DIPOLE_LOSS=total CONFIG=$REPO/d1/mace_torch_polar_totmu.yaml NGPU=4 $REPO/d1/mace_torch_polar.sh ~/pah/dipole/dataset_v7_com $R/polar_v7_totmu --max_num_epochs 200 --patience 40 > $R/polar_v7_totmu.log 2>&1
    M2=$R/polar_v7_totmu/polar.model
    CUDA_VISIBLE_DEVICES=1 $P -W ignore $REPO/d1/mlff_dipole_eval.py ~/pah/dipole/dataset_v7 $R/polar_v7_totmu.json --models q0=$M2,q1=$M2 --model-type PolarMACE > $R/polar_v7_totmu_eval.log 2>&1
    echo "== totmu done $(date)" ) &
fi
echo "== G5 $(date)"
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 $P -W ignore $REPO/d1/dipole_g5.py run $G/models/v7.model $G/seed/species_v2.csv,$G/seed/transfer2/species_transfer2.csv $G/seed/xyz,$G/seed/transfer2/xyz $G/data/pahdb-complete-theoretical-v4.00.xml.gz $G/v7 --workers 16 \
  --dipole polar:PolarMACE:$M:$M > $G/v7_polar.log 2>&1
$P -W ignore $REPO/d1/dipole_g5.py report $G/v7 $G/v7_report.json --pahdb $G/data/pahdb-complete-theoretical-v4.00.xml --ref-flags $G/data/pahdb_v4_replacements_v1.csv \
  --combo xtb --combo xtb+dxtb_dip --combo dip --combo diel --combo polar 2>&1 | grep -v "Warning\|nanmedian\|^  "
echo "== after done $(date)"
wait
echo "== all done $(date)"
