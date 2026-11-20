#!/bin/bash
# High-spin seed frames for the charged-MLFF work (docs/plan/handoff_highspin_to_charged_mlff.md) on ONE Hangang GH200
# node: (1) GFN2-xTB sampling with the charged-MLFF sampler (mlff_sample.py; neutral mult 3 from OUR triplet minimum,
# cation mult 2 from PAHdb), (2) the manifest with g16_io.py write, (3) B3LYP/6-31G* labels with dft_seed_run.py
# (GPU4PySCF, container D, NPROC processes per GPU, every frame stability-tested: --stab-all). The charged-MLFF code is
# used unchanged from a detached worktree of feat-gpu-gh200 (CODE_D1). Usage, from the repository root under
# /appsdata/$USER:
#   sbatch d1/slurm_highspin_seed.sbatch D_gpu4pyscf.sif XTB_ENV.tgz SEL_DIR CODE_D1 OUT
# SEL_DIR: species.csv and xyz/ (d1/highspin_seed_v1). XTB_ENV.tgz: a conda env (tblite-python, ase, numpy) packed
# from /tmp/t641a02-xtb; it is unpacked to the node's /tmp at the same path (no relocation, nothing on Lustre).
# Environment: PAIRS (comma list; default every pair), SAMPLE_ARGS (override the sampler settings for every run, for
# tests), NPROC (default 2), NSAMPLE (parallel sampler runs, default 20).
# Sampler settings as in the charged-MLFF seed set: 32 frames per pair and charge (min, 3 disp, NMS 300/1000 K x2,
# 4 boundary, MD 4 x 300/600/900/1200/1500 K); transfer_test pairs as transfer2's MD: min + MD 300/900/1500 K x1.
# Restartable: existing frame files, gjf manifests and finished DFT frames are skipped.
# Output: OUT/frames/<pair>_q<q>.extxyz, OUT/gjf/manifest.json, OUT/dft/<frame>.json, OUT/logs/.
#SBATCH --job-name=hs_seed
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --comment="field=aac;appl=misc"
#SBATCH --output=hs_seed_%j.log
set -euo pipefail
usage="usage: sbatch slurm_highspin_seed.sbatch D_gpu4pyscf.sif XTB_ENV.tgz SEL_DIR CODE_D1 OUT"
SIF=$(realpath "${1:?$usage}"); ENVTGZ=$(realpath "${2:?$usage}"); SEL=$(realpath "${3:?$usage}")
CODE=$(realpath "${4:?$usage}"); mkdir -p "${5:?$usage}"; OUT=$(realpath "$5")
NPROC=${NPROC:-2}; NSAMPLE=${NSAMPLE:-20}
CACHE=/appsdata/$USER/cache/cupy; mkdir -p "$CACHE" "$OUT/frames" "$OUT/gjf" "$OUT/dft" "$OUT/logs"
module load apptainer/1.4.5
jid=${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}; jid=${jid:-$SLURM_JOB_ID}
echo "host $(hostname)  job $jid  $(date '+%F %T')  code $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo ?)"

# 1. sampling (CPU)
[ -x /tmp/t641a02-xtb/bin/python ] || tar xzf "$ENVTGZ" -C /tmp
PY=/tmp/t641a02-xtb/bin/python
jobs=$OUT/logs/sample_jobs_$jid.txt; : > "$jobs"
tail -n +2 "$SEL/species.csv" | while IFS=, read -r pair formula nC bin split rest; do
  [ -z "${PAIRS:-}" ] || [[ ",$PAIRS," == *",$pair,"* ]] || continue
  if [ -n "${SAMPLE_ARGS:-}" ]; then args=$SAMPLE_ARGS
  elif [ "$split" = transfer_test ]; then args="--temps 300,900,1500 --per-T 1 --n-disp 0 --nms '' --n-ch 0 --n-oop 0"
  else args=""; fi
  for q in 0 1; do
    m=$([ $q = 0 ] && echo 3 || echo 2); f=$OUT/frames/${pair}_q$q.extxyz
    [ -s "$f" ] || echo "$PY -I $CODE/mlff_sample.py $SEL/xyz/${pair}_q$q.xyz $f --charge $q --mult $m $args > $OUT/logs/sample_${pair}_q$q.log 2>&1" >> "$jobs"
  done
done
echo "sampling: $(wc -l < "$jobs") runs, $NSAMPLE at a time  $(date '+%T')"
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 xargs -d "\n" -P "$NSAMPLE" -I{} bash -c "{}" < "$jobs" || echo "WARNING: some sampler runs failed (see $OUT/logs/sample_*.log)"
echo "sampling done $(date '+%T'): $(ls "$OUT/frames" | grep -c '\.extxyz$') frame files"

# 2. provenance key and manifest (frames written by this job only; q0 = our own triplet minimum, q1 = PAHdb cation)
for f in "$OUT"/frames/*.extxyz; do
  g=$OUT/gjf/.done_$(basename "$f"); [ -e "$g" ] && continue
  case $f in *_q0.extxyz) o=own_highspin_v1;; *) o=pahdb_v4.00;; esac
  sed -i "/^Properties=/{/ ref_origin=/!s/\$/ ref_origin=$o/}" "$f"
  apptainer exec --bind "$CODE":/w:ro --bind "$OUT":/o "$SIF" python -I /w/g16_io.py write "/o/frames/$(basename "$f")" /o/gjf
  touch "$g"
done
names=$(python3 -c "
import json; print(','.join(sorted(json.load(open('$OUT/gjf/manifest.json')))))")
echo "frames in manifest: $(echo "$names" | tr ',' '\n' | wc -l)"

# 3. DFT, every frame stability-tested (open shells: Stable=Opt)
limit=$(squeue -h -j "$jid" -o %L || true)
hours=$(python3 -c "
import sys; s = sys.argv[1]; d, s = (s.split('-') + [''])[:2] if '-' in s else ('0', s)
p = [int(x) for x in s.split(':')]; p = [0] * (3 - len(p)) + p
h = int(d) * 24 + p[0] + p[1] / 60 + p[2] / 3600; print(round(h - min(0.5, 0.15 * h), 3))" "$limit" 2>/dev/null) || { echo "cannot read the time limit of $jid ('$limit')"; exit 1; }
ngpu=$(nvidia-smi -L | wc -l); N=$((ngpu * NPROC))
topo=$(nvidia-smi topo -m | sed 's/\x1b\[[0-9;]*m//g')
echo "DFT: $ngpu GPUs x $NPROC, --hours $hours  $(date '+%T')"
pids=()
for g in $(seq 0 $((ngpu - 1))); do
  cpus=$(printf '%s\n' "$topo" | awk -v r="GPU$g" '$1 == r {for (i = 2; i <= NF; i++) if ($i ~ /^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$/) {print $i; exit}}')
  if [ -z "$cpus" ]; then per=$(( $(nproc --all) / ngpu )); cpus="$((g * per))-$(((g + 1) * per - 1))"; fi
  for p in $(seq 0 $((NPROC - 1))); do
    i=$((g * NPROC + p))
    taskset -c "$cpus" apptainer exec --nv \
      --env CUDA_VISIBLE_DEVICES=$g,CUPY_CACHE_DIR=/cupy,PYTHONWARNINGS=ignore,CUPY_ACCELERATORS=cub \
      --bind "$CODE":/w:ro --bind "$OUT":/seed --bind "$CACHE":/cupy "$SIF" \
      python -I /w/dft_seed_run.py run /seed/frames --out /seed/dft --shard $i/$N --hours "$hours" --stab-all --names "$names" \
      > "$OUT/logs/dft_${jid}_g${g}_p$p.log" 2>&1 &
    pids+=($!)
  done
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
for f in "$OUT"/logs/dft_${jid}_*.log; do echo "===== $(basename $f)"; grep -E '^shard|stopping|Error|Traceback' "$f" || true; tail -1 "$f"; done
echo "frames done: $(ls "$OUT/dft" | grep -c '\.json$') of $(echo "$names" | tr ',' '\n' | wc -l)  $(date '+%F %T')"
exit $rc
