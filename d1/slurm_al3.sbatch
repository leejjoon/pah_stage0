#!/bin/bash
# AL round 3, track 1 (shared-notes handoff 2026-11-21-pah_stage0-z2mini-to-glogin04-al3; package README
# al3_track1/README.md) on Hangang: one GH200 node per array task. Per node, for its share of the pairs:
#   (A) cation minima: mlff_species_r2.py opt (container D, NPROC processes per GPU, pairs dealt out in rank order);
#       meanwhile the neutrals are sampled on the CPU (they need no optimisation);
#   (B) bond-graph check q1 vs q0 (mlff_sample.graph); a pair whose cation is not ok or changed its graph is dropped
#       and listed in OUT/dropped.txt; then the cations are sampled;
#   (C) manifest (g16_io.py write, defaults) and B3LYP/6-31G* labels (dft_seed_run.py run, manifest stability tests),
#       in chunks of CHUNK pairs in rank order, so that a job stopped by its time budget loses whole low-rank species.
# Usage, from the repository root under /appsdata/$USER:
#   sbatch --array=0-(NT-1) d1/slurm_al3.sbatch D_gpu4pyscf.sif XTB_ENV.tgz PKG OUTBASE
# PKG: the unpacked al3_track1/ (species_r3.csv, xyz/, xyz_start/); opt results are written into PKG/xyz and PKG/opt.
# Task k of NT (NT = array size, or NTASK) takes the pairs with (rank-1) % NT == k and MAXRANK >= rank, and writes
# OUTBASE/t<k>/{frames,gjf,dft,logs}. Environment: MAXRANK (default all), NTASK, NPROC (default 2), NSAMPLE (default
# 20), CHUNK (default 4), SAMPLE_ARGS (test override for every sampler run).
# Sampler: seed recipe (defaults, 32 frames), --seed (cksum(pair) % 100000 + q) as in the package README.
# Restartable: finished opt, frame files, manifests and DFT frames are skipped.
#SBATCH --job-name=al3
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --comment="field=aac;appl=misc"
#SBATCH --output=al3_%A_%a.log
set -euo pipefail
usage="usage: sbatch --array=0-N d1/slurm_al3.sbatch D_gpu4pyscf.sif XTB_ENV.tgz PKG OUTBASE"
SIF=$(realpath "${1:?$usage}"); ENVTGZ=$(realpath "${2:?$usage}"); PKG=$(realpath "${3:?$usage}")
mkdir -p "${4:?$usage}"; BASE=$(realpath "$4"); CODE=$(realpath "$(dirname "$0")")
[ -f "$CODE/mlff_species_r2.py" ] || CODE=$(realpath d1)
K=${SLURM_ARRAY_TASK_ID:-0}; NT=${NTASK:-${SLURM_ARRAY_TASK_COUNT:-1}}
NPROC=${NPROC:-2}; NSAMPLE=${NSAMPLE:-20}; CHUNK=${CHUNK:-4}; MAXRANK=${MAXRANK:-100000}
OUT=$BASE/t$K; CACHE=/appsdata/$USER/cache/cupy
mkdir -p "$CACHE" "$OUT/frames" "$OUT/gjf" "$OUT/dft" "$OUT/logs" "$OUT/tmp" "$PKG/opt"
module load apptainer/1.4.5
jid=${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}; jid=${jid:-$SLURM_JOB_ID}
T0=$(date +%s)
echo "host $(hostname)  job $jid  task $K/$NT  $(date '+%F %T')  code $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo ?)"

# time budget (hours) from the job's time limit, with a margin
limit=$(squeue -h -j "$jid" -o %L || true)
BUDGET=$(python3 -c "
import sys; s = sys.argv[1]; d, s = (s.split('-') + [''])[:2] if '-' in s else ('0', s)
p = [int(x) for x in s.split(':')]; p = [0] * (3 - len(p)) + p
h = int(d) * 24 + p[0] + p[1] / 60 + p[2] / 3600; print(round(h - min(0.4, 0.1 * h), 3))" "$limit") || { echo "cannot read the time limit of $jid ('$limit')"; exit 1; }
left() { python3 -c "print(round($BUDGET - ($(date +%s) - $T0) / 3600, 3))"; }

# this task's pairs, in rank order
PAIRS=$(python3 -c "
import csv; r = sorted(csv.DictReader(open('$PKG/species_r3.csv')), key=lambda x: int(x['rank']))
print(' '.join(x['pair'] for x in r if int(x['rank']) <= $MAXRANK and (int(x['rank']) - 1) % $NT == $K))")
echo "pairs ($(echo $PAIRS | wc -w)): $PAIRS"; echo "budget $BUDGET h"

ngpu=$(nvidia-smi -L | wc -l); N=$((ngpu * NPROC))
topo=$(nvidia-smi topo -m | sed 's/\x1b\[[0-9;]*m//g')
cpus_of() {
  local c; c=$(printf '%s\n' "$topo" | awk -v r="GPU$1" '$1 == r {for (i = 2; i <= NF; i++) if ($i ~ /^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$/) {print $i; exit}}')
  if [ -z "$c" ]; then local per=$(( $(nproc --all) / ngpu )); c="$(($1 * per))-$((($1 + 1) * per - 1))"; fi; echo "$c"; }
dexec() {  # dexec GPU cmd... : container D on one GPU, pinned to its CPUs
  local g=$1; shift
  taskset -c "$(cpus_of $g)" apptainer exec --nv \
    --env CUDA_VISIBLE_DEVICES=$g,CUPY_CACHE_DIR=/cupy,PYTHONWARNINGS=ignore,CUPY_ACCELERATORS=cub \
    --bind "$CODE":/w:ro --bind "$PKG":/pkg --bind "$OUT":/o --bind "$CACHE":/cupy "$SIF" "$@"; }

[ -x /tmp/t641a02-xtb/bin/python ] || tar xzf "$ENVTGZ" -C /tmp
PY=/tmp/t641a02-xtb/bin/python
sample_cmds() {  # sample_cmds q pair... : one sampler command line per missing frame file
  local q=$1; shift
  for pair in "$@"; do
    f=$OUT/frames/${pair}_q$q.extxyz; [ -s "$f" ] && continue
    seed=$(( $(echo $pair | cksum | cut -d' ' -f1) % 100000 + q ))
    echo "OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $PY -I $CODE/mlff_sample.py $PKG/xyz/${pair}_q$q.xyz $OUT/tmp/${pair}_q$q.extxyz --charge $q --mult $((q + 1)) --seed $seed ${SAMPLE_ARGS:-} > $OUT/logs/sample_${pair}_q$q.log 2>&1 && mv $OUT/tmp/${pair}_q$q.extxyz $f"
  done; }

# (A) neutral sampling in the background, cation optimisation on the GPUs
sample_cmds 0 $PAIRS > "$OUT/logs/sample0_$jid.txt"
xargs -d "\n" -P "$NSAMPLE" -I{} bash -c "{}" < "$OUT/logs/sample0_$jid.txt" &
spid=$!
need=(); for pair in $PAIRS; do
  [ -f "$PKG/xyz/${pair}_q1.xyz" ] || [ -f "$PKG/opt/${pair}_q1.json" ] || need+=("$pair"); done
echo "opt: ${#need[@]} cations over $N processes  $(date '+%T')"
pids=()
for i in $(seq 0 $((N - 1))); do
  mine=(); for j in "${!need[@]}"; do [ $((j % N)) = $i ] && mine+=("${need[$j]}"); done
  [ ${#mine[@]} = 0 ] && continue
  g=$((i / NPROC)); only=$(IFS=,; echo "${mine[*]}")
  dexec $g python -I /w/mlff_species_r2.py opt /pkg --only "$only" > "$OUT/logs/opt_${jid}_p$i.log" 2>&1 &
  pids+=($!)
done
for p in "${pids[@]}"; do wait $p || echo "WARNING: an opt process failed (see $OUT/logs/opt_${jid}_*.log)"; done
echo "opt done $(date '+%T')"; grep -h -E 'ok|NOT OK|Error|Traceback' "$OUT"/logs/opt_${jid}_*.log 2>/dev/null || true

# (B) keep pairs with an ok cation of the same bond graph
KEEP=$(PYTHONPATH=$CODE $PY -c "
import sys, json, os, numpy as np
from mlff_sample import graph
def rd(f):
    L = open(f).read().split('\n'); n = int(L[0]); a = [l.split() for l in L[2:2 + n]]
    return [x[0] for x in a], np.array([[float(v) for v in x[1:4]] for x in a])
keep = []; drop = []
for pair in sys.argv[2:]:
    f1 = f'{sys.argv[1]}/xyz/{pair}_q1.xyz'; js = f'{sys.argv[1]}/opt/{pair}_q1.json'
    if not os.path.exists(f1):
        drop.append((pair, 'cation not ok' if os.path.exists(js) else 'cation not optimised')); continue
    s0, p0 = rd(f'{sys.argv[1]}/xyz/{pair}_q0.xyz'); s1, p1 = rd(f1)
    if s0 != s1 or graph(s0, p0) != graph(s1, p1): drop.append((pair, 'bond graph changed')); continue
    keep.append(pair)
open('$OUT/dropped.txt', 'w').write(''.join(f'{p}\t{w}\n' for p, w in drop))
print(' '.join(keep))" "$PKG" $PAIRS)
echo "kept $(echo $KEEP | wc -w) pairs; dropped: $(cut -f1 "$OUT/dropped.txt" | tr '\n' ' ')"
sample_cmds 1 $KEEP > "$OUT/logs/sample1_$jid.txt"
wait $spid || echo "WARNING: some neutral sampler runs failed"
xargs -d "\n" -P "$NSAMPLE" -I{} bash -c "{}" < "$OUT/logs/sample1_$jid.txt" || echo "WARNING: some cation sampler runs failed"
echo "sampling done $(date '+%T'): $(ls "$OUT/frames" | grep -c '\.extxyz$') frame files"

# (C) manifest and DFT, CHUNK pairs at a time in rank order
set -- $KEEP; c=0
while [ $# -gt 0 ]; do
  chunk=(); for _ in $(seq 1 $CHUNK); do [ $# -gt 0 ] && { chunk+=("$1"); shift; }; done
  files=(); for pair in "${chunk[@]}"; do for q in 0 1; do
    f=$OUT/frames/${pair}_q$q.extxyz; [ -s "$f" ] && files+=("$f"); done; done
  for f in "${files[@]}"; do
    g=$OUT/gjf/.done_$(basename "$f"); [ -e "$g" ] && continue
    apptainer exec --bind "$CODE":/w:ro --bind "$OUT":/o "$SIF" python -I /w/g16_io.py write "/o/frames/$(basename "$f")" /o/gjf
    touch "$g"
  done
  names=$(python3 -c "
import json, sys; m = json.load(open('$OUT/gjf/manifest.json')); p = sys.argv[1:]
print(','.join(sorted(n for n in m if any(n.startswith(x + '_q') for x in p))))" "${chunk[@]}")
  h=$(left); c=$((c + 1))
  echo "chunk $c: ${chunk[*]}: $(echo "$names" | tr ',' '\n' | grep -c .) frames, $h h left  $(date '+%T')"
  if python3 -c "import sys; sys.exit(0 if $h > 0.2 else 1)"; then :; else echo "no time left; stopping"; break; fi
  pids=()
  for i in $(seq 0 $((N - 1))); do
    dexec $((i / NPROC)) python -I /w/dft_seed_run.py run /o/frames --out /o/dft --shard $i/$N --hours "$h" --names "$names" \
      >> "$OUT/logs/dft_${jid}_p$i.log" 2>&1 &
    pids+=($!)
  done
  rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
  [ $rc = 0 ] || echo "WARNING: a DFT process failed in chunk $c"
  grep -h stopping "$OUT"/logs/dft_${jid}_p*.log 2>/dev/null | tail -2 || true
done
echo "frames done: $(ls "$OUT/dft" | grep -c '\.json$')  $(date '+%F %T')"
