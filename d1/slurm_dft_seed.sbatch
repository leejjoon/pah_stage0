#!/bin/bash
# Seed-set DFT (d1/dft_seed_run.py, GPU4PySCF, container D) on Hangang GH200 nodes: NPROC processes per GPU (default
# 2, the GH200 optimum; no MPS), each pinned to its GPU's CPU cores and running one shard. Usage, from the repository
# root under /appsdata/$USER (docs/plan/cluster_migration.md, "Seed-set DFT"):
#   sbatch [--array=0-4] [--time=HH:MM:SS] d1/slurm_dft_seed.sbatch D_gpu4pyscf.sif SEED_DIR OUT [run options]
# SEED_DIR holds frames/, frames_transfer/, gjf/manifest.json and gjf_transfer/manifest.json (seed_hangang.tgz).
# Task k of K runs shards (k*ngpu + g)*NPROC + p of K*ngpu*NPROC. Finished frames are skipped by file name, so a
# resubmit (any array size) continues; do not run two jobs on the same OUT at once (they would repeat frames).
# Each process starts no new frame within 30 min of the time limit (--hours). Per-process output:
# OUT/logs/<jobid>_t<task>_g<gpu>_p<proc>.log.
#SBATCH --job-name=seed_dft
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --comment="field=aac;appl=misc"
#SBATCH --output=seed_dft_%A_%a.log
set -euo pipefail
usage="usage: sbatch slurm_dft_seed.sbatch D_gpu4pyscf.sif SEED_DIR OUT [run options]"
SIF=$(realpath "${1:?$usage}"); SEED=$(realpath "${2:?$usage}"); mkdir -p "${3:?$usage}"; OUT=$(realpath "$3"); shift 3
NPROC=${NPROC:-2}
CACHE=${CUPY_CACHE:-/appsdata/$USER/cache/cupy}; mkdir -p "$CACHE" "$OUT/logs"
module load apptainer/1.4.5
limit=$(squeue -h -j "$SLURM_JOB_ID" -o %L || true)  # time left, [D-]HH:MM:SS
hours=$(python3 -c "
import sys; s = sys.argv[1]; d, s = (s.split('-') + [''])[:2] if '-' in s else ('0', s)
p = [int(x) for x in s.split(':')]; p = [0] * (3 - len(p)) + p
h = int(d) * 24 + p[0] + p[1] / 60 + p[2] / 3600; print(round(h - min(0.5, 0.15 * h), 3))" "$limit" 2>/dev/null || echo 23.5)
task=${SLURM_ARRAY_TASK_ID:-0}; ntask=${SLURM_ARRAY_TASK_COUNT:-1}
[ "${SLURM_ARRAY_TASK_MIN:-0}" = 0 ] || { echo "array indices must start at 0"; exit 1; }
ngpu=$(nvidia-smi -L | wc -l); N=$((ntask * ngpu * NPROC))
echo "task $task/$ntask  host $(hostname)  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)  ${ngpu} GPUs x $NPROC  shards of $N  --hours $hours $*"
topo=$(nvidia-smi topo -m | sed 's/\x1b\[[0-9;]*m//g')    # read once: a pipe that closes early would trip pipefail
printf '%s\n' "$topo" | sed -n "1,$((ngpu + 1))p" 
pids=()
for g in $(seq 0 $((ngpu - 1))); do
  # the GPU's cores from nvidia-smi's "CPU Affinity" column (e.g. 0-71); fall back to an even split of the node
  cpus=$(printf '%s\n' "$topo" | awk -v r="GPU$g" '$1 == r {for (i = 2; i <= NF; i++) if ($i ~ /^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$/) {print $i; exit}}')
  if [ -z "$cpus" ]; then per=$(( $(nproc --all) / ngpu )); cpus="$((g * per))-$(((g + 1) * per - 1))"; fi
  for p in $(seq 0 $((NPROC - 1))); do
    i=$(((task * ngpu + g) * NPROC + p))
    taskset -c "$cpus" apptainer exec --nv \
      --env CUDA_VISIBLE_DEVICES=$g,CUPY_CACHE_DIR=/cupy,PYTHONWARNINGS=ignore \
      --bind "${SLURM_SUBMIT_DIR}/d1":/w:ro --bind "$SEED":/seed:ro --bind "$OUT":/out --bind "$CACHE":/cupy "$SIF" \
      python -I /w/dft_seed_run.py run /seed/frames /seed/frames_transfer --out /out --shard $i/$N --hours "$hours" "$@" \
      > "$OUT/logs/${SLURM_JOB_ID}_t${task}_g${g}_p$p.log" 2>&1 &
    pids+=($!)
    echo "shard $i/$N: GPU $g, cores $cpus"
  done
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
for f in "$OUT"/logs/${SLURM_JOB_ID}_t${task}_*.log; do
  echo "===== $(basename $f)"; grep -E '^shard|stopping|Error|Traceback' "$f" || true; tail -1 "$f"
done
echo "frames done in OUT: $(ls "$OUT" | grep -c '\.json$')"
exit $rc
