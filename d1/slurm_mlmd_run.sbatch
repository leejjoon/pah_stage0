#!/bin/bash
# D1 MLMD production (d1/mlmd_run.py) on Hangang GH200 nodes: one process per GPU, each a shard of the plan's
# units; checkpoints let the next job resume. Usage, from the repository root under /appsdata/$USER:
#   apptainer exec A_jax.sif python d1/mlmd_run.py plan /appsdata/$USER/d1/XYZ_pahdb_1704 OUT [...]    # once
#   sbatch [--time=HH:MM:SS] [--array=0-4] d1/slurm_mlmd_run.sbatch A_jax.sif D1_DIR OUT [run options, e.g. --units ...]
# Several nodes on one OUT: use an array; task k of K runs shards k*ngpu+g of K*ngpu. Never run two separate jobs
# (or arrays) on the same OUT at once, and resubmit with the same array size so each unit stays with one shard.
# D1_DIR holds mai2025/src/Supplementary_Data_Code (Zenodo 15771437); the species come from the plan's xyz_dir. The run stops itself
# before the job's time limit (30 min, or 15 % of a short limit), so resubmitting the same command continues.
# Per-GPU output: OUT/mlmd_<array or job id>_t<task>_g<k>.log; progress: apptainer exec A_jax.sif python d1/mlmd_run.py status OUT.
#SBATCH --job-name=mlmd
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --comment="field=aac;appl=jax"
#SBATCH --output=mlmd_%A_%a.log
set -euo pipefail
usage="usage: sbatch slurm_mlmd_run.sbatch A_jax.sif D1_DIR OUT [run options]"
SIF=$(realpath "${1:?$usage}"); D1=$(realpath "${2:?$usage}"); OUT=$(realpath "${3:?$usage}"); shift 3
[ -f "$OUT/plan.json" ] || { echo "no $OUT/plan.json; run mlmd_run.py plan first"; exit 1; }
XYZ=$(python3 -c "import json, sys; print(json.load(open(sys.argv[1]))['xyz_dir'])" "$OUT/plan.json")  # the plan's species
[ -d "$XYZ" ] || { echo "plan's xyz_dir $XYZ not found"; exit 1; }
module load apptainer/1.4.5
export APPTAINERENV_JAX_PLATFORMS=cuda,cpu   # the self-check reference may use the CPU (--env would split the comma)
# This task's own id: in an array, SLURM_JOB_ID of one task equals the array id, and squeue -j <array id>
# lists every task (job 1265580: two lines, so the parse failed and the run ignored the time limit).
jid=${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}; jid=${jid:-$SLURM_JOB_ID}
limit=$(squeue -h -j "$jid" -o %L || true)  # time left, [D-]HH:MM:SS
hours=$(python3 -c "
import sys; s = sys.argv[1]; d, s = (s.split('-') + [''])[:2] if '-' in s else ('0', s)
p = [int(x) for x in s.split(':')]; p = [0] * (3 - len(p)) + p
h = int(d) * 24 + p[0] + p[1] / 60 + p[2] / 3600; print(round(h - min(0.5, 0.15 * h), 3))" "$limit" 2>/dev/null) || { echo "cannot read the time limit of $jid (squeue: '$limit')"; exit 1; }
echo "task ${SLURM_ARRAY_TASK_ID:-0}/${SLURM_ARRAY_TASK_COUNT:-1}  host $(hostname)  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)  run --hours $hours $*"
# JAX compilation cache on /scratch, shared by all runs: its XLA autotune part writes ~10^5 small files, which
# filled the group's /appsdata file quota on 2026-11-13. /scratch purges files unused for 15 days (only a recompile).
CACHE=/scratch/$USER/jax_cache; mkdir -p "$CACHE"
ngpu=$(nvidia-smi -L | wc -l); pids=()
task=${SLURM_ARRAY_TASK_ID:-0}; ntask=${SLURM_ARRAY_TASK_COUNT:-1}
[ "${SLURM_ARRAY_TASK_MIN:-0}" = 0 ] || { echo "array indices must start at 0"; exit 1; }
for g in $(seq 0 $((ngpu - 1))); do
  apptainer exec --nv --env CUDA_VISIBLE_DEVICES=$g,XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --bind "$D1/mai2025/src/Supplementary_Data_Code":/mai:ro --bind "$XYZ":/xyz:ro \
    --bind "${SLURM_SUBMIT_DIR}/d1":/w:ro --bind "$OUT":/out --bind "$CACHE":/cache "$SIF" \
    python -I /w/mlmd_run.py run /mai /xyz /out --cache-dir /cache --shard $((task * ngpu + g)) --nshards $((ntask * ngpu)) --hours "$hours" "$@" \
    > "$OUT/mlmd_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_t${task}_g$g.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
for g in $(seq 0 $((ngpu - 1))); do echo "===== GPU $g"; grep -E 'done |finished|paused|Error|Traceback|assert' "$OUT/mlmd_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_t${task}_g$g.log" || true; done
apptainer exec --bind "$OUT":/out "$SIF" python -I "${SLURM_SUBMIT_DIR}/d1/mlmd_run.py" status /out || true
exit $rc
