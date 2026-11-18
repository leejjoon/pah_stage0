#!/bin/bash
# High-spin recomputation of the PAHdb v4.00 non-Kekule species (d1/triplet_batch.py -> d1/triplet_freq.py --opt) in
# container D: one process per GPU, each a shard of d1/triplet_species_v1.csv. Usage, from the repository root under
# /appsdata/$USER:
#   sbatch [--array=0-4] [--time=HH:MM:SS] d1/slurm_triplet_batch.sbatch D_gpu4pyscf.sif PAHDB_XML_GZ OUT [batch options]
# Task k of K runs shards k*ngpu+g of K*ngpu; resubmit with the same array size to continue (finished species are
# skipped). Never run two jobs on the same OUT at once. Per-GPU output: OUT/batch_<array or job id>_t<task>_g<gpu>.log.
#SBATCH --job-name=triplet
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --comment="field=aac;appl=misc"
#SBATCH --output=triplet_%A_%a.log
set -euo pipefail
usage="usage: sbatch slurm_triplet_batch.sbatch D_gpu4pyscf.sif PAHDB_XML_GZ OUT [batch options]"
SIF=$(realpath "${1:?$usage}"); XML=$(realpath "${2:?$usage}"); mkdir -p "${3:?$usage}"; OUT=$(realpath "$3"); shift 3
CACHE=/appsdata/$USER/cache/cupy; mkdir -p "$CACHE"
module load apptainer/1.4.5
# this task's own id (in an array, SLURM_JOB_ID of one task equals the array id and squeue would list every task)
jid=${SLURM_ARRAY_JOB_ID:+${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}}; jid=${jid:-$SLURM_JOB_ID}
limit=$(squeue -h -j "$jid" -o %L || true)
hours=$(python3 -c "
import sys; s = sys.argv[1]; d, s = (s.split('-') + [''])[:2] if '-' in s else ('0', s)
p = [int(x) for x in s.split(':')]; p = [0] * (3 - len(p)) + p
h = int(d) * 24 + p[0] + p[1] / 60 + p[2] / 3600; print(round(h - min(0.25, 0.1 * h), 3))" "$limit" 2>/dev/null) || { echo "cannot read the time limit of $jid ('$limit')"; exit 1; }
task=${SLURM_ARRAY_TASK_ID:-0}; ntask=${SLURM_ARRAY_TASK_COUNT:-1}
[ "${SLURM_ARRAY_TASK_MIN:-0}" = 0 ] || { echo "array indices must start at 0"; exit 1; }
ngpu=$(nvidia-smi -L | wc -l); N=$((ntask * ngpu)); pids=()
echo "task $task/$ntask  host $(hostname)  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)  --hours $hours $*"
for g in $(seq 0 $((ngpu - 1))); do
  apptainer exec --nv --env CUDA_VISIBLE_DEVICES=$g,CUPY_CACHE_DIR=/cupy,PYTHONWARNINGS=ignore,OMP_NUM_THREADS=16,CUPY_ACCELERATORS=cub \
    --bind "${SLURM_SUBMIT_DIR}/d1":/w:ro --bind "$(dirname "$XML")":/x:ro --bind "$OUT":/out --bind "$CACHE":/cupy "$SIF" \
    python -I /w/triplet_batch.py /w/triplet_species_v1.csv "/x/$(basename "$XML")" /out --shard $((task * ngpu + g)) --nshards $N --hours "$hours" "$@" \
    > "$OUT/batch_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_t${task}_g$g.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
for f in "$OUT"/batch_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_t${task}_g*.log; do echo "===== $(basename $f)"; grep -E "^shard|done|FAILED|stop" "$f" || true; done
echo "species done in OUT: $(ls "$OUT" | grep -c '\.json$'); failed: $(ls "$OUT/failed" | wc -l)"
exit $rc
