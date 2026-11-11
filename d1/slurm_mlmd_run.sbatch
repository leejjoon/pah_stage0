#!/bin/bash
# D1 MLMD production (d1/mlmd_run.py) on one Hangang GH200 node: one process per GPU, each taking every 4th unit of
# the plan; checkpoints let the next job resume. Usage, from the repository root under /appsdata/$USER:
#   apptainer exec A_jax.sif python d1/mlmd_run.py plan /appsdata/$USER/d1/XYZ_pahdb_1704 OUT [...]    # once
#   sbatch [--time=HH:MM:SS] d1/slurm_mlmd_run.sbatch A_jax.sif D1_DIR OUT [mlmd_run.py run options, e.g. --units ...]
# D1_DIR holds mai2025/src/Supplementary_Data_Code (Zenodo 15771437) and XYZ_pahdb_1704/. The run stops itself
# before the job's time limit (30 min, or 15 % of a short limit), so resubmitting the same command continues.
# Per-GPU output: OUT/mlmd_<jobid>_g<k>.log; progress: apptainer exec A_jax.sif python d1/mlmd_run.py status OUT.
#SBATCH --job-name=mlmd
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --comment="field=aac;appl=jax"
#SBATCH --output=mlmd_%j.log
set -euo pipefail
usage="usage: sbatch slurm_mlmd_run.sbatch A_jax.sif D1_DIR OUT [run options]"
SIF=$(realpath "${1:?$usage}"); D1=$(realpath "${2:?$usage}"); OUT=$(realpath "${3:?$usage}"); shift 3
[ -f "$OUT/plan.json" ] || { echo "no $OUT/plan.json; run mlmd_run.py plan first"; exit 1; }
module load apptainer/1.4.5
export APPTAINERENV_JAX_PLATFORMS=cuda,cpu   # the self-check reference may use the CPU (--env would split the comma)
limit=$(squeue -h -j "$SLURM_JOB_ID" -o %L)  # time left, [D-]HH:MM:SS
hours=$(python3 -c "
import sys; s = sys.argv[1]; d, s = (s.split('-') + [''])[:2] if '-' in s else ('0', s)
p = [int(x) for x in s.split(':')]; p = [0] * (3 - len(p)) + p
h = int(d) * 24 + p[0] + p[1] / 60 + p[2] / 3600; print(round(h - min(0.5, 0.15 * h), 3))" "$limit")
echo "host $(hostname)  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)  run --hours $hours $*"
ngpu=$(nvidia-smi -L | wc -l); pids=()
for g in $(seq 0 $((ngpu - 1))); do
  apptainer exec --nv --env CUDA_VISIBLE_DEVICES=$g,XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --bind "$D1/mai2025/src/Supplementary_Data_Code":/mai:ro --bind "$D1/XYZ_pahdb_1704":/xyz:ro \
    --bind "${SLURM_SUBMIT_DIR}/d1":/w:ro --bind "$OUT":/out "$SIF" \
    python -I /w/mlmd_run.py run /mai /xyz /out --shard $g --nshards $ngpu --hours "$hours" "$@" \
    > "$OUT/mlmd_${SLURM_JOB_ID}_g$g.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
for g in $(seq 0 $((ngpu - 1))); do echo "===== GPU $g"; grep -E 'done |finished|paused|Error|Traceback|assert' "$OUT/mlmd_${SLURM_JOB_ID}_g$g.log" || true; done
apptainer exec --bind "$OUT":/out "$SIF" python -I "${SLURM_SUBMIT_DIR}/d1/mlmd_run.py" status /out || true
exit $rc
