#!/bin/bash
# Port test of the Mai et al. 2025 NNFF + EPNN models in container A on one Hangang GH200 (d1/port/mai_port_test.py).
# Usage, from the repository root under /appsdata/$USER:
#   sbatch d1/port/slurm_mai_port_test.sbatch /appsdata/$USER/sif/A_jax.sif /appsdata/$USER/d1
# The second argument holds mai2025/src/Supplementary_Data_Code (extracted from Zenodo 15771437) and XYZ_pahdb_1704/.
# GPU nodes are allocated whole, so one process runs per GPU: each sweeps a quarter of the geometries, and
# GPU 0 also runs the load, GPU-vs-CPU and NVE checks. Per-GPU output goes to mai_port_<jobid>_g<k>.log.
#SBATCH --job-name=mai_port
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=00:30:00
#SBATCH --comment="field=aac;appl=jax"
#SBATCH --output=mai_port_%j.log
set -euo pipefail
SIF=$(realpath "${1:?usage: sbatch slurm_mai_port_test.sbatch A_jax.sif D1_DIR}")
D1=$(realpath "${2:?usage: sbatch slurm_mai_port_test.sbatch A_jax.sif D1_DIR}")
module load apptainer/1.4.5
export APPTAINERENV_JAX_PLATFORMS=cuda,cpu   # the image allows only cuda; check 2 needs cpu too (--env would split the comma)
echo "host $(hostname)  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
ngpu=$(nvidia-smi -L | wc -l); pids=()
for g in $(seq 0 $((ngpu - 1))); do
  apptainer exec --nv --env CUDA_VISIBLE_DEVICES=$g,XLA_PYTHON_CLIENT_PREALLOCATE=false \
    --bind "$D1/mai2025/src/Supplementary_Data_Code":/mai:ro --bind "$D1/XYZ_pahdb_1704":/xyz:ro \
    --bind "${SLURM_SUBMIT_DIR}/d1/port":/w:ro "$SIF" \
    python -I /w/mai_port_test.py /mai /xyz 900 $g $ngpu > "mai_port_${SLURM_JOB_ID}_g$g.log" 2>&1 &
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
for g in $(seq 0 $((ngpu - 1))); do echo "===== GPU $g"; grep -E '^\[|^ {4}|^jax|Error|Traceback' "mai_port_${SLURM_JOB_ID}_g$g.log" || true; done
exit $rc
