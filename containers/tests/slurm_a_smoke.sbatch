#!/bin/bash
# Container A smoke test on a Slurm GH200 cluster (aarch64; A_jax.sif copied from the project GH200, not rebuilt).
# Usage: sbatch -p <partition> containers/tests/slurm_a_smoke.sbatch /path/to/A_jax.sif
# Runs containers/tests/a_jax.sh (JAX on GPU, cueq import, x64 MACE-OFF conversion and check, predict, jax-md)
# once per visible GPU. It downloads MACE-OFF23_small into $HOME/.cache/mace (needs internet on the compute node).
#SBATCH --job-name=a_smoke
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=a_smoke_%j.log
set -euo pipefail
SIF=$(realpath "${1:?usage: sbatch slurm_a_smoke.sbatch /path/to/A_jax.sif}")
module load apptainer
HERE=${SLURM_SUBMIT_DIR}/containers/tests
WORK=${TMPDIR:-/tmp}/$USER-a_smoke-$SLURM_JOB_ID          # tmpfs on this cluster: small files only
mkdir -p "$WORK"; cp "$HERE"/* "$WORK"/
echo "host $(hostname)  arch $(uname -m)  apptainer $(apptainer --version)"
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv
echo "A_jax.sif sha256 $(sha256sum "$SIF" | cut -c1-16)  (GH200 build: 7304b1becb903c16)"
ngpu=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((ngpu - 1))); do
  echo "===== GPU $g"
  mkdir -p "$WORK/g$g"; cp "$HERE"/* "$WORK/g$g"/
  apptainer exec --nv --env CUDA_VISIBLE_DEVICES=$g --bind "$WORK/g$g":/t "$SIF" bash /t/a_jax.sh
done
# Same timing probe as the GH200 feasibility check: force-call median for MACE-OFF small, float32, cueq off/on.
for v in "" _cueq; do for n in 1 10 100; do
  apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0 --bind "$WORK/g0":/t "$SIF" \
    python /t/jax_time.py /t/off_small_f32$v.npz float32 $n 30 2>&1 | grep "call median" | sed "s/^/[$v n=$n] /"
done; done
echo "outputs in $WORK (tmpfs: copy anything you want to keep before the node is released)"
