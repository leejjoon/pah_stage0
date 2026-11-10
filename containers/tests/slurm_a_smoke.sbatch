#!/bin/bash
# Container A smoke test on KISTI Hangang GH200 nodes (aarch64; uses the T0 A_jax.sif pulled from myhub).
# Hangang rules (docs/external/hangang-user-guide, system/slurm.md):
#   - jobs can be submitted only from /scratch/$USER or /appsdata/$USER: keep the clone there and sbatch from it;
#   - --comment="field=...;appl=..." is mandatory (values from `showappl`); --time is mandatory;
#   - GPU nodes are allocated whole (--exclusive): 4 GH200, 288 cores, about 850 GiB per node.
# Usage, from the repository root under /appsdata/$USER (or /scratch/$USER):
#   sbatch [-p normal] containers/tests/slurm_a_smoke.sbatch /appsdata/$USER/sif/A_jax.sif
# Runs containers/tests/a_jax.sh (JAX on GPU, cueq import, x64 MACE-OFF conversion and check, predict, jax-md)
# once per GPU, then the force-call timing probe. MACE-OFF23_small is downloaded into $HOME/.cache/mace.
#SBATCH --job-name=a_smoke
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --comment="field=aac;appl=jax"
#SBATCH --output=a_smoke_%j.log
set -euo pipefail
SIF=$(realpath "${1:?usage: sbatch slurm_a_smoke.sbatch /path/to/A_jax.sif}")
module load apptainer/1.4.5
HERE=${SLURM_SUBMIT_DIR}/containers/tests
WORK=/scratch/$USER/a_smoke/$SLURM_JOB_ID      # /scratch: purged after 15 days without access
mkdir -p "$WORK"
echo "host $(hostname)  arch $(uname -m)  $(apptainer --version)"
nvidia-smi --query-gpu=index,name,driver_version,memory.used --format=csv
echo "A_jax.sif sha256 $(sha256sum "$SIF" | cut -c1-16)  (T0 image: 7304b1becb903c16)"
ngpu=$(nvidia-smi -L | wc -l)
for g in $(seq 0 $((ngpu - 1))); do
  echo "===== GPU $g"
  mkdir -p "$WORK/g$g"; cp "$HERE"/* "$WORK/g$g"/
  apptainer exec --nv --env CUDA_VISIBLE_DEVICES=$g --bind "$WORK/g$g":/t "$SIF" bash /t/a_jax.sh
done
# Same probe as the GH200 feasibility check: force-call median for MACE-OFF small, float32, cueq off/on.
# Its input, eq_frames.xyz, is rebuilt from the T0 species (on the project node it came from container B).
cp "${SLURM_SUBMIT_DIR}/t0/species_T0.xyz" "$WORK/g0"/
apptainer exec --nv --bind "$WORK/g0":/t "$SIF" python /t/eq_frames_t0.py
for v in "" _cueq; do for n in 1 10 100; do
  log="$WORK/g0/time_f32${v}_n$n.log"
  apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0 --bind "$WORK/g0":/t "$SIF" \
    python /t/jax_time.py /t/off_small_f32$v.npz float32 $n 30 > "$log" 2>&1 || { echo "[$v n=$n] FAILED:"; tail -5 "$log"; exit 1; }
  grep "call median" "$log" | sed "s/^/[$v n=$n] /"
done; done
echo "outputs in $WORK"
