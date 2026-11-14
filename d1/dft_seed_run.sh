#!/bin/bash
# Launch dft_seed_run.py (G4) in container D: NPROC processes per GPU on the listed GPUs, each pinned to its GPU's
# NUMA node, one shard each (shard i of n = NPROC x number of GPUs). Resumable: rerun the same command.
#   GPUS="0 1" NPROC=1 d1/dft_seed_run.sh [extra dft_seed_run.py run args, e.g. --only C10H8_330,...]
# Logs: $OUT/logs/. Frames: mlff_seed/frames (v2, 7,424) and mlff_seed/frames_transfer (80).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
SEED=/var/tmp/$USER-pah/d1/mlff_seed
OUT=${OUT:-/var/tmp/$USER-pah/d1/seed_dft}
SIF=${SIF:-/var/tmp/$USER-pah/sif/D_gpu4pyscf.sif}
GPUS=(${GPUS:-0}); NPROC=${NPROC:-1}
N=$(( ${#GPUS[@]} * NPROC )); i=0
mkdir -p $OUT/logs /var/tmp/$USER-pah/cache/cupy
for g in "${GPUS[@]}"; do
  node=$([ "$g" = 1 ] && echo 1 || echo 0)      # GPU 0: cores 0-71 (node 0); GPU 1: cores 72-143 (node 1)
  for p in $(seq $NPROC); do
    CUDA_VISIBLE_DEVICES=$g numactl --cpunodebind=$node --membind=$node \
      apptainer exec --nv --bind "$REPO":/r --bind /var/tmp/$USER-pah \
      --env CUPY_CACHE_DIR=/var/tmp/$USER-pah/cache/cupy,PYTHONWARNINGS=ignore "$SIF" \
      python /r/d1/dft_seed_run.py run $SEED/frames $SEED/frames_transfer --out $OUT --shard $i/$N "$@" \
      >> $OUT/logs/shard$i-of$N.log 2>&1 &
    i=$((i + 1))
  done
done
wait
