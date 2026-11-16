#!/bin/bash
# mace-jax baseline (docs/plan/charged_mlff_plan.md, Phase 2): one model with a head per charge state (q0, q1) on a
# curated dataset (d1/mlff_dataset.py), container A, one GPU. Usage on the project node:
#   GPU=1 d1/mace_jax_baseline.sh <dataset dir> <run dir> [extra mace-jax-train options, e.g. --max-epochs 2]
# 1. mace-jax-preprocess per charge: <split>_q<q>.extxyz (train, validation) -> <run dir>/h5/q<q>_/{train,val}
#    (energy dft_energy, forces dft_forces, dipole dft_dipole; r_max 5.0); skipped if present.
# 2. mace-jax-train with d1/mace_jax_baseline.gin (model, loss, optimizer, E0s by z_table index: see the gin file).
# Float32 without TF32 (JAX_DEFAULT_MATMUL_PRECISION=highest); no memory preallocation so the GPU can be shared.
set -euo pipefail
DS=$(realpath "${1:?dataset dir}"); mkdir -p "${2:?run dir}"; RUN=$(realpath "$2"); shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
SIF=${SIF:-/var/tmp/jjlee-pah/sif/A_jax.sif}
ENV=CUDA_VISIBLE_DEVICES=${GPU:-1},XLA_PYTHON_CLIENT_PREALLOCATE=false,JAX_DEFAULT_MATMUL_PRECISION=highest,JAX_PLATFORMS=cuda
X="apptainer exec --nv --env $ENV --bind $DS:/ds:ro --bind $RUN:/run --bind $REPO/d1:/w:ro --pwd /run $SIF"
for q in 0 1; do
  [ -d "$RUN/h5/q${q}_/train" ] || $X mace-jax-preprocess --train_file /ds/train_q$q.extxyz \
    --valid_file /ds/validation_q$q.extxyz --energy_key dft_energy --forces_key dft_forces --dipole_key dft_dipole \
    --h5_prefix h5/q${q}_ --r_max 5.0 --E0s '{1: -13.613116, 6: -1029.849742}' --compute_statistics
done
HEADS='{"q0": {"train_path": "h5/q0_/train", "valid_path": "h5/q0_/val"}, "q1": {"train_path": "h5/q1_/train", "valid_path": "h5/q1_/val"}}'
$X mace-jax-train /w/mace_jax_baseline.gin --name baseline --log-dir logs --checkpoint-dir ckpt --checkpoint-best \
  --train-path h5/q0_/train --valid-path h5/q0_/val --heads-config "$HEADS" --atomic-numbers 1 6 \
  --energy-key dft_energy --forces-key dft_forces "$@"
