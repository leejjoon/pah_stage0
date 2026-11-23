#!/bin/bash
# PolarMACE dipole/energy/force model (d1/mace_torch_polar.yaml) on a COM-centred dataset (d1/mlff_dipole_prep.py),
# one model for q0 + q1 (train/validation files concatenated here), plain venv, NGPU-way torchrun data parallel.
#   NGPU=4 PY=~/pah/mace-venv/bin/python d1/mace_torch_polar.sh <ds>_com <run dir> [extra run_train args]
# Model: <run dir>/polar.model; evaluate with
#   d1/mlff_dipole_eval.py <ds> <out.json> --models q0=<m>,q1=<m> --model-type PolarMACE
set -euo pipefail
DS=$1; RUN=$2; shift 2
PY=${PY:-$HOME/pah/mace-venv/bin/python}; NGPU=${NGPU:-4}; CONFIG=${CONFIG:-$(dirname "$0")/mace_torch_polar.yaml}
mkdir -p "$RUN"
cat "$DS/train_q0.extxyz" "$DS/train_q1.extxyz" > "$RUN/train.extxyz"
cat "$DS/validation_q0.extxyz" "$DS/validation_q1.extxyz" > "$RUN/valid.extxyz"
cp "$CONFIG" "$RUN/config.yaml"
cd "$RUN"
export NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 OMP_NUM_THREADS=4
DIST=(); [ "$NGPU" -gt 1 ] && DIST=(--distributed --launcher torchrun)
"$PY" -m torch.distributed.run --standalone --nproc_per_node "$NGPU" -m mace.cli.run_train --config config.yaml \
  --name polar --work_dir "$RUN" --train_file train.extxyz --valid_file valid.extxyz "${DIST[@]}" "$@"
