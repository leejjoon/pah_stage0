#!/bin/bash
# Train d1/mace_torch_dielectric.yaml with a plain Python env (x86 venv on wasp33a, or the VESSL job image), no container:
#   GPU=0 PY=~/pah/mace-venv/bin/python d1/mace_torch_dielectric.sh <COM-centred dataset dir> <run dir> \
#     --train_file <ds>/train_q1.extxyz --valid_file <ds>/validation_q1.extxyz [--max_num_epochs 2 ...]
# The env needs mace-torch==0.3.17 e3nn==0.4.4 (container B's pins). TF32 off as on gh200 (d1/mace_torch_baseline.sh);
# torch's own matmul TF32 default is already off. The run dir gets config.yaml, logs/, checkpoints/, results/, models.
set -euo pipefail
DS=$(realpath "${1:?dataset dir}"); mkdir -p "${2:?run dir}"; RUN=$(realpath "$2"); shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
PY=${PY:-python}
cp "${CONFIG:-$REPO/d1/mace_torch_dielectric.yaml}" "$RUN/config.yaml"
echo "dataset $DS" > "$RUN/dataset.txt"
cd "$RUN"
CUDA_VISIBLE_DEVICES=${GPU:-0} NVIDIA_TF32_OVERRIDE=0 TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  "$PY" -m mace.cli.run_train --config config.yaml --name dielectric --work_dir "$RUN" "$@"
