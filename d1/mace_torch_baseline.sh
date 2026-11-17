#!/bin/bash
# mace-torch baseline (d1/mace_torch_baseline.yaml) on the project node, container B, one GPU:
#   GPU=1 d1/mace_torch_baseline.sh <dataset dir> <run dir> [extra mace_run_train options, e.g. --max_num_epochs 2]
# The run dir gets config.yaml (DATASET filled in), logs/, checkpoints/, results/ and the model files.
set -euo pipefail
DS=$(realpath "${1:?dataset dir}"); mkdir -p "${2:?run dir}"; RUN=$(realpath "$2"); shift 2
REPO=$(cd "$(dirname "$0")/.." && pwd)
SIF=${SIF:-/var/tmp/jjlee-pah/sif/B_torch.sif}
sed "s#DATASET#/ds#g" "$REPO/d1/mace_torch_baseline.yaml" > "$RUN/config.yaml"
apptainer exec --nv --env CUDA_VISIBLE_DEVICES=${GPU:-1},NVIDIA_TF32_OVERRIDE=0,TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
  --bind "$DS":/ds:ro --bind "$RUN":/run --pwd /run "$SIF" \
  mace_run_train --config config.yaml --name baseline --work_dir /run "$@"
