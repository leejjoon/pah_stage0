#!/bin/bash
# G2/G3 of docs/plan/handoff_gpu4pyscf_gh200.md on the GH200 (H100), container D (GPU4PySCF 1.9.2).
#   d1/h100_bench.sh <step> [gpu]      steps: validate | stab | series | floor | batch | batch_mps
# One GPU (default 0) and its Grace NUMA node; outputs under /var/tmp/$USER-pah/d1/h100_*. Results that contain the
# Mai et al. geometries (C90H24_618, C144H30_641) stay there and are never committed.
set -euo pipefail
STEP=$1; GPU=${2:-0}
REPO=$(cd "$(dirname "$0")/.." && pwd)
D=/var/tmp/$USER-pah/d1; P=$D/g16_phase0/pahdb; X=$D/dft_pilot/xyz
SIF=${SIF:-/var/tmp/$USER-pah/sif/D_gpu4pyscf.sif}
NODE=$([ "$GPU" = 1 ] && echo 1 || echo 0)        # GPU 0 sits on NUMA node 0, GPU 1 on node 1
mkdir -p /var/tmp/$USER-pah/cache/cupy
run() {   # run a command in container D on $GPU (extra apptainer options in $EXTRA)
  CUDA_VISIBLE_DEVICES=${CLIENT_CVD:-$GPU} numactl --cpunodebind=$NODE --membind=$NODE \
    apptainer exec --nv ${EXTRA:-} --bind "$REPO":/r --bind /var/tmp/$USER-pah \
    --env CUPY_CACHE_DIR=/var/tmp/$USER-pah/cache/cupy,PYTHONWARNINGS=ignore "$SIF" "$@"
}
VALID=("$P/naph_uid330_q0_freq.gjf naph_q0 0 1" "$P/naph_uid359_q1_freq.gjf naph_q1 1 2"
       "$X/coronene_q0.xyz coro_q0 0 1" "$X/coronene_q1.xyz coro_q1 1 2")
case $STEP in
validate)   # energy, forces, dipole, <S2> against CPU PySCF (h100_match/cpu, made with the dft env)
  for j in "${VALID[@]}"; do set -- $j
    run python /r/d1/pyscf_match.py $1 $D/h100_match/gpu --tag $2 --charge $3 --mult $4 --displace 0.05 --seed 0 \
        --only 99x590_t9g6 --gpu; done ;;
stab)       # stability analysis, GPU (CPU references: stability_check.py in the dft env)
  for j in "${VALID[@]}"; do set -- $j
    run python /r/d1/stability_check.py $1 --charge $3 --mult $4 --gpu --out $D/h100_match/stab/$2_gpu.json; done ;;
series)     # C10-C144 timings, q0 and q1, production setting
  for m in naphthalene coronene ovalene circumcoronene C90H24_618 C144H30_641; do
    for q in 0 1; do
      run python /r/d1/pyscf_match.py $X/${m}_q$q.xyz $D/h100_timing --tag ${m}_q$q --charge $q --mult $((q + 1)) \
          --displace 0.05 --seed 0 --only 99x590_t9g6 --gpu; done; done ;;
floor)      # SCF noise floor: does conv_tol 1e-11 converge for circumcoronene on the H100?
  for q in 0 1; do
    run python /r/d1/pyscf_match.py $X/circumcoronene_q$q.xyz $D/h100_floor --tag circumcoronene_q$q --charge $q \
        --mult $((q + 1)) --displace 0.05 --seed 0 --only 99x590 --gpu; done ;;
batch|batch_mps)   # T2.5: 1, 2, 4 processes per GPU, 8 frames each (batch_mps: under a host MPS daemon, see below)
  OUT=$D/h100_batch/$STEP; mkdir -p $OUT
  if [ $STEP = batch_mps ]; then
    # The image's CUDA 12.9 runs on the host driver (595.45) directly, without forward compatibility, so the host
    # daemon serves its clients (container B needed the daemon inside the image). Short pipe directory: Unix socket
    # paths are limited to 108 bytes.
    export CUDA_MPS_PIPE_DIRECTORY=/dev/shm/mps-$USER/pipe CUDA_MPS_LOG_DIRECTORY=/dev/shm/mps-$USER/log
    mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
    if pgrep -u "$USER" -f "nvidia-cuda-mps-contro[l]" >/dev/null; then echo "an MPS daemon already runs"; exit 1; fi
    CUDA_VISIBLE_DEVICES=$GPU nvidia-cuda-mps-control -d
    trap 'echo quit | nvidia-cuda-mps-control' EXIT
    EXTRA="--env CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
    CLIENT_CVD=0      # a client's CUDA_VISIBLE_DEVICES indexes the devices its MPS daemon sees
  fi
  for m in "naphthalene 0 1" "coronene 0 1" "ovalene 0 1"; do set -- $m
    for np in 1 2 4; do
      for k in $(seq 0 $((np - 1))); do
        run python /r/d1/dft_gpu_batch.py $X/$1_q$2.xyz --charge $2 --mult $3 --n 8 --seed0 $((100 * k)) \
            --out $OUT/$1_q$2_np${np}_p$k.json > $OUT/$1_q$2_np${np}_p$k.log 2>&1 &
      done; wait
      echo "$1 np=$np: $(python3 -c "import json,glob; r=[json.load(open(f)) for f in glob.glob('$OUT/$1_q$2_np${np}_p*.json')]; print(round(sum(x['points_per_hour'] for x in r),1), 'points/h total;', [x['s_per_point'] for x in r], 's/point')")"
    done
  done
  if [ $STEP = batch_mps ]; then
    echo "MPS clients seen: $(grep -ho "NEW CLIENT [0-9]*" $CUDA_MPS_LOG_DIRECTORY/control.log | sort -u | wc -l)"
  fi ;;
*) echo "unknown step $STEP"; exit 1 ;;
esac
