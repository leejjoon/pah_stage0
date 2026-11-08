#!/bin/bash
# Re-run the GH200 feasibility checks (docs/plan/GH200_feasibility_check_2026-11-07.md) against the built images.
# Usage: containers/tests/run_all.sh [sif_dir] [work_dir]
# Scripts are copied to work_dir (local NVMe) and bound as /t; outputs land there, not in the repo.
set -euo pipefail
SIF=${1:-/var/tmp/jjlee-pah/sif}
WORK=${2:-/var/tmp/$USER-pah/testrun}
export SCRATCH=${SCRATCH:-/var/tmp/$USER-pah/scratch}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK" "$SCRATCH/julia_depot"; cp "$HERE"/* "$WORK"/
PIN=(numactl --cpunodebind=0 --membind=0)
export CUDA_VISIBLE_DEVICES=0
GPU=(apptainer exec --nv --env CUDA_VISIBLE_DEVICES=0 --bind "$WORK":/t)
[ -d "$SCRATCH/julia_env" ] || apptainer exec "$SIF/C_cpu.sif" cp -r /opt/julia_env "$SCRATCH/julia_env"
CPU=(apptainer exec --containall --env SCRATCH="$SCRATCH" --bind "$SCRATCH" --bind "$SCRATCH/julia_env:/opt/julia_env" --bind "$WORK":/t)
ONE=(--env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1)

echo "== B: MACE-OFF small, float32, cueq off/on (writes torch forces)";  "${PIN[@]}" "${GPU[@]}" "$SIF/B_torch.sif" python /t/b_torch.py
echo "== A: JAX GPU, convert, predict, jax-md";                         "${PIN[@]}" "${GPU[@]}" "$SIF/A_jax.sif" bash /t/a_jax.sh
echo "== force gate jax vs torch (tolerance 1e-3 eV/A)";                 "${GPU[@]}" "$SIF/A_jax.sif" python /t/compare.py
echo "== B+TorchSim: batched forces vs ASE";                             "${PIN[@]}" "${GPU[@]}" "$SIF/B_torchsim.sif" python /t/b_torchsim.py
echo "== B+TorchSim: 1 ps, 100-molecule pilot (~5 min)";                 "${PIN[@]}" "${GPU[@]}" "$SIF/B_torchsim.sif" python /t/b_torchsim_pilot.py
echo "== B+TorchSim: force-call timing and profile";                     "${PIN[@]}" "${GPU[@]}" "$SIF/B_torchsim.sif" python /t/b_torchsim_fwd.py
echo "== C: PySR, PAHFIT packs, PAHdb suite";                            "${PIN[@]}" "${CPU[@]}" "$SIF/C_cpu.sif" python /t/c_tools.py
echo "== C: PAHFIT closure fit, maxiter 10000, single-threaded";         "${PIN[@]}" "${CPU[@]}" "${ONE[@]}" --env MAXIT=10000 "$SIF/C_cpu.sif" python /t/c_pahfit.py
PAHDB=${PAHDB:-/var/tmp/$USER-pah/data/pahdb-complete-theoretical-v4.00.xml}
echo "== 20-species equivalence (plan 4.2) on PAHdb v4.00 geometries (falls back to idealised lattice PAHs if PAHDB is missing)"
if [ -f "$PAHDB" ]; then
  mkdir -p "$SCRATCH/pahdb_cache"
  "${PIN[@]}" apptainer exec --containall --env TMPDIR=/cache --bind "$SCRATCH/pahdb_cache":/cache --bind "$(dirname "$PAHDB")":/data --bind "$WORK":/t \
    "$SIF/C_cpu.sif" python /t/pahdb_candidates.py "/data/$(basename "$PAHDB")" /t/pahdb_candidates.json
  apptainer exec --bind "$WORK":/t "$SIF/B_torch.sif" python /t/pahdb_select.py /t/pahdb_candidates.json
else
  apptainer exec --bind "$WORK":/t "$SIF/B_torch.sif" python /t/eq_species.py
fi
# Plan 4.2 gates (revised 2026-11-08): float32 forces and interaction energies, float64 totals from the x64 bundle.
"${PIN[@]}" "${GPU[@]}" "$SIF/B_torch.sif" python /t/eq_torch_gates.py
for p in f32:float32 x64:float64; do for cq in "" _cueq _cueqf; do "${PIN[@]}" "${GPU[@]}" --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env JAX_DEFAULT_MATMUL_PRECISION=highest \
  "$SIF/A_jax.sif" python /t/jax_predict_logged.py /t/off_small_${p%%:*}$cq.npz /t/eq_frames.xyz --dtype ${p##*:} --full-outputs --no-progress \
  --output /t/jax_${p##*:}$cq.npz 2>&1 | grep -E "cuex|x64|rror"; done; done
apptainer exec --bind "$WORK":/t "$SIF/B_torch.sif" python /t/eq_gates.py
echo "== JAX force-call timing per bundle variant (float32; quick check, not the T0 benchmark)"
for n in 100 1; do for v in "" _cueq _cueqf; do "${PIN[@]}" "${GPU[@]}" --env XLA_PYTHON_CLIENT_PREALLOCATE=false --env JAX_DEFAULT_MATMUL_PRECISION=highest \
  "$SIF/A_jax.sif" python /t/jax_time.py /t/off_small_f32$v.npz float32 $n 30 2>&1 | grep "call median"; done; done
