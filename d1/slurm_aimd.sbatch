#!/bin/bash
# Naphthalene DFT MD reference (B3 part one; d1/aimd_pyscf.py --gpu, container D) on one Hangang GH200 node: one
# trajectory per GPU (two per GPU gain nothing: naphthalene is launch-bound), each pinned to its GPU's cores.
# Usage, from the repository root under /appsdata/$USER (docs/plan/hangang_aimd_handoff.md):
#   sbatch d1/slurm_aimd.sbatch D_gpu4pyscf.sif RUN_DIR naph_q1_300K_r0,naph_q1_300K_r1,naph_q1_900K_r0,naph_q1_900K_r1
# RUN_DIR holds C10H8_330_q{0,1}.xyz and the checkpoints naph_q<q>_<T>K_r<r>.npz (resumed if present). Run names
# give charge q (mult q+1), T and replica r; the seed is T*10 + 2r + q, as in the original launcher (run_all.sh).
# Each process checkpoints and stops 20 min before the time limit (--hours); if every process stopped that way and at
# least one advanced, the job resubmits itself with the same arguments (at most AIMD_RESUBMITS times, default 8).
# Logs: RUN_DIR/logs/<name>.log (appended across jobs).
#SBATCH --job-name=aimd
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --comment="field=aac;appl=misc"
#SBATCH --output=aimd_%j.log
set -euo pipefail
usage="usage: sbatch slurm_aimd.sbatch D_gpu4pyscf.sif RUN_DIR name1,name2,..."
SIF=$(realpath "${1:?$usage}"); RUN=$(realpath "${2:?$usage}"); NAMES=${3:?$usage}
LEFT=${AIMD_RESUBMITS:-8}
CACHE=${CUPY_CACHE:-/appsdata/$USER/cache/cupy}; mkdir -p "$CACHE" "$RUN/logs"
module load apptainer/1.4.5
limit=$(squeue -h -j "$SLURM_JOB_ID" -o %L || true)
hours=$(python3 -c "
import sys; s = sys.argv[1]; d, s = (s.split('-') + [''])[:2] if '-' in s else ('0', s)
p = [int(x) for x in s.split(':')]; p = [0] * (3 - len(p)) + p
h = int(d) * 24 + p[0] + p[1] / 60 + p[2] / 3600; print(round(h - min(1 / 3, 0.15 * h), 3))" "$limit" 2>/dev/null || echo 23.6)
ngpu=$(nvidia-smi -L | wc -l)
topo=$(nvidia-smi topo -m | sed 's/\x1b\[[0-9;]*m//g')
IFS=, read -ra runs <<< "$NAMES"
[ ${#runs[@]} -le "$ngpu" ] || { echo "more runs (${#runs[@]}) than GPUs ($ngpu)"; exit 1; }
echo "host $(hostname)  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)  --hours $hours  resubmits left $LEFT"
# "step total" of a checkpoint (0 0 if none), read with the image's numpy (the host Python may have none)
step_of() { [ -f "$1" ] || { echo "0 0"; return; }
  apptainer exec --bind "$RUN" "$SIF" python -c "import numpy as np, sys; z = np.load(sys.argv[1])
print(int(z['step']), int(round((float(z['nvt_ps']) + float(z['nve_ps'])) * 2000)))" "$1"; }
before=(); pids=(); g=0
for n in "${runs[@]}"; do
  [[ $n =~ ^naph_q([01])_([0-9]+)K_r([0-9]+)$ ]] || { echo "bad run name $n"; exit 1; }
  q=${BASH_REMATCH[1]}; T=${BASH_REMATCH[2]}; r=${BASH_REMATCH[3]}
  read -r s0 _ <<< "$(step_of "$RUN/$n.npz")"; before+=("$s0")
  cpus=$(printf '%s\n' "$topo" | awk -v r="GPU$g" '$1 == r {for (i = 2; i <= NF; i++) if ($i ~ /^[0-9]+(-[0-9]+)?(,[0-9]+(-[0-9]+)?)*$/) {print $i; exit}}')
  if [ -z "$cpus" ]; then per=$(( $(nproc --all) / ngpu )); cpus="$((g * per))-$(((g + 1) * per - 1))"; fi
  echo "$n: GPU $g, cores $cpus, from step ${before[-1]}"
  { echo "=== job $SLURM_JOB_ID $(date '+%F %T') $(hostname) GPU $g"; } >> "$RUN/logs/$n.log"
  taskset -c "$cpus" apptainer exec --nv \
    --env CUDA_VISIBLE_DEVICES=$g,CUPY_CACHE_DIR=/cupy,PYTHONWARNINGS=ignore \
    --bind "${SLURM_SUBMIT_DIR}/d1":/w:ro --bind "$RUN":/md --bind "$CACHE":/cupy "$SIF" \
    python -I /w/aimd_pyscf.py /md/C10H8_330_q$q.xyz /md/$n.npz --charge $q --mult $((q + 1)) --T $T \
      --seed $((T * 10 + r * 2 + q)) --gpu --hours "$hours" >> "$RUN/logs/$n.log" 2>&1 &
  pids+=($!); g=$((g + 1))
done
rcs=(); rc=0; for p in "${pids[@]}"; do if wait $p; then rcs+=(0); else rcs+=(1); rc=1; fi; done
paused=0; done_=0; advanced=0; i=0
for n in "${runs[@]}"; do
  read -r s tot <<< "$(step_of "$RUN/$n.npz")"
  echo "$n: exit ${rcs[$i]}, step ${before[$i]} -> $s of $tot"
  [ "$s" -gt "${before[$i]}" ] && advanced=1
  if [ "$tot" -gt 0 ] && [ "$s" -ge "$tot" ]; then done_=$((done_ + 1)); elif [ "${rcs[$i]}" = 0 ]; then paused=$((paused + 1)); fi
  i=$((i + 1))
done
if [ $rc = 0 ] && [ $paused -gt 0 ] && [ $((paused + done_)) = ${#runs[@]} ] && [ $advanced = 1 ] && [ "$LEFT" -gt 0 ]; then
  cd "$SLURM_SUBMIT_DIR"
  sbatch --export=ALL,AIMD_RESUBMITS=$((LEFT - 1)) d1/slurm_aimd.sbatch "$SIF" "$RUN" "$NAMES"
elif [ $done_ = ${#runs[@]} ]; then
  echo "all runs done"
else
  echo "not resubmitting (rc $rc, paused $paused, done $done_, advanced $advanced, resubmits left $LEFT); check the logs"
fi
exit $rc
