#!/bin/bash
# Naphthalene DFT MD reference (B3 part one) on a cloud VM's CPUs: conda-forge PySCF (env from dft_env_*.lock),
# one process per trajectory, each pinned to its own block of cores. Replaces the project node's run_all.sh, which
# points at a checkout without the Newton fix (64b3aba); never reuse that one. Handoff: docs/plan/cloud_aimd_handoff.md
#   launch_aimd.sh run    RUN_DIR   resume every unfinished trajectory in RUN_DIR; exit 0 only when all are done
#   launch_aimd.sh timing RUN_DIR   20 MD steps per trajectory, all at once with the production pinning (s/step, peak RSS)
#   launch_aimd.sh status RUN_DIR   step reached, engines, last QA record per trajectory
# RUN_DIR holds C10H8_330_q{0,1}.xyz and naph_q<q>_<T>K_r<r>.npz. Environment:
#   RUNS       comma-separated run names (default: all 8 that have an npz or are in the standard matrix)
#   THREADS_Q0 / THREADS_Q1   cores per neutral / cation run (default 8 / 8). Only unfinished runs get cores, so
#              once the neutrals are done the cations can be restarted with more threads (results unaffected)
#   CKPT_FS    checkpoint interval in fs (default 20: 40 steps, 6-9 min). SIGTERM also saves, after the current step.
#   AIMD_TAG   engine tag recorded in the npz `engines` field (default cloud-<arch>)
#   AIMD_PY    python of the env (default $HOME/aimd/env/bin/python); AIMD_CODE: dir of aimd_pyscf.py (default: ..)
# Each run gets whole physical cores (one logical CPU per core, from lscpu), whatever the SMT numbering. On x86 VMs
# still prefer one thread per core (GCP --threads-per-core=1): the idle siblings are otherwise paid for.
# Exit: 0 all done, 3 some paused or stopped (rerun to continue), 1 a run crashed (read RUN_DIR/logs/<run>.log).
set -uo pipefail
mode=${1:?usage: launch_aimd.sh run|timing|status RUN_DIR}; RUN=$(realpath "${2:?RUN_DIR}")
HERE=$(cd "$(dirname "$0")" && pwd); CODE=${AIMD_CODE:-$(dirname "$HERE")}
PY=${AIMD_PY:-$HOME/aimd/env/bin/python}; CK=${CKPT_FS:-20}
T0=${THREADS_Q0:-8}; T1=${THREADS_Q1:-8}
[ -x "$PY" ] || { echo "no python at $PY (set AIMD_PY)"; exit 1; }
[ -f "$CODE/aimd_pyscf.py" ] || { echo "no aimd_pyscf.py in $CODE (set AIMD_CODE)"; exit 1; }
if [ -n "${RUNS:-}" ]; then IFS=, read -ra runs <<< "$RUNS"; else
  runs=(); for q in 1 0; do for T in 300 900; do for r in 0 1; do runs+=("naph_q${q}_${T}K_r$r"); done; done; done; fi
mkdir -p "$RUN/logs"

# "step total" of a checkpoint (0 0 if none)
step_of() { [ -f "$1" ] || { echo "0 0"; return; }
  "$PY" -c "import numpy as np, sys; z = np.load(sys.argv[1])
print(int(z['step']), int(round((float(z['nvt_ps']) + float(z['nve_ps'])) * 2000)))" "$1"; }

if [ "$mode" = status ]; then
  "$PY" - "$RUN" "${runs[@]}" <<'EOF'
import sys, os, numpy as np
d = sys.argv[1]
print(f"{'run':18s} {'step':>6s} {'of':>6s}  {'%':>5s}  last QA (step, T/K, Etot/eV, <S2>, SCF cycles)  engines")
for n in sys.argv[2:]:
    p = os.path.join(d, n + ".npz")
    if not os.path.exists(p): print(f"{n:18s} no checkpoint"); continue
    z = np.load(p); s = int(z["step"]); tot = int(round((float(z["nvt_ps"]) + float(z["nve_ps"])) * 2000))
    qa = np.array(z["qa"]); last = np.round(qa[-1], 3).tolist() if len(qa) else "-"
    eng = z["engines"] if "engines" in z.files else "-"   # checkpoints written before the field existed have none
    print(f"{n:18s} {s:6d} {tot:6d}  {100 * s / tot:5.1f}  {last}  {eng}")
EOF
  exit 0
fi

if [ "$mode" = run ]; then   # finished runs need no cores
  todo=(); for n in "${runs[@]}"; do read -r s tot <<< "$(step_of "$RUN/$n.npz")"
    if [ "$tot" -gt 0 ] && [ "$s" -ge "$tot" ]; then echo "$n: done ($s of $tot)"; else todo+=("$n"); fi; done
  all=("${runs[@]}"); runs=("${todo[@]}")
  [ ${#runs[@]} = 0 ] && { echo "all runs done"; touch "$RUN/ALL_DONE"; exit 0; }
fi
# one logical CPU per physical core, in core order: SMT siblings are numbered 0,1 on some machines (Ryzen) and
# 0,N on others, so "cores 0-7" would put 8 threads on 4 cores on the former
mapfile -t pcpu < <(lscpu -p=CPU,CORE,SOCKET | grep -v '^#' | sort -t, -k3,3n -k2,2n -k1,1n | awk -F, '!seen[$3","$2]++ {print $1}')
ncpu=${#pcpu[@]}; need=0
for n in "${runs[@]}"; do [[ $n == naph_q1_* ]] && need=$((need + T1)) || need=$((need + T0)); done
[ $need -le "$ncpu" ] || { echo "runs need $need cores, this machine has $ncpu"; exit 1; }
echo "$(date '+%F %T') $(hostname) $mode: ${#runs[@]} runs, $need of $ncpu cores, q0 $T0 / q1 $T1 threads, ckpt ${CK} fs"

pids=(); names=(); c=0
for n in "${runs[@]}"; do
  [[ $n =~ ^naph_q([01])_([0-9]+)K_r([0-9]+)$ ]] || { echo "bad run name $n"; exit 1; }
  q=${BASH_REMATCH[1]}; T=${BASH_REMATCH[2]}; r=${BASH_REMATCH[3]}
  th=$([ "$q" = 1 ] && echo "$T1" || echo "$T0"); cpus=$(IFS=,; echo "${pcpu[*]:c:th}"); c=$((c + th))
  args=("$RUN/C10H8_330_q$q.xyz")
  if [ "$mode" = timing ]; then
    args+=(- --time-steps 20)
  else
    read -r s tot <<< "$(step_of "$RUN/$n.npz")"
    args+=("$RUN/$n.npz" --ckpt-fs "$CK" --tag "${AIMD_TAG:-cloud-$(uname -m)}")
  fi
  args+=(--charge "$q" --mult $((q + 1)) --T "$T" --seed $((T * 10 + r * 2 + q)) --threads "$th")
  log=$RUN/logs/$n.log; [ "$mode" = timing ] && log=$RUN/logs/timing_$n.log
  echo "$n: cores $cpus${s:+, from step $s}"
  echo "=== $mode $(date '+%F %T') $(hostname) cores $cpus" >> "$log"
  # flock: a second launcher (a manual start while the service runs) skips runs that are already going
  OMP_NUM_THREADS=$th OPENBLAS_NUM_THREADS=$th PYTHONWARNINGS=ignore \
    flock -n -E 75 "$RUN/.$n.lock" taskset -c "$cpus" \
    "$PY" -I "$CODE/aimd_pyscf.py" "${args[@]}" >> "$log" 2>&1 &
  pids+=($!); names+=("$n")
done
# pass a stop on to the Python processes themselves (flock and time would just die); each saves after its step
trap 'pkill -TERM -f "^$PY -I $CODE/aimd_pyscf.py $RUN/" ; echo "stop requested: runs save after their current step"' TERM INT
rc=0; i=0
for p in "${pids[@]}"; do
  wait "$p"; e=$?; [ $e = 143 ] && { wait "$p"; e=$?; }   # wait returns early when the trap fires
  [ $e = 75 ] && echo "${names[$i]}: already running elsewhere, skipped"
  [ $e != 0 ] && [ $e != 75 ] && { echo "${names[$i]}: exit $e"; rc=1; }
  i=$((i + 1))
done
if [ "$mode" = timing ]; then
  for n in "${names[@]}"; do
    printf '%-18s %s\n' "$n" "$(grep 's per step' "$RUN/logs/timing_$n.log" | tail -1 | sed 's/.*q=[01]: //')"
  done
  exit $rc
fi
left=0
for n in "${all[@]}"; do read -r s tot <<< "$(step_of "$RUN/$n.npz")"; echo "$n: step $s of $tot"
  { [ "$tot" -gt 0 ] && [ "$s" -ge "$tot" ]; } || left=$((left + 1)); done
[ $rc = 1 ] && { echo "a run crashed: read $RUN/logs/"; exit 1; }
[ $left = 0 ] && { echo "all runs done"; touch "$RUN/ALL_DONE"; exit 0; }
echo "$left runs unfinished; rerun to continue"; exit 3
