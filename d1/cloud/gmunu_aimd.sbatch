#!/bin/bash
# Naphthalene DFT MD reference (B3 part one) on one gmunu (KASI) CPU node: launch_aimd.sh `run` for a set of runs,
# THREADS cores each (32 is where a run stops getting faster on these EPYC nodes; 4 runs fill a 128-core node).
# Handoff: docs/plan/cloud_aimd_handoff.md ("gmunu"). Submit from ~/aimd/run (Slurm output goes to logs/):
#   cd ~/aimd/run && AIMD_SBATCH_ARGS="-J aimd-q0 --constraint=AMD-R6525-7702" \
#     sbatch $AIMD_SBATCH_ARGS ~/aimd/code/cloud/gmunu_aimd.sbatch ~/aimd/run naph_q0_300K_r0,naph_q0_300K_r1,... 32
# Restarts (the runs always resume from their checkpoints; at most one step, or 20 fs after a crash, is lost):
#   - time limit: USR1 5 min before it -> every run saves after its current step -> the job resubmits itself with the
#     same arguments and AIMD_SBATCH_ARGS (at most AIMD_RESUBMITS times, default 20; 3 days each).
#   - node failure: --requeue; Slurm restarts this script, which resumes. If a dead node's lock is still held, the
#     launcher skips that run and the job resubmits itself 10 minutes later.
#   - scancel: every run saves and the job does NOT resubmit (sbatch the same command again to continue).
#   - a run crashed (launcher exit 1): no resubmit; read RUN_DIR/logs/<run>.log.
# The same run can never go on two nodes at once: launch_aimd.sh holds a flock per run (works across gmunu nodes).
#SBATCH --job-name=aimd
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@300
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=logs/slurm_%x_%j.out
set -uo pipefail
usage="usage: sbatch gmunu_aimd.sbatch RUN_DIR name1,name2,... [THREADS=32]"
RUN=$(realpath "${1:?$usage}"); NAMES=${2:?$usage}; TH=${3:-32}
LEFT=${AIMD_RESUBMITS:-20}
SELF=${AIMD_SBATCH:-$HOME/aimd/code/cloud/gmunu_aimd.sbatch}
echo "=== job $SLURM_JOB_ID ${SLURM_JOB_NAME} restart ${SLURM_RESTART_COUNT:-0} $(date '+%F %T') $(hostname)" \
  "runs $NAMES, $TH threads, resubmits left $LEFT, extra sbatch args: ${AIMD_SBATCH_ARGS:-none}"
why=""
trap 'why=timelimit; kill -TERM "$lp" 2>/dev/null' USR1
trap 'why=${why:-cancel}; kill -TERM "$lp" 2>/dev/null' TERM INT
RUNS=$NAMES THREADS_Q0=$TH THREADS_Q1=$TH AIMD_TAG=gmunu-$(hostname -s) \
  "$(dirname "$SELF")/launch_aimd.sh" run "$RUN" &
lp=$!
while :; do wait "$lp"; rc=$?; kill -0 "$lp" 2>/dev/null || break; done   # a trap ends `wait` early: wait again
echo "$(date '+%F %T') launcher exit $rc${why:+ ($why)}"
case $rc in
  0) echo "all runs in this job done" ;;
  3) if [ "$why" = cancel ]; then echo "cancelled: not resubmitting (sbatch the same command again to continue)"
     elif [ "$LEFT" -le 0 ]; then echo "no resubmits left: sbatch the same command again to continue"
     else begin=""; [ -z "$why" ] && begin="--begin=now+600"   # stopped without a signal: e.g. a stale lock
       cd "$SLURM_SUBMIT_DIR" && sbatch ${AIMD_SBATCH_ARGS:-} $begin --export=ALL,AIMD_RESUBMITS=$((LEFT - 1)) \
         "$SELF" "$RUN" "$NAMES" "$TH"
     fi ;;
  *) echo "a run crashed: not resubmitting; read $RUN/logs/" ;;
esac
exit "$rc"
