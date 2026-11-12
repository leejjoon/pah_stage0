#!/bin/bash
# Gaussian 16 single points for charged-state MLFF training data on KISTI Hangang CPU nodes (normal_cpu: 256 CPUs and
# ~770 GB per node, 24 h, nodes shared between jobs). docs/plan/charged_mlff_plan.md.
#   ls $PWD/gjf/*.gjf > list.txt
#   N=$(( ($(wc -l < list.txt) + K - 1) / K ))          # K inputs per array task
#   sbatch --array=0-$((N-1))%50 --cpus-per-task=8 --mem=20G d1/slurm_g16.sbatch list.txt K
# Each task runs inputs [task*K, task*K+K) one after another, in each input's own directory (the .log lands next to
# the .gjf), and skips inputs whose .log already ends in "Normal termination": resubmitting the same array resumes.
# --cpus-per-task must equal %nprocshared in the inputs (g16_io.py write --nproc), and --mem must exceed %mem by a
# few GB. Use 8 CPUs for species up to ~60 C, 32 for larger ones (dft_pilot thread scaling).
# CHECK BEFORE FIRST USE: the --comment application code (`showappl | grep -i gauss`).
#SBATCH --job-name=g16
#SBATCH --partition=normal_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=24:00:00
#SBATCH --comment="field=aac;appl=gaussian"
#SBATCH --output=g16_%A_%a.log
set -u
LIST=$1; K=${2:-20}
module load gaussian/g16.c02
T=${SLURM_ARRAY_TASK_ID:-0}
SCR=/scratch/$USER/g16scr/${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$T
mkdir -p "$SCR"; export GAUSS_SCRDIR=$SCR
trap 'rm -rf "$SCR"' EXIT
echo "$(date '+%F %T') task $T on $(hostname): inputs $((T * K))..$((T * K + K - 1)) of $LIST, ${SLURM_CPUS_PER_TASK:-?} CPUs"
sed -n "$((T * K + 1)),$((T * K + K))p" "$LIST" | while read -r gjf; do
  [ -z "$gjf" ] && continue
  log=${gjf%.gjf}.log
  if [ -f "$log" ] && tail -n 5 "$log" | grep -q "Normal termination"; then continue; fi
  t0=$(date +%s)
  ( cd "$(dirname "$gjf")" && g16 < "$(basename "$gjf")" > "$(basename "$log")" 2>&1 )
  st=$?; rm -f "$SCR"/*
  echo "$(date '+%T') $(basename "$gjf") exit $st $(( $(date +%s) - t0 )) s $(tail -n 1 "$log" | cut -c1-60)"
done
echo "$(date '+%F %T') task $T done"
