#!/bin/bash
# Size scaling and geometry optimisation test of d1/triplet_freq.py (container D) on one Hangang GH200 node, one species
# per GPU: GPU 0 C30H14 uid 10770 with --opt (steps, energy change, spectrum after re-optimisation); GPUs 1-3 C60H20 uid
# 10169, C84H24 uid 6512, C96H26 uid 9390 at the PAHdb geometry (analytic Hessian + IR timing vs size). Usage, from the
# repository root under /appsdata/$USER:
#   sbatch d1/slurm_triplet_test.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif /appsdata/$USER/d1/pahdb/pahdb-complete-theoretical-v4.00.xml.gz OUT
# Output: OUT/<uid>.json, OUT/<uid>.log (and OUT/10770.json.opt.log).
#SBATCH --job-name=triplet_test
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --comment="field=aac;appl=misc"
#SBATCH --output=triplet_test_%j.log
set -euo pipefail
usage="usage: sbatch slurm_triplet_test.sbatch D_gpu4pyscf.sif PAHDB_XML_GZ OUT"
SIF=$(realpath "${1:?$usage}"); XML=$(realpath "${2:?$usage}"); mkdir -p "${3:?$usage}"; OUT=$(realpath "$3")
CACHE=/appsdata/$USER/cache/cupy; mkdir -p "$CACHE"
module load apptainer/1.4.5
echo "host $(hostname)  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
run() {  # gpu uid [options]
  local g=$1 u=$2; shift 2
  apptainer exec --nv --env CUDA_VISIBLE_DEVICES=$g,CUPY_CACHE_DIR=/cupy,PYTHONWARNINGS=ignore,OMP_NUM_THREADS=16 \
    --bind "${SLURM_SUBMIT_DIR}/d1":/w:ro --bind "$(dirname "$XML")":/x:ro --bind "$OUT":/out --bind "$CACHE":/cupy "$SIF" \
    python -I /w/triplet_freq.py "/x/$(basename "$XML")" $u /out/$u.json "$@" > "$OUT/$u.log" 2>&1
}
run 0 10770 --opt & p0=$!
run 1 10169 & p1=$!
run 2 6512 & p2=$!
run 3 9390 & p3=$!
rc=0; for p in $p0 $p1 $p2 $p3; do wait $p || rc=1; done
for u in 10770 10169 6512 9390; do echo "===== $u"; grep -vE "^INFO" "$OUT/$u.log" | grep -E "Traceback|Error|^\{|^ \"|^\}" | head -30 || true; done
exit $rc
