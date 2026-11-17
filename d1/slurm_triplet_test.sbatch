#!/bin/bash
# Size scaling and geometry optimisation test of d1/triplet_freq.py (container D) on one Hangang GH200 node, one species
# per GPU: GPU 0 C60H20 uid 10169 with --opt; GPU 1 C84H24 uid 6512 and GPU 2 C96H26 uid 9390 at the PAHdb geometry;
# GPU 3 C96H26 uid 9392 with --opt (analytic Hessian + IR and optimisation cost vs size). Second version: job 1294335
# (first version) failed on cuTENSOR (fixed with CUPY_ACCELERATORS=cub) and on SCF convergence (fixed in
# triplet_freq.py); the fixes were checked in interactive allocation 1294706. Usage, from the repository root under
# /appsdata/$USER:
#   sbatch d1/slurm_triplet_test.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif /appsdata/$USER/d1/pahdb/pahdb-complete-theoretical-v4.00.xml.gz OUT
# Output: OUT/<uid>.json, OUT/<uid>.log (and OUT/<uid>.json.opt.log for the --opt runs).
#SBATCH --job-name=triplet_test
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=04:00:00
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
  apptainer exec --nv --env CUDA_VISIBLE_DEVICES=$g,CUPY_CACHE_DIR=/cupy,PYTHONWARNINGS=ignore,OMP_NUM_THREADS=16,CUPY_ACCELERATORS=cub \
    --bind "${SLURM_SUBMIT_DIR}/d1":/w:ro --bind "$(dirname "$XML")":/x:ro --bind "$OUT":/out --bind "$CACHE":/cupy "$SIF" \
    python -I /w/triplet_freq.py "/x/$(basename "$XML")" $u /out/$u.json "$@" > "$OUT/$u.log" 2>&1
}
run 0 10169 --opt & p0=$!
run 1 6512 & p1=$!
run 2 9390 & p2=$!
run 3 9392 --opt & p3=$!
rc=0; for p in $p0 $p1 $p2 $p3; do wait $p || rc=1; done
for u in 10169 6512 9390 9392; do echo "===== $u"; grep -vE "^INFO" "$OUT/$u.log" | grep -E "Traceback|Error|^\{|^ \"|^\}" | head -30 || true; done
exit $rc
