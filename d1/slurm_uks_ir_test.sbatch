#!/bin/bash
# Test of the analytic open-shell IR intensities (d1/uks_ir.py, d1/uks_ir_test.py) in container D on one Hangang
# GH200 node: three tests in parallel on GPUs 0-2 (closed: naphthalene RKS vs UKS; cation: naphthalene cation vs
# PAHdb; triplet: C30H14 uid 10770 as singlet and triplet). Usage, from the repository root under /appsdata/$USER:
#   sbatch d1/slurm_uks_ir_test.sbatch /appsdata/$USER/sif/D_gpu4pyscf.sif /appsdata/$USER/d1/pahdb/pahdb-complete-theoretical-v4.00.xml.gz OUT
# Output: OUT/<test>.json and OUT/<test>.log.
#SBATCH --job-name=uks_ir
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --time=01:00:00
#SBATCH --comment="field=aac;appl=misc"
#SBATCH --output=uks_ir_%j.log
set -euo pipefail
usage="usage: sbatch slurm_uks_ir_test.sbatch D_gpu4pyscf.sif PAHDB_XML_GZ OUT"
SIF=$(realpath "${1:?$usage}"); XML=$(realpath "${2:?$usage}"); mkdir -p "${3:?$usage}"; OUT=$(realpath "$3")
CACHE=/appsdata/$USER/cache/cupy; mkdir -p "$CACHE"
module load apptainer/1.4.5
echo "host $(hostname)  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)"
pids=(); g=0
for t in closed cation triplet; do
  apptainer exec --nv --env CUDA_VISIBLE_DEVICES=$g,CUPY_CACHE_DIR=/cupy,PYTHONWARNINGS=ignore,OMP_NUM_THREADS=16 \
    --bind "${SLURM_SUBMIT_DIR}/d1":/w:ro --bind "$(dirname "$XML")":/x:ro --bind "$OUT":/out --bind "$CACHE":/cupy "$SIF" \
    python -I /w/uks_ir_test.py "/x/$(basename "$XML")" $t /out/$t.json > "$OUT/$t.log" 2>&1 &
  pids+=($!); g=$((g + 1))
done
rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
for t in closed cation triplet; do echo "===== $t"; grep -vE "^INFO" "$OUT/$t.log" | grep -E "Traceback|Error|^\{|^ |^\}" | head -40 || true; done
exit $rc
