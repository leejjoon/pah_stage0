#!/bin/bash
# Container C checks on their own, for a CPU-only machine (e.g. the x86-64 build from C_cpu_x86.def).
# Same checks as the C lines of run_all.sh: PySR (serial deterministic + multiprocessing), PAHFIT packs and a
# closure fit with maxiter 10000, the PAHdb suite import.
# Usage: containers/tests/run_c.sh path/to/C.sif [work_dir]
set -euo pipefail
SIF=$(realpath "${1:?usage: run_c.sh path/to/C.sif [work_dir]}")
WORK=${2:-/var/tmp/$USER-pah/testrun_c}
export SCRATCH=${SCRATCH:-/var/tmp/$USER-pah/scratch_c}
HERE=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$WORK" "$SCRATCH/julia_depot"; cp "$HERE"/c_tools.py "$HERE"/c_pahfit.py "$WORK"/
# juliapkg takes a lock file in its project, so give it a writable copy (one per image)
JENV="$SCRATCH/julia_env_$(basename "$SIF" .sif)"
[ -d "$JENV" ] || apptainer exec "$SIF" cp -r /opt/julia_env "$JENV"
CPU=(apptainer exec --containall --env SCRATCH="$SCRATCH" --bind "$SCRATCH" --bind "$JENV:/opt/julia_env" --bind "$WORK":/t)
ONE=(--env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1)
echo "== arch: $(uname -m), $(nproc) cores, $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ //')"
echo "== C: versions";                                     "${CPU[@]}" "$SIF" python -c "import sys, ssl, json; print(sys.version.split()[0], ssl.OPENSSL_VERSION, 'julia', json.load(open('/opt/julia_env/pyjuliapkg/meta.json'))['version'])"
echo "== C: PySR, PAHFIT packs, PAHdb suite";              "${CPU[@]}" "$SIF" python /t/c_tools.py
echo "== C: PAHFIT closure fit, maxiter 10000, single-threaded"; "${CPU[@]}" "${ONE[@]}" --env MAXIT=10000 "$SIF" python /t/c_pahfit.py
