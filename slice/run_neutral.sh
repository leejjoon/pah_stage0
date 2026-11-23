#!/bin/bash
# Neutral symbolic-regression prototype, end to end (also `make -C slice neutral`). Run on the GH200 node:
#   slice/run_neutral.sh [out dir, default /var/tmp/$USER-pah/sr_proto/out] [workers, default 56]
# Stages (each one script, each re-runnable on its own):
#   1 neutral_pool.py   species table from the D1 libraries, the exclusion list, the spin screen and our own spectra
#   2 emission.py       cascade emission per absorbed photon, D1 temperature rules (tdep) and harmonic reference (harm)
#   3 populations.py    2,000 populations (size x photon energy; x cation fraction once charged rows exist)
#   4 measure.py        MIRI MRS model + PAHFIT: main (tdep, harm), clean, out-of-basis continuum, Orion templates
#   5 regress.py        features, random forest, one-ratio baselines, PySR, controls, Orion application
#   6 figures.py
# To add cations: append their rows to the species table (same structure_id, charge 1, status use) and their line
# lists to emission.py's sources; nothing else changes (populations.py then draws f, regress.py takes --targets
# f_cation,logNc,E_eV).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-/var/tmp/$USER-pah/sr_proto/out}; J=${2:-56}
PAH=/var/tmp/$USER-pah; SIF=$PAH/sif/C_cpu.sif; export SCRATCH=$PAH/scratch_c
SPECIES=${SPECIES:-/r/slice/neutral_species_v1.csv}; TARGETS=${TARGETS:-logNc,E_eV}
mkdir -p "$OUT" "$SCRATCH/julia_depot"
[ -d "$SCRATCH/julia_env_C_cpu" ] || apptainer exec "$SIF" cp -r /opt/julia_env "$SCRATCH/julia_env_C_cpu"
C=(apptainer exec --containall --env SCRATCH="$SCRATCH" --env TMPDIR=/cache --env PYTHONUNBUFFERED=1 --env MPLCONFIGDIR=/o
   --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1
   --bind "$SCRATCH" --bind "$SCRATCH/julia_env_C_cpu:/opt/julia_env" --bind "$PAH/data/cache:/cache" --bind "$PAH/data:/data"
   --bind "$REPO:/r" --bind "$OUT:/o" --pwd /o "$SIF")
XML=/data/pahdb-complete-theoretical-v4.00.xml
t0=$(date +%s)
python3 "$REPO/slice/neutral_pool.py"
"${C[@]}" python /r/slice/emission.py "$SPECIES" $XML /o/emission_v1.h5 --jobs "$J"
"${C[@]}" python /r/slice/populations.py "$SPECIES" /o/emission_v1.h5 /o/pop_v1.h5
for run in "tdep main all" "harm main all" "tdep clean 300" "tdep oob 300"; do
  set -- $run
  "${C[@]}" python /r/slice/measure.py /o/pop_v1.h5 "/o/feat_$1_$2.csv" --variant "$1" --mode "$2" --n "$3" --jobs "$J"
done
"${C[@]}" python /r/slice/measure.py orion /o/feat_orion.csv --nmc 30 --jobs "$J"
"${C[@]}" python /r/slice/regress.py /o/pop_v1_labels.csv /o/feat_tdep_main.csv /o/reg_v1 --targets "$TARGETS" --pysr \
  --clean /o/feat_tdep_clean.csv --oob /o/feat_tdep_oob.csv --other /o/feat_harm_main.csv --other-name harm --orion /o/feat_orion.csv
"${C[@]}" python /r/slice/figures.py "$SPECIES" /o/emission_v1.h5 /o/pop_v1.h5 /o/reg_v1 /o/reg_v1/figs
{ echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "repo commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "container C sha256: $(sha256sum "$SIF" | cut -d' ' -f1)"; echo "PAHdb: $(cat "$PAH/data/pahdb-v4.00.sha256" 2>/dev/null | head -1)"
  "${C[@]}" python -c "import pahfit, pysr, sklearn, importlib.metadata as m; print('pahfit', m.version('pahfit'), '| pysr', pysr.__version__, '| scikit-learn', sklearn.__version__)"
  echo "wall time: $(( $(date +%s) - t0 )) s"; } | tee "$OUT/reg_v1/provenance.txt"
