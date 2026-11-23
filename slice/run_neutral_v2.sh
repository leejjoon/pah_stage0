#!/bin/bash
# Neutral symbolic-regression prototype, version 2, end to end. Run on the GH200 node:
#   slice/run_neutral_v2.sh [out dir, default /var/tmp/$USER-pah/sr_proto/out_v2] [workers, default 48]
# Check the node's load first and lower the worker count if other jobs need the cores (the stages are niced).
# Differences from run_neutral.sh (version 1):
#   emission     --clamps: three ways to extend the temperature rules above 900 K (tdep, tdep900, tdeplin)
#   populations  --design v2: edge structure (solo, trio, quartet H shares), N and own fractions and the continuum
#                are hypercube axes, independent of size
#   measure      three continuum shapes, a NIRSpec 3.3 um segment, local-continuum band integrals next to PAHFIT;
#                the control tables hold the validation + test populations (600)
#   regress_v2   observable sets side by side: PAHFIT or local integrals, MIRI only or with 3.3 um
#   endpoint     analytic-endpoint control on the 25 neutral/cation pairs of slice/species.csv
# To add cations to the main run: append their rows to the species table (same structure_id, charge 1, status use)
# and add f_cation to TARGETS; populations.py then draws f, and nothing else changes.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=${1:-/var/tmp/$USER-pah/sr_proto/out_v2}; J=${2:-48}
PAH=/var/tmp/$USER-pah; SIF=$PAH/sif/C_cpu.sif; export SCRATCH=$PAH/scratch_c
SPECIES=${SPECIES:-/r/slice/neutral_species_v1.csv}; TARGETS=${TARGETS:-logNc,E_eV}; PYSR=${PYSR:-P,P33,L,L33}
mkdir -p "$OUT" "$SCRATCH/julia_depot"
[ -d "$SCRATCH/julia_env_C_cpu" ] || apptainer exec "$SIF" cp -r /opt/julia_env "$SCRATCH/julia_env_C_cpu"
C=(nice -n 5 apptainer exec --containall --env SCRATCH="$SCRATCH" --env TMPDIR=/cache --env PYTHONUNBUFFERED=1 --env MPLCONFIGDIR=/o
   --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1 --env JULIA_NUM_THREADS=1
   --bind "$SCRATCH" --bind "$SCRATCH/julia_env_C_cpu:/opt/julia_env" --bind "$PAH/data/cache:/cache" --bind "$PAH/data:/data"
   --bind "$REPO:/r" --bind "$OUT:/o" --pwd /o "$SIF")
XML=/data/pahdb-complete-theoretical-v4.00.xml
t0=$(date +%s)
python3 "$REPO/slice/neutral_pool.py"
"${C[@]}" python /r/slice/emission.py "$SPECIES" $XML /o/emission_v2.h5 --jobs "$J" --clamps
"${C[@]}" python /r/slice/populations.py "$SPECIES" /o/emission_v2.h5 /o/pop_v2.h5 --design v2
M=(python /r/slice/measure.py /o/pop_v2.h5)
"${C[@]}" "${M[@]}" /o/feat_tdep_main.csv --jobs "$J"
for v in harm tdep900 tdeplin; do "${C[@]}" "${M[@]}" "/o/feat_${v}_main.csv" --variant "$v" --pick heldout --jobs "$J"; done
"${C[@]}" "${M[@]}" /o/feat_tdep_clean.csv --mode clean --pick heldout --jobs "$J"
"${C[@]}" python /r/slice/measure.py orion /o/feat_orion.csv --nmc 30 --jobs "$J"
"${C[@]}" python /r/slice/regress_v2.py /o/pop_v2_labels.csv /o/feat_tdep_main.csv /o/reg_v2 --targets "$TARGETS" --pysr "$PYSR" \
  --clean /o/feat_tdep_clean.csv --ctl harm=/o/feat_harm_main.csv --ctl tdep900=/o/feat_tdep900_main.csv \
  --ctl tdeplin=/o/feat_tdeplin_main.csv --orion /o/feat_orion.csv
# analytic-endpoint control (plan 5.9, control 2)
python3 "$REPO/slice/endpoint.py" table "$REPO/slice/pairs_species_v1.csv"
"${C[@]}" python /r/slice/emission.py /r/slice/pairs_species_v1.csv $XML /o/pairs_emission.h5 --jobs 16
"${C[@]}" python /r/slice/populations.py /r/slice/pairs_species_v1.csv /o/pairs_emission.h5 /o/pairs_pop.h5 --fixed 30,50,80@6
for v in harm tdep; do for m in clean main; do
  "${C[@]}" python /r/slice/measure.py /o/pairs_pop.h5 "/o/pairs_feat_${v}_${m}.csv" --variant $v --mode $m --jobs 33
done; done
"${C[@]}" python /r/slice/endpoint.py check /o/pairs_pop_labels.csv /o/reg_v2/endpoint.json /o/pairs_feat_harm_clean.csv \
  /o/pairs_feat_harm_main.csv /o/pairs_feat_tdep_clean.csv /o/pairs_feat_tdep_main.csv
"${C[@]}" python /r/slice/figures_v2.py /o/reg_v2 /o/reg_v2/figs
{ echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "repo commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "container C sha256: $(sha256sum "$SIF" | cut -d' ' -f1)"; echo "PAHdb: $(cat "$PAH/data/pahdb-v4.00.sha256" 2>/dev/null | head -1)"
  "${C[@]}" python -c "import pahfit, pysr, sklearn, importlib.metadata as m; print('pahfit', m.version('pahfit'), '| pysr', pysr.__version__, '| scikit-learn', sklearn.__version__)"
  echo "wall time: $(( $(date +%s) - t0 )) s"; } | tee "$OUT/reg_v2/provenance.txt"
