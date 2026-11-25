#!/bin/bash
# Neutral + cation symbolic-regression prototype, version 3 (2026-11-25), in two parts:
#   slice/run_nc_v3.sh spectra [out dir, default /var/tmp/$USER-pah/sr_proto/out_v3] [workers, default 48]
#       on the GH200 node, in container C, CPU only and niced (check the node's load first and lower the worker
#       count if other jobs need the cores): species table, emission, populations, measurements
#   slice/run_nc_v3.sh regress <dir with the tables of part 1> [out dir, default <dir>/reg_v3]
#       on an x86-64 machine with uv (PySR segfaults on the GH200 node; CLAUDE.md): random forests, PySR, controls,
#       Orion templates, Gate B4 step b. Needs pop_v3_labels.csv and feat_*.csv only (about 30 MB).
# Differences from run_neutral_v2.sh:
#   species      slice/species_nc_v1.csv: the 819 neutrals plus every screened PAHdb cation with 26-196 C (nc_pool.py)
#   rules        d1/tdep_rules_v1_clean.csv (385 spin-consistent closed-shell singlets), applied equally to cations
#   emission     --b4: six variants that change the rules of the cations only (cc17, cc30, shp30, shm30, wdp30, wdm30)
#   populations  --design v3: cation fraction f as a hypercube axis; neutrals and cations as two sub-populations
#   measure      orion mode also reads the NIRSpec templates (3.3 um) and removes gas lines before local integration
#   regress_v2   targets f_cation, logNc, E_eV; sets without the 17 um ratio; Orion predictions with Monte Carlo scatter
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
B4="harmq cc17 cc30 shp30 shm30 wdp30 wdm30"     # harmq: cations harmonic, neutrals with the rules (populations.py)
if [ "${1:-}" = spectra ]; then
  OUT=${2:-/var/tmp/$USER-pah/sr_proto/out_v3}; J=${3:-48}
  PAH=/var/tmp/$USER-pah; SIF=$PAH/sif/C_cpu.sif; export SCRATCH=$PAH/scratch_c
  mkdir -p "$OUT"
  C=(nice -n 10 apptainer exec --containall --env SCRATCH="$SCRATCH" --env TMPDIR=/cache --env PYTHONUNBUFFERED=1 --env MPLCONFIGDIR=/o
     --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1
     --bind "$SCRATCH" --bind "$PAH/data/cache:/cache" --bind "$PAH/data:/data" --bind "$REPO:/r" --bind "$OUT:/o" --pwd /o "$SIF")
  XML=/data/pahdb-complete-theoretical-v4.00.xml; SP=/r/slice/species_nc_v1.csv; t0=$(date +%s)
  [ -n "${SKIP_POOL:-}" ] || "${C[@]}" python /r/slice/nc_pool.py $XML $SP
  [ -n "${SKIP_EMISSION:-}" ] || "${C[@]}" python /r/slice/emission.py $SP $XML /o/emission_v3.h5 --jobs "$J" --b4 --rules /r/d1/tdep_rules_v1_clean.csv
  "${C[@]}" python /r/slice/populations.py $SP /o/emission_v3.h5 /o/pop_v3.h5 --design v3
  M=(python /r/slice/measure.py /o/pop_v3.h5)
  "${C[@]}" "${M[@]}" /o/feat_tdep_main.csv --jobs "$J"
  for v in harm $B4; do "${C[@]}" "${M[@]}" "/o/feat_${v}_main.csv" --variant "$v" --pick heldout --jobs "$J"; done
  "${C[@]}" "${M[@]}" /o/feat_tdep_clean.csv --mode clean --pick heldout --jobs "$J"
  "${C[@]}" python /r/slice/measure.py orion /o/feat_orion.csv --nmc 30 --jobs "$J"
  { echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "repo commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "container C sha256: $(sha256sum "$SIF" | cut -d' ' -f1)"; echo "PAHdb: $(cat "$PAH/data/pahdb-v4.00.sha256" 2>/dev/null | head -1)"
    echo "rules: d1/tdep_rules_v1_clean.csv $(sha256sum "$REPO/d1/tdep_rules_v1_clean.csv" | cut -d' ' -f1)"
    "${C[@]}" python -c "import pahfit, sklearn, importlib.metadata as m; print('pahfit', m.version('pahfit'))"
    echo "wall time: $(( $(date +%s) - t0 )) s"; } | tee "$OUT/provenance_spectra.txt"
elif [ "${1:-}" = regress ]; then
  IN=$2; OUT=${3:-$IN/reg_v3}; mkdir -p "$OUT"
  export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/var/tmp/$USER-pah/julia_depot_x86} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  # keep Julia and its project outside uv's throw-away environment, or every `uv run` downloads Julia again (290 MB)
  export PYTHON_JULIAPKG_PROJECT=${PYTHON_JULIAPKG_PROJECT:-/var/tmp/$USER-pah/juliapkg_x86}
  PY=(uv run -q --python 3.12 --with pysr==2.7.2 --with scikit-learn==1.9.1 --with matplotlib python)
  CTL=(); for v in harm $B4; do CTL+=(--ctl "$v=$IN/feat_${v}_main.csv"); done
  A=("$IN/pop_v3_labels.csv" "$IN/feat_tdep_main.csv"); PYSR=${PYSR:-P,P33,L,L33,Ln,L33n}
  B=(--targets "${TARGETS:-f_cation,logNc,E_eV}" --pysr "$PYSR" --clean "$IN/feat_tdep_clean.csv" "${CTL[@]}" --orion "$IN/feat_orion.csv"
     --procs "${PROCS:-14}" --rf-jobs "${PROCS:-14}")
  # two runs side by side (the sets with formulas split between them), then merged
  "${PY[@]}" "$REPO/slice/regress_v2.py" "${A[@]}" "$OUT/part1" "${B[@]}" --sets P,P33,Pn,P33n,LL,LL33,P33x,L33x,T,T33,Tn,T33n,Ln > "$OUT/part1.log" 2>&1 &
  "${PY[@]}" "$REPO/slice/regress_v2.py" "${A[@]}" "$OUT/part2" "${B[@]}" --sets L,L33,L33n > "$OUT/part2.log" 2>&1 &
  wait
  cp "$OUT"/part1/features.csv "$OUT"/part?/pred_*.csv "$OUT"/part?/front_*.csv "$OUT"/
  "${PY[@]}" "$REPO/slice/regress_v2.py" x x "$OUT" --merge "$OUT/part1" "$OUT/part2"
  "${PY[@]}" "$REPO/slice/figures_v3.py" "$OUT" "$OUT/figs"
  { echo "date: $(date -Is)"; echo "host: $(hostname) ($(uname -m))"; echo "repo commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    "${PY[@]}" -c "import pysr, sklearn, juliacall; print('pysr', pysr.__version__, '| scikit-learn', sklearn.__version__, '| julia', juliacall.Main.seval('string(VERSION)'))"
  } | tee "$OUT/provenance_regress.txt"
else
  echo "usage: $0 spectra [out] [workers] | regress <tables dir> [out]"; exit 2
fi
