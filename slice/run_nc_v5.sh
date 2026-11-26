#!/bin/bash
# Neutral + cation symbolic regression, version 5 (2026-11-27): the Stage-0 production run. Two parts, as run_nc_v4.sh:
#   slice/run_nc_v5.sh spectra [out dir, default /var/tmp/$USER-pah/sr_proto/out_v5] [workers, default 48]
#       on the GH200 node, in container C, CPU only and niced (check the node's load first)
#   slice/run_nc_v5.sh regress <dir with the tables of part 1> [out dir, default <dir>/reg_v5]
#       on an x86-64 machine with uv
# Differences from run_nc_v4.sh:
#   cation rules the charged species take d1/tdep_rules_cation_v1.csv and d1/perline_widths_cation_v1.csv (cation MLMD,
#                L = 16 ps); the neutrals keep d1/tdep_rules_v1_clean.csv and d1/perline_widths_v2_clean.csv
#   sensitivity  a second variant of every spectrum, s112: the cation 11.2 um strength slope -0.4 % per 100 K (the
#                L = 5.1 ps value) instead of -4.7 (emission.py --alt); its own feature table (feat_s112_main.csv)
#   regress      two runs: base (trained on the tdep table; formulas for the Lpch and PAHFIT sets, with and without
#                3.3 um) and s112 (trained on the s112 table); each is applied to the other table
#   measure      Lcrv_17: the 17 um complex above a curved continuum (sets Hc, Hc33)
#   not repeated the plateau and wing nuisance tables of v4 (they did not close the Orion gap)
# The Orion continuum checks are slice/orion_cont.py (see reports/nc_sr_v5): run it in container C after part 1 and copy
# its feat_orion_cont.csv next to the tables; part 2 then applies the models to those refits as well.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
if [ "${1:-}" = spectra ]; then
  OUT=${2:-/var/tmp/$USER-pah/sr_proto/out_v5}; J=${3:-48}
  PAH=/var/tmp/$USER-pah; SIF=$PAH/sif/C_cpu.sif; export SCRATCH=$PAH/scratch_c
  mkdir -p "$OUT"
  C=(nice -n 10 apptainer exec --containall --env SCRATCH="$SCRATCH" --env TMPDIR=/cache --env PYTHONUNBUFFERED=1 --env MPLCONFIGDIR=/o
     --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1
     --bind "$SCRATCH" --bind "$PAH/data/cache:/cache" --bind "$PAH/data:/data" --bind "$REPO:/r" --bind "$OUT:/o" --pwd /o "$SIF")
  XML=/data/pahdb-complete-theoretical-v4.00.xml; SP=/r/slice/species_nc_v1.csv; t0=$(date +%s)
  RUL=(--rules /r/d1/tdep_rules_v1_clean.csv --widths /r/d1/perline_widths_v2_clean.csv)
  RULES_CATION=${RULES_CATION:-/r/d1/tdep_rules_cation_v1.csv}; WIDTHS_CATION=${WIDTHS_CATION:-/r/d1/perline_widths_cation_v1.csv}
  RUL+=(--rules-cation "$RULES_CATION" --widths-cation "$WIDTHS_CATION"); ALT=${ALT:-s112:11.2:-0.004}
  [ -n "${SKIP_EMISSION:-}" ] || "${C[@]}" python /r/slice/emission.py $SP $XML /o/emission_v5.h5 --jobs "$J" "${RUL[@]}" --alt "$ALT"
  "${C[@]}" python /r/slice/populations.py $SP /o/emission_v5.h5 /o/pop_v5.h5 --design v3
  M=(python /r/slice/measure.py /o/pop_v5.h5)
  "${C[@]}" "${M[@]}" /o/feat_tdep_main.csv --jobs "$J"
  "${C[@]}" "${M[@]}" /o/feat_s112_main.csv --variant s112 --jobs "$J"
  "${C[@]}" "${M[@]}" /o/feat_harm_main.csv --variant harm --pick heldout --jobs "$J"
  "${C[@]}" "${M[@]}" /o/feat_tdep_clean.csv --mode clean --pick heldout --jobs "$J"
  "${C[@]}" python /r/slice/measure.py orion /o/feat_orion.csv --nmc 30 --jobs "$J"
  for snr in 20 50 100; do for k in 1 2 3 4 5 6 7 8 9 10; do
    "${C[@]}" "${M[@]}" "/o/feat_mc_s${snr}_k${k}.csv" --pick test --seed "$k" --snr "$snr" --jobs "$J" | tail -n 2 | head -n 1
  done; done
  # analytic-endpoint control (plan 5.9, control 2) with the clean rules and widths
  "${C[@]}" python /r/slice/emission.py /r/slice/pairs_species_v1.csv $XML /o/pairs_emission.h5 --jobs 16 "${RUL[@]}"
  "${C[@]}" python /r/slice/populations.py /r/slice/pairs_species_v1.csv /o/pairs_emission.h5 /o/pairs_pop.h5 --fixed 30,50,80@6
  for v in harm tdep; do for m in clean main; do
    "${C[@]}" python /r/slice/measure.py /o/pairs_pop.h5 "/o/pairs_feat_${v}_${m}.csv" --variant $v --mode $m --jobs 33
  done; done
  "${C[@]}" python /r/slice/endpoint.py check /o/pairs_pop_labels.csv /o/endpoint.json /o/pairs_feat_harm_clean.csv \
    /o/pairs_feat_harm_main.csv /o/pairs_feat_tdep_clean.csv /o/pairs_feat_tdep_main.csv
  { echo "date: $(date -Is)"; echo "host: $(hostname)"; echo "repo commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    echo "container C sha256: $(sha256sum "$SIF" | cut -d' ' -f1)"; echo "PAHdb: $(cat "$PAH/data/pahdb-v4.00.sha256" 2>/dev/null | head -1)"
    for f in tdep_rules_v1_clean.csv perline_widths_v2_clean.csv tdep_rules_cation_v1.csv perline_widths_cation_v1.csv; do echo "$f $(sha256sum "$REPO/d1/$f" | cut -d' ' -f1)"; done
    echo "cation rules: $RULES_CATION; cation widths: $WIDTHS_CATION; alternative variants: $ALT"
    "${C[@]}" python -c "import pahfit, sklearn, importlib.metadata as m; print('pahfit', m.version('pahfit'))"
    echo "wall time: $(( $(date +%s) - t0 )) s"; } | tee "$OUT/provenance_spectra.txt"
elif [ "${1:-}" = regress ]; then
  IN=$2; OUT=${3:-$IN/reg_v5}; mkdir -p "$OUT"
  export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/var/tmp/$USER-pah/julia_depot_x86} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export PYTHON_JULIAPKG_PROJECT=${PYTHON_JULIAPKG_PROJECT:-/var/tmp/$USER-pah/juliapkg_x86}
  PY=(uv run -q --python 3.12 --with pysr==2.7.2 --with scikit-learn==1.9.1 --with matplotlib python)
  MC=(); for s in 20 50 100; do MC+=(--mc "s$s=$(ls "$IN"/feat_mc_s${s}_k*.csv | paste -sd,)"); done
  SETS=${SETS:-Pn,P33n,Ln,L33n,Hn,H33n,Hc,Hc33,LLn,LL33n,Tn,T33n,P,L,H}; PYSR=${PYSR:-}
  ORI=$IN/feat_orion.csv
  if [ -f "$IN/feat_orion_cont.csv" ]; then                                         # the refits of slice/orion_cont.py as further "templates"
    ORI=$OUT/feat_orion_all.csv
    python3 -c "
import csv, sys
rows = [r for f in sys.argv[1:3] for r in csv.DictReader(open(f))]; cols = list(dict.fromkeys(c for r in rows for c in r))
w = csv.DictWriter(open(sys.argv[3], 'w', newline=''), fieldnames=cols, restval=''); w.writeheader(); w.writerows(rows)" "$IN/feat_orion.csv" "$IN/feat_orion_cont.csv" "$ORI"
  fi
  B=(--targets "${TARGETS:-f_cation,logNc,E_eV}" --clean "$IN/feat_tdep_clean.csv" --ctl "harm=$IN/feat_harm_main.csv" --orion "$ORI"
     --procs "${PROCS:-14}" --rf-jobs "${PROCS:-14}" --sets "$SETS")
  # base: trained on the tdep table (cation 11.2 um slope of L = 16 ps); s112: trained on the table with the 5.1 ps slope
  "${PY[@]}" "$REPO/slice/regress_v2.py" "$IN/pop_v5_labels.csv" "$IN/feat_tdep_main.csv" "$OUT/base" "${B[@]}" "${MC[@]}" \
     --ctl "s112=$IN/feat_s112_main.csv" --pysr "${PYSR_BASE:-Hn,H33n,Pn,P33n}" > "$OUT/base.log" 2>&1 &
  "${PY[@]}" "$REPO/slice/regress_v2.py" "$IN/pop_v5_labels.csv" "$IN/feat_s112_main.csv" "$OUT/s112" "${B[@]}" \
     --ctl "l16=$IN/feat_tdep_main.csv" --pysr "$PYSR" > "$OUT/s112.log" 2>&1 &
  wait
  { echo "date: $(date -Is)"; echo "host: $(hostname) ($(uname -m))"; echo "repo commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    "${PY[@]}" -c "import pysr, sklearn, juliacall; print('pysr', pysr.__version__, '| scikit-learn', sklearn.__version__, '| julia', juliacall.Main.seval('string(VERSION)'))"
  } | tee "$OUT/provenance_regress.txt"
else
  echo "usage: $0 spectra [out] [workers] | regress <tables dir> [out]"; exit 2
fi
