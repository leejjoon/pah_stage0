#!/bin/bash
# Neutral + cation symbolic-regression prototype, version 4 (2026-11-25): preparation run. Two parts, as run_nc_v3.sh:
#   slice/run_nc_v4.sh spectra [out dir, default /var/tmp/$USER-pah/sr_proto/out_v4] [workers, default 48]
#       on the GH200 node, in container C, CPU only and niced (check the node's load first)
#   slice/run_nc_v4.sh regress <dir with the tables of part 1> [out dir, default <dir>/reg_v4]
#       on an x86-64 machine with uv
# Differences from run_nc_v3.sh:
#   widths       d1/perline_widths_v2_clean.csv next to d1/tdep_rules_v1_clean.csv (emission.py --widths)
#   cation rules RULES_CATION=<csv> [WIDTHS_CATION=<csv>] in the environment are passed to emission.py
#                (--rules-cation, --widths-cation): d1/tdep_rules_cation_v1.csv once it exists. Unset: neutral rules.
#   measure      a third local continuum (Lpch, monotone piecewise cubic); a second main table with broad plateau
#                emission added to every spectrum (--plateau, feat_tdep_plat.csv) and a third with band wings
#                (--wings, feat_tdep_wing.csv)
#   Monte Carlo  the 300 test populations measured with 10 more noise draws at S/N 20, 50 and 100 (feat_mc_s*_k*.csv)
#   endpoint     the analytic-endpoint control of v2 (plan 5.9, control 2) with the clean rules and widths
#   regress      three runs: trained without nuisance (base, with formulas for the Lpch sets), with the plateau
#                emission (plat) and with band wings (wing); each is applied to the other
#                table and to the Orion Bar templates; observable sets also with the Lpch and straight continua
#   no B4        the cation-rule variants of v3 are not repeated
# The Orion diagnosis itself is slice/orion_gap.py (run on the v3 emission file; see reports/nc_sr_v4).
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
if [ "${1:-}" = spectra ]; then
  OUT=${2:-/var/tmp/$USER-pah/sr_proto/out_v4}; J=${3:-48}
  PAH=/var/tmp/$USER-pah; SIF=$PAH/sif/C_cpu.sif; export SCRATCH=$PAH/scratch_c
  mkdir -p "$OUT"
  C=(nice -n 10 apptainer exec --containall --env SCRATCH="$SCRATCH" --env TMPDIR=/cache --env PYTHONUNBUFFERED=1 --env MPLCONFIGDIR=/o
     --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 --env MKL_NUM_THREADS=1
     --bind "$SCRATCH" --bind "$PAH/data/cache:/cache" --bind "$PAH/data:/data" --bind "$REPO:/r" --bind "$OUT:/o" --pwd /o "$SIF")
  XML=/data/pahdb-complete-theoretical-v4.00.xml; SP=/r/slice/species_nc_v1.csv; t0=$(date +%s)
  RUL=(--rules /r/d1/tdep_rules_v1_clean.csv --widths /r/d1/perline_widths_v2_clean.csv)
  [ -z "${RULES_CATION:-}" ] || RUL+=(--rules-cation "$RULES_CATION")
  [ -z "${WIDTHS_CATION:-}" ] || RUL+=(--widths-cation "$WIDTHS_CATION")
  [ -n "${SKIP_EMISSION:-}" ] || "${C[@]}" python /r/slice/emission.py $SP $XML /o/emission_v4.h5 --jobs "$J" "${RUL[@]}"
  "${C[@]}" python /r/slice/populations.py $SP /o/emission_v4.h5 /o/pop_v4.h5 --design v3
  M=(python /r/slice/measure.py /o/pop_v4.h5)
  "${C[@]}" "${M[@]}" /o/feat_tdep_main.csv --jobs "$J"
  "${C[@]}" "${M[@]}" /o/feat_tdep_plat.csv --plateau --jobs "$J"
  "${C[@]}" "${M[@]}" /o/feat_tdep_wing.csv --wings --jobs "$J"
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
    for f in tdep_rules_v1_clean.csv perline_widths_v2_clean.csv; do echo "$f $(sha256sum "$REPO/d1/$f" | cut -d' ' -f1)"; done
    echo "cation rules: ${RULES_CATION:-as neutral}; cation widths: ${WIDTHS_CATION:-as neutral}"
    "${C[@]}" python -c "import pahfit, sklearn, importlib.metadata as m; print('pahfit', m.version('pahfit'))"
    echo "wall time: $(( $(date +%s) - t0 )) s"; } | tee "$OUT/provenance_spectra.txt"
elif [ "${1:-}" = regress ]; then
  IN=$2; OUT=${3:-$IN/reg_v4}; mkdir -p "$OUT"
  export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-/var/tmp/$USER-pah/julia_depot_x86} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  export PYTHON_JULIAPKG_PROJECT=${PYTHON_JULIAPKG_PROJECT:-/var/tmp/$USER-pah/juliapkg_x86}
  PY=(uv run -q --python 3.12 --with pysr==2.7.2 --with scikit-learn==1.9.1 --with matplotlib python)
  MC=(); for s in 20 50 100; do MC+=(--mc "s$s=$(ls "$IN"/feat_mc_s${s}_k*.csv | paste -sd,)"); done
  SETS=${SETS:-Pn,P33n,Ln,L33n,Hn,H33n,LLn,LL33n,Tn,T33n,P,L,H}; PYSR=${PYSR:-}
  B=(--targets "${TARGETS:-f_cation,logNc,E_eV}" --clean "$IN/feat_tdep_clean.csv" --ctl "harm=$IN/feat_harm_main.csv" --orion "$IN/feat_orion.csv"
     --procs "${PROCS:-14}" --rf-jobs "${PROCS:-14}" --sets "$SETS")
  # base: trained on the spectra without plateau emission; plat: trained with it. Each is applied to the other table.
  "${PY[@]}" "$REPO/slice/regress_v2.py" "$IN/pop_v4_labels.csv" "$IN/feat_tdep_main.csv" "$OUT/base" "${B[@]}" "${MC[@]}" \
     --ctl "plat=$IN/feat_tdep_plat.csv" --ctl "wing=$IN/feat_tdep_wing.csv" --pysr "${PYSR_BASE:-Hn,H33n}" > "$OUT/base.log" 2>&1 &
  "${PY[@]}" "$REPO/slice/regress_v2.py" "$IN/pop_v4_labels.csv" "$IN/feat_tdep_plat.csv" "$OUT/plat" "${B[@]}" \
     --ctl "noplat=$IN/feat_tdep_main.csv" --pysr "$PYSR" > "$OUT/plat.log" 2>&1 &
  "${PY[@]}" "$REPO/slice/regress_v2.py" "$IN/pop_v4_labels.csv" "$IN/feat_tdep_wing.csv" "$OUT/wing" "${B[@]}" \
     --ctl "nowing=$IN/feat_tdep_main.csv" --pysr "" > "$OUT/wing.log" 2>&1 &
  wait
  { echo "date: $(date -Is)"; echo "host: $(hostname) ($(uname -m))"; echo "repo commit: $(git -C "$REPO" rev-parse HEAD 2>/dev/null || echo unknown)"
    "${PY[@]}" -c "import pysr, sklearn, juliacall; print('pysr', pysr.__version__, '| scikit-learn', sklearn.__version__, '| julia', juliacall.Main.seval('string(VERSION)'))"
  } | tee "$OUT/provenance_regress.txt"
else
  echo "usage: $0 spectra [out] [workers] | regress <tables dir> [out]"; exit 2
fi
