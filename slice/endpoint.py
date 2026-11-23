# Analytic-endpoint control (plan 5.9, control 2): at fixed size distributions the population spectrum is linear in
# the cation fraction f, so a band ratio r = A/B follows r(f) = (A0 + f (A1 - A0)) / (B0 + f (B1 - B0)) and
#   f = (A0 - r B0) / (r (B1 - B0) - (A1 - A0)),
# with A0, A1 (numerator band) and B0, B1 (11.2 um) the band strengths at f = 0 and f = 1 on a common normalisation.
# The control mixes known intermediate f, measures them like any other spectrum and checks that f comes back.
#   python slice/endpoint.py table <out csv>
#       species table (neutral_pool.py columns) of the 25 neutral/cation pairs of slice/species.csv; both charge
#       states take PAHdb v4.00 harmonic line lists. Then, with the usual scripts:
#         emission.py <table> <xml> pairs_emission.h5
#         populations.py <table> pairs_emission.h5 pairs_pop.h5 --fixed 30,50,80@6
#         measure.py pairs_pop.h5 pairs_feat_<variant>_<mode>.csv --variant harm|tdep --mode clean|main
#   python slice/endpoint.py check <labels csv> <out json> <features csv> [more csv ...]
#       per features table, size distribution, method (PAHFIT powers; local spline integrals) and numerator band
#       (6.2, 7.7): endpoint strengths, whether the ratio varies with f, linearity of the strengths in f, and the
#       recovered f at the nine intermediate fractions (rms and largest error). Pass: f recovered, judged by the
#       errors, not by coefficients. With B1 = B0 the inverse is linear in r; the formula above covers it.
# The temperature rules come from neutral MLMD spectra; applying them to cations (variant tdep) is an assumption, so
# the harmonic variant is the reference for this control.
import csv, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PF = {"6.2": ["PAH_6.2"], "7.7": ["PAH_7.7a", "PAH_7.7b", "PAH_7.7c"], "11.2": ["PAH_11.3a", "PAH_11.3b"]}


def table(out):
    cols = ["species", "source", "uid", "structure_id", "formula", "charge", "mult", "nC", "nH", "nN", "cls", "solo", "duo", "trio",
            "quartet", "lib300", "lib600", "lib900", "status"]
    rows = []
    for r in csv.DictReader(open(os.path.join(HERE, "species.csv"))):
        rows.append(dict(species=f"{r['formula']}_v4u{r['uid']}", source="pahdb", uid=r["uid"], structure_id=f"pair{r['pair']}",
                         formula=r["formula"], charge=r["charge"], mult=r["symmetry"].split("-")[0], nC=r["n_c"], nH=r["n_h"], nN=0,
                         cls=r["class"], solo=r["n_solo"], duo=r["n_duo"], trio=r["n_trio"], quartet=r["n_quartet"],
                         lib300=0, lib600=0, lib900=0, status="use"))
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rows ({sum(r['charge'] == '1' for r in rows)} cations) -> {out}")


def check(labels, out, files):
    lab = list(csv.DictReader(open(labels))); f = np.array([float(r["f_cation"]) for r in lab]); mu = np.array([float(r["logNc0"]) for r in lab])
    res = {}
    for path in files:
        rows = list(csv.DictReader(open(path))); assert len(rows) == len(lab)
        sc = np.array([float(r["scale"]) for r in rows]); name = os.path.basename(path).replace(".csv", "")
        S = {"pahfit": {b: sum(np.array([float(r.get(f"P_{n}") or "nan") for r in rows]) for n in ns) / sc for b, ns in PF.items()},
             "local": {b: np.array([float(r[f"Lspl_{b}"]) for r in rows]) / sc for b in PF}}
        res[name] = {}
        for c in np.unique(mu):
            m = mu == c; o = np.argsort(f[m]); ff = f[m][o]; e = {}
            for meth, Sm in S.items():
                Bv = Sm["11.2"][m][o]; B0, B1 = Bv[0], Bv[-1]
                for band in ("6.2", "7.7"):
                    Av = Sm[band][m][o]; A0, A1 = Av[0], Av[-1]; r = Av / Bv
                    with np.errstate(all="ignore"):
                        frec = (A0 - r * B0) / (r * (B1 - B0) - (A1 - A0))
                    mid = (ff > 0) & (ff < 1); d = frec[mid] - ff[mid]
                    linA = np.max(np.abs(Av - ((1 - ff) * A0 + ff * A1))) / max(abs(A0), abs(A1))
                    linB = np.max(np.abs(Bv - ((1 - ff) * B0 + ff * B1))) / max(abs(B0), abs(B1))
                    e[f"{meth}_{band}"] = dict(A0=float(A0), A1=float(A1), B0=float(B0), B1=float(B1), r0=float(r[0]), r1=float(r[-1]),
                                               ratio_varies=bool(abs(r[-1] - r[0]) > 0.05 * max(abs(r[0]), abs(r[-1]))),
                                               monotone=bool(np.all(np.diff(r) > 0) or np.all(np.diff(r) < 0)),
                                               nonlinearity_A=float(linA), nonlinearity_B=float(linB),
                                               f_rms=float(np.sqrt(np.nanmean(d ** 2))), f_max=float(np.nanmax(np.abs(d))),
                                               f_recovered=[round(float(x), 4) for x in frec])
            res[name][f"Nc0={10 ** c:.0f}"] = e
        print(name)
        for c, e in res[name].items():
            for k, v in e.items():
                print(f"  {c:8s} {k:11s} r(f=0) {v['r0']:6.3f}  r(f=1) {v['r1']:6.3f}  monotone {str(v['monotone']):5s}  strengths off linear by "
                      f"{100 * v['nonlinearity_A']:.1f} % / {100 * v['nonlinearity_B']:.1f} %  f error rms {v['f_rms']:.4f}, max {v['f_max']:.4f}")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "table":
        table(sys.argv[2])
    else:
        check(sys.argv[2], sys.argv[3], sys.argv[4:])
