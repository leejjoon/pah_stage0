# Empirical extra broadening for the 6.2 and 7.7 um families (docs/plan/d1_hangang_new_species.md, "Per-line widths",
# closure): with the per-line widths of perline_widths_v1.csv, PAHdb sticks reproduce the MLMD width growth of the 3.3
# and 8.6 um features but not of 6.2 (+1.9 vs +5.2 cm^-1 rms, 300 -> 900 K) and 7.7 um (+0.3 vs +3.6). Here an extra
# FWHM slope k (cm^-1 per 100 K, zero at 300 K) is added to the lines of one family and scanned; for each k the median
# over species of (rules growth - MLMD growth) of that family's feature is computed, and k is where it crosses zero.
# Species: 60 random of the 900 K library with a normal PAHdb spectrum (not in pahdb_v4_exclude_nonkekule_v1.csv).
#   python width_term_fit.py <pahdb v4.00 xml.gz> <repo d1 dir> <spectra dir> <out csv>
import csv, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules_closure as rc  # noqa: E402
from profile_fit import complex_metrics  # noqa: E402
from issue_plots import load_sticks  # noqa: E402

FAMILIES = {"6.2": (1540, 1700, (1540, 1660)), "7.7": (1220, 1470, (1220, 1470))}   # line range, feature window
KGRID = np.arange(0, 16.1, 1.0)


def spectrum(nu, I, T, nc, R, W, extra):
    dT = (T - 300) / 100; y = np.zeros_like(rc.GRID)
    for f, a in zip(nu, I):
        fam = next((b for b, lo, hi in rc.FAMILY if lo <= f < hi), None)
        f2 = f + (rc.rule(R, fam, "position", nc) * dT if fam else 0)
        w = next((w0 + sl * dT for lo, hi, w0, sl in W if lo <= f < hi), 4.5 + 0.67 * dT)
        for lo, hi, k in extra:
            if lo <= f < hi:
                w += k * dT
        y += a * (w / 2) ** 2 / ((rc.GRID - f2) ** 2 + (w / 2) ** 2) / (np.pi * w / 2)
    return y


def main(xml, d1, spec, out):
    R = {(r["band"], r["quantity"]): r for r in csv.DictReader(open(f"{d1}/tdep_rules_v1.csv"))}
    W = [(float(r["lo_cm1"]), float(r["hi_cm1"]), float(r["fwhm_300K"]), float(r["fwhm_slope_per_100K"]))
         for r in csv.DictReader(open(f"{d1}/perline_widths_v1.csv"))]
    bad = {r["uid"] for r in csv.DictReader(open(f"{d1}/pahdb_v4_exclude_nonkekule_v1.csv"))}
    lib = [r["species"] for r in csv.DictReader(open(f"{d1}/new900_library_v1.csv")) if r["species"].split("_v4u")[1] not in bad]
    pick = list(np.random.default_rng(3).choice(lib, 60, replace=False)); st = load_sticks(pick)
    md = {}
    for fn in sorted(glob.glob(f"{spec}/profile_fits_v1_c*.csv")):
        for r in csv.DictReader(open(fn)):
            if r["region"] == "complex" and r["species"] in st:
                md.setdefault((r["species"], r["band"], int(r["T"])), []).append(float(r["rmsw"]))
    rows = []; best = {}
    for fam, (lo, hi, (wlo, whi)) in FAMILIES.items():
        g_md = {s: np.mean(md[(s, fam, 900)]) - np.mean(md[(s, fam, 300)]) for s in st}
        w300 = {s: complex_metrics(rc.GRID, spectrum(*st[s], 300, int(s[1:s.index("H")]), R, W, []), wlo, whi)[1] for s in st}
        diffs = []
        for k in KGRID:
            d = [complex_metrics(rc.GRID, spectrum(*st[s], 900, int(s[1:s.index("H")]), R, W, [(lo, hi, k)]), wlo, whi)[1] - w300[s] - g_md[s]
                 for s in st]
            diffs.append(np.median(d))
            rows.append(dict(family=fam, k=k, median_gap=round(float(np.median(d)), 3), p16=round(float(np.percentile(d, 16)), 3),
                             p84=round(float(np.percentile(d, 84)), 3)))
            print(f"{fam} k={k:4.1f}: median (rules - MD) growth {np.median(d):+6.2f} [{np.percentile(d, 16):+.2f}..{np.percentile(d, 84):+.2f}]", flush=True)
        diffs = np.array(diffs); i = np.nonzero(np.diff(np.sign(diffs)))[0]
        best[fam] = float(np.interp(0, diffs[i[0]:i[0] + 2], KGRID[i[0]:i[0] + 2])) if len(i) else None
        print(f"=> {fam}: extra FWHM slope k = {best[fam]} cm^-1 per 100 K" if best[fam] is not None else f"=> {fam}: no zero crossing up to k = {KGRID[-1]}")
    # side effect on 8.6 um with both extras
    ex = [(FAMILIES[f][0], FAMILIES[f][1], best[f]) for f in best if best[f] is not None]
    g = [complex_metrics(rc.GRID, spectrum(*st[s], 900, int(s[1:s.index("H")]), R, W, ex), 1100, 1220)[1]
         - complex_metrics(rc.GRID, spectrum(*st[s], 300, int(s[1:s.index("H")]), R, W, ex), 1100, 1220)[1]
         - (np.mean(md[(s, "8.6", 900)]) - np.mean(md[(s, "8.6", 300)])) for s in st]
    print(f"8.6 um with both extras: median (rules - MD) growth {np.median(g):+.2f}")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


if __name__ == "__main__":
    main(*sys.argv[1:5])
