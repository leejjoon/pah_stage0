# Temperature dependence of the bands in the Hangang neutral MLMD libraries (300/600/900 K), and whether it is linear
# enough to extrapolate to 1200 K (docs/plan/d1_hangang_new_species.md, "Band trends").
#   python tdep_fit.py <library csv> <spectra dir 300> <dir 600> <dir 900> <out csv>
# Spectra: mlmd_spectra.py npz (production convention: qm=3, unclipped, each replica's NVE mean T in prefactor_T).
# Bands as in tdep_compare.py: the 6 strongest peaks of the 300 K replica-mean spectrum, followed to 600 and 900 K
# (maximum of the replica-mean spectrum within -30/+12 cm-1 of the previous position). Per replica and band, in a
# +-20 cm-1 window around the band's position at that T: intensity-weighted centroid and rms width (non-negative
# part), and the window's share of the 300-3800 cm-1 total relative to the 300 K replica mean. Each quantity is
# fitted against the replicas' NVE mean T (12 points) with a line and with a quadratic. Per band: slope per 100 K,
# curvature and its significance, the 900 K residual of the 300-600 K line, and the 1200 K values of both fits.
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tdep_compare import peaks_300, follow, HALF

FAMILIES = [("CH oop", 600, 1000), ("CC/CH in-plane", 1000, 1700), ("CH stretch", 2800, 3300)]


def window_metrics(f, s, fc):
    m = (f >= fc - HALF) & (f <= fc + HALF); x = f[m]; w = np.clip(s[m], 0, None)
    c = (w * x).sum() / w.sum()
    return c, np.sqrt((w * (x - c) ** 2).sum() / w.sum()), s[m].sum() / s.sum()


def fits(x, y):
    """Line and quadratic in t = (T - 600)/100; returns slope, its s.e., curvature (per (100 K)^2), its s.e.,
    1200 K value of each fit."""
    t = (np.asarray(x) - 600.0) / 100.0; y = np.asarray(y)
    out = []
    for deg in (1, 2):
        A = np.vstack([t ** k for k in range(deg + 1)]).T
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        r = y - A @ c; s2 = (r ** 2).sum() / max(1, len(y) - deg - 1)
        cov = s2 * np.linalg.inv(A.T @ A)
        out.append((c, np.sqrt(np.diag(cov)), np.polyval(c[::-1], 6.0)))
    (c1, e1, y1), (c2, e2, y2) = out
    return c1[1], e1[1], c2[2], e2[2], y1, y2


def main():
    lib, d300, d600, d900, out_csv = sys.argv[1:6]
    species = [r["species"] for r in csv.DictReader(open(lib))]
    rows = []
    for sp in species:
        S = {}
        for T, d in ((300, d300), (600, d600), (900, d900)):
            z = np.load(f"{d}/{sp}_{T}K.npz")
            S[T] = [(z["freq"], z["sigma"][i], float(z["prefactor_T"][i])) for i in range(len(z["replicas"]))]
        nc = int(sp[1:sp.index("H")])
        pk = peaks_300([(f, s) for f, s, _ in S[300]])
        for label, f0 in pk.items():
            pos = {300: f0}
            pos[600] = follow([(f, s) for f, s, _ in S[600]], pos[300])
            pos[900] = follow([(f, s) for f, s, _ in S[900]], pos[600])
            pts = [(Tm, *window_metrics(f, s, pos[T])) for T in (300, 600, 900) for f, s, Tm in S[T]]
            x = [p[0] for p in pts]
            share0 = np.mean([p[3] for p, (T) in zip(pts, [300] * 4 + [600] * 4 + [900] * 4) if T == 300])
            fam = next((n for n, lo, hi in FAMILIES if lo <= f0 < hi), "other")
            row = dict(species=sp, nC=nc, band=round(float(f0), 1), family=fam, T_nve_min=round(min(x)), T_nve_max=round(max(x)))
            for q, i in (("centroid", 1), ("width", 2), ("share", 3)):
                y = [p[i] / share0 if q == "share" else p[i] for p in pts]
                sl, se, cu, ce, y1, y2 = fits(x, y)
                # 900 K check: line through the 300 and 600 K replicas, residual of the 900 K replica mean
                lo = [(a, b) for a, b in zip(x, y) if a < 750]; hi = [b for a, b in zip(x, y) if a >= 750]
                A = np.vstack([np.array([a for a, _ in lo]), np.ones(len(lo))]).T
                c, *_ = np.linalg.lstsq(A, np.array([b for _, b in lo]), rcond=None)
                t9 = np.mean([a for a in x if a >= 750])
                row.update({f"{q}_slope": sl, f"{q}_slope_se": se, f"{q}_curv": cu, f"{q}_curv_se": ce,
                            f"{q}_900_resid": np.mean(hi) - (c[0] * t9 + c[1]),
                            f"{q}_900_se": np.std(hi, ddof=1) / np.sqrt(len(hi)),
                            f"{q}_1200_lin": y1, f"{q}_1200_quad": y2})
            rows.append(row)
    with open(out_csv, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(species)} species, {len(rows)} bands -> {out_csv}")
    summarise(rows)


def summarise(rows):
    print("Per family: median slope per 100 K (centroid, width in cm-1; share as a fraction of its 300 K value);"
          " fraction of bands with |curvature| > 2 s.e.; median |900 K residual of the 300-600 K line| / its s.e.;"
          " median |quadratic - linear| at 1200 K")
    for fam in [f[0] for f in FAMILIES]:
        R = [r for r in rows if r["family"] == fam]
        if not R:
            continue
        print(f"\n{fam}: {len(R)} bands")
        for q in ("centroid", "width", "share"):
            sl = np.array([r[f"{q}_slope"] for r in R])
            curved = np.mean([abs(r[f"{q}_curv"]) > 2 * r[f"{q}_curv_se"] for r in R])
            z9 = np.median([abs(r[f"{q}_900_resid"]) / r[f"{q}_900_se"] for r in R if r[f"{q}_900_se"] > 0])
            dx = np.median([abs(r[f"{q}_1200_quad"] - r[f"{q}_1200_lin"]) for r in R])
            sign = np.median([r[f"{q}_curv"] for r in R])
            print(f"   {q:8s} slope {np.median(sl):+.3f} [{np.percentile(sl, 10):+.3f}..{np.percentile(sl, 90):+.3f}]  "
                  f"curved {curved:.0%} (median curvature {sign:+.3f})  900K-resid/se {z9:.1f}  |quad-lin| at 1200 K {dx:.3f}")
    print("\nCentroid slope (cm-1 per 100 K) by Nc, per family (median):")
    for fam in [f[0] for f in FAMILIES]:
        cells = []
        for lo, hi in ((0, 40), (40, 60), (60, 90), (90, 130), (130, 999)):
            v = [r["centroid_slope"] for r in rows if r["family"] == fam and lo <= r["nC"] < hi]
            if v:
                cells.append(f"Nc {lo}-{hi}: {np.median(v):+.2f} (n={len(v)})")
        print(f"   {fam:15s} " + "; ".join(cells))


if __name__ == "__main__":
    main()
