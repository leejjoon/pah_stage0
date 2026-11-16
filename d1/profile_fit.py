# Profile fits of the bands in the Hangang neutral MLMD libraries at 300/600/900 K, for band position, width and
# strength against temperature (docs/plan/d1_hangang_new_species.md, "Band profiles").
#   python profile_fit.py fit <library csv> <spectra dir 300> <dir 600> <dir 900> <out csv> [--n N]
#   python profile_fit.py summary <fits csv>
# Profile: the exact spectrum of a Lorentzian ACF truncated at the ACF length L the spectra use (7.987 % of the NVE
# run; profile_calib.py: recovers FWHM down to 0.5 cm^-1, where a plain Drude fails below ~1.5), one component per
# band plus a linear baseline per region (its integral is the "pedestal", kept apart from the bands). Components:
# peaks of the 300 K replica-mean spectrum (prominence >= 3 % of the region maximum, up to MAXC per region),
# refitted at 600 and 900 K starting from the previous temperature; then each replica is fitted from the replica-mean
# fit at its T, against its own NVE mean T. Strength: the component's Lorentzian area, and the window integral of the
# data over +-max(15, 1.5 FWHM) around the centre (the integral the B1 estimator is validated for); both as a fraction
# of the 300-3800 cm^-1 total of that spectrum.
import csv, os, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.constants import c as C_LIGHT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_spectra import SCALE, FRACTION  # noqa: E402

REGIONS = [("oop", 650, 1000), ("inplane", 1100, 1700), ("CHstr", 2950, 3150)]
MAXC = {"oop": 8, "inplane": 12, "CHstr": 4}
CM = 2 * np.pi * C_LIGHT * 100
L = int(200000 * FRACTION) * 1e-15                   # ACF length (s) for 200 ps NVE at 1 fs
# diagnostic bands (scaled cm^-1, as the spectra), wavelength labels
BANDS = [("3.3", 3000, 3110), ("6.2", 1570, 1640), ("7.7", 1240, 1450), ("8.6", 1130, 1210), ("11.2", 860, 925),
         ("12.0", 815, 860), ("12.7", 765, 815), ("13.5", 720, 765)]


def model(p, x):
    k = (len(p) - 2) // 3; y = p[-2] + p[-1] * (x - x.mean())
    for i in range(k):
        A, x0, fw = p[3 * i:3 * i + 3]
        gam = np.pi * C_LIGHT * 100 * fw / SCALE; z = gam - 1j * CM * (x - x0) / SCALE
        y = y + A * np.real((1 - np.exp(-z * L)) / z) * gam
    return y


def fit_region(x, y, comps, shift=(-40.0, 15.0)):
    """comps: [(A, x0, fwhm)] start values. Returns fitted comps, baseline (b0, b1) and rms residual / max."""
    p0 = [v for c in comps for v in c] + [0.0, 0.0]
    lb = [v for A, x0, fw in comps for v in (0.0, x0 + shift[0], 0.3)] + [-np.inf, -np.inf]
    ub = [v for A, x0, fw in comps for v in (np.inf, x0 + shift[1], 150.0)] + [np.inf, np.inf]
    p0 = np.clip(p0, np.array(lb) + 1e-9, np.array(ub) - 1e-9)
    r = least_squares(lambda p: model(p, x) - y, p0, bounds=(lb, ub), x_scale="jac", max_nfev=2000)
    k = len(comps)
    return [tuple(r.x[3 * i:3 * i + 3]) for i in range(k)], (r.x[-2], r.x[-1]), np.sqrt(np.mean(r.fun ** 2)) / max(y.max(), 1e-30)


def start_comps(x, y, name):
    pk, pr = find_peaks(y, prominence=0.03 * y.max(), distance=4)
    top = sorted(pk[np.argsort(pr["prominences"])[::-1][:MAXC[name]]])
    return [(y[i], x[i], 3.0) for i in top]


def band_of(x0):
    return next((b for b, lo, hi in BANDS if lo <= x0 < hi), "")


def fit_species(sp, dirs):
    S = {}
    for T, d in dirs.items():
        z = np.load(f"{d}/{sp}_{T}K.npz")
        S[T] = (z["freq"], z["sigma"], z["prefactor_T"], z["replicas"])
    rows = []
    for name, lo, hi in REGIONS:
        comps = None
        for T in sorted(S):
            f, sig, Tp, reps = S[T]; m = (f >= lo) & (f < hi); x = f[m]
            ymean = sig.mean(0)[m]; scale = ymean.max()
            if comps is None:
                comps = start_comps(x, ymean / scale, name)
                if not comps:
                    break
            comps, base, rms = fit_region(x, ymean / scale, comps)
            fits = [("mean", float(np.mean(Tp)), sig.mean(0), comps, base, rms)]
            for i, r in enumerate(reps):
                yi = sig[i][m]
                ci, bi, ri = fit_region(x, yi / scale, comps, shift=(-10.0, 10.0))
                fits.append((int(r), float(Tp[i]), sig[i], ci, bi, ri))
            for rep, Tm, s, cs, b, rms in fits:
                tot = s.sum() / scale                                  # 300-3800 total, same units as the fit
                ped = (b[0] * len(x))                                  # baseline integral over the region (1 cm^-1 bins)
                cen = sorted(c[1] for c in cs)
                for j, (A, x0, fw) in enumerate(cs):
                    near = min([abs(x0 - o) for o in cen if o != x0] or [np.inf])
                    w = np.abs(x - x0) <= max(15.0, 1.5 * fw)
                    rows.append(dict(species=sp, nC=int(sp[1:sp.index("H")]), region=name, comp=j, band=band_of(comps[j][1] if T == 300 else x0),
                                     T=T, replica=rep, T_nve=round(Tm, 1), center=round(x0, 2), fwhm=round(fw, 3),
                                     area=A * np.pi / 2 * fw / tot, win=(s[m][w].sum() / scale) / tot,
                                     pedestal=ped / tot, overlap=int(near < fw), rms=round(rms, 4)))
    return rows


def cmd_fit(a):
    lib, d300, d600, d900, out = a[:5]
    n = int(a[a.index("--n") + 1]) if "--n" in a else None
    species = [r["species"] for r in csv.DictReader(open(lib))]
    if n:
        species = species[::max(1, len(species) // n)][:n]
    rows = []
    for k, sp in enumerate(species):
        rows += fit_species(sp, {300: d300, 600: d600, 900: d900})
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(species)} species, {len(rows)} component fits -> {out}")


def cmd_summary(a):
    R = list(csv.DictReader(open(a[0])))
    by = {}
    for r in R:
        if r["replica"] == "mean":
            continue
        by.setdefault((r["species"], r["region"], r["comp"]), []).append(r)
    stats = {}
    for key, v in by.items():
        b300 = [r for r in v if r["T"] == "300"]
        if not b300 or not b300[0]["band"]:
            continue
        band = b300[0]["band"]; ov = any(r["overlap"] == "1" for r in v if r["T"] == "900")
        x = np.array([float(r["T_nve"]) for r in v]) / 100.0
        out = {}
        for q in ("center", "fwhm", "area", "win"):
            y = np.array([float(r[q]) for r in v])
            if q in ("area", "win"):
                y0 = np.mean([float(r[q]) for r in b300]); y = y / y0
            c1 = np.polyfit(x, y, 1); c2 = np.polyfit(x, y, 2)
            out[q] = (c1[0], np.polyval(c2, 12) - np.polyval(c1, 12), float(np.mean([float(r[q]) for r in b300])))
        stats.setdefault(band, []).append((ov, int(v[0]["nC"]), out))
    print("Per diagnostic band (components assigned by their 300 K centre; isolated = no neighbour within one FWHM at "
          "900 K). Slopes per 100 K: centre and FWHM in cm^-1, area and window integral as a fraction of the 300 K value. "
          "Median [10-90 %]; quad-lin = quadratic minus linear fit at 1200 K.")
    for band, *_ in BANDS:
        v = [s for s in stats.get(band, []) if not s[0]]
        if not v:
            continue
        line = f"{band:>5s} um: {len(v):4d} isolated ({len(stats[band]) - len(v)} overlapping)"
        for q in ("center", "fwhm", "area", "win"):
            sl = np.array([s[2][q][0] for s in v]); ql = np.array([s[2][q][1] for s in v])
            line += f" | {q} {np.median(sl):+.3f} [{np.percentile(sl, 10):+.3f}..{np.percentile(sl, 90):+.3f}] q-l {np.median(ql):+.3f}"
        line += f" | FWHM at 300 K {np.median([s[2]['fwhm'][2] for s in v]):.1f}"
        print(line)


if __name__ == "__main__":
    {"fit": cmd_fit, "summary": cmd_summary}[sys.argv[1]](sys.argv[2:])
