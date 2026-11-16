# Profile fits of the bands in the Hangang neutral MLMD libraries at 300/600/900 K, for band position, width and
# strength against temperature (docs/plan/d1_hangang_new_species.md, "Band profiles").
#   python profile_fit.py fit <library csv> <spectra dir 300> <dir 600> <dir 900> <out csv> [--n N | --start i --stop j]
#   python profile_fit.py summary <fits csv> [more csv ...]
#   python profile_fit.py bysize <fits csv> [more csv ...]     per-band trends by Nc bin and per decade of Nc
#   python profile_fit.py rules <out csv> <fits csv> [more csv ...]   rules table: value at Nc = 50 + per decade of Nc
# Blended features (3.3 um CH stretch; the 6.2, 7.7 and 8.6 um complexes) are measured as whole complexes in fixed
# windows (COMPLEXES): intensity-weighted centroid and rms width of the non-negative part, the FWHM read directly from
# the spectrum (5 cm^-1 boxcar), and the window's share of the total. Multi-component fits of these blends are
# degenerate (20-species test, 2026-11-16: the 3.3 um components shifted -1.2 and narrowed, against -9.9 and +6.6
# cm^-1 per 100 K at region level). Profile fits are kept for the out-of-plane region, where bands are isolated.
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

REGIONS = [("oop", 650, 1000)]
ALL_REGIONS = {"oop": (650, 1000), "inplane": (1100, 1700)}   # --regions inplane: per-line widths of isolated in-plane bands
COMPLEXES = [("3.3", 2850, 3250), ("6.2", 1540, 1660), ("7.7", 1220, 1470), ("8.6", 1100, 1220)]
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


def complex_metrics(f, s, lo, hi):
    m = (f >= lo) & (f < hi); x = f[m]; y = s[m]; w = np.clip(y, 0, None)
    c = (w * x).sum() / w.sum(); rw = np.sqrt((w * (x - c) ** 2).sum() / w.sum())
    ys = np.convolve(y, np.ones(5) / 5, mode="same"); k = int(np.argmax(ys)); h = ys[k] / 2
    i = k
    while i > 0 and ys[i] > h: i -= 1
    j = k
    while j < len(ys) - 1 and ys[j] > h: j += 1
    return c, rw, x[j] - x[i], x[k], y.sum() / s.sum()


def band_of(x0):
    return next((b for b, lo, hi in BANDS if lo <= x0 < hi), "")


def fit_species(sp, dirs, regions=None, complexes=True):
    regions = regions or REGIONS
    S = {}
    for T, d in dirs.items():
        z = np.load(f"{d}/{sp}_{T}K.npz")
        S[T] = (z["freq"], z["sigma"], z["prefactor_T"], z["replicas"])
    rows = []
    nc = int(sp[1:sp.index("H")])
    for T in sorted(S):
        f, sig, Tp, reps = S[T]
        for i, r in enumerate(reps):
            for b, lo, hi in (COMPLEXES if complexes else []):
                c, rw, fw, pk, sh = complex_metrics(f, sig[i], lo, hi)
                rows.append(dict(species=sp, nC=nc, region="complex", comp=0, band=b, T=T, replica=int(r),
                                 T_nve=round(float(Tp[i]), 1), center=round(c, 2), fwhm=round(fw, 2), rmsw=round(rw, 2),
                                 peak=round(pk, 1), area=np.nan, win=sh, pedestal=np.nan, overlap=0, rms=np.nan))
    for name, lo, hi in regions:
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
                    rows.append(dict(species=sp, nC=nc, region=name, comp=j, band=band_of(comps[j][1] if T == 300 else x0),
                                     T=T, replica=rep, T_nve=round(Tm, 1), center=round(x0, 2), fwhm=round(fw, 3),
                                     rmsw=np.nan, peak=np.nan,
                                     area=A * np.pi / 2 * fw / tot, win=(s[m][w].sum() / scale) / tot,
                                     pedestal=ped / tot, overlap=int(near < fw), rms=round(rms, 4)))
    return rows


def cmd_fit(a):
    lib, d300, d600, d900, out = a[:5]
    n = int(a[a.index("--n") + 1]) if "--n" in a else None
    species = [r["species"] for r in csv.DictReader(open(lib))]
    if n:
        species = species[::max(1, len(species) // n)][:n]
    if "--sample" in a:                                   # random sample (seed 5), before chunking
        species = sorted(np.random.default_rng(5).choice(species, int(a[a.index("--sample") + 1]), replace=False))
    if "--start" in a:
        species = species[int(a[a.index("--start") + 1]):int(a[a.index("--stop") + 1])]
    regions = [(r, *ALL_REGIONS[r]) for r in a[a.index("--regions") + 1].split(",")] if "--regions" in a else None
    rows = []
    for k, sp in enumerate(species):
        rows += fit_species(sp, {300: d300, 600: d600, 900: d900}, regions, complexes=regions is None)
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(species)} species, {len(rows)} component fits -> {out}")


def cmd_summary(a):
    R = [r for f in a for r in csv.DictReader(open(f))]
    complexes_summary([r for r in R if r["region"] == "complex"])
    by = {}
    for r in R:
        if r["replica"] == "mean" or r["region"] == "complex":
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


def complexes_summary(R):
    by = {}
    for r in R:
        by.setdefault((r["species"], r["band"]), []).append(r)
    print("Complexes (whole-window measures; per 100 K, median [10-90 %] over species; share relative to 300 K; "
          "quad-lin = quadratic minus linear at 1200 K):")
    for b, *_ in COMPLEXES:
        out = {q: [] for q in ("center", "rmsw", "fwhm", "peak", "win")}; f300 = []
        for (sp, bb), v in by.items():
            if bb != b:
                continue
            x = np.array([float(r["T_nve"]) for r in v]) / 100.0
            for q in out:
                y = np.array([float(r[q]) for r in v])
                if q == "win":
                    y = y / np.mean([float(r[q]) for r in v if r["T"] == "300"])
                c1 = np.polyfit(x, y, 1); c2 = np.polyfit(x, y, 2)
                out[q].append((c1[0], np.polyval(c2, 12) - np.polyval(c1, 12)))
            f300.append(np.mean([float(r["fwhm"]) for r in v if r["T"] == "300"]))
        line = f"{b:>5s} um ({len(f300)} species, FWHM at 300 K {np.median(f300):.1f})"
        for q, v in out.items():
            v = np.array(v)
            line += f" | {q} {np.median(v[:, 0]):+.3f} [{np.percentile(v[:, 0], 10):+.3f}..{np.percentile(v[:, 0], 90):+.3f}] q-l {np.median(v[:, 1]):+.2f}"
        print(line)
    print()


def cmd_bysize(a):
    """Per species and band: linear slopes of position, width and strength against the replicas' NVE T; then medians
    per Nc bin and an OLS trend per decade of Nc. Complexes: peak (centroid for 3.3), rms width (FWHM for 3.3), window
    share; oop components: centre, FWHM, area, grouped by component and named by their 300 K band; components that
    overlap a neighbour at 900 K are left out."""
    import warnings; warnings.simplefilter("ignore")
    R = [r for f in a for r in csv.DictReader(open(f))]
    by = {}
    for r in R:
        if r["replica"] != "mean":
            by.setdefault((r["species"], r["region"], r["band"] if r["region"] == "complex" else r["comp"]), []).append(r)
    res = {}
    for (sp, reg, k), v in by.items():
        b300 = [r for r in v if r["T"] == "300"]
        if len(b300) < 3 or len(v) < 9 or not b300[0]["band"]:
            continue
        band = b300[0]["band"]
        if reg != "complex" and any(r["overlap"] == "1" for r in v if r["T"] == "900"):
            continue
        x = np.array([float(r["T_nve"]) for r in v]) / 100
        pos = "peak" if reg == "complex" and band != "3.3" else "center"
        wid = "rmsw" if reg == "complex" and band != "3.3" else "fwhm"
        st = "win" if reg == "complex" else "area"
        y0 = np.mean([float(r[st]) for r in b300])
        sl = [np.polyfit(x, np.array([float(r[q]) for r in v]) / (y0 if q == st else 1), 1)[0] for q in (pos, wid, st)]
        res.setdefault(band, []).append((int(v[0]["nC"]), *sl, np.mean([float(r[wid]) for r in b300])))
    bins = [(20, 40), (40, 60), (60, 90), (90, 130), (130, 220)]
    for q, i, unit in (("position shift", 1, "cm-1 per 100 K"), ("broadening", 2, "cm-1 per 100 K"),
                       ("strength", 3, "fraction of 300 K per 100 K"), ("width at 300 K", 4, "cm-1; FWHM for 3.3 and oop, rms for 6-9 um")):
        print(f"\n{q} ({unit}): median per Nc bin (n); trend = OLS slope per decade of Nc +- s.e.")
        print("  band " + "".join(f"{f'{lo}-{hi}':>13s}" for lo, hi in bins) + "      trend")
        for b, *_ in COMPLEXES + [x for x in BANDS if x[0] in ("11.2", "12.0", "12.7", "13.5")]:
            v = np.array(res.get(b, []))
            if not len(v):
                continue
            cells = ""
            for lo, hi in bins:
                m = (v[:, 0] >= lo) & (v[:, 0] < hi)
                cells += f"{np.median(v[m, i]):>8.3f}({m.sum():3d})" if m.sum() >= 5 else f"{'-':>13s}"
            X = np.log10(v[:, 0]); A = np.vstack([X, np.ones_like(X)]).T; c, *_ = np.linalg.lstsq(A, v[:, i], rcond=None)
            s2 = ((v[:, i] - A @ c) ** 2).sum() / (len(X) - 2); se = np.sqrt(s2 / ((X - X.mean()) ** 2).sum())
            print(f"  {b:>5s}{cells}  {c[0]:+.3f} +- {se:.3f}")


def per_species(files):
    """{band: [(nC, {quantity: (slope per 100 K, quad - lin at 1200 K, value at 300 K)})]}, as in cmd_bysize."""
    R = [r for f in files for r in csv.DictReader(open(f))]
    by = {}
    for r in R:
        if r["replica"] != "mean":
            by.setdefault((r["species"], r["region"], r["band"] if r["region"] == "complex" else r["comp"]), []).append(r)
    res = {}
    for (sp, reg, k), v in by.items():
        b300 = [r for r in v if r["T"] == "300"]
        if len(b300) < 3 or len(v) < 9 or not b300[0]["band"]:
            continue
        band = b300[0]["band"]
        if reg != "complex" and any(r["overlap"] == "1" for r in v if r["T"] == "900"):
            continue
        x = np.array([float(r["T_nve"]) for r in v]) / 100
        qs = {"position": "peak" if reg == "complex" and band != "3.3" else "center",
              "width": "rmsw" if reg == "complex" and band != "3.3" else "fwhm",
              "strength": "win" if reg == "complex" else "area"}
        out = {}
        for name, q in qs.items():
            y = np.array([float(r[q]) for r in v]); y300 = np.mean([float(r[q]) for r in b300])
            if name == "strength":
                y = y / y300
            c1 = np.polyfit(x, y, 1); c2 = np.polyfit(x, y, 2)
            out[name] = (c1[0], np.polyval(c2, 12) - np.polyval(c1, 12), y300)
        res.setdefault(band, []).append((int(v[0]["nC"]), out))
    return res


def cmd_rules(a):
    """Per band and quantity, OLS in log10(Nc/50): value at Nc = 50 and per decade of Nc (+- s.e.), robust scatter
    between species (1.4826 MAD of the residuals), and the median quad - lin difference at 1200 K."""
    import warnings; warnings.simplefilter("ignore")
    out, files = a[0], a[1:]
    res = per_species(files); rows = []
    kind = {"3.3": "complex (centroid, FWHM)", "6.2": "complex (peak, rms width)", "7.7": "complex (peak, rms width)",
            "8.6": "complex (peak, rms width)"}
    for b in [c[0] for c in COMPLEXES] + ["11.2", "12.0", "12.7", "13.5"]:
        v = res.get(b, [])
        if len(v) < 10:
            continue
        X = np.log10(np.array([n for n, _ in v]) / 50.0)
        for name, unit in (("position", "cm-1 per 100 K"), ("width", "cm-1 per 100 K"), ("strength", "fraction of 300 K per 100 K"),
                           ("width300", "cm-1 at 300 K")):
            q, i = ("width", 2) if name == "width300" else (name, 0)
            y = np.array([o[q][i] for _, o in v])
            A = np.vstack([np.ones_like(X), X]).T; c, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = y - A @ c; s2 = (r ** 2).sum() / (len(y) - 2); cov = s2 * np.linalg.inv(A.T @ A)
            ql = np.median([o[q][1] for _, o in v]) if name != "width300" else np.nan
            rows.append(dict(band=b, kind=kind.get(b, "isolated oop band (truncated-Lorentzian fit)"), quantity=name, unit=unit,
                             n_species=len(v), at_Nc50=round(c[0], 4), se=round(np.sqrt(cov[0, 0]), 4),
                             per_decade_Nc=round(c[1], 4), se_decade=round(np.sqrt(cov[1, 1]), 4),
                             scatter=round(1.4826 * np.median(np.abs(r - np.median(r))), 4),
                             quad_minus_lin_1200K=round(ql, 4)))
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rules -> {out}")
    for r in rows:
        print(f"{r['band']:>5s} {r['quantity']:9s} {r['at_Nc50']:+9.4f} +- {r['se']:.4f}  per decade {r['per_decade_Nc']:+8.4f} +- "
              f"{r['se_decade']:.4f}  scatter {r['scatter']:.4f}  q-l {r['quad_minus_lin_1200K']:+.3f}  (n={r['n_species']})")


if __name__ == "__main__":
    {"fit": cmd_fit, "summary": cmd_summary, "bysize": cmd_bysize, "rules": cmd_rules}[sys.argv[1]](sys.argv[2:])
