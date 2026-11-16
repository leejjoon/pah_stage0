# Broad background ("pedestal") of the MLMD spectra against temperature, for the emission model
# (docs/plan/d1_hangang_new_species.md, "Pedestal"). Per region, the lower envelope of the spectrum (running minimum
# over ENV cm^-1, then a running mean over the same width) is integrated and divided by the 300-3800 cm^-1 total.
# Lorentzian wings of the bands also raise that floor, so the same is measured on PAHdb harmonic sticks broadened
# with the per-line widths of perline_widths_v1.csv and shifted by the position rules (rules_closure.spectrum logic);
# the MLMD excess over that is the pedestal the per-line model does not produce. Then the rms widths of the 6.2/7.7/
# 8.6 um complexes are recomputed with the excess added as a flat pedestal, to see whether it closes the gap.
#   python pedestal.py <pahdb xml.gz> <rules csv> <perline csv> <library csv> <spectra 300> <600> <900> [--n 60]
import csv, gzip, os, sys
import xml.etree.ElementTree as ET
import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rules_closure as rc  # noqa: E402
from profile_fit import complex_metrics, COMPLEXES  # noqa: E402

ENV = 41
REGIONS = [("oop", 650, 1000), ("in-plane", 1100, 1700), ("gap", 1700, 2800), ("CH stretch", 2800, 3300)]


def floor_share(f, s, lo, hi):
    env = uniform_filter1d(minimum_filter1d(s, ENV), ENV)
    m = (f >= lo) & (f < hi)
    return env[m].sum() / s.sum(), env


def stick_spec(nu, I, T, nc, R, W):
    dT = (T - 300) / 100; y = np.zeros_like(rc.GRID)
    for fr, a in zip(nu, I):
        fam = next((b for b, lo, hi in rc.FAMILY if lo <= fr < hi), None)
        f2 = fr + (rc.rule(R, fam, "position", nc) * dT if fam else 0)
        w = next((w0 + sl * dT for lo, hi, w0, sl in W if lo <= fr < hi), 4.5 + 0.67 * dT)
        y += a * (w / 2) ** 2 / ((rc.GRID - f2) ** 2 + (w / 2) ** 2) / (np.pi * w / 2)
    return y


def main(a):
    xml, rules_csv, perline_csv, lib, d300, d600, d900 = a[:7]
    n = int(a[a.index("--n") + 1]) if "--n" in a else 60
    R = {(r["band"], r["quantity"]): r for r in csv.DictReader(open(rules_csv))}
    W = [(float(r["lo_cm1"]), float(r["hi_cm1"]), float(r["fwhm_300K"]), float(r["fwhm_slope_per_100K"]))
         for r in csv.DictReader(open(perline_csv))]
    species = [r["species"] for r in csv.DictReader(open(lib))]
    dirs = {300: d300, 600: d600, 900: d900}
    # 1. MLMD floor for every species, per replica, against its NVE T
    md = {name: [] for name, *_ in REGIONS}; md_T = {}
    for sp in species:
        for T, d in dirs.items():
            z = np.load(f"{d}/{sp}_{T}K.npz")
            for i in range(len(z["replicas"])):
                for name, lo, hi in REGIONS:
                    sh, _ = floor_share(z["freq"], z["sigma"][i], lo, hi)
                    md[name].append((sp, T, float(z["prefactor_T"][i]), sh))
    print(f"MLMD pedestal (lower envelope, {ENV} cm^-1) as a share of the total, median over {len(species)} species "
          "at 300/600/900 K, and the median per-species slope per 100 K:")
    slopes = {}
    for name, *_ in REGIONS:
        v = md[name]
        med = [np.median([x[3] for x in v if x[1] == T]) for T in (300, 600, 900)]
        sl = []
        for sp in species:
            p = [(x[2], x[3]) for x in v if x[0] == sp]
            sl.append(np.polyfit([q[0] / 100 for q in p], [q[1] for q in p], 1)[0])
        slopes[name] = np.median(sl)
        print(f"  {name:10s} {med[0]:.4f} {med[1]:.4f} {med[2]:.4f} | slope {np.median(sl):+.4f} [{np.percentile(sl, 16):+.4f}.."
              f"{np.percentile(sl, 84):+.4f}] per 100 K")
    # 2. the same on broadened PAHdb sticks for a random subset, and the excess
    pick = list(np.random.default_rng(3).choice(species, n, replace=False)); uid = {int(s.split("_v4u")[1]): s for s in pick}
    st = {}
    with gzip.open(xml) as fh:
        for ev, el in ET.iterparse(fh, events=("end",)):
            if el.tag == rc.NS + "specie" and int(el.get("uid")) in uid:
                ms = el.find(rc.NS + "transitions").findall(rc.NS + "mode")
                st[uid[int(el.get("uid"))]] = (np.array([float(m.find(rc.NS + "frequency").text) for m in ms]),
                                               np.array([float(m.find(rc.NS + "intensity").text) for m in ms]))
            if el.tag == rc.NS + "specie":
                el.clear()
    ex = {name: {T: [] for T in dirs} for name, *_ in REGIONS}; close = {b: [] for b, *_ in COMPLEXES if b != "3.3"}
    for sp, (nu, I) in st.items():
        nc = int(sp[1:sp.index("H")])
        row = {}
        for T, d in dirs.items():
            y = stick_spec(nu, I, T, nc, R, W)
            z = np.load(f"{d}/{sp}_{T}K.npz"); smd = z["sigma"].mean(0)
            for name, lo, hi in REGIONS:
                ex[name][T].append(floor_share(z["freq"], smd, lo, hi)[0] - floor_share(rc.GRID, y, lo, hi)[0])
            # closure: add the in-plane excess as a flat pedestal over 1100-1700 cm^-1, then complex rms widths
            e = max(0.0, floor_share(z["freq"], smd, 1100, 1700)[0] - floor_share(rc.GRID, y, 1100, 1700)[0])
            m = (rc.GRID >= 1100) & (rc.GRID < 1700); y2 = y.copy(); y2[m] += e * y.sum() / m.sum()
            row[T] = {b: (complex_metrics(rc.GRID, y, lo, hi)[1], complex_metrics(rc.GRID, y2, lo, hi)[1],
                          complex_metrics(z["freq"], smd, lo, hi)[1]) for b, lo, hi in COMPLEXES if b != "3.3"}
        for b in row[300]:
            close[b].append([row[T][b] for T in (300, 600, 900)])
    print(f"\nExcess of the MLMD floor over broadened PAHdb sticks ({len(st)} species), share of the total, median at "
          "300/600/900 K:")
    for name, *_ in REGIONS:
        print(f"  {name:10s} " + " ".join(f"{np.median(ex[name][T]):+.4f}" for T in (300, 600, 900)))
    print("\nComplex rms width growth 300 -> 900 K (cm^-1), median: sticks | sticks + in-plane pedestal | MLMD")
    for b, v in close.items():
        v = np.array(v)                                      # species, T, (sticks, sticks+ped, md)
        g = [np.median(v[:, 2, k] - v[:, 0, k]) for k in range(3)]
        print(f"  {b:>4s} um: {g[0]:+5.1f} | {g[1]:+5.1f} | {g[2]:+5.1f}")


if __name__ == "__main__":
    main(sys.argv[1:])
