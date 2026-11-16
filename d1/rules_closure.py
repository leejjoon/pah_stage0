# Closure test of the temperature rules (d1/tdep_rules_v1.csv): broaden each species' PAHdb v4.00 harmonic sticks
# the way an emission model would (each line shifted by its family's position rule, Lorentzian of the per-line width
# from the isolated oop bands), and compare the blended features (3.3, 6.2, 7.7, 8.6 um; complex_metrics of
# profile_fit.py, same windows) with those measured on our MLMD spectra at 300, 600 and 900 K.
# PAHdb (B3LYP/6-31G*, scale 0.9794) and Mai's model (B3LYP/4-31G training, MD scaled by 0.9578) put lines at
# different absolute positions, so only widths and their change with T are compared.
#   python rules_closure.py <pahdb xml.gz> <rules csv> <library csv> <fits csv glob> [--n 60] [--invert out.csv]
# --invert: per species, feature and T, the per-line FWHM (applied to that feature's family of sticks) for which the
# broadened sticks reproduce the MLMD feature width (bisection); prints medians and their T and Nc dependence.
import csv, glob, gzip, os, sys
import xml.etree.ElementTree as ET
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from profile_fit import complex_metrics, COMPLEXES  # noqa: E402

NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
FAMILY = [("3.3", 2800, 3300), ("6.2", 1540, 1660), ("7.7", 1220, 1470), ("8.6", 1100, 1220), ("11.2", 860, 925),
          ("12.0", 815, 860), ("12.7", 765, 815), ("13.5", 650, 765), ("11.2", 925, 1000)]
GRID = np.arange(300.0, 3800.0, 1.0)


def rule(R, band, q, nc):
    r = R[(band, q)]
    return float(r["at_Nc50"]) + float(r["per_decade_Nc"]) * np.log10(nc / 50.0)


def spectrum(nu, I, T, nc, R, w=None):
    dT = (T - 300.0) / 100.0
    if w is None:
        w = rule(R, "11.2", "width300", nc) + rule(R, "11.2", "width", nc) * dT  # per-line FWHM (oop rule)
    y = np.zeros_like(GRID)
    for f, a in zip(nu, I):
        fam = next((b for b, lo, hi in FAMILY if lo <= f < hi), None)
        f2 = f + (rule(R, fam, "position", nc) * dT if fam else 0.0)
        y += a * (w / 2) ** 2 / ((GRID - f2) ** 2 + (w / 2) ** 2) / (np.pi * w / 2)
    return y


def main():
    xml, rules_csv, lib, fits = sys.argv[1:5]
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 60
    R = {(r["band"], r["quantity"]): r for r in csv.DictReader(open(rules_csv))}
    species = [r["species"] for r in csv.DictReader(open(lib))]
    pick = list(np.random.default_rng(3).choice(species, n, replace=False))
    uid = {int(s.split("_v4u")[1]): s for s in pick}
    sticks = {}
    with gzip.open(xml) as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag == NS + "specie" and int(el.get("uid")) in uid:
                ms = el.find(NS + "transitions").findall(NS + "mode")
                sticks[uid[int(el.get("uid"))]] = (np.array([float(m.find(NS + "frequency").text) for m in ms]),
                                                     np.array([float(m.find(NS + "intensity").text) for m in ms]))
            if el.tag == NS + "specie":
                el.clear()
    md = {}
    for fn in sorted(glob.glob(fits)):
        for r in csv.DictReader(open(fn)):
            if r["region"] == "complex" and r["species"] in sticks:
                md.setdefault((r["species"], r["band"], int(r["T"])), []).append(r)
    res = {b: [] for b, *_ in COMPLEXES}
    for sp, (nu, I) in sticks.items():
        nc = int(sp[1:sp.index("H")])
        for b, lo, hi in COMPLEXES:
            row = []
            for T in (300, 600, 900):
                y = spectrum(nu, I, T, nc, R)
                _, rw, fw, _, _ = complex_metrics(GRID, y, lo, hi)
                v = md[(sp, b, T)]
                row.append((fw if b == "3.3" else rw, np.mean([float(r["fwhm" if b == "3.3" else "rmsw"]) for r in v])))
            res[b].append(row)
    print(f"{len(sticks)} species. Width of each blended feature (FWHM for 3.3, rms in the window for 6.2/7.7/8.6), "
          "median over species: rules applied to PAHdb sticks vs MLMD; growth = 900 K minus 300 K")
    for b, v in res.items():
        v = np.array(v)                       # species, T, (rules, md)
        g_r = v[:, 2, 0] - v[:, 0, 0]; g_m = v[:, 2, 1] - v[:, 0, 1]
        print(f"  {b:>4s} um: 300 K {np.median(v[:, 0, 0]):6.1f} vs {np.median(v[:, 0, 1]):6.1f} | 600 K {np.median(v[:, 1, 0]):6.1f} vs "
              f"{np.median(v[:, 1, 1]):6.1f} | 900 K {np.median(v[:, 2, 0]):6.1f} vs {np.median(v[:, 2, 1]):6.1f} | growth "
              f"{np.median(g_r):+5.1f} vs {np.median(g_m):+5.1f} (median ratio rules/MD {np.median(g_r / g_m):.2f})")


def invert(sticks, md, R, out):
    rows = []
    for sp, (nu, I) in sticks.items():
        nc = int(sp[1:sp.index("H")])
        for b, lo, hi in COMPLEXES:
            for T in (300, 600, 900):
                v = md[(sp, b, T)]; q = "fwhm" if b == "3.3" else "rmsw"
                target = np.mean([float(r[q]) for r in v])
                def width(w):
                    y = spectrum(nu, I, T, nc, R, w)
                    _, rw, fw, _, _ = complex_metrics(GRID, y, lo, hi)
                    return fw if b == "3.3" else rw
                a, z = 0.3, 300.0
                if not (width(a) <= target <= width(z)):
                    rows.append(dict(species=sp, nC=nc, band=b, T=T, target=target, w=np.nan)); continue
                for _ in range(30):
                    m = np.sqrt(a * z); a, z = (m, z) if width(m) < target else (a, m)
                rows.append(dict(species=sp, nC=nc, band=b, T=T, target=round(target, 2), w=round(np.sqrt(a * z), 2)))
    with open(out, "w", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    print(f"Per-line FWHM (cm^-1) that reproduces the MLMD feature width, median [16-84 %] (n with a solution), -> {out}")
    for b, *_ in COMPLEXES:
        cells = []; per = {}
        for T in (300, 600, 900):
            w = np.array([r["w"] for r in rows if r["band"] == b and r["T"] == T]); ok = w[np.isfinite(w)]
            cells.append(f"{T} K {np.median(ok):6.1f} [{np.percentile(ok, 16):.1f}..{np.percentile(ok, 84):.1f}] ({len(ok)}/{len(w)})")
        # per species slope vs T, then median and size trend
        sl = []
        for sp in {r["species"] for r in rows if r["band"] == b}:
            v = [r for r in rows if r["species"] == sp and r["band"] == b and np.isfinite(r["w"])]
            if len(v) == 3:
                sl.append((v[0]["nC"], np.polyfit([r["T"] / 100 for r in v], [r["w"] for r in v], 1)[0]))
        sl = np.array(sl)
        X = np.log10(sl[:, 0] / 50); c = np.polyfit(X, sl[:, 1], 1)
        print(f"  {b:>4s} um: " + " | ".join(cells) + f" | slope {np.median(sl[:, 1]):+.2f} per 100 K (at Nc 50 {c[1]:+.2f}, {c[0]:+.2f}/dex)")


if __name__ == "__main__":
    if "--invert" in sys.argv:
        xml, rules_csv, lib, fits = sys.argv[1:5]
        n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 60
        R = {(r["band"], r["quantity"]): r for r in csv.DictReader(open(rules_csv))}
        species = [r["species"] for r in csv.DictReader(open(lib))]
        pick = list(np.random.default_rng(3).choice(species, n, replace=False)); uid = {int(s.split("_v4u")[1]): s for s in pick}
        sticks = {}
        with gzip.open(xml) as f:
            for ev, el in ET.iterparse(f, events=("end",)):
                if el.tag == NS + "specie" and int(el.get("uid")) in uid:
                    ms = el.find(NS + "transitions").findall(NS + "mode")
                    sticks[uid[int(el.get("uid"))]] = (np.array([float(m.find(NS + "frequency").text) for m in ms]),
                                                         np.array([float(m.find(NS + "intensity").text) for m in ms]))
                if el.tag == NS + "specie":
                    el.clear()
        md = {}
        for fn in sorted(glob.glob(fits)):
            for r in csv.DictReader(open(fn)):
                if r["region"] == "complex" and r["species"] in sticks:
                    md.setdefault((r["species"], r["band"], int(r["T"])), []).append(r)
        invert(sticks, md, R, sys.argv[sys.argv.index("--invert") + 1])
    else:
        main()
