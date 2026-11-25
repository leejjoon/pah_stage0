# Per-line widths refitted on the spin-consistent closed-shell singlets (coordinator request 2026-11-25): the clean
# counterpart of d1/perline_widths_v2.csv, next to d1/tdep_rules_v1_clean.csv.
#   python d1/perline_widths_clean.py <pahdb xml.gz> <dir with the Hangang fit tables> [out csv] [--jobs 16]
# The directory holds profile_fits_v1_c{0,1,2}.csv, inplane_fits_v1_c{0,1}.csv and perline_width_v1_kekule_ok.csv
# (gh200:~/projects/hangang_appdata/d1/spectra/). Needs numpy and scipy.
# Clean set: verdict ok and multiplicity 1 in d1/pahdb_v4_screen_spin_v1.csv (385 of the 549 fitted species).
# Each family is refitted the way version 2 was made, on the clean species only:
#   CH stretch (3.3 um)   per-line FWHM for which broadened PAHdb sticks reproduce the MLMD feature width (the 60
#                         species of perline_width_v1_kekule_ok.csv): median at 300 K, median slope
#   in-plane families     isolated bands of the in-plane profile fits (inplane_widths.py; 200 species were fitted)
#   CH out-of-plane       the 11.2 um rule at Nc = 50 of the clean rules table (isolated oop bands of the profile fits)
#   6.2 um extra term     the extra FWHM slope k for which sticks with the clean rules and the clean widths above
#                         reproduce the MLMD growth of the 6.2 um feature width, 300 -> 900 K (width_term_fit.py), on
#                         the clean species among the 60 that version 2 used plus 60 other clean species (the
#                         table takes the pooled value; both halves are printed). The 7.7 um gap is reported, as
#                         in version 2.
# With --all the same code runs on all fitted species and must reproduce perline_widths_v2.csv (a check of this
# script, not a product).
import csv, gzip, os, sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
import numpy as np

D1 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D1)
import inplane_widths as iw  # noqa: E402
import profile_fit as pf  # noqa: E402
import rules_closure as rc  # noqa: E402

KGRID = np.arange(0, 16.1, 1.0)
FAM = {"6.2": (1540, 1700, (1540, 1660)), "7.7": (1220, 1470, (1220, 1470))}          # line range, feature window
_G = {}


def spectrum(nu, I, T, nc, extra):
    R, W = _G["R"], _G["W"]; dT = (T - 300) / 100; y = np.zeros_like(rc.GRID)
    for f, a in zip(nu, I):
        fam = next((b for b, lo, hi in rc.FAMILY if lo <= f < hi), None)
        f2 = f + (rc.rule(R, fam, "position", nc) * dT if fam else 0)
        w = next((w0 + sl * dT for lo, hi, w0, sl in W if lo <= f < hi), W[-1][2] + W[-1][3] * dT)
        for lo, hi, k in extra:
            if lo <= f < hi:
                w += k * dT
        y += a * (w / 2) ** 2 / ((rc.GRID - f2) ** 2 + (w / 2) ** 2) / (np.pi * w / 2)
    return y


def growth(job):
    """Rules growth of the feature rms width, 300 -> 900 K, per k of the scan, for one species and family."""
    sp, nu, I, fam = job; lo, hi, (wlo, whi) = FAM[fam]; nc = int(sp[1:sp.index("H")])
    w300 = pf.complex_metrics(rc.GRID, spectrum(nu, I, 300, nc, []), wlo, whi)[1]
    return sp, fam, [pf.complex_metrics(rc.GRID, spectrum(nu, I, 900, nc, [(lo, hi, k)]), wlo, whi)[1] - w300 for k in KGRID]


def gaps(pool, st, md):
    """{(species, family): (rules - MLMD) growth of the feature rms width per k}."""
    res = {}
    for sp, fam, g in pool.imap_unordered(growth, [(s, *st[s], f) for s in st for f in FAM]):
        res[(sp, fam)] = np.array(g) - (np.mean(md[(sp, fam, 900)]) - np.mean(md[(sp, fam, 300)]))
    return res


def scan(res, species, tag):
    """Extra slope k per family where the median gap crosses zero; None without a crossing."""
    best = {}
    for fam in FAM:
        d = np.array([res[(s, fam)] for s in species]); med = np.median(d, 0); i = np.nonzero(np.diff(np.sign(med)))[0]
        best[fam] = float(np.interp(0, med[i[0]:i[0] + 2], KGRID[i[0]:i[0] + 2])) if len(i) else None
        j = int(np.argmin(np.abs(KGRID - (best[fam] or 0))))
        print(f"  {tag}, {fam} um, {len(species)} species: median (rules - MLMD) growth {med[0]:+.2f} at k = 0, {med[-1]:+.2f} at k = {KGRID[-1]:.0f}; "
              + (f"k = {best[fam]:.2f} cm^-1 per 100 K (16-84 % of the gap there {np.percentile(d[:, j], 16):+.1f}..{np.percentile(d[:, j], 84):+.1f})"
                 if best[fam] is not None else "no zero crossing"), flush=True)
    return best


def main():
    a = sys.argv[1:]; jobs = 16; allsp = "--all" in a
    if "--jobs" in a:
        jobs = int(a[a.index("--jobs") + 1]); del a[a.index("--jobs"):a.index("--jobs") + 2]
    a = [x for x in a if x != "--all"]
    xml, fits = a[0], a[1]; out = a[2] if len(a) > 2 else os.path.join(D1, "perline_widths_v2_clean.csv")
    scr = {r["uid"]: r for r in csv.DictReader(open(os.path.join(D1, "pahdb_v4_screen_spin_v1.csv")))}
    ok = lambda s: allsp or ("_v4u" in s and scr[s.split("_v4u")[1]]["verdict"] == "ok" and scr[s.split("_v4u")[1]]["mult"] == "1")
    P = [os.path.join(fits, f"profile_fits_v1_c{i}.csv") for i in range(3)]
    prof = [r for f in P for r in csv.DictReader(open(f))]
    keep = {r["species"] for r in prof if ok(r["species"])}
    print(f"{len({r['species'] for r in prof})} fitted species, {len(keep)} kept ({'all' if allsp else 'clean'})")
    # CH stretch
    by = {}
    for r in csv.DictReader(open(os.path.join(fits, "perline_width_v1_kekule_ok.csv"))):
        if r["band"] == "3.3" and r["species"] in keep and r["w"] not in ("", "nan"):
            by.setdefault(r["species"], {})[int(r["T"])] = float(r["w"])
    w = np.array([[v[300], v[600], v[900]] for v in by.values() if len(v) == 3])
    ch = (float(np.median(w[:, 0])), float(np.median(np.polyfit([3, 6, 9], w.T, 1)[0])), len(w))
    print(f"CH stretch: {ch[2]} species, FWHM 300 K {ch[0]:.2f}, slope {ch[1]:+.2f} per 100 K")
    # in-plane
    fam, n_sp, n_all = iw.collect([os.path.join(fits, f"inplane_fits_v1_c{i}.csv") for i in range(2)], keep)
    ip = {}
    for name, lo, hi in iw.FAMILIES:
        v = np.array(fam[name]); ip[name] = (lo, hi, float(np.median(v[:, 1])), float(np.median(v[:, 2])), len(v))
        print(f"{name}: {len(v)} isolated bands of {n_sp} species, FWHM 300 K {ip[name][2]:.2f}, slope {ip[name][3]:+.2f}")
    # out-of-plane: the 11.2 um rule at Nc = 50, as version 2 (4.5, 0.67 from tdep_rules_v1.csv)
    rules = os.path.join(D1, "tdep_rules_v1.csv" if allsp else "tdep_rules_v1_clean.csv")
    _G["R"] = {(r["band"], r["quantity"]): r for r in csv.DictReader(open(rules))}
    nb = sum(int(_G["R"][(b, "width")]["n_species"]) for b in ("11.2", "12.0", "12.7", "13.5"))
    oop = (float(_G["R"][("11.2", "width300")]["at_Nc50"]), float(_G["R"][("11.2", "width")]["at_Nc50"]), nb)
    print(f"CH out-of-plane: 11.2 um rule of {os.path.basename(rules)}: FWHM 300 K {oop[0]:.2f}, slope {oop[1]:+.2f}; {nb} isolated oop bands in the four rules; "
          "the other three: " + ", ".join(f"{b} {float(_G['R'][(b, 'width300')]['at_Nc50']):.2f} {float(_G['R'][(b, 'width')]['at_Nc50']):+.2f}" for b in ("12.0", "12.7", "13.5")))
    # extra 6.2 um term
    r2 = lambda x: round(x, 2)
    base = [(2800, 3300, r2(ch[0]) if not allsp else 9.9, r2(ch[1]) if not allsp else 5.8)] + \
           [(lo, hi, r2(a_), r2(s_)) for lo, hi, a_, s_, _ in ip.values()] + [(650, 1000, r2(oop[0]), r2(oop[1]))]
    _G["W"] = base                                                                    # version 2 scanned with the v1 table (CH stretch 9.9, 5.8)
    bad = {r["uid"] for r in csv.DictReader(open(os.path.join(D1, "pahdb_v4_exclude_nonkekule_v1.csv")))}
    lib = [r["species"] for r in csv.DictReader(open(os.path.join(D1, "new900_library_v1.csv"))) if r["species"].split("_v4u")[1] not in bad]
    pick = list(np.random.default_rng(3).choice(lib, 60, replace=False))             # the 60 species of version 2
    s1 = [s for s in pick if s in keep]
    rest = sorted(s for s in keep if s in set(lib) and s not in set(pick))
    s2 = [] if allsp else list(np.random.default_rng(4).choice(rest, 60, replace=False))
    st = load_sticks(xml, set(s1) | set(s2))
    md = {}
    for r in prof:
        if r["region"] == "complex" and r["species"] in st:
            md.setdefault((r["species"], r["band"], int(r["T"])), []).append(float(r["rmsw"]))
    with Pool(jobs) as pool:
        res = gaps(pool, st, md)
    k1 = scan(res, s1, "version-2 sample" + ("" if allsp else ", clean species"))
    k2 = scan(res, s2, "60 other clean species") if s2 else {}
    k3 = scan(res, s1 + s2, "both samples") if s2 else k1
    k = k3["6.2"]                                                                     # the table takes the pooled value
    rows = [("CH stretch (3.3 um)", 2800, 3300, ch[0], ch[1], f"{ch[2]} species", "inversion of the 3.3 um feature width on PAHdb sticks (perline_width_v1_kekule_ok.csv)")]
    for name, (lo, hi, a_, s_, n) in ip.items():
        if name.startswith("CC stretch"):
            rows.append((name, lo, hi, a_, s_ + k, n, f"isolated bands {s_:+.2f}, plus an empirical {k:+.2f} fitted so that broadened PAHdb sticks reproduce "
                         f"the MLMD 6.2 um width growth ({len(s1 + s2)} species; {k1['6.2']:+.2f} on the {len(s1)} clean species of the version-2 sample"
                         + (f", {k2['6.2']:+.2f} on 60 others)" if s2 else ")")))
        else:
            rows.append((name, lo, hi, a_, s_, n, "isolated in-plane bands (inplane_widths.py)"))
    rows.append(("CH out-of-plane (11.2-13.5 um)", 650, 1000, oop[0], oop[1], oop[2], "isolated oop bands (profile_fit.py; tdep_rules_v1_clean.csv, 11.2 um rule at Nc = 50)"))
    with open(out, "w", newline="") as fo:
        wr = csv.writer(fo); wr.writerow(["family", "lo_cm1", "hi_cm1", "fwhm_300K", "fwhm_slope_per_100K", "n_bands", "source"])
        for name, lo, hi, a_, s_, n, src in rows:
            wr.writerow([name, lo, hi, round(a_, 2), round(s_, 2), n, src])
        wr.writerow(["note: 7.7 um complex", 1220, 1470, "", "", "", "no extra line broadening closes the 7.7 um width growth "
                     f"(k = {k3['7.7']}; see the log of perline_widths_clean.py); carried as an uncertainty, as in version 2"])
    print("->", out)
    old = {r["family"]: r for r in csv.DictReader(open(os.path.join(D1, "perline_widths_v2.csv")))}
    for name, lo, hi, a_, s_, n, _ in rows:
        o = old[name]; print(f"  {name:32s} {a_:5.2f} {s_:+5.2f}   (version 2: {o['fwhm_300K']}, {o['fwhm_slope_per_100K']})")


def load_sticks(xml, species):
    uid = {int(s.split("_v4u")[1]): s for s in species}; st = {}
    with (gzip.open(xml) if xml.endswith(".gz") else open(xml, "rb")) as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag == rc.NS + "specie":
                if int(el.get("uid")) in uid:
                    ms = el.find(rc.NS + "transitions").findall(rc.NS + "mode")
                    st[uid[int(el.get("uid"))]] = (np.array([float(m.find(rc.NS + "frequency").text) for m in ms]),
                                                   np.array([float(m.find(rc.NS + "intensity").text) for m in ms]))
                el.clear()
    return st


if __name__ == "__main__":
    main()
