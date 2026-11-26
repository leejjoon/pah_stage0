# D1 cation finite-T rules (d1/tdep_rules_cation_v1; coordinator 2026-11-26 08:58, step 4) from the cation MLMD
# spectra (cation_mlmd_produce.py -> cation_mlmd_dipoles.py -> cation_spec_export.py):
#   python cation_rules.py fit <species csv> <spectra dir with 300/600/900> <fits csv> [--jobs 32]
#   python cation_rules.py rules <fits csv> <rules csv> [--fixed <fixed-window rules csv>]
#   python cation_rules.py widths <fits csv> <rules csv> <species csv> <pahdb xml> <widths csv> [--jobs 32]
# fit: profile_fit.fit_species per species (L from the files), regions oop (650-1000) and in-plane (1100-1700), and the
#   fixed-window complexes; one csv.
# rules: same table as tdep_rules_v1_clean.csv (profile_fit.py rules: per species the slope against each replica's
#   NVE T, then OLS in log10(Nc/50) over species: value at Nc = 50, per decade, s.e., scatter, quad - lin at 1200 K),
#   one row per band and quantity (emission.py keys the table by band and quantity and uses position and strength):
#   - 6.2, 7.7, 8.6 um position and strength: shifting band profiles, i.e. the in-plane profile components tracked
#     from 300 to 900 K (profile_fit refits each T from the previous one, shift bounds -40/+15 cm^-1), assigned to a
#     band by their 300 K replica-mean centre in emission.py's line ranges (6.2: 1540-1700, 7.7: 1220-1470, 8.6:
#     1100-1220); position = area-weighted mean centre, strength = summed area (fraction of the 300-3800 total),
#     relative to its 300 K value; tracked components whose FWHM reaches MAXW = 100 cm^-1 at any T are dropped (failed
#     fits; also applied to the isolated oop components). 8.6 um strength: the fixed-window area instead (tracked sum not estimable; cmd_rules, --tracked-strength).
#     Width and width300 of these bands: the fixed-window rms width, as the neutral table (not used by emission.py).
#   - 3.3 um: the fixed-window complex (2850-3250; the band stays inside it to 900 K), as the neutral table.
#   - 11.2, 12.0, 12.7, 13.5 um: isolated oop profile components, as the neutral table (components overlapping
#     another at 900 K dropped).
#   --fixed: the neutral method throughout (fixed windows for 3.3-8.6), for comparison with tdep_rules_v1_clean.csv.
# widths: per-line widths in the perline_widths_v2_clean.csv layout (same family lo_cm1 values, which emission.py
#   matches), made the way perline_widths_clean.py makes the neutral table, on the cation data:
#   CH stretch: per species and T the per-line FWHM for which the PAHdb cation sticks, broadened with the cation
#   rules, reproduce the MLMD 3.3 um FWHM (rules_closure.invert's bisection); median at 300 K, median slope.
#   In-plane families: isolated in-plane bands (inplane_widths.collect). Out-of-plane: the cation 11.2 um rule at
#   Nc = 50. 6.2 um extra slope k: scanned so that sticks broadened with the cation rules and widths reproduce the
#   MLMD growth of the 6.2 um feature rms width, 300 -> 900 K (median over species crosses zero); 7.7 um reported.
import argparse, csv, gzip, os, sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import profile_fit as pf  # noqa: E402
import rules_closure as rc  # noqa: E402
import inplane_widths as iw  # noqa: E402

TS = (300, 600, 900)
MAXW = 100.0     # cm^-1: a tracked component whose replica-mean FWHM reaches this at any T is a failed fit, not a band (a weak
                 # narrow line refitted from the previous T into a broad pedestal-like component, up to the 150 bound, that
                 # takes up background: C34H16_v4u7230 8.6 um area 0.01 -> 0.65, FWHM 4 -> 148, 300 -> 900 K); dropped from
                 # the band's sum at every T (cf. the oop rule's overlap drop). Real lines: median FWHM 5, 99th pct ~50 at 300 K.
TRACKED = [("6.2", 1540, 1700), ("7.7", 1220, 1470), ("8.6", 1100, 1220)]       # emission.py FEATURES ranges
OOP = ["11.2", "12.0", "12.7", "13.5"]
KIND = {"3.3": "complex (centroid, FWHM)", "6.2": "tracked in-plane components (area-weighted centre, summed area); width: complex rms",
        "7.7": "tracked in-plane components (area-weighted centre, summed area); width: complex rms",
        "8.6": "tracked in-plane components (area-weighted centre, summed area); width: complex rms"}
_A = {}


def _fit(sp):
    import warnings; warnings.simplefilter("ignore")
    try:
        return pf.fit_species(sp, {T: f"{_A['spec']}/{T}" for T in TS}, [("oop", 650, 1000), ("inplane", 1100, 1700)], complexes=True)
    except Exception as e:                                       # noqa: BLE001
        print(f"{sp}: fit failed ({e})", flush=True); return []


def cmd_fit(a):
    sp = [r["species"] for r in csv.DictReader(open(a.species))]
    sp = [s for s in sp if all(os.path.exists(f"{a.spec}/{T}/{s}_{T}K.npz") for T in TS)]
    _A["spec"] = a.spec; rows = []
    with Pool(a.jobs) as p:
        for r in p.imap(_fit, sp):
            rows += r
    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(sp)} species, {len(rows)} rows -> {a.out}")


def slope(v, q, rel=False):
    x = np.array([r["x"] for r in v]); y = np.array([r[q] for r in v]); y300 = np.mean([r[q] for r in v if r["T"] == 300])
    if rel:
        y = y / y300
    c1 = np.polyfit(x, y, 1); c2 = np.polyfit(x, y, 2)
    return c1[0], np.polyval(c2, 12) - np.polyval(c1, 12), y300


def per_species(rows, fixed=False, maxw=None):
    """{band: [(nC, {quantity: (slope per 100 K, quad - lin at 1200 K, value at 300 K)})]}"""
    res = {}; by = {}
    for r in rows:
        if r["replica"] != "mean":
            by.setdefault((r["species"], r["region"]), []).append(r)
    for (sp, reg), v in sorted(by.items()):                     # "complex" before "inplane": tracked widths use the complex rms
        nc = int(v[0]["nC"])
        if reg == "complex":
            for b in ("3.3",) if not fixed else ("3.3", "6.2", "7.7", "8.6"):
                w = [dict(T=int(r["T"]), x=float(r["T_nve"]) / 100, peak=float(r["peak"]), center=float(r["center"]),
                          rmsw=float(r["rmsw"]), fwhm=float(r["fwhm"]), win=float(r["win"])) for r in v if r["band"] == b]
                if len(w) < 9:
                    continue
                pq, wq = ("center", "fwhm") if b == "3.3" else ("peak", "rmsw")
                res.setdefault(b, []).append((nc, dict(position=slope(w, pq), width=slope(w, wq), strength=slope(w, "win", True))))
            if not fixed:                                        # widths of the tracked bands: complex rms, as the neutrals
                for b in ("6.2", "7.7", "8.6"):
                    w = [dict(T=int(r["T"]), x=float(r["T_nve"]) / 100, rmsw=float(r["rmsw"])) for r in v if r["band"] == b]
                    if len(w) >= 9:
                        _A.setdefault("cw", {})[(sp, b)] = slope(w, "rmsw")
        elif reg == "oop":
            comps = {}; owmax = {}
            for r in v:
                comps.setdefault(r["comp"], []).append(r)
            for r in rows:                                       # the MAXW failed-fit cut, as for the tracked bands
                if r["species"] == sp and r["region"] == "oop" and r["replica"] == "mean":
                    owmax[r["comp"]] = max(owmax.get(r["comp"], 0.0), float(r["fwhm"]))
            for c, w in comps.items():
                b300 = [r for r in w if r["T"] == "300"]
                if len(b300) < 3 or len(w) < 9 or not b300[0]["band"] or any(r["overlap"] == "1" for r in w if r["T"] == "900") \
                        or owmax.get(c, 0.0) >= (MAXW if maxw is None else maxw):
                    continue
                ww = [dict(T=int(r["T"]), x=float(r["T_nve"]) / 100, center=float(r["center"]), fwhm=float(r["fwhm"]), area=float(r["area"])) for r in w]
                res.setdefault(b300[0]["band"], []).append((nc, dict(position=slope(ww, "center"), width=slope(ww, "fwhm"), strength=slope(ww, "area", True))))
        elif reg == "inplane" and not fixed:
            c300, wmax = {}, {}                                  # component -> 300 K replica-mean centre, max FWHM over T
            for r in rows:
                if r["species"] == sp and r["region"] == "inplane" and r["replica"] == "mean":
                    wmax[r["comp"]] = max(wmax.get(r["comp"], 0.0), float(r["fwhm"]))
                    if r["T"] == "300":
                        c300[r["comp"]] = float(r["center"])
            mw = MAXW if maxw is None else maxw
            for b, lo, hi in TRACKED:
                cs = {c for c, x0 in c300.items() if lo <= x0 < hi and wmax.get(c, 0.0) < mw}
                if not cs:
                    continue
                agg = {}
                for r in v:
                    if r["comp"] in cs:
                        k = (int(r["T"]), r["replica"]); a_ = float(r["area"])
                        g = agg.setdefault(k, dict(T=int(r["T"]), x=float(r["T_nve"]) / 100, s=0.0, sc=0.0))
                        g["s"] += a_; g["sc"] += a_ * float(r["center"])
                w = [dict(g, center=g["sc"] / g["s"], area=g["s"]) for g in agg.values() if g["s"] > 0]
                if len(w) < 9 or sum(1 for g in w if g["T"] == 300) < 3:
                    continue
                cw = _A.get("cw", {}).get((sp, b))
                if cw is None:
                    continue
                res.setdefault(b, []).append((nc, dict(position=slope(w, "center"), strength=slope(w, "area", True), width=cw)))
    return res


def table(res, kinds):
    import warnings; warnings.simplefilter("ignore")
    rows = []
    for b in ["3.3", "6.2", "7.7", "8.6"] + OOP:
        v = res.get(b, [])
        if len(v) < 10:
            print(f"{b}: {len(v)} species, no rule"); continue
        X = np.log10(np.array([n for n, _ in v]) / 50.0)
        for name, unit in (("position", "cm-1 per 100 K"), ("width", "cm-1 per 100 K"), ("strength", "fraction of 300 K per 100 K"),
                           ("width300", "cm-1 at 300 K")):
            q, i = ("width", 2) if name == "width300" else (name, 0)
            y = np.array([o[q][i] for _, o in v])
            A = np.vstack([np.ones_like(X), X]).T; c, *_ = np.linalg.lstsq(A, y, rcond=None)
            r = y - A @ c; s2 = (r ** 2).sum() / (len(y) - 2); cov = s2 * np.linalg.inv(A.T @ A)
            ql = np.median([o[q][1] for _, o in v]) if name != "width300" else np.nan
            rows.append(dict(band=b, kind=kinds.get(b, "isolated oop band (truncated-Lorentzian fit)"), quantity=name, unit=unit,
                             n_species=len(v), at_Nc50=round(c[0], 4), se=round(np.sqrt(cov[0, 0]), 4),
                             per_decade_Nc=round(c[1], 4), se_decade=round(np.sqrt(cov[1, 1]), 4),
                             scatter=round(1.4826 * np.median(np.abs(r - np.median(r))), 4), quad_minus_lin_1200K=round(ql, 4)))
    return rows


def write(rows, out):
    with open(out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"{len(rows)} rules -> {out}")
    for r in rows:
        print(f"  {r['band']:>5s} {r['quantity']:9s} {r['at_Nc50']:+9.4f} +- {r['se']:.4f}  per decade {r['per_decade_Nc']:+8.4f} +- "
              f"{r['se_decade']:.4f}  scatter {r['scatter']:.4f}  q-l {r['quad_minus_lin_1200K']:+.3f}  (n={r['n_species']})")


FIXED_KIND = {"3.3": "complex (centroid, FWHM)", "6.2": "complex (peak, rms width)", "7.7": "complex (peak, rms width)",
              "8.6": "complex (peak, rms width)"}
WINDOW_STRENGTH = ("8.6",)   # bands whose strength row is the fixed-window area (see below)


def cmd_rules(a):
    """8.6 um strength: the fixed-window (1100-1220 = emission.py's FEATURES range, whose lines the strength rule scales)
    area, not the tracked sum. The cation 8.6 band is weak (tracked 300 K area 0.4-4 % of the total), so area / area(300 K)
    of its tracked components is not estimable: the fit moves intensity between overlapping weak components as they
    broaden (per-species slopes up to +3.4 per 100 K after the MAXW cut; OLS +0.15 vs median +0.03, scatter 0.41). The band
    shifts by only ~2 cm^-1 per 100 K, so it stays inside the window to 900 K (unlike 6.2, whose tracked strength is kept).
    --tracked-strength keeps the tracked row."""
    rows = list(csv.DictReader(open(a.fits)))
    main = table(per_species(rows, maxw=a.maxw), KIND); fixed = table(per_species(rows, fixed=True), FIXED_KIND)
    if not a.tracked_strength:
        fx = {(r["band"], r["quantity"]): r for r in fixed}
        for i, r in enumerate(main):
            if r["band"] in WINDOW_STRENGTH and r["quantity"] == "strength":
                main[i] = dict(fx[(r["band"], "strength")], kind="complex window area (1100-1220; tracked sum not estimable, see md)")
    write(main, a.out)
    if a.fixed:
        write(fixed, a.fixed)


# ---- per-line widths ------------------------------------------------------------------------------------------------
KGRID = np.arange(0, 16.1, 1.0)   # cap: k > 16 cm^-1 per 100 K (3x the neutral +5.26) would not be line broadening. Cations: the
                                  # 6.2 um gap is -0.65 at k = 16, crosses at k ~ 19 (scan to 48), so no extra term is applied
FAM = {"6.2": (1540, 1700, (1540, 1660)), "7.7": (1220, 1470, (1220, 1470))}


def load_sticks(xml, uids):
    st = {}
    with (gzip.open(xml) if xml.endswith(".gz") else open(xml, "rb")) as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag == rc.NS + "specie":
                if el.get("uid") in uids:
                    ms = el.find(rc.NS + "transitions").findall(rc.NS + "mode")
                    st[uids[el.get("uid")]] = (np.array([float(m.find(rc.NS + "frequency").text) for m in ms]),
                                               np.array([float(m.find(rc.NS + "intensity").text) for m in ms]))
                el.clear()
    return st


def spectrum(nu, I, T, nc, extra):
    R, W = _A["R"], _A["W"]; dT = (T - 300) / 100; y = np.zeros_like(rc.GRID)
    for f, a_ in zip(nu, I):
        fam = next((b for b, lo, hi in rc.FAMILY if lo <= f < hi), None)
        f2 = f + (rc.rule(R, fam, "position", nc) * dT if fam else 0)
        w = next((w0 + sl * dT for lo, hi, w0, sl in W if lo <= f < hi), W[-1][2] + W[-1][3] * dT)
        for lo, hi, k in extra:
            if lo <= f < hi:
                w += k * dT
        y += a_ * (w / 2) ** 2 / ((rc.GRID - f2) ** 2 + (w / 2) ** 2) / (np.pi * w / 2)
    return y


def growth(job):
    sp, nu, I, fam = job; lo, hi, (wlo, whi) = FAM[fam]; nc = int(sp[1:sp.index("H")])
    w300 = pf.complex_metrics(rc.GRID, spectrum(nu, I, 300, nc, []), wlo, whi)[1]
    return sp, fam, [pf.complex_metrics(rc.GRID, spectrum(nu, I, 900, nc, [(lo, hi, k)]), wlo, whi)[1] - w300 for k in KGRID]


def ch_invert(job):
    """Per-line FWHM of the CH stretch for which the broadened sticks reproduce the MLMD 3.3 um FWHM at each T."""
    sp, nu, I, targets = job; nc = int(sp[1:sp.index("H")]); out = {}
    for T, target in targets.items():
        def width(w):
            dT = (T - 300) / 100; y = np.zeros_like(rc.GRID)
            for f, a_ in zip(nu, I):
                fam = next((b for b, lo, hi in rc.FAMILY if lo <= f < hi), None)
                f2 = f + (rc.rule(_A["R"], fam, "position", nc) * dT if fam else 0)
                y += a_ * (w / 2) ** 2 / ((rc.GRID - f2) ** 2 + (w / 2) ** 2) / (np.pi * w / 2)
            return pf.complex_metrics(rc.GRID, y, 2850, 3250)[2]
        lo, hi = 0.3, 300.0
        if not (width(lo) <= target <= width(hi)):
            out[T] = np.nan; continue
        for _ in range(30):
            m = np.sqrt(lo * hi); lo, hi = (m, hi) if width(m) < target else (lo, m)
        out[T] = np.sqrt(lo * hi)
    return sp, out


def cmd_widths(a):
    fits = list(csv.DictReader(open(a.fits)))
    R = {(r["band"], r["quantity"]): r for r in csv.DictReader(open(a.rules))}
    for b, *_ in rc.FAMILY:                                      # bands without a cation rule would fail in rc.rule
        assert (b, "position") in R, f"no cation position rule for {b}"
    _A["R"] = R
    spl = list(csv.DictReader(open(a.species))); uids = {r["uid"]: r["species"] for r in spl}
    st = load_sticks(a.xml, uids)
    md = {}
    for r in fits:
        if r["region"] == "complex" and r["replica"] != "mean":
            md.setdefault((r["species"], r["band"], int(r["T"])), []).append(r)
    st = {s: v for s, v in st.items() if all((s, "3.3", T) in md for T in TS)}
    print(f"{len(st)} species with sticks and MLMD complexes")
    # CH stretch
    jobs = [(s, *st[s], {T: np.mean([float(r["fwhm"]) for r in md[(s, "3.3", T)]]) for T in TS}) for s in st]
    with Pool(a.jobs) as p:
        inv = dict(p.map(ch_invert, jobs))
    w = np.array([[inv[s][T] for T in TS] for s in inv if all(np.isfinite(inv[s][T]) for T in TS)])
    ch = (float(np.median(w[:, 0])), float(np.median(np.polyfit([3, 6, 9], w.T, 1)[0])), len(w))
    print(f"CH stretch: {ch[2]} species, FWHM 300 K {ch[0]:.2f}, slope {ch[1]:+.2f} per 100 K")
    # in-plane
    tmp = a.out + ".inplane_rows.csv"
    with open(tmp, "w", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=list(fits[0])); wr.writeheader(); wr.writerows([r for r in fits if r["region"] == "inplane"])
    fam, n_sp, n_all = iw.collect([tmp], None); os.remove(tmp)
    ip = {}
    for name, lo, hi in iw.FAMILIES:
        v = np.array(fam[name]) if fam[name] else np.zeros((0, 4))
        ip[name] = (lo, hi, float(np.median(v[:, 1])) if len(v) else np.nan, float(np.median(v[:, 2])) if len(v) else np.nan, len(v))
        print(f"{name}: {len(v)} isolated bands of {n_sp} species, FWHM 300 K {ip[name][2]:.2f}, slope {ip[name][3]:+.2f}")
    oop = (float(R[("11.2", "width300")]["at_Nc50"]), float(R[("11.2", "width")]["at_Nc50"]),
           sum(int(R[(b, "width")]["n_species"]) for b in OOP if (b, "width") in R))
    print(f"CH out-of-plane: cation 11.2 um rule at Nc = 50: FWHM 300 K {oop[0]:.2f}, slope {oop[1]:+.2f}")
    r2 = lambda x: round(x, 2)
    _A["W"] = [(2800, 3300, r2(ch[0]), r2(ch[1]))] + [(lo, hi, r2(a_), r2(s_)) for lo, hi, a_, s_, n in ip.values() if n] + \
              [(650, 1000, r2(oop[0]), r2(oop[1]))]
    with Pool(a.jobs) as p:
        res = {}
        for sp, f_, g in p.imap_unordered(growth, [(s, *st[s], f) for s in st for f in FAM]):
            if all((sp, f_, T) in md for T in (300, 900)):
                res[(sp, f_)] = np.array(g) - (np.mean([float(r["rmsw"]) for r in md[(sp, f_, 900)]]) - np.mean([float(r["rmsw"]) for r in md[(sp, f_, 300)]]))
    best = {}
    for f_ in FAM:
        d = np.array([v for (s, ff), v in res.items() if ff == f_]); med = np.median(d, 0); i = np.nonzero(np.diff(np.sign(med)))[0]
        if f_ == "6.2":
            _A["gap62"] = (med[0], med[-1])
        best[f_] = float(np.interp(0, med[i[0]:i[0] + 2], KGRID[i[0]:i[0] + 2])) if len(i) and med[i[0]] < med[i[0] + 1] else None
        print(f"  {f_} um, {len(d)} species: median (rules - MLMD) growth of the rms width {med[0]:+.2f} at k = 0, {med[-1]:+.2f} at k = {KGRID[-1]:.0f}; "
              + (f"k = {best[f_]:.2f} cm^-1 per 100 K" if best[f_] is not None else "no zero crossing"))
    k = best["6.2"] if best["6.2"] is not None else 0.0
    rows = [("CH stretch (3.3 um)", 2800, 3300, ch[0], ch[1], f"{ch[2]} species",
             "inversion of the 3.3 um feature FWHM on PAHdb cation sticks broadened with tdep_rules_cation_v1 (rules_closure bisection)")]
    for name, (lo, hi, a_, s_, n) in ip.items():
        if not n:
            continue
        if name.startswith("CC stretch"):
            rows.append((name, lo, hi, a_, s_ + k, n, f"isolated bands {s_:+.2f}, plus an empirical {k:+.2f} fitted so that broadened PAHdb cation sticks "
                         f"reproduce the cation MLMD 6.2 um width growth ({len(d)} species)" if best["6.2"] is not None else
                         f"isolated bands {s_:+.2f}; no extra term: the 6.2 um growth gap (rules - MLMD rms width growth, median) is {_A['gap62'][0]:+.2f} at k = 0 and "
                         f"{_A['gap62'][1]:+.2f} at k = {KGRID[-1]:.0f}, i.e. no crossing for k <= {KGRID[-1]:.0f} cm^-1 per 100 K (see md)"))
        else:
            rows.append((name, lo, hi, a_, s_, n, "isolated in-plane bands (inplane_widths.py), cation MLMD"))
    rows.append(("CH out-of-plane (11.2-13.5 um)", 650, 1000, oop[0], oop[1], oop[2], "isolated oop bands (profile_fit.py; tdep_rules_cation_v1.csv, 11.2 um rule at Nc = 50)"))
    with open(a.out, "w", newline="") as fo:
        wr = csv.writer(fo); wr.writerow(["family", "lo_cm1", "hi_cm1", "fwhm_300K", "fwhm_slope_per_100K", "n_bands", "source"])
        for name, lo, hi, a_, s_, n, src in rows:
            wr.writerow([name, lo, hi, round(a_, 2), round(s_, 2), n, src])
        wr.writerow(["note: 7.7 um complex", 1220, 1470, "", "", "", f"extra line broadening that closes the cation 7.7 um width growth: k = {best['7.7']}; not applied"])
    print("->", a.out)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit"); f.add_argument("species"); f.add_argument("spec"); f.add_argument("out"); f.add_argument("--jobs", type=int, default=32)
    r = sp.add_parser("rules"); r.add_argument("fits"); r.add_argument("out"); r.add_argument("--fixed", default="")
    r.add_argument("--tracked-strength", action="store_true", help="keep the tracked 8.6 um strength row")
    r.add_argument("--maxw", type=float, default=None, help=f"tracked-component FWHM cut (default MAXW = {MAXW}; 1e9: no cut)")
    w = sp.add_parser("widths"); w.add_argument("fits"); w.add_argument("rules"); w.add_argument("species"); w.add_argument("xml"); w.add_argument("out")
    w.add_argument("--jobs", type=int, default=32)
    a = ap.parse_args()
    {"fit": cmd_fit, "rules": cmd_rules, "widths": cmd_widths}[a.cmd](a)


if __name__ == "__main__":
    main()
