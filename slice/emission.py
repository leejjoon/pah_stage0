# Emission spectra per absorbed photon for the species pool (prototype stage 2; plan 5.3 with the D1 rules).
#   python slice/emission.py <species csv> <pahdb xml or xml.gz> <out h5> [--jobs 32] [--energies 3,4,5,6,8,10,12]
# Species rows with status = use are computed; source = pahdb takes the PAHdb v4.00 line list of `uid`, source = own
# takes d1/own_spectra/ (our high-spin recomputations, PAHdb frequency scaling).
#
# Cascade (thermal approximation, as d1/b4_impact.py and docs/plan/intensity_convention.md): a photon of energy E
# heats the molecule to T_max with U(T_max) = E (harmonic heat capacity of all modes). While it cools by
# dU = C(T) dT, that energy leaves in line i in proportion to sigma_i(T) nu_i^3 / (exp(h c nu_i / k T) - 1).
# Two variants are written:
#   tdep   D1 temperature rules (d1/tdep_rules_v1.md): at each temperature step every line is shifted by its
#          feature's position rule, scaled by its feature's strength rule and broadened by a Lorentzian of its line
#          family's width (d1/perline_widths_v2.csv); a flat background in 1700-2800 cm^-1 holds 1.6 % of the total
#          intensity at 300 K and +0.9 % per 100 K. Size terms are used where they exceed 2 standard errors, as in
#          the rules page. The rules are valid for 300-900 K: below 300 K the 300 K values are used, above 1200 K the
#          1200 K values (linear extrapolation in between), and strength factors are kept within [0.5, 2].
#   harm   the PAHdb-style reference: fixed harmonic lines, no shift, Gaussian of FWHM 15 cm^-1 (plan 5.3).
# --clamps adds two variants that bound the extrapolation of the rules above 900 K (small molecules reach
# 1,000-1,600 K): tdep900 holds the rules at their 900 K values; tdeplin extrapolates them linearly without an upper
# limit (strength factors within [0.05, 20] only). tdep itself is the middle case.
# Output (HDF5): grid (cm^-1), energies (eV), species, tmax[species, E] (K), tdep and harm[species, E, grid] in eV
# per cm^-1 per absorbed photon, in_grid[species, E, variant] = share of E emitted inside the grid.
import argparse, csv, gzip, os, sys
import xml.etree.ElementTree as ET
from multiprocessing import Pool
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); D1 = os.path.join(HERE, "..", "d1")
sys.path.insert(0, D1)
from own_spectra import load_own_sticks  # noqa: E402

NS = "{http://www.astrochemistry.org/pahdb/theoretical}"
HC_K = 1.438776877           # h c / k, cm K
EV_CM = 8065.543937          # cm^-1 per eV
GRID = np.arange(450.0, 3400.25, 0.5)
T_LO, T_HI, T_START, DT = 300.0, 1200.0, 30.0, 15.0
FEATURES = [("3.3", 2800, 3300), ("6.2", 1540, 1700), ("7.7", 1220, 1470), ("8.6", 1100, 1220), ("11.2", 860, 925),
            ("12.0", 815, 860), ("12.7", 765, 815), ("13.5", 650, 765)]            # tdep_rules_v1.md, table 1
# line families of perline_widths_v2.csv; lines outside every family take the nearest family's width
WIDTH_FALLBACK = [(0, 650, 650), (1000, 1100, 1100), (1700, 2800, 1470), (3300, 1e9, 2800)]   # lo, hi, family lo
PED = (1700.0, 2800.0, 0.016, 0.009)                                                # range, share at 300 K, per 100 K
VARIANTS = {"tdep": (T_HI, 0.5, 2.0), "tdep900": (900.0, 0.5, 2.0), "tdeplin": (np.inf, 0.05, 20.0)}   # upper T of the rules, strength clip


def load_rules():
    R = {(r["band"], r["quantity"]): r for r in csv.DictReader(open(os.path.join(D1, "tdep_rules_v1.csv")))}

    def rule(band, q):
        r = R[(band, q)]; a, b, sb = float(r["at_Nc50"]), float(r["per_decade_Nc"]), float(r["se_decade"])
        return a, (b if abs(b) > 2 * sb else 0.0)
    W = [(float(r["lo_cm1"]), float(r["hi_cm1"]), float(r["fwhm_300K"]), float(r["fwhm_slope_per_100K"]))
         for r in csv.DictReader(open(os.path.join(D1, "perline_widths_v2.csv"))) if r["fwhm_300K"]]
    return {b: dict(pos=rule(b, "position"), strength=rule(b, "strength")) for b, *_ in FEATURES}, W


RULES, WIDTHS = load_rules()


def line_rules(nu, nc):
    """Per line: position slope (cm^-1 per 100 K), strength slope (fraction per 100 K), FWHM at 300 K and its slope."""
    x = np.log10(nc / 50.0)
    ps = np.zeros_like(nu); ss = np.zeros_like(nu); w0 = np.zeros_like(nu); ws = np.zeros_like(nu)
    for b, lo, hi in FEATURES:
        m = (nu >= lo) & (nu < hi)
        ps[m] = RULES[b]["pos"][0] + RULES[b]["pos"][1] * x
        ss[m] = RULES[b]["strength"][0] + RULES[b]["strength"][1] * x
    fam = {lo: (a, s) for lo, hi, a, s in WIDTHS}
    for lo, hi, a, s in WIDTHS:
        m = (nu >= lo) & (nu < hi); w0[m] = a; ws[m] = s
    for lo, hi, f in WIDTH_FALLBACK:
        m = (nu >= lo) & (nu < hi); w0[m], ws[m] = fam[f]
    assert (w0 > 0).all()
    return ps, ss, w0, ws


def cascade(args):
    name, nc, nu, A, energies, variants = args
    keep = nu > 0; nu, A = nu[keep], A[keep]
    Tg = np.geomspace(10.0, 6000.0, 4000)
    x = HC_K * nu[None, :] / Tg[:, None]; ex = np.exp(-x)
    C = (x ** 2 * ex / (1 - ex) ** 2).sum(1) / HC_K                                  # cm^-1 per K
    U = np.concatenate([[0.0], np.cumsum(0.5 * (C[1:] + C[:-1]) * np.diff(Tg))])
    tmax = np.interp(np.array(energies) * EV_CM, U, Tg)
    edges = np.unique(np.concatenate([np.arange(T_START, tmax.max(), DT), tmax]))
    ps, ss, w0, ws = line_rules(nu, nc)
    act = A > 1e-6 * A.sum()                                                          # lines that can emit
    nua, Aa, psa, ssa, w0a, wsa = nu[act], A[act], ps[act], ss[act], w0[act], ws[act]
    ped = (GRID >= PED[0]) & (GRID < PED[1]); dg = GRID[1] - GRID[0]
    harm = np.zeros((len(energies), GRID.size), np.float32); tdep = {v: np.zeros_like(harm) for v in variants}
    acc = {v: np.zeros(GRID.size) for v in variants}; eline = np.zeros(nua.size); done = 0
    sig = 15.0 / 2.3548
    for lo, hi in zip(edges[:-1], edges[1:]):
        T = 0.5 * (lo + hi); dU = np.interp(hi, Tg, U) - np.interp(lo, Tg, U)
        # harmonic reference
        w = Aa * nua ** 3 / np.expm1(HC_K * nua / T)
        eline += dU * w / w.sum()
        # temperature rules (variants that share the rule temperature and the clip share the step)
        step = {}
        for v in variants:
            thi, c0, c1 = VARIANTS[v]; dTe = (min(max(T, T_LO), thi) - 300.0) / 100.0
            if (dTe, c0, c1) not in step:
                p = nua + psa * dTe; g = np.clip(1 + ssa * dTe, c0, c1); gam = 0.5 * (w0a + wsa * dTe)
                w = Aa * g * p ** 3 / np.expm1(HC_K * p / T)
                wped = np.zeros(GRID.size)
                wped[ped] = (PED[2] + PED[3] * dTe) * (Aa * g).sum() / (PED[1] - PED[0]) * GRID[ped] ** 3 / np.expm1(HC_K * GRID[ped] / T)
                norm = w.sum() + wped.sum() * dg
                big = w > 1e-7 * norm
                L = (gam[big, None] / np.pi) / ((GRID[None, :] - p[big, None]) ** 2 + gam[big, None] ** 2)
                step[(dTe, c0, c1)] = dU / norm * ((w[big, None] * L).sum(0) + wped)
            acc[v] += step[(dTe, c0, c1)]
        while done < len(energies) and hi >= tmax[done] - 1e-9:                      # snapshot when T_max(E) is reached
            for v in variants:
                tdep[v][done] = acc[v] / EV_CM
            G = np.exp(-0.5 * ((GRID[None, :] - nua[:, None]) / sig) ** 2) / (sig * np.sqrt(2 * np.pi))
            harm[done] = (eline[:, None] * G).sum(0) / EV_CM
            done += 1
    ing = np.stack([tdep["tdep"].sum(1) * dg / np.array(energies), harm.sum(1) * dg / np.array(energies)], 1)
    return name, tmax, tdep, harm, ing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species"); ap.add_argument("xml"); ap.add_argument("out")
    ap.add_argument("--jobs", type=int, default=32); ap.add_argument("--energies", default="3,4,5,6,8,10,12")
    ap.add_argument("--clamps", action="store_true")
    a = ap.parse_args()
    import h5py
    energies = [float(x) for x in a.energies.split(",")]
    rows = [r for r in csv.DictReader(open(a.species)) if r["status"] == "use"]
    own = load_own_sticks() if any(r["source"] == "own" for r in rows) else {}
    want = {int(r["uid"]) for r in rows if r["source"] == "pahdb"}; sticks = {}
    with (gzip.open(a.xml) if a.xml.endswith(".gz") else open(a.xml, "rb")) as f:
        for ev, el in ET.iterparse(f, events=("end",)):
            if el.tag != NS + "specie":
                continue
            if int(el.get("uid")) in want:
                ms = el.find(NS + "transitions").findall(NS + "mode")
                sticks[int(el.get("uid"))] = (np.array([float(m.find(NS + "frequency").text) for m in ms]),
                                              np.array([float(m.find(NS + "intensity").text) for m in ms]))
            el.clear()
    tasks = []; variants = list(VARIANTS) if a.clamps else ["tdep"]
    for r in rows:
        nu, A = own[r["species"]] if r["source"] == "own" else sticks[int(r["uid"])]
        tasks.append((r["species"], int(r["nC"]), nu, A, energies, variants))
    print(f"{len(tasks)} species, {len(energies)} photon energies, grid {GRID[0]:.0f}-{GRID[-1]:.0f} cm^-1 ({GRID.size} points)", flush=True)
    with Pool(a.jobs) as pool, h5py.File(a.out, "w") as h:
        h["grid"] = GRID; h["energies"] = energies
        h["species"] = np.array([t[0] for t in tasks], dtype="S")
        dt = {v: h.create_dataset(v, (len(tasks), len(energies), GRID.size), "f4") for v in variants}
        dh = h.create_dataset("harm", (len(tasks), len(energies), GRID.size), "f4")
        tm = np.zeros((len(tasks), len(energies))); ing = np.zeros((len(tasks), len(energies), 2))
        idx = {t[0]: i for i, t in enumerate(tasks)}
        for k, (name, tmax, td, hm, ig) in enumerate(pool.imap_unordered(cascade, tasks, chunksize=2)):
            i = idx[name]; dh[i] = hm; tm[i] = tmax; ing[i] = ig
            for v in variants:
                dt[v][i] = td[v]
            if (k + 1) % 100 == 0:
                print(f"  {k + 1}/{len(tasks)}", flush=True)
        h["tmax"] = tm; h["in_grid"] = ing
        h.attrs["variants"] = "tdep: d1/tdep_rules_v1 + perline_widths_v2; harm: fixed lines, Gaussian FWHM 15 cm^-1"
    nc = np.array([t[1] for t in tasks])
    for j, E in enumerate(energies):
        print(f"  {E:4.0f} eV: T_max median {np.median(tm[:, j]):5.0f} K (Nc<40: {np.median(tm[nc < 40, j]):5.0f}, Nc>=100: "
              f"{np.median(tm[nc >= 100, j]):5.0f}); share of E inside the grid tdep {np.median(ing[:, j, 0]):.3f}, harm {np.median(ing[:, j, 1]):.3f}")


if __name__ == "__main__":
    main()
