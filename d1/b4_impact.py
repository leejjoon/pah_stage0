# Diagnostic-impact gate, prototype (committee review B4; docs/plan/b4_diagnostic_impact.md): how much does the
# temperature dependence of band strengths, and its charge dependence, bias the cation fraction recovered from a
# 6.2/11.2 um band ratio?
#   python b4_impact.py <pahdb.xml> <out prefix> [--cmin 20] [--cmax 100] [--energies 6,8,10,12] [--tol 0.05]
# Emission: thermal cascade on PAHdb v4.00 harmonic spectra of closed-shell neutral / doublet cation pairs of the same
# structure. A molecule absorbing a photon of energy E0 reaches T_max with E0 = int C(T) dT (harmonic heat capacity
# from all its modes); while it cools by dE = C(T) dT the energy goes into band i in proportion to sigma_i(T) B(nu_i(T), T)
# (absorption net of stimulated emission times the Planck function, counted once; docs/plan/intensity_convention.md).
# Variants (band classes by scaled harmonic frequency: oop < 1000 cm^-1 (600-1000 for the trends), inplane 1000-1700,
# CHstr > 2500; below 600 cm^-1 no T dependence):
#   v1 harmonic   fixed strengths, 15 cm^-1 redshift, FWHM 15 cm^-1 (the PAHdb standard model)
#   v2 neutral    strengths x (1 + s_class (T - 300 K)/100 K) above 300 K with the MLMD neutral slopes for both charges;
#                 positions shift by p_class T/100 K, FWHM 15 + 1 cm^-1 per 100 K above 300 K
#   v3 charge     as v2, but cations take the GFN2-xTB cation slopes (preliminary, untrustworthy for in-plane
#                 intensities); v3x0.5 and v3x2 scale the cation-minus-neutral slope difference to bracket it
# Slope factors are held at their 1200 K value above 1200 K and clipped to [0.5, 2].
# Diagnostic: band strengths in two window conventions (narrow: 6.2 1570-1650, 11.2 870-910 cm^-1; broad: 1500-1700,
# 850-940 cm^-1) summed over the species of a size bin, per absorbed photon. The cation fraction f (of absorbed
# photons) is recovered by inverting R(f) calibrated with v1, from R observed with v2/v3; bias = f_recovered - f_true.
import argparse, csv, json, math, os, sys
from collections import defaultdict
from xml.etree.ElementTree import iterparse
import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlff_species import graph_hash  # noqa: E402

HC_K = 1.438776877          # h c / k in cm K
EV_CM = 8065.543937         # cm^-1 per eV
L = lambda e: e.tag.split("}")[-1]
# per 100 K: strength fraction (from d1/estimator-convention MD fits, xtb_test/class_slopes.json, 2026-11-12)
SLOPES = {
    "neutral_mlmd": {"oop": +0.008, "inplane": +0.022, "CHstr": -0.019},   # mean of naphthalene, pyrene, coronene
    "cation_xtb": {"oop": +0.029, "inplane": -0.009, "CHstr": +0.152},     # mean of the three cations
}
POS = {"oop": -2.5, "inplane": -5.0, "CHstr": -10.5}                       # cm^-1 per 100 K (band-level MD slopes)
WINDOWS = {"narrow": ((1570, 1650), (870, 910)), "broad": ((1500, 1700), (850, 940))}


def kid(e, t):
    for k in e:
        if L(k) == t:
            return k


def read_pairs(xml, cmin, cmax):
    rec = {}
    for _, el in iterparse(open(xml, "rb"), events=("end",)):
        if L(el) != "specie":
            continue
        atoms = [(int(kid(a, "type").text), [float(kid(a, c).text) for c in "xyz"]) for a in kid(el, "geometry")]
        Z = [z for z, _ in atoms]
        if set(Z) == {1, 6} and cmin <= Z.count(6) <= cmax:
            q = int(kid(el, "charge").text); meth = kid(el, "method").text
            if (q, meth) in ((0, "RB3LYP"), (1, "UB3LYP")):
                tr = [(float(kid(m, "frequency").text), float(kid(m, "intensity").text)) for m in kid(el, "transitions")]
                rec[int(el.get("uid"))] = dict(q=q, nC=Z.count(6), nat=len(Z), Z=Z, P=np.array([p for _, p in atoms]),
                                               nu=np.array([t[0] for t in tr]), A=np.array([t[1] for t in tr]))
        el.clear()
    by = defaultdict(dict)
    for uid, r in rec.items():
        by[graph_hash(r["Z"], r["P"])].setdefault(r["q"], (uid, r))
    return [(d[0][1], d[1][1]) for d in by.values() if 0 in d and 1 in d]


def heat_capacity(nu, T):
    x = HC_K * nu[None, :] / T[:, None]
    ex = np.exp(-x)
    return (x ** 2 * ex / (1 - ex) ** 2).sum(1)                             # in units of k


def cascade(mol, E0_ev, variant, slopes_q, scale=1.0, nT=300, shifts=True):
    nu0, A = mol["nu"], mol["A"]
    E0 = E0_ev * EV_CM                                                        # cm^-1 (energy / hc)
    Tg = np.geomspace(20.0, 4000.0, 4000); C = heat_capacity(nu0, Tg) * (1 / HC_K)   # cm^-1 per K
    U = np.concatenate([[0], np.cumsum(0.5 * (C[1:] + C[:-1]) * np.diff(Tg))])
    Tmax = float(np.interp(E0, U, Tg))
    T = np.geomspace(20.0, Tmax, nT); Cm = np.interp(T, Tg, C); dT = np.gradient(T)
    cls = np.where(nu0 > 2500, "CHstr", np.where(nu0 > 1000, "inplane", np.where(nu0 > 600, "oop", "low")))
    out = np.zeros((nT, len(nu0))); pos = np.zeros((nT, len(nu0))); fwhm = np.zeros(nT)
    for k, t in enumerate(T):
        if variant == "v1":
            g = np.ones_like(A); p = nu0 - 15.0; fw = 15.0
        else:
            te = min(t, 1200.0)
            s = np.array([slopes_q.get(c, 0.0) for c in cls]) * scale
            g = np.clip(1 + s * max(te - 300.0, 0.0) / 100.0, 0.5, 2.0)
            if shifts:
                p = nu0 + np.array([POS.get(c, 0.0) for c in cls]) * t / 100.0
                fw = 15.0 + max(t - 300.0, 0.0) / 100.0
            else:                                                             # strengths only: v1 positions and width
                p = nu0 - 15.0; fw = 15.0
        w = A * g * p ** 3 / np.expm1(HC_K * p / t)                         # sigma_i B(nu_i, T), up to constants
        out[k] = w / w.sum() * Cm[k] * dT[k]                                  # energy (cm^-1) into band i at this T
        pos[k] = p; fwhm[k] = fw
    return out, pos, fwhm, Tmax


def window_energy(out, pos, fwhm, lo, hi):
    s = fwhm[:, None] / 2.3548
    frac = 0.5 * (erf((hi - pos) / (s * 2 ** 0.5)) - erf((lo - pos) / (s * 2 ** 0.5)))
    return float((out * frac).sum())


def pair_contrib(neu, cat, bins, energies, variants):
    """Energy fractions in the band windows per absorbed photon, for one neutral/cation pair, all E0 and variants."""
    E = {}
    b = next((bb for bb in bins if bb[0] <= neu["nC"] < bb[1]), None)
    if b is None:
        return E
    for E0 in energies:
        for vname, sc, shifts in variants:
            for q, mol in ((0, neu), (1, cat)):
                if vname == "v1":
                    sl = {}
                elif vname.startswith("v2") or q == 0:
                    sl = SLOPES["neutral_mlmd"]
                else:                                                          # cation: neutral + scale x (cation - neutral)
                    only = vname.split(":")[1] if ":" in vname else None        # "v3s:CHstr": that class only
                    sl = {c: SLOPES["neutral_mlmd"][c] + (sc if only in (None, c) else 0.0)
                          * (SLOPES["cation_xtb"][c] - SLOPES["neutral_mlmd"][c]) for c in SLOPES["neutral_mlmd"]}
                out, pos, fw, Tmax = cascade(mol, E0, "v1" if vname == "v1" else "vT", sl, 1.0, shifts=shifts)
                tot = out.sum()
                for conv, (w62, w112) in WINDOWS.items():
                    E[(b, E0, conv, "6.2", q, vname)] = window_energy(out, pos, fw, *w62) / tot
                    E[(b, E0, conv, "11.2", q, vname)] = window_energy(out, pos, fw, *w112) / tot
    return E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xml"); ap.add_argument("prefix")
    ap.add_argument("--cmin", type=int, default=20); ap.add_argument("--cmax", type=int, default=100)
    ap.add_argument("--energies", default="6,8,10,12"); ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--by-class", action="store_true", help="v3s with the charge difference in one band class at a time")
    a = ap.parse_args()
    pairs = read_pairs(a.xml, a.cmin, a.cmax)
    print(f"{len(pairs)} neutral/cation pairs with {a.cmin}-{a.cmax} C")
    bins = [(20, 30), (30, 40), (40, 60), (60, 101)]
    variants = [("v1", 1.0, True), ("v2", 1.0, True), ("v3", 1.0, True), ("v3x0.5", 0.5, True), ("v3x2", 2.0, True),
                ("v2s", 1.0, False), ("v3s", 1.0, False), ("v3sx0.5", 0.5, False), ("v3sx2", 2.0, False)]
    if a.by_class:                                                             # which band class drives the charge effect
        variants = [("v1", 1.0, True), ("v2s", 1.0, False), ("v3s", 1.0, False),
                    ("v3s:CHstr", 1.0, False), ("v3s:inplane", 1.0, False), ("v3s:oop", 1.0, False)]
    energies = [float(x) for x in a.energies.split(",")]
    from multiprocessing import Pool
    with Pool(a.jobs) as pool:
        parts = pool.starmap(pair_contrib, [(neu, cat, bins, energies, variants) for neu, cat in pairs], chunksize=4)
    E = defaultdict(float)
    for part in parts:
        for k, v in part.items():
            E[k] += v
    fgrid = np.linspace(0, 1, 1001); rows = []
    for b in bins:
        for E0 in [float(x) for x in a.energies.split(",")]:
            for conv in WINDOWS:
                g = lambda band, q, v: E[(b, E0, conv, band, q, v)]
                if g("11.2", 0, "v1") == 0:
                    continue
                R1 = ((1 - fgrid) * g("6.2", 0, "v1") + fgrid * g("6.2", 1, "v1")) / ((1 - fgrid) * g("11.2", 0, "v1") + fgrid * g("11.2", 1, "v1"))
                for v, *_ in variants[1:]:
                    for ft in (0.1, 0.3, 0.5, 0.7, 0.9):
                        R = ((1 - ft) * g("6.2", 0, v) + ft * g("6.2", 1, v)) / ((1 - ft) * g("11.2", 0, v) + ft * g("11.2", 1, v))
                        fr = float(np.interp(R, R1, fgrid)) if R1[-1] > R1[0] else float("nan")
                        rows.append(dict(bin=f"{b[0]}-{b[1] - 1}", E0=E0, conv=conv, variant=v, f_true=ft, f_rec=round(fr, 4),
                                         bias=round(fr - ft, 4)))
    with open(f"{a.prefix}_bias.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"\nbias in recovered cation fraction (f_rec - f_true), max |bias| over f_true 0.1-0.9; tolerance {a.tol}")
    print(f"{'bin':8s} {'E0':>4s} {'conv':7s} " + " ".join(f"{v:>8s}" for v, *_ in variants[1:]) + "   v3-v2  v3s-v2s")
    for b in bins:
        for E0 in [float(x) for x in a.energies.split(",")]:
            for conv in WINDOWS:
                sel = lambda v: [r for r in rows if r["bin"] == f"{b[0]}-{b[1] - 1}" and r["E0"] == E0 and r["conv"] == conv and r["variant"] == v]
                if not sel("v2s"):
                    continue
                mx = {v: max(abs(r["bias"]) for r in sel(v)) for v, *_ in variants[1:]}
                d32 = max((abs(r3["bias"] - r2["bias"]) for r2, r3 in zip(sel("v2"), sel("v3"))), default=float("nan"))
                d32s = max(abs(r3["bias"] - r2["bias"]) for r2, r3 in zip(sel("v2s"), sel("v3s")))
                print(f"{b[0]:3d}-{b[1] - 1:<4d} {E0:4.0f} {conv:7s} " + " ".join(f"{mx[v]:8.3f}" for v, *_ in variants[1:]) + f"   {d32:6.3f}  {d32s:6.3f}")


if __name__ == "__main__":
    main()
