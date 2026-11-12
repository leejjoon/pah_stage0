# Temperature dependence of IR bands: GFN2-xTB neutral vs cation (d1/xtb_md.py), and the neutral MLMD reference
# (d1/mlmd_run.py), for docs/plan/charged_tdep_validation.md. Spectra with the authors' calc_MD_IR maths
# (mlmd_spectra.ir_sigma, qm 3 unclipped by default (see below), frequencies scaled by 0.9578 for both methods: only shifts and widths relative to
# 300 K are compared, never absolute positions across methods).
#   python tdep_compare.py <xtb runs dir> <mlmd run dir> <png or ->
# Per species, method/charge and temperature, per replica: for three regions (CH out-of-plane 600-1000, CC/CH
# in-plane 1000-1700, CH stretch 2700-3300 cm^-1) the intensity-weighted centroid, rms width and share of the
# 300-3800 total; then the replica mean and its standard error. Prints the change from 300 K and the
# cation-minus-neutral difference with its error.
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_spectra import ir_sigma
from mlmd_integrity import bonds, read_xyz

# Intensity convention (committee review B1; d1/estimator_check.py): qm=3 (harmonic quantum correction) without
# clipping negative values, with each replica's own mean NVE temperature in the prefactor, reproduces harmonic
# integrated intensities at every T. The authors' convention (qm=0, clipped, nominal T) inflates high-frequency bands
# with T (a 1613/893 cm^-1 ratio by 27 % from 300 to 1200 K for purely harmonic data) and narrow-line integrals.
# Environment: QM (default 3), CLIP (default 0), TNOM=1 to use the nominal T in the prefactor (old behaviour).
QM = int(os.environ.get("QM", 3)); CLIP = os.environ.get("CLIP", "0") == "1"; TNOM = os.environ.get("TNOM", "0") == "1"
SKIPPED = []                                    # trajectories left out by the integrity check

REGIONS = {"oop 600-1000": (600, 1000), "inplane 1000-1700": (1000, 1700), "CHstr 2700-3300": (2700, 3300)}
MLMD_NAME = {"C10H8_330": "naphthalene", "C16H10_334": "pyrene", "C24H12_18": "coronene"}


def metrics(f, s):
    """Per window: centroid and rms width from the non-negative part of the spectrum (shape), and the share of the
    300-3800 cm^-1 total from the full spectrum (an unclipped spectrum's ripples cancel over a window)."""
    tot = s.sum(); out = {}
    for name, (lo, hi) in REGIONS.items():
        m = (f >= lo) & (f < hi); x = f[m]; w = np.clip(s[m], 0, None)
        c = (w * x).sum() / w.sum()
        out[name] = (c, np.sqrt((w * (x - c) ** 2).sum() / w.sum()), s[m].sum() / tot)
    return out


def load_xtb(d):
    R = defaultdict(list)                      # (species, label, T) -> [metrics per replica]
    for f in sorted(glob.glob(f"{d}/*.npz")):
        z = np.load(f)
        if int(z["frames_done"]) < len(z["dipole"]) or len(z["dipole"]) < 1000:
            continue
        sp, q, T = re.match(r"(\w+?)_q(\d)_(\d+)K_r\d", os.path.basename(f)).groups()
        sym, p0 = read_xyz(str(z["xyz"]))
        if bonds(sym, p0) != bonds(sym, z["positions"]):          # bond graph changed by the end of the run
            SKIPPED.append(os.path.basename(f)); continue
        Tm = float(np.mean(np.array(z["qa"])[:, 1]))
        fr, s = ir_sigma(z["dipole"], float(T) if TNOM else Tm, 1.0, QM, CLIP)
        R[(sp, f"xTB q={q}", float(T))].append((fr, s, Tm))
    return R


def load_mlmd(d, R):
    for meta in sorted(glob.glob(f"{d}/units/*/meta.json")):
        u = os.path.dirname(meta)
        if not os.path.exists(f"{u}/DONE"):
            continue
        m = json.load(open(meta)); mu = np.load(f"{u}/dipole.npy"); st = np.load(f"{u}/state.npz"); qT = st["qa_T"]
        xyz = os.path.join(os.path.dirname(os.path.abspath(d)), "xyz")      # <run>/../xyz, as the MLMD runs are laid out
        for k, s in enumerate(m["systems"]):
            if os.path.isdir(xyz):
                sym, p0 = read_xyz(f"{xyz}/{s['species']}.xyz")
                if bonds(sym, p0) != bonds(sym, st["p"][k]):
                    SKIPPED.append(f"{s['species']} {s['T']:.0f}K r{s['replica']}"); continue
            Tm = float(qT[:, k].mean())
            fr, sg = ir_sigma(mu[:, k], s["T"] if TNOM else Tm, 1.0, QM, CLIP)
            R[(MLMD_NAME.get(s["species"], s["species"]), "MLMD q=0", s["T"])].append((fr, sg, Tm))
    return R


HALF = 20.0                                    # peak window half-width (cm-1)
SEARCH = (-30.0, 12.0)                         # where to look for a band at the next T, around its previous position
MATCH = 40.0                                   # pair bands between two methods/charges within this distance


def peaks_300(spectra, n=6, half=HALF):
    """The n strongest peaks of the 300 K replica-mean spectrum (separated by > 2 half-widths): {label: f0}."""
    from scipy.signal import find_peaks
    f = spectra[0][0]; y = np.mean([x[1] for x in spectra], 0)
    k, pr = find_peaks(y, distance=max(1, int(2 * half / (f[1] - f[0]))), prominence=0.02 * y.max())
    top = sorted(k[np.argsort(pr["prominences"])[::-1][:n]])
    return {f"peak {f[i]:.0f}": f[i] for i in top}


def follow(spectra, f0):
    """New position of a band last seen at f0: the maximum of the replica-mean spectrum in f0 + SEARCH."""
    f = spectra[0][0]; y = np.mean([x[1] for x in spectra], 0)
    m = (f >= f0 + SEARCH[0]) & (f <= f0 + SEARCH[1])
    return f[m][np.argmax(y[m])]


def stat(lst, region, i):
    v = np.array([x[region][i] for x in lst])
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan


def main():
    S = load_mlmd(sys.argv[2], load_xtb(sys.argv[1]))
    print(f"intensity convention: qm={QM}, clip={CLIP}, prefactor T={'nominal' if TNOM else 'replica NVE mean'}; "
          f"left out by the integrity check: {len(SKIPPED)} {SKIPPED}")
    S_T = {k: [x[2] for x in v] for k, v in S.items()}       # each replica's NVE mean temperature
    global REGIONS
    base = dict(REGIONS); R = {}; PK = {}
    for (sp, lab, T), lst in S.items():          # per (species, method): regions + its own 300 K peaks
        if (sp, lab) not in PK:
            PK[(sp, lab)] = peaks_300(S[(sp, lab, 300.0)]) if (sp, lab, 300.0) in S else {}
    for (sp, lab), pk in PK.items():               # follow each band up in temperature, step by step
        pos = dict(pk)
        for T in sorted(T_ for s_, l_, T_ in S if (s_, l_) == (sp, lab)):
            lst = S[(sp, lab, T)]
            if T > 300.0:
                pos = {k: follow(lst, c) for k, c in pos.items()}
            REGIONS = {**base, **{k: (c - HALF, c + HALF) for k, c in pos.items()}}
            R[(sp, lab, T)] = [metrics(x[0], x[1]) for x in lst]
    for (sp, lab, T), lst in S.items():            # species/methods without a 300 K run: regions only
        if (sp, lab, T) not in R:
            REGIONS = base; R[(sp, lab, T)] = [metrics(x[0], x[1]) for x in lst]
    REGIONS = base
    species = sorted({k[0] for k in R}); labels = ["MLMD q=0", "xTB q=0", "xTB q=1"]
    temps = sorted({k[2] for k in R})
    qname = ["centroid shift (cm-1)", "width change (cm-1)", "share ratio"]
    for sp in species:
        print(f"\n=== {sp}: change from 300 K (replica mean +- s.e.), and xTB cation - neutral")
        regs = list(base) + sorted({r for (s_, l_), pk in PK.items() if s_ == sp for r in pk},
                                       key=lambda r: float(r.split()[1]))
        for reg in regs:
            for i, qn in enumerate(qname):
                line = f"{reg:18s} {qn:22s}"
                for T in [t for t in temps if t != 300.0]:
                    vals = {}
                    for lab in labels:
                        a, b = R.get((sp, lab, T)), R.get((sp, lab, 300.0))
                        if not a or not b or reg not in a[0] or reg not in b[0]:
                            continue
                        (ma, ea), (mb, eb) = stat(a, reg, i), stat(b, reg, i)
                        vals[lab] = ((ma / mb, np.hypot(ea / mb, ma * eb / mb ** 2)) if i == 2 else (ma - mb, np.hypot(ea, eb)))
                    cell = " ".join(f"{l.split()[0][0]}{l[-1]}:{v[0]:+.3g}" for l, v in vals.items())
                    if "xTB q=0" in vals and "xTB q=1" in vals:
                        (a1, e1), (a0, e0) = vals["xTB q=1"], vals["xTB q=0"]
                        d, e = a1 - a0, np.hypot(e1, e0)
                        cell += f" | d={d:+.3g}+-{e:.2g}{' *' if abs(d) > 2 * e else ''}"
                    line += f" | {T:.0f}K {cell}"
                if "|" in line[41:]:
                    print(line)
    print("\nM0 = MLMD neutral, x0 = xTB neutral, x1 = xTB cation; d = cation - neutral change; * = |d| > 2 s.e.")
    print("Peak windows (+-20 cm-1) follow each 300 K band up in T: maximum within -30..+12 cm-1 of its previous position.")

    def change(sp, lab, reg, T, i):
        a, b = R.get((sp, lab, T)), R.get((sp, lab, 300.0))
        if not a or not b:
            return None
        (ma, ea), (mb, eb) = stat(a, reg, i), stat(b, reg, i)
        return (ma / mb, np.hypot(ea / mb, ma * eb / mb ** 2)) if i == 2 else (ma - mb, np.hypot(ea, eb))

    for la, lb, what in (("xTB q=0", "xTB q=1", "cation vs neutral (xTB)"), ("MLMD q=0", "xTB q=0", "control: xTB vs MLMD neutral")):
        print(f"\n### matched bands, {what}: change from 300 K of {lb} - {la}  (* = > 2 s.e.)")
        for sp in species:
            pa, pb = PK.get((sp, la), {}), PK.get((sp, lb), {})
            for ka, fa in pa.items():
                kb = min(pb, key=lambda k: abs(pb[k] - fa), default=None)
                if kb is None or abs(pb[kb] - fa) > MATCH:
                    continue
                for i, qn in enumerate(qname):
                    line = f"{sp:12s} {fa:5.0f}/{pb[kb]:5.0f} {qn:22s}"
                    for T in [t for t in temps if t != 300.0]:
                        va, vb = change(sp, la, ka, T, i), change(sp, lb, kb, T, i)
                        if va and vb:
                            d, e = vb[0] - va[0], np.hypot(va[1], vb[1])
                            line += f" | {T:.0f}K {va[0]:+.3g} {vb[0]:+.3g} d={d:+.3g}+-{e:.2g}{' *' if abs(d) > 2 * e else ''}"
                    print(line)

    # Linear trends against each replica's own NVE <T> (one NVE run of a small molecule is one canonical energy
    # sample, so its mean T can be far from the nominal one), 300 K to TMAX: slope per 100 K +- s.e. (residuals).
    # Share is relative to the 300 K replica mean, so its slope is a fractional change per 100 K.
    tmax = float(os.environ.get("TMAX", 900))

    def slope(sp, lab, reg, i):
        x, y = [], []
        for T in temps:
            if 300.0 <= T <= tmax and (sp, lab, T) in R and reg in R[(sp, lab, T)][0]:
                x += S_T[(sp, lab, T)]; y += [m[reg][i] for m in R[(sp, lab, T)]]
        if len(x) < 4:
            return None
        if i == 2:
            y = list(np.array(y) / stat(R[(sp, lab, 300.0)], reg, 2)[0])
        x = np.array(x) / 100.0; y = np.array(y)
        A = np.vstack([x, np.ones_like(x)]).T; c, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        s2 = ((y - A @ c) ** 2).sum() / (len(x) - 2)
        return c[0], np.sqrt(s2 / ((x - x.mean()) ** 2).sum())

    sn = ["centroid", "width", "share"]
    print(f"\n### slopes against each replica's NVE <T>, 300-{tmax:.0f} K, per 100 K: centroid and width in cm-1, "
          "share as a fraction of its 300 K value  (* = |d| > 2 s.e.)")
    print("replica NVE <T> (min-max) per nominal T: " + "; ".join(
        f"{sp} {lab} {T:.0f}: {min(v):.0f}-{max(v):.0f}" for (sp, lab, T), v in sorted(S_T.items()) if T in (300.0, 900.0)))
    for la, lb, what in (("xTB q=0", "xTB q=1", "cation vs neutral (xTB)"), ("MLMD q=0", "xTB q=0", "control: xTB vs MLMD neutral")):
        print(f"\n# matched bands, {what}: {la} / {lb}, d = {lb} - {la}")
        for sp in species:
            pa, pb = PK.get((sp, la), {}), PK.get((sp, lb), {})
            pairs = [(r, r) for r in base]
            for ka, fa in pa.items():
                kb = min(pb, key=lambda k: abs(pb[k] - fa), default=None)
                if kb is not None and abs(pb[kb] - fa) <= MATCH:
                    pairs.append((ka, kb))
            for ka, kb in pairs:
                line = f"{sp:12s} {ka:18s}/{kb.split()[-1]:>5s}"
                for i in range(3):
                    va, vb = slope(sp, la, ka, i), slope(sp, lb, kb, i)
                    if va and vb:
                        d, e = vb[0] - va[0], np.hypot(va[1], vb[1])
                        line += f" | {sn[i]} {va[0]:+.3g}+-{va[1]:.2g} {vb[0]:+.3g}+-{vb[1]:.2g} d={d:+.3g}{' *' if abs(d) > 2 * e else ''}"
                if "|" in line:
                    print(line)
    print("\n# every followed band, per method: centroid / width slope per 100 K (cm-1)")
    for sp in species:
        for lab in labels:
            cells = []
            for k in PK.get((sp, lab), {}):
                c, w = slope(sp, lab, k, 0), slope(sp, lab, k, 1)
                if c and w:
                    cells.append(f"{k.split()[-1]}: {c[0]:+.2f}+-{c[1]:.2f} / {w[0]:+.2f}")
            if cells:
                print(f"{sp:12s} {lab:9s} " + "; ".join(cells))
    if sys.argv[3] != "-":
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(len(species), 3, figsize=(13, 3.2 * len(species)), squeeze=False)
        for r, sp in enumerate(species):
            for c, reg in enumerate(REGIONS):
                for lab, st in zip(labels, ("k-o", "C0-s", "C3-^")):
                    ts = [T for T in temps if R.get((sp, lab, T)) and R.get((sp, lab, 300.0))]
                    if not ts:
                        continue
                    b = stat(R[(sp, lab, 300.0)], reg, 0)[0]
                    y = [stat(R[(sp, lab, T)], reg, 0) for T in ts]
                    ax[r, c].errorbar(ts, [v[0] - b for v in y], [v[1] for v in y], fmt=st, label=lab, capsize=2)
                ax[r, c].set_title(f"{sp}: {reg}"); ax[r, c].set_xlabel("T (K)"); ax[r, c].set_ylabel("centroid shift (cm$^{-1}$)")
        ax[0, 0].legend(); fig.tight_layout(); fig.savefig(sys.argv[3], dpi=110); print("figure:", sys.argv[3])


if __name__ == "__main__":
    main()
