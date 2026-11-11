# Temperature dependence of IR bands: GFN2-xTB neutral vs cation (d1/xtb_md.py), and the neutral MLMD reference
# (d1/mlmd_run.py), for docs/plan/charged_tdep_validation.md. Spectra with the authors' calc_MD_IR maths
# (mlmd_spectra.ir_sigma, qm 0, frequencies scaled by 0.9578 for both methods: only shifts and widths relative to
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

REGIONS = {"oop 600-1000": (600, 1000), "inplane 1000-1700": (1000, 1700), "CHstr 2700-3300": (2700, 3300)}
MLMD_NAME = {"C10H8_330": "naphthalene", "C16H10_334": "pyrene", "C24H12_18": "coronene"}


def metrics(f, s):
    tot = s.sum(); out = {}
    for name, (lo, hi) in REGIONS.items():
        m = (f >= lo) & (f < hi); w = s[m]; x = f[m]
        c = (w * x).sum() / w.sum()
        out[name] = (c, np.sqrt((w * (x - c) ** 2).sum() / w.sum()), w.sum() / tot)
    return out


def load_xtb(d):
    R = defaultdict(list)                      # (species, label, T) -> [metrics per replica]
    for f in sorted(glob.glob(f"{d}/*.npz")):
        z = np.load(f)
        if int(z["frames_done"]) < len(z["dipole"]) or len(z["dipole"]) < 1000:
            continue
        sp, q, T = re.match(r"(\w+?)_q(\d)_(\d+)K_r\d", os.path.basename(f)).groups()
        fr, s = ir_sigma(z["dipole"], float(T), 1.0, 0)
        R[(sp, f"xTB q={q}", float(T))].append((fr, s))
    return R


def load_mlmd(d, R):
    for meta in sorted(glob.glob(f"{d}/units/*/meta.json")):
        u = os.path.dirname(meta)
        if not os.path.exists(f"{u}/DONE"):
            continue
        m = json.load(open(meta)); mu = np.load(f"{u}/dipole.npy")
        for k, s in enumerate(m["systems"]):
            fr, sg = ir_sigma(mu[:, k], s["T"], 1.0, 0)
            R[(MLMD_NAME.get(s["species"], s["species"]), "MLMD q=0", s["T"])].append((fr, sg))
    return R


HALF = 20.0                                    # peak window half-width (cm-1)
SEARCH = (-30.0, 12.0)                         # where to look for a band at the next T, around its previous position
MATCH = 40.0                                   # pair bands between two methods/charges within this distance


def peaks_300(spectra, n=6, half=HALF):
    """The n strongest peaks of the 300 K replica-mean spectrum (separated by > 2 half-widths): {label: f0}."""
    from scipy.signal import find_peaks
    f = spectra[0][0]; y = np.mean([s for _, s in spectra], 0)
    k, pr = find_peaks(y, distance=max(1, int(2 * half / (f[1] - f[0]))), prominence=0.02 * y.max())
    top = sorted(k[np.argsort(pr["prominences"])[::-1][:n]])
    return {f"peak {f[i]:.0f}": f[i] for i in top}


def follow(spectra, f0):
    """New position of a band last seen at f0: the maximum of the replica-mean spectrum in f0 + SEARCH."""
    f = spectra[0][0]; y = np.mean([s for _, s in spectra], 0)
    m = (f >= f0 + SEARCH[0]) & (f <= f0 + SEARCH[1])
    return f[m][np.argmax(y[m])]


def stat(lst, region, i):
    v = np.array([x[region][i] for x in lst])
    return v.mean(), v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan


def main():
    S = load_mlmd(sys.argv[2], load_xtb(sys.argv[1]))
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
            R[(sp, lab, T)] = [metrics(f, y) for f, y in lst]
    for (sp, lab, T), lst in S.items():            # species/methods without a 300 K run: regions only
        if (sp, lab, T) not in R:
            REGIONS = base; R[(sp, lab, T)] = [metrics(f, y) for f, y in lst]
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
