# Band-strength temperature slopes from the naphthalene DFT MD (B3 part one; docs/plan/cloud_aimd_handoff.md step 8),
# for the B4 gate (docs/plan/b4_diagnostic_impact.md): the deciding number is the cation in-plane (CC) slope.
#   python aimd_slopes.py <dir with naph_q{0,1}_{300,900}K_r*.npz> [--json out.json]
# Same convention as the MLMD/xTB class slopes b4_impact.py uses (xtb_test/class_slopes.py, d1/tdep_compare.py):
# mlmd_spectra.ir_sigma with qm=3, unclipped, each replica's mean NVE temperature in the prefactor; band classes on
# the scaled axis oop 600-1000, inplane 1000-1700, CHstr 2700-3300 cm^-1; strength = integral in km/mol; a straight
# line in T (replica NVE mean) through all replicas; slope reported as a fraction of the fitted 300 K value per 100 K.
# Here qa also holds the NVT steps, so the replica temperature is the mean over NVE records only (step > NVT steps).
# Spread: the slope from each of the 4 pairings of one 300 K and one 900 K replica (min..max).
# Also: the 6.2/11.2-type windows of b4_impact.py (narrow and broad), NVE energy drift, max SCF cycles and <S^2>.
import argparse, glob, itertools, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_spectra import ir_sigma, SCALE  # noqa: E402

KM = 6.02214076e23 * 1e-5          # cm^2 per molecule * cm^-1 -> km/mol
CL = {"oop": (600, 1000), "inplane": (1000, 1700), "CHstr": (2700, 3300),
      "6.2narrow": (1570, 1650), "11.2narrow": (870, 910), "6.2broad": (1500, 1700), "11.2broad": (850, 940)}
# reference slopes (fraction per 100 K) from xtb_test/class_slopes.json (2026-11-12), naphthalene and 3-species mean
REF = {"MLMD q=0 naph": {"oop": -0.025, "inplane": +0.025, "CHstr": -0.005},
       "xTB q=1 naph": {"oop": +0.040, "inplane": -0.013, "CHstr": +0.124},
       "b4 neutral_mlmd (3-sp mean)": {"oop": +0.008, "inplane": +0.022, "CHstr": -0.019},
       "b4 cation_xtb (3-sp mean)": {"oop": +0.029, "inplane": -0.009, "CHstr": +0.152}}


def load(path):
    z = np.load(path); qa = np.array(z["qa"]); n_nvt = int(round(float(z["nvt_ps"]) * 2000))
    nve = qa[qa[:, 0] > n_nvt]
    mu = np.array(z["dipole"]); assert np.isfinite(mu).all() and int(z["step"]) == n_nvt + 2 * len(mu), path
    Tm = float(nve[:, 1].mean())
    f, s = ir_sigma(mu, Tm, float(z["dt_fs"]), 3, False)
    df = (f[1] - f[0]) / SCALE
    st = {c: float(s[(f >= lo) & (f < hi)].sum() * df * KM) for c, (lo, hi) in CL.items()}
    cen = {}
    for c, (lo, hi) in CL.items():
        m = (f >= lo) & (f < hi); w = np.clip(s[m], 0, None); cen[c] = float((w * f[m]).sum() / w.sum())
    e = nve[:, 2]; t_ps = (nve[:, 0] - n_nvt) / 2000
    drift = float(np.polyfit(t_ps, e, 1)[0]) * 1000      # meV/ps
    return dict(T=Tm, strength=st, centroid=cen, drift_meV_ps=drift, e_std_meV=float(e.std() * 1000),
                scf_max=int(qa[:, 4].max()), s2_max=float(qa[:, 3].max()), engines=str(z["engines"]),
                seconds=float(z["seconds"]) if "seconds" in z.files else None)


def fit(pts):
    """pts: [(T, y)] -> (fraction per 100 K of the fitted 300 K value, fitted 300 K value, abs slope per 100 K)"""
    v = np.array(pts); a, b = np.polyfit(v[:, 0] / 100, v[:, 1], 1)
    return a / (3 * a + b), 3 * a + b, a


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir"); ap.add_argument("--json")
    a = ap.parse_args()
    R = {}
    for p in sorted(glob.glob(f"{a.dir}/naph_q[01]_*K_r*.npz")):
        q, T, r = map(int, re.search(r"naph_q(\d)_(\d+)K_r(\d+)\.npz$", p).groups()); R[(q, T, r)] = load(p)
    print(f"{'run':16s} {'T_NVE':>6s} {'drift':>9s} {'E std':>6s} {'SCF':>3s} {'<S2>':>5s}  "
          + "  ".join(f"{c:>9s}" for c in CL) + "   (km/mol)")
    for (q, T, r), x in sorted(R.items()):
        print(f"q{q}_{T}K_r{r}      {x['T']:6.1f} {x['drift_meV_ps']:+6.2f}m/ps {x['e_std_meV']:6.1f} {x['scf_max']:3d} "
              f"{x['s2_max']:5.3f}  " + "  ".join(f"{x['strength'][c]:9.2f}" for c in CL))
    out = {}
    print("\nslope, fraction of the fitted 300 K strength per 100 K (all replicas; [min, max] over the 4 replica pairings)"
          "; centroid shift cm^-1 per 100 K")
    for q in (0, 1):
        print(f"--- {'neutral' if q == 0 else 'cation'} (q={q})")
        for c in CL:
            pts = [(x["T"], x["strength"][c]) for (qq, T, r), x in R.items() if qq == q]
            frac, s300, _ = fit(pts)
            pair = [fit([(R[(q, 300, i)]["T"], R[(q, 300, i)]["strength"][c]),
                         (R[(q, 900, j)]["T"], R[(q, 900, j)]["strength"][c])])[0]
                    for i, j in itertools.product((0, 1), (0, 1))]
            cs = np.polyfit([x["T"] / 100 for (qq, T, r), x in R.items() if qq == q],
                            [x["centroid"][c] for (qq, T, r), x in R.items() if qq == q], 1)[0]
            out[f"q{q}|{c}"] = dict(frac_per_100K=frac, pair_min=min(pair), pair_max=max(pair), s300_km_mol=s300,
                                    centroid_per_100K=float(cs))
            print(f"  {c:11s} {frac:+.4f}  [{min(pair):+.4f}, {max(pair):+.4f}]   300 K {s300:8.2f} km/mol   "
                  f"centroid {cs:+6.2f}")
    print("\ncation minus neutral (fraction per 100 K):")
    for c in CL:
        d = out[f"q1|{c}"]["frac_per_100K"] - out[f"q0|{c}"]["frac_per_100K"]; out[f"diff|{c}"] = d
        print(f"  {c:11s} {d:+.4f}")
    print("\nreference slopes (fraction per 100 K): " + "; ".join(
        f"{k}: " + ", ".join(f"{c} {v:+.3f}" for c, v in d.items()) for k, d in REF.items()))
    if a.json:
        json.dump(dict(slopes=out, runs={f"q{q}_{T}K_r{r}": x for (q, T, r), x in R.items()}), open(a.json, "w"),
                  indent=1)


if __name__ == "__main__":
    main()
