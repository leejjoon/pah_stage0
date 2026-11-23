# Summary of frame_sp.py output per run, next to the run's 1000-1300 / 1300-1700 cm^-1 IR strengths (aimd_slopes.py
# convention), to see whether the persistent per-replica spread of the cation's 300 K in-plane strength tracks the
# electronic state.
#   python frame_sp_summary.py <dir with *.jsonl> <dir with the run npz> [--json out.json]
import argparse, glob, json, os, re, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_spectra import ir_sigma, SCALE  # noqa: E402

KM = 6.02214076e23 * 1e-5


def strengths(npz):
    z = np.load(npz); qa = z["qa"]; n = int(round(float(z["nvt_ps"]) * 2000))
    f, s = ir_sigma(z["dipole"], qa[qa[:, 0] > n, 1].mean(), float(z["dt_fs"]), 3, False); df = (f[1] - f[0]) / SCALE
    return [float(s[(f >= lo) & (f < hi)].sum() * df * KM) for lo, hi in ((1000, 1300), (1300, 1700))]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("spdir"); ap.add_argument("npzdir"); ap.add_argument("--json")
    a = ap.parse_args()
    R = defaultdict(list)
    for p in glob.glob(f"{a.spdir}/*.jsonl"):
        for l in open(p):
            if l.strip():
                r = json.loads(l); R[r["run"]].append(r)
    out = {}
    print(f"{'run':16s} {'n':>4s} {'IR1000-1300':>11s} {'1300-1700':>9s} {'max|dmu|':>9s} {'max|dE|':>8s} {'stab':>6s} "
          f"{'reopt dE':>9s} {'<S2>':>6s}  {'|chg A-B|':>9s} {'|chg up-dn|':>11s} {'|spin A-B|':>10s} {'|spin up-dn|':>12s}  "
          f"rms dipole (long, short, normal) e*A")
    for run in sorted(R):
        v = sorted(R[run], key=lambda r: r["frame"])
        ir = strengths(os.path.join(a.npzdir, run + ".npz"))
        dmu = max(r["d_dipole"] for r in v); de = [abs(r["d_e_meV"]) for r in v if "d_e_meV" in r]
        st = [r["stability"] for r in v if "stability" in r]
        reo = [s.get("d_e_meV", 0.0) for s in st if s["rounds"]]
        ca = np.abs([r["charge_asym"] for r in v]).mean(0); sa = np.abs([r["spin_asym"] for r in v]).mean(0)
        mu = np.array([r["dipole_mol"] for r in v]); s2 = np.mean([r["s2"] for r in v])
        # axis signs from eigh are arbitrary frame to frame, so only rms (not mean) dipole components are meaningful
        o = dict(n=len(v), ir_1000_1300=ir[0], ir_1300_1700=ir[1], max_d_dipole=dmu, max_abs_d_e_meV=max(de) if de else None,
                 n_stab=len(st), n_stable=sum(s["stable"] for s in st), reopt_d_e_meV=reo, s2_mean=float(s2),
                 charge_asym_abs=ca.tolist(), spin_asym_abs=sa.tolist(), dipole_mol_rms=np.sqrt((mu ** 2).mean(0)).tolist(),
                 not_converged=sum(not r["converged"] for r in v), newton=sum(r["newton"] for r in v))
        out[run] = o
        print(f"{run[5:]:16s} {len(v):4d} {ir[0]:11.0f} {ir[1]:9.0f} {dmu:9.1e} {(max(de) if de else float('nan')):8.3f} "
              f"{o['n_stable']:2d}/{len(st):<3d} {(min(reo) if reo else 0):9.3f} {s2:6.4f}  {ca[0]:9.3f} {ca[1]:11.3f} "
              f"{sa[0]:10.3f} {sa[1]:12.3f}  rms {np.round(o['dipole_mol_rms'], 4).tolist()}")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
