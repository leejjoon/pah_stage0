# Harmonic normal-mode energies of the naphthalene MD runs, to test whether the persistent per-replica spread of the
# cation's 300 K in-plane IR strength (docs/plan/cloud_aimd_handoff.md, "Extra cation replicas") is vibrational
# energy frozen in a few IR-active modes (electronic-state differences were ruled out by frame_sp.py).
#   python mode_energy.py ref <xyz> <ref.npz> --charge 1 --mult 2 [--threads 32]
#       optimise at the MD level (b3lypg, 6-31G* 6D, default grid, conv_tol 1e-9; scipy L-BFGS on the analytic
#       gradient), analytic Hessian, harmonic modes (translations/rotations projected out), IR intensity of each mode
#       from a central difference of the COM dipole along it.
#   python mode_energy.py analyse <ref.npz> <run.npz ...> [--json out.json]
#       per run: energy in each mode, E_k = 1/2 qdot_k^2 + 1/2 w_k^2 q_k^2 after an Eckart-like (mass-weighted Kabsch)
#       alignment to the reference, averaged over the saved NVE frames (frames_pos/frames_vel), or from the final
#       positions/velocities alone for runs without frames; ratio to k_B T (run's NVE mean T); and the predicted band
#       strength factor F = sum_W I_k E_k/kT / sum_W I_k for the bands W of aimd_slopes.py (scaled axis), next to the
#       measured strength (qm=3 makes the classical estimator give the harmonic intensity when E_k = kT).
import argparse, json, os, re, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
HA_EV, BOHR_A, KB_EV, D_PER_EA = 27.211386245988, 0.529177210903, 8.617333262e-5, 4.80320471
KM_FACTOR = 42.2561                       # km/mol per (D / (A amu^1/2))^2
BANDS = {"1000-1300": (1000, 1300), "1300-1700": (1300, 1700), "inplane": (1000, 1700), "oop": (600, 1000)}


def ref(a):
    from pyscf import gto, dft, lib
    from ase.io import read
    from scipy.optimize import minimize
    lib.num_threads(a.threads)
    at = read(a.xyz); sym = at.get_chemical_symbols(); m = at.get_masses(); x0 = at.positions.copy()
    state = {"dm": None, "n": 0}

    def scf(pos, grad=True):
        mol = gto.M(atom=list(zip(sym, pos)), basis="6-31g*", charge=a.charge, spin=a.mult - 1, unit="Angstrom",
                    cart=True, verbose=0)
        mf = (dft.RKS if a.mult == 1 else dft.UKS)(mol); mf.xc = "b3lypg"; mf.conv_tol = 1e-9; mf.max_cycle = 200
        mf.kernel(dm0=state["dm"]); assert mf.converged
        state["dm"] = mf.make_rdm1()
        return mf

    def fg(xf):
        pos = xf.reshape(-1, 3); mf = scf(pos); g = mf.nuc_grad_method().kernel()
        state["n"] += 1
        return mf.e_tot, (g / BOHR_A).ravel()          # Ha, Ha/A

    t0 = time.time()
    res = minimize(fg, x0.ravel(), jac=True, method="L-BFGS-B", options=dict(gtol=1e-6, maxiter=200))
    pos = res.x.reshape(-1, 3); com = (m[:, None] * pos).sum(0) / m.sum(); pos = pos - com
    gmax = float(np.abs(res.jac).max() * BOHR_A)
    print(f"opt: {state['n']} evaluations, {time.time() - t0:.0f} s, max |grad| {gmax:.2e} Ha/bohr, "
          f"moved {np.abs(pos - (x0 - (m[:, None] * x0).sum(0) / m.sum())).max():.4f} A", flush=True)
    mf = scf(pos); e0 = mf.e_tot; s2 = float(mf.spin_square()[0]) if a.mult > 1 else 0.0
    t0 = time.time(); H = mf.Hessian().kernel()                     # (N, N, 3, 3) Ha/bohr^2
    n = len(sym); H = H.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n) * HA_EV / BOHR_A ** 2   # eV/A^2
    print(f"hessian: {time.time() - t0:.0f} s", flush=True)
    sm = np.repeat(np.sqrt(m), 3); Hm = H / np.outer(sm, sm)        # eV/(A^2 amu)
    # translations and rotations, mass-weighted, orthonormalised; project them out
    T = []
    for i in range(3):
        v = np.zeros((n, 3)); v[:, i] = 1; T.append((v * np.sqrt(m)[:, None]).ravel())
    for i in range(3):
        e = np.zeros(3); e[i] = 1; T.append((np.cross(e, pos) * np.sqrt(m)[:, None]).ravel())
    Q, _ = np.linalg.qr(np.array(T).T); P = np.eye(3 * n) - Q @ Q.T
    w2, L = np.linalg.eigh(P @ Hm @ P)
    keep = np.argsort(np.abs(w2))[6:]; keep = keep[np.argsort(w2[keep])]
    w2, L = w2[keep], L[:, keep]                                    # eV/(A^2 amu), columns mass-weighted unit modes
    # wavenumber: w = sqrt(w2) in sqrt(eV/amu)/A -> s^-1 -> cm^-1
    w_si = np.sqrt(np.abs(w2) * 1.602176634e-19 / 1.66053906660e-27) * 1e10
    wn = np.sign(w2) * w_si / (2 * np.pi * 2.99792458e10)
    print("frequencies (cm^-1, unscaled):", np.round(wn, 1).tolist(), flush=True)
    # IR intensities: central difference of the COM dipole along each mass-weighted mode, step h A amu^1/2
    h = 0.01; dmu = []
    t0 = time.time()

    def dip(p):
        mf = scf(p, grad=False); c = (m[:, None] * p).sum(0) / m.sum()
        return np.asarray(mf.dip_moment(unit="Debye", verbose=0)) - a.charge * c * D_PER_EA   # Debye, about the COM
    for k in range(L.shape[1]):
        d = (L[:, k] / sm).reshape(n, 3) * h
        dmu.append((dip(pos + d) - dip(pos - d)) / (2 * h))          # D / (A amu^1/2)
    dmu = np.array(dmu); I = KM_FACTOR * (dmu ** 2).sum(1)
    print(f"dipole derivatives: {time.time() - t0:.0f} s; total IR {I.sum():.1f} km/mol", flush=True)
    np.savez(a.out, symbols=np.array(sym), masses=m, pos=pos, e0=e0, s2=s2, w2=w2, wn=wn, L=L, dmu=dmu, I=I,
             charge=a.charge, mult=a.mult, gmax=gmax, method="B3LYP(VWN3)/6-31G* 6D PySCF, default grid")
    from mlmd_spectra import SCALE
    for b, (lo, hi) in BANDS.items():
        s = (wn * SCALE >= lo) & (wn * SCALE < hi)
        print(f"  {b:10s} {s.sum():2d} modes, harmonic {I[s].sum():7.1f} km/mol")


def kabsch(P, Q, w):
    """rotation R minimising sum w |R p - q|^2 (P, Q centred)"""
    C = (P * w[:, None]).T @ Q; U, S, Vt = np.linalg.svd(C); d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def mode_E(rf, pos, vel):
    m = rf["masses"]; sm = np.sqrt(m)[:, None]; com = (m[:, None] * pos).sum(0) / m.sum()
    p = pos - com; v = vel - (m[:, None] * vel).sum(0) / m.sum()
    R = kabsch(p, rf["pos"], m); p = p @ R.T; v = v @ R.T
    q = rf["L"].T @ ((p - rf["pos"]) * sm).ravel(); qd = rf["L"].T @ (v * sm).ravel()
    return 0.5 * qd ** 2 + 0.5 * rf["w2"] * q ** 2, 0.5 * qd ** 2         # eV: total, kinetic


def analyse(a):
    from aimd_slopes import load, CL
    from mlmd_spectra import SCALE
    rf = dict(np.load(a.ref)); wn_s = rf["wn"] * SCALE; I = rf["I"]
    out = {}
    hdr = "  ".join(f"{b:>17s}" for b in BANDS)
    print(f"{'run':16s} {'T':>5s} {'src':>6s}  E/kT of the 3 strongest IR modes in 1000-1300   per band: measured km/mol /"
          f" predicted (harmonic x F)\n{'':16s} {'':5s} {'':6s}  {'':44s}  {hdr}")
    top = [k for k in np.argsort(-I) if 1000 <= wn_s[k] < 1300][:3]
    for p in a.runs:
        z = np.load(p); run = os.path.basename(p)[:-4]; x = load(p); T = x["T"]
        if "frames_pos" in z.files and len(z["frames_pos"]):
            E = np.array([mode_E(rf, P, V)[0] for P, V in zip(z["frames_pos"], z["frames_vel"])])
            Ek = E.mean(0); src = f"{len(E)}fr"
        else:
            Ek = mode_E(rf, z["positions"], z["velocities"])[0]; src = "final"
        r = Ek / (KB_EV * T)
        o = dict(T=T, src=src, E_over_kT=r.tolist(), bands={})
        cells = []
        for b, (lo, hi) in BANDS.items():
            s = (wn_s >= lo) & (wn_s < hi); harm = I[s].sum(); F = (I[s] * r[s]).sum() / harm
            meas = x["strength"][b] if b in x["strength"] else None
            if meas is None:     # 1000-1300 / 1300-1700 are not aimd_slopes classes: integrate here
                from mlmd_spectra import ir_sigma
                f, sg = ir_sigma(z["dipole"], T, float(z["dt_fs"]), 3, False); df = (f[1] - f[0]) / SCALE
                meas = float(sg[(f >= lo) & (f < hi)].sum() * df * 6.02214076e23 * 1e-5)
            o["bands"][b] = dict(measured=meas, harmonic=float(harm), F=float(F), predicted=float(harm * F))
            cells.append(f"{meas:7.0f} /{harm * F:7.0f}")
        out[run] = o
        print(f"{run[5:]:16s} {T:5.0f} {src:>6s}  " + " ".join(f"{wn_s[k]:6.0f}:{r[k]:5.2f}" for k in top).ljust(44)
              + "  " + "  ".join(cells))
    # normalised slope: strength per unit mode energy (measured / predicted) for runs with frames, mean and standard
    # error per nominal temperature, then the change from the lowest to the highest temperature as a fraction of the
    # low-T value per 100 K (actual mean NVE temperatures), errors propagated from the two standard errors
    slopes = {}
    grp = {}
    for run, o in out.items():
        if o["src"] != "final":
            grp.setdefault(int(re.search(r"_(\d+)K_", run).group(1)), []).append(o)
    if len(grp) >= 2:
        lo, hi = min(grp), max(grp)
        print(f"\nnormalised slope (runs with frames: {len(grp[lo])} at {lo} K, {len(grp[hi])} at {hi} K): "
              "measured/predicted mean +- s.e.; slope % per 100 K")
        for b in BANDS:
            st = {}
            for T in (lo, hi):
                rr = np.array([o["bands"][b]["measured"] / o["bands"][b]["predicted"] for o in grp[T]])
                st[T] = (rr.mean(), rr.std(ddof=1) / np.sqrt(len(rr)) if len(rr) > 1 else np.nan,
                         np.mean([o["T"] for o in grp[T]]))
            (m0, e0, T0), (m1, e1, T1) = st[lo], st[hi]
            dT = (T1 - T0) / 100; s = (m1 / m0 - 1) / dT
            es = np.hypot(e1 / m0, m1 * e0 / m0 ** 2) / dT
            slopes[b] = dict(ratio_lo=m0, se_lo=e0, T_lo=T0, ratio_hi=m1, se_hi=e1, T_hi=T1, pct_per_100K=100 * s,
                             se_pct_per_100K=100 * es)
            print(f"  {b:10s} {m0:.3f} +- {e0:.3f} ({T0:.0f} K)   {m1:.3f} +- {e1:.3f} ({T1:.0f} K)   "
                  f"{100 * s:+.2f} +- {100 * es:.2f} %/100 K")
    if a.json:
        json.dump(dict(top_modes=[dict(wn_scaled=float(wn_s[k]), I=float(I[k])) for k in top], runs=out,
                       normalised_slopes=slopes), open(a.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("ref"); r.add_argument("xyz"); r.add_argument("out")
    r.add_argument("--charge", type=int, default=0); r.add_argument("--mult", type=int, default=1)
    r.add_argument("--threads", type=int, default=32)
    an = sp.add_parser("analyse"); an.add_argument("ref"); an.add_argument("runs", nargs="+"); an.add_argument("--json")
    a = ap.parse_args()
    ref(a) if a.cmd == "ref" else analyse(a)


if __name__ == "__main__":
    main()
