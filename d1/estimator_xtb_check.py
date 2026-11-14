# Check of the IR intensity convention on a real molecule (committee review B1): GFN2-xTB harmonic intensities at the
# xTB minimum against band integrals of GFN2-xTB MD spectra (d1/xtb_md.py runs) at low temperature, where the
# vibrations are nearly harmonic. Same potential and dipole on both sides, so the only difference is the estimator.
#   python estimator_xtb_check.py <xtb runs dir> <xyz> <species tag> --charge 0 --mult 1 [--T 300] [--win 25]
# Harmonic: xTB geometry optimisation, finite-difference Hessian and dipole derivatives (about the centre of mass),
# A_i [km/mol] = 42.2561 |dmu/dQ_i|^2 (D / (A amu^1/2)). MD: replica-mean spectrum at T for qm 0 and 3, clipped and
# not, integrated over +-win cm^-1 around each strong harmonic band (frequencies scaled by 0.9578 like the MD axis).
import argparse, glob, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_spectra import ir_sigma, SCALE  # noqa: E402

AVOGADRO = 6.02214076e23
EA_TO_D = 4.80320471
EV_A2_AMU_TO_CM = 521.4709                      # sqrt(eV / (A^2 amu)) in cm^-1


def harmonic(xyz, q, m):
    from ase.io import read
    from ase.optimize import BFGS
    from tblite.ase import TBLite
    at = read(xyz); at.calc = TBLite(method="GFN2-xTB", charge=q, multiplicity=m, verbosity=0)
    BFGS(at, logfile=None).run(fmax=1e-4, steps=2000)
    p0 = at.positions.copy(); mass = at.get_masses(); n = len(at); h = 0.005
    com = lambda p: (mass[:, None] * p).sum(0) / mass.sum()

    def fmu(p):
        at.positions = p
        return at.get_forces().ravel(), at.get_dipole_moment() - q * com(p)

    H = np.zeros((3 * n, 3 * n)); dmu = np.zeros((3 * n, 3))
    for k in range(3 * n):
        pp = p0.copy(); pp.flat[k] += h; fp, mp = fmu(pp)
        pm = p0.copy(); pm.flat[k] -= h; fm, mm = fmu(pm)
        H[k] = -(fp - fm) / (2 * h); dmu[k] = (mp - mm) / (2 * h)            # dmu in e*A / A
    H = 0.5 * (H + H.T); w = 1 / np.sqrt(np.repeat(mass, 3))
    ev, vec = np.linalg.eigh(H * w[:, None] * w[None])
    keep = np.argsort(np.abs(ev))[6:]                                         # drop translations and rotations
    freq = np.sqrt(np.clip(ev[keep], 0, None)) * EV_A2_AMU_TO_CM
    dQ = (vec[:, keep] * w[:, None]).T @ dmu * EA_TO_D                         # D / (A amu^1/2)
    return freq, 42.2561 * (dQ ** 2).sum(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs"); ap.add_argument("xyz"); ap.add_argument("tag")
    ap.add_argument("--charge", type=int, default=0); ap.add_argument("--mult", type=int, default=1)
    ap.add_argument("--T", type=float, default=300.0); ap.add_argument("--win", type=float, default=25.0)
    a = ap.parse_args()
    freq, A = harmonic(a.xyz, a.charge, a.mult)
    files = sorted(glob.glob(f"{a.runs}/{a.tag}_q{a.charge}_{a.T:.0f}K_r*.npz"))
    spec = {}
    for qm in (0, 3):
        for clip in (True, False):
            ys = []
            for f in files:
                z = np.load(f); Tm = float(np.mean(np.array(z["qa"])[:, 1]))
                fr, s = ir_sigma(z["dipole"], Tm, 1.0, qm, clip); ys.append(s)
            spec[(qm, clip)] = (fr, np.mean(ys, 0))
    fr = spec[(3, False)][0]; df = (fr[1] - fr[0]) / SCALE
    strong = [i for i in np.argsort(A)[::-1] if A[i] > 0.05 * A.max() and 300 < freq[i] * SCALE < 3800]
    print(f"{a.tag} q={a.charge}: {len(files)} replicas at {a.T:.0f} K; harmonic total (300-3800 scaled) "
          f"{A[(freq * SCALE > 300) & (freq * SCALE < 3800)].sum():.1f} km/mol")
    print("band(scaled)  A_harm  | MD/harmonic: qm0 clip   qm0      qm3 clip   qm3")
    rows = []
    for i in sorted(strong, key=lambda i: freq[i]):
        c = freq[i] * SCALE
        harm = A[np.abs(freq * SCALE - c) <= a.win].sum()                        # all harmonic modes in the window
        r = []
        for key in ((0, True), (0, False), (3, True), (3, False)):
            f, s = spec[key]; m = np.abs(f - c) <= a.win
            r.append(s[m].sum() * df * AVOGADRO * 1e-5 / harm)
        rows.append((c, harm, r))
        print(f"{c:9.0f}  {harm:7.1f}  | " + "  ".join(f"{x:8.3f}" for x in r))
    for key, lab in zip(((0, True), (0, False), (3, True), (3, False)), ("qm0 clip", "qm0", "qm3 clip", "qm3")):
        f, s = spec[key]; m = (f > 300) & (f < 3800)
        print(f"total MD / harmonic, {lab:8s}: {s[m].sum() * df * AVOGADRO * 1e-5 / A[(freq * SCALE > 300) & (freq * SCALE < 3800)].sum():.3f}")


if __name__ == "__main__":
    main()
