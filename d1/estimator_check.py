# Check of the IR intensity estimator in mlmd_spectra.ir_sigma (committee review B1) on classical harmonic
# oscillators with linear dipoles, where the exact answer is known: the harmonic integrated absorption intensity
#   A [km/mol] = 42.2561 x |dmu/dQ|^2   (dmu/dQ in D / (A amu^1/2), Q mass-weighted),
# independent of temperature (it is the absorption net of stimulated emission).
#   python estimator_check.py [--temps 50,300,600,900,1200] [--real 64] [--ps 100]
# For each temperature: --real independent canonical realisations (each mode gets an energy drawn from the classical
# kT and a random phase; for a harmonic oscillator that gives the canonical mean ACF exactly), dipole every 1 fs for --ps ps, spectrum with ir_sigma for qm = 0..3,
# with and without the authors' clipping of negative values, band intensities integrated over +-40 cm^-1 of each
# scaled band and converted to km/mol. Prints the ratio to the
# exact value per band, qm and T, and the T-dependence of a band ratio. No MD and no JAX: numpy and scipy only.
import argparse, os, sys
import numpy as np
from scipy.constants import physical_constants as pc, Avogadro, atomic_mass, angstrom, c as C_LIGHT, k as KB

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_spectra import ir_sigma, SCALE  # noqa: E402

DEBYE = 3.33564e-30                                   # C m
E_A = pc["elementary charge"][0] * angstrom           # C m per e*A
# modes: unscaled harmonic wavenumber (cm^-1) and |dmu/dQ| (D / (A amu^1/2)); intensities 10-60 km/mol
MODES = [(893.0 / SCALE, 1.0), (1613.0 / SCALE, 0.7), (3060.0 / SCALE, 0.5), (500.0 / SCALE, 0.6)]


def realisation(T, n, dt, rng):
    t = np.arange(n) * dt
    mu = np.zeros((n, 3))
    for k, (wn, d) in enumerate(MODES):
        w = 2 * np.pi * C_LIGHT * 100 * wn                   # rad/s
        # Energy kT per mode: for a harmonic oscillator the dipole ACF is linear in the mode energy, so this gives the
        # canonical mean ACF exactly, without the sampling noise of drawing E from the Boltzmann distribution.
        E = KB * T
        amp = np.sqrt(2 * E) / w                              # Q amplitude in kg^1/2 m
        dmu = d * DEBYE / (angstrom * np.sqrt(atomic_mass))   # C m per (kg^1/2 m)
        mu[:, k % 3] += dmu * amp * np.cos(w * t + rng.uniform(0, 2 * np.pi)) / E_A
    return mu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temps", default="50,300,600,900,1200"); ap.add_argument("--real", type=int, default=64)
    ap.add_argument("--ps", type=float, default=100.0); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed); n = int(a.ps * 1000); dt = 1e-15
    exact = [42.2561 * d ** 2 for _, d in MODES]
    res = {}
    for T in [float(x) for x in a.temps.split(",")]:
        reals = [realisation(T, n, dt, rng) for _ in range(a.real)]
        for qm in range(4):
          for clip in (True, False):
            acc = np.zeros(len(MODES))
            for mu in reals:
                f, s = ir_sigma(mu, T, 1.0, qm, clip)
                df = f[1] - f[0]                                       # scaled cm^-1 per bin
                for k, (wn, _) in enumerate(MODES):
                    m = np.abs(f - wn * SCALE) <= 40
                    # integrated over the scaled axis; undo the scaling to integrate over true wavenumber
                    acc[k] += s[m].sum() * df / SCALE * Avogadro * 1e-5  # cm^2 cm^-1 per molecule -> km/mol
            res[(T, qm, clip)] = acc / a.real
    print("ratio of the estimated to the exact harmonic intensity (1 = correct), per band (scaled cm^-1)")
    print("   T   qm clip " + "".join(f"{wn * SCALE:>9.0f}" for wn, _ in MODES))
    for (T, qm, clip), v in sorted(res.items()):
        print(f"{T:5.0f}  {qm:2d}  {'yes' if clip else ' no'} " + "".join(f"{x / e:9.3f}" for x, e in zip(v, exact)))
    i893, i1613 = 0, 1
    print("\nband ratio I(1613)/I(893) relative to 300 K, unclipped, per qm (harmonic truth: 1 at every T)")
    for qm in range(4):
        r300 = res[(300.0, qm, False)][i1613] / res[(300.0, qm, False)][i893]
        print(f"qm {qm}: " + "  ".join(f"{T:.0f} K {res[(T, qm, False)][i1613] / res[(T, qm, False)][i893] / r300:.3f}"
                                         for T in [float(x) for x in a.temps.split(",")]))
    x = lambda wn, T: 6.62607015e-34 * C_LIGHT * 100 * wn / (KB * T)
    wq = lambda wn, T: (1 - np.exp(-x(wn, T))) / x(wn, T)
    print("\nanalytic qm=0 weight (1-e^-x)/x for the same ratio, relative to 300 K: "
          + "  ".join(f"{T:.0f} K {wq(1613 / SCALE, T) / wq(893 / SCALE, T) / (wq(1613 / SCALE, 300) / wq(893 / SCALE, 300)):.3f}"
                      for T in [float(x) for x in a.temps.split(",")]))


if __name__ == "__main__":
    main()
