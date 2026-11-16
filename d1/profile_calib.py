# Can a profile fit recover band widths from mlmd_spectra spectra? The spectrum is the DCT of the dipole ACF truncated
# at 7.987 % of the run (16 ps for 200 ps NVE), with no taper: an instrumental line of ~2 cm^-1 with sinc ringing.
# Synthetic lines with known Lorentzian FWHM (phase-diffusing oscillators: ACF exp(-pi c G t) cos(w t), i.e. a
# Lorentzian of FWHM G), run through ir_sigma at production settings (1 fs, --ps ps, qm 3, unclipped, 4 replicas
# averaged), then fitted with
#   drude:  A g^2 / ((x0/x - x/x0)^2 + g^2) + baseline, FWHM = g x0 (PAHFIT's profile);
#   trunc:  the exact spectrum of a Lorentzian ACF truncated at L: Re[(1 - exp(-(gam - i d) L)) / (gam - i d)] + baseline.
#   python profile_calib.py [--ps 200] [--real 4] [--sets 8]
import argparse, os, sys
import numpy as np
from scipy.optimize import least_squares
from scipy.constants import c as C_LIGHT

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_spectra import ir_sigma, SCALE, FRACTION  # noqa: E402

CM = 2 * np.pi * C_LIGHT * 100          # rad/s per cm^-1


def synth(wn_true, fwhm_true, n, dt, rng):
    """One replica: dipole along x of a phase-diffusing oscillator (Lorentzian of FWHM fwhm_true, cm^-1)."""
    D = np.pi * C_LIGHT * 100 * fwhm_true             # phase diffusion constant (1/s): ACF decays as exp(-D t)
    phi = rng.uniform(0, 2 * np.pi) + np.cumsum(np.sqrt(2 * D * dt) * rng.standard_normal(n))
    mu = np.zeros((n, 3)); mu[:, 0] = 0.1 * np.cos(CM * wn_true * np.arange(n) * dt + phi)
    return mu


def drude(p, x):
    A, x0, g, b = p
    return A * g ** 2 / ((x0 / x - x / x0) ** 2 + g ** 2) + b


def trunc(p, x, L):
    """x, x0 and fwhm in scaled cm^-1; L the ACF length in s."""
    A, x0, fw, b = p
    gam = np.pi * C_LIGHT * 100 * fw / SCALE          # 1/s (scaled -> true wavenumber)
    d = CM * (x - x0) / SCALE                          # rad/s
    z = gam - 1j * d
    return A * np.real((1 - np.exp(-z * L)) / z) * gam + b


def fit(model, x, y, x0, L=None):
    p0 = [y.max(), x0, 0.003 if model == "drude" else 3.0, 0.0]
    f = (lambda p: drude(p, x) - y) if model == "drude" else (lambda p: trunc(p, x, L) - y)
    lb = [0, x0 - 10, 1e-5, -np.inf] if model == "drude" else [0, x0 - 10, 0.01, -np.inf]
    ub = [np.inf, x0 + 10, 0.2, np.inf] if model == "drude" else [np.inf, x0 + 10, 200, np.inf]
    r = least_squares(f, p0, bounds=(lb, ub), x_scale="jac")
    return r.x[2] * r.x[1] if model == "drude" else r.x[2]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ps", type=float, default=200); ap.add_argument("--real", type=int, default=4)
    ap.add_argument("--sets", type=int, default=8); a = ap.parse_args()
    dt = 1e-15; n = int(a.ps * 1000); L = int(n * FRACTION) * dt
    rng = np.random.default_rng(1)
    print(f"ACF length {L * 1e12:.1f} ps; 1/(c L) = {1 / (C_LIGHT * 100 * L):.2f} cm^-1 (true), "
          f"{SCALE / (C_LIGHT * 100 * L):.2f} scaled. Recovered FWHM (scaled cm^-1): mean +- sd over {a.sets} sets of "
          f"{a.real} replicas")
    print(f"{'band':>6s} {'true FWHM':>9s} | {'drude':>15s} | {'trunc':>15s}")
    for wn_s in (893.0, 1613.0, 3060.0):
        for fw_s in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
            res = {"drude": [], "trunc": []}
            for _ in range(a.sets):
                S = [ir_sigma(synth(wn_s / SCALE, fw_s / SCALE, n, dt, rng), 600.0, 1.0, 3, False) for _ in range(a.real)]
                f = S[0][0]; y = np.mean([s for _, s in S], 0)
                half = max(25.0, 4 * fw_s); m = np.abs(f - wn_s) <= half
                for k in res:
                    res[k].append(fit(k, f[m], y[m] / y[m].max(), wn_s, L))
            print(f"{wn_s:6.0f} {fw_s:9.1f} | " + " | ".join(f"{np.mean(v):7.2f} +- {np.std(v):5.2f}" for v in res.values()))


if __name__ == "__main__":
    main()
