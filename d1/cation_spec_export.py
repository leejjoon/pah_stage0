# Rule B1 spectra of the cation MLMD dipoles (cation_mlmd_dipoles.py <id>_spec.npz) in profile_fit.py's input format:
#   python cation_spec_export.py <dipoles dir> <out dir> [--acf-ps 16] [--grid-ps 15.974] [--ids a,b] [--check]
# writes <out>/<T>/<species>_<T>K.npz (freq, sigma [replicas, K], prefactor_T = each replica's T_nve, replicas, mean,
# acf_len_s, grid_cm1). The spectra are ir_sigma's (qm=3, unclipped, per-replica T_nve, the DCT-I of the truncated ACF).
# --acf-ps 16 (the primary Step 4 input; coordinator 2026-11-26 16:50): ACF kept to 16 ps (8,000 lags at 2 fs), grid
# ~1.0 cm^-1 after scaling as the neutral 200 ps libraries (15,974 lags at 1 fs), no refinement.
# Default (the 5.1 ps sensitivity case): ACF at FRACTION of the series as the production _spec.npz (2,555 lags, grid
# 3.13 cm^-1), refined to the integer factor closest to --grid-ps / L (3) by cubic-spline interpolation of the spectrum
# (every 3rd point is the DCT value), so profile_fit.py's bin-based steps (unit-bin area normalisation, 5-bin boxcar,
# peak spacing) work as for the neutrals; its truncated-Lorentzian model takes L = acf_len_s from the file.
# Not zero padding: the cation dipole ACF has not decayed at 5.1 ps (rotating permanent dipole: ~50 % of ACF(0); 11 %
# at 16 ps), DCT-I handles the non-zero tail exactly but zeros after it make a step whose ringing, times the ~w^2
# prefactor, puts 5-90 % of a band's maximum on the bands (75d119e-48de4f0 did that; the gate's rule-level numbers of
# 11-26 13:07 came from such spectra). --check: refined vs stored _spec.npz spectrum (integral, values at shared points).
import argparse, glob, os, sys
import numpy as np
from scipy import fftpack, signal
from scipy.interpolate import CubicSpline
from scipy.constants import physical_constants as pc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_spectra import FRACTION, SCALE, FMIN, FMAX  # noqa: E402


def ir_sigma_padded(mu, T, dt_fs, grid_fs, qm=3, nlag=None):
    """mlmd_spectra.ir_sigma(mu, T, dt_fs, qm, clip=False) with nlag ACF lags (default int(n x FRACTION), as ir_sigma),
    the spectrum refined by the integer factor round(grid_fs / (dt_fs nlag)) with a cubic spline through the DCT points."""
    boltz = pc["Boltzmann constant"][0]; c = pc["speed of light in vacuum"][0]; hbar = pc["reduced Planck constant"][0]
    e_c = pc["elementary charge"][0]; AA_0 = pc["Angstrom star"][0]; eps0 = pc["vacuum electric permittivity"][0]
    timestep = dt_fs * 1e-15; n = len(mu)
    d = np.asarray(mu, np.float64) * e_c * AA_0
    acf = 0
    for k in range(3):
        x = d[:, k]
        sh = np.zeros(n * 2 if n % 2 == 0 else n * 2 - 1)
        sh[n // 2:n // 2 + n] = x
        acf = acf + signal.fftconvolve(sh, x[::-1], mode="same")[-n:] / np.arange(n, 0, -1)
    acf = acf[:nlag or int(n * FRACTION)]
    lineshape = fftpack.dct(acf, type=1)[1:] * timestep
    f_hz = np.linspace(0, 0.5 / timestep, len(acf))[1:]
    k = max(1, int(round(grid_fs / dt_fs / len(acf))))
    if k > 1:
        fine = np.linspace(f_hz[0], f_hz[-1], (len(f_hz) - 1) * k + 1)
        lineshape = CubicSpline(f_hz, lineshape)(fine); f_hz = fine
    wn = f_hz / (100 * c); w = f_hz * 2 * np.pi; x = hbar * w / (boltz * T)
    field = w * (1.0 - np.exp(-x))
    qc = {0: 1.0, 1: 2 / (1.0 + np.exp(-x)), 2: x / (2 * np.tanh(x / 2)), 3: x / (1.0 - np.exp(-x))}[qm]
    sigma = lineshape * field * (10000 / (6 * hbar * c * eps0)) * qc
    wn = wn * SCALE; k = (wn >= FMIN) & (wn <= FMAX)
    return wn[k], sigma[k]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dipoles"); ap.add_argument("out")
    ap.add_argument("--grid-ps", type=float, default=int(200000 * FRACTION) / 1000); ap.add_argument("--ids", default="")
    ap.add_argument("--check", action="store_true"); ap.add_argument("--suffix", default="_spec.npz")
    ap.add_argument("--acf-ps", type=float, default=None, help="ACF window (ps) instead of FRACTION of the series, e.g. 16 "
                    "(8,000 lags at 2 fs: the ~1.0 cm^-1 grid of the neutral 200 ps libraries, no padding; coordinator 11-26 16:50)")
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.dipoles}/*{a.suffix}"))
    if a.ids:
        files = [f for f in files if os.path.basename(f)[:-len(a.suffix)] in a.ids.split(",")]
    for f in files:
        z = np.load(f); T = int(round(float(z["temp"]))); sp = str(z["species"]); dt = float(z["dt_fs"])
        if not len(z["replicas"]):
            print(f"{os.path.basename(f)}: no intact replica, skipped"); continue
        nlag = int(round(a.acf_ps * 1000 / dt)) if a.acf_ps else None
        L = nlag * dt * 1e-15 if nlag else float(z["acf_len_s"])
        out = [ir_sigma_padded(mu, float(t), dt, a.grid_ps * 1000, nlag=nlag) for mu, t in zip(z["dipole"], z["T_nve"])]
        fr = out[0][0]; sig = np.array([s for _, s in out]); mean = sig.mean(0) / sig.mean(0).max()
        os.makedirs(f"{a.out}/{T}", exist_ok=True)
        np.savez(f"{a.out}/{T}/{sp}_{T}K.npz", freq=fr, sigma=sig, prefactor_T=z["T_nve"], replicas=z["replicas"], mean=mean,
                 acf_len_s=L, grid_cm1=float(fr[1] - fr[0]), dt_fs=dt, n_frames=int(z["n_frames"]), qm=3)
        msg = f"{os.path.basename(f)} -> {T}/{sp}_{T}K.npz: {len(sig)} replicas, grid {fr[1] - fr[0]:.3f} cm-1, L {L * 1e12:.2f} ps"
        if a.check and not nlag:
            f0, s0 = np.asarray(z["freq_cm1"]), np.asarray(z["sigma"])
            I0 = s0.sum(1) * (f0[1] - f0[0]); I1 = sig.sum(1) * (fr[1] - fr[0])
            on = np.interp(f0, fr, sig[0])                     # grid points coincide (integer refinement)
            msg += f"; integral refined/stored {np.max(np.abs(I1 / I0 - 1)):.1e}; max |diff| on the coarse grid {np.max(np.abs(on - s0[0])) / s0[0].max():.1e} of max"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
