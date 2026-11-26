# Rule B1 spectra of the cation MLMD dipoles (cation_mlmd_dipoles.py <id>_spec.npz) in profile_fit.py's input format:
#   python cation_spec_export.py <dipoles dir> <out dir> [--grid-ps 15.974] [--ids a,b] [--check]
# writes <out>/<T>/<species>_<T>K.npz (freq, sigma [replicas, K], prefactor_T = each replica's T_nve, replicas, mean,
# acf_len_s, grid_cm1). The spectra are ir_sigma's (qm=3, unclipped, per-replica T_nve, ACF truncated at FRACTION of the
# series: 5.1 ps for 32,000 frames at 2 fs) with the truncated ACF zero-padded by the integer factor closest to
# --grid-ps / L before the DCT (3 here), so they are the same spectra sampled 3x finer, near the grid of the neutral
# rules' inputs (200 ps at 1 fs: 15,974 fs of ACF, ~1.0 cm^-1 after scaling), every 3rd point equal to the unpadded one. profile_fit.py's bin-based steps (unit-bin area normalisation, 5-bin boxcar, peak spacing) then work
# as for the neutrals, and its truncated-Lorentzian model takes L = acf_len_s from the file. --check: compares the
# padded spectrum with the unpadded one stored in the _spec.npz (total integral and the values at the shared grid
# points) for every file.
import argparse, glob, os, sys
import numpy as np
from scipy import fftpack, signal
from scipy.constants import physical_constants as pc

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlmd_spectra import FRACTION, SCALE, FMIN, FMAX  # noqa: E402


def ir_sigma_padded(mu, T, dt_fs, grid_fs, qm=3):
    """mlmd_spectra.ir_sigma(mu, T, dt_fs, qm, clip=False) with the truncated ACF zero-padded to grid_fs."""
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
    acf = acf[:int(n * FRACTION)]
    k = max(1, int(round(grid_fs / dt_fs / len(acf))))         # integer refinement: the DCT-I grid of the unpadded ACF
    acf = np.concatenate([acf, np.zeros(k * (len(acf) - 1) + 1 - len(acf))])   # is every k-th point of the padded one
    lineshape = fftpack.dct(acf, type=1)[1:] * timestep
    f_hz = np.linspace(0, 0.5 / timestep, len(acf))[1:]
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
    a = ap.parse_args()
    files = sorted(glob.glob(f"{a.dipoles}/*{a.suffix}"))
    if a.ids:
        files = [f for f in files if os.path.basename(f)[:-len(a.suffix)] in a.ids.split(",")]
    for f in files:
        z = np.load(f); T = int(round(float(z["temp"]))); sp = str(z["species"]); dt = float(z["dt_fs"])
        if not len(z["replicas"]):
            print(f"{os.path.basename(f)}: no intact replica, skipped"); continue
        out = [ir_sigma_padded(mu, float(t), dt, a.grid_ps * 1000) for mu, t in zip(z["dipole"], z["T_nve"])]
        fr = out[0][0]; sig = np.array([s for _, s in out]); mean = sig.mean(0) / sig.mean(0).max()
        os.makedirs(f"{a.out}/{T}", exist_ok=True)
        np.savez(f"{a.out}/{T}/{sp}_{T}K.npz", freq=fr, sigma=sig, prefactor_T=z["T_nve"], replicas=z["replicas"], mean=mean,
                 acf_len_s=float(z["acf_len_s"]), grid_cm1=float(fr[1] - fr[0]), dt_fs=dt, n_frames=int(z["n_frames"]), qm=3)
        msg = f"{os.path.basename(f)} -> {T}/{sp}_{T}K.npz: {len(sig)} replicas, grid {fr[1] - fr[0]:.3f} cm-1, L {float(z['acf_len_s']) * 1e12:.2f} ps"
        if a.check:
            f0, s0 = np.asarray(z["freq_cm1"]), np.asarray(z["sigma"])
            I0 = s0.sum(1) * (f0[1] - f0[0]); I1 = sig.sum(1) * (fr[1] - fr[0])
            on = np.interp(f0, fr, sig[0])                     # grid points coincide (integer refinement)
            msg += f"; integral padded/unpadded {np.max(np.abs(I1 / I0 - 1)):.1e}; max |diff| on the coarse grid {np.max(np.abs(on - s0[0])) / s0[0].max():.1e} of max"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
