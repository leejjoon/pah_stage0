# IR spectra from mlmd_run.py dipoles, with the authors' 3_calc_IR.py maths (calc_MD_IR; Mai et al. 2025, code
# Apache 2.0): autocorrelation of the dipole, first 7.987 % of it, DCT-I, field factor w (1 - exp(-hbar w / kT)),
# optional quantum correction (qm 0-3), negative values set to 0, frequencies scaled by 0.9578, 300-3800 cm^-1.
#   python mlmd_spectra.py <run outdir> <spectra outdir> [--qm 0]
# writes, per species and temperature, <spectra outdir>/<species>_<T>K.npz: freq (cm^-1, scaled), sigma (replicas,
# freq) in cm^2 (unnormalised), mean (replica mean of sigma, normalised to a peak of 1), and <..>_qm<k>.txt in the
# authors' format (normalised replica mean). Replicas are averaged before normalising. Flagged systems are skipped.
# Hidden-event check: the screen (screen_check.py) sees only the final geometry, so an H that hops and returns
# passes. A hop changes the slowly varying dipole. Over 1 ps windows: vec_shift = largest distance (e*A) of a window
# mean of the dipole vector from the median window mean; mag_shift = the same for the window mean of |mu|; mu_med =
# median of those |mu| means. The molecule tumbles, so for a polar molecule vec_shift reflects rotation, not
# chemistry (overlap_v1: C11H7N_465, C59H20_96, 0.7-1.2 in every replica). mag_shift is rotation-invariant but
# misses many events (sp3 ring openings barely change |mu|). Flag if mag_shift > MAG_FLAG, or if the molecule has
# no permanent dipole (mu_med < POLAR) and vec_shift > VEC_FLAG. Calibrated 2026-11-14 on overlap_v1 and four
# new300_v1 units: species passing the screen reach vec 0.16 (nonpolar) and mag 0.114. A partial check: it
# catches most hops in nonpolar species, few in polar ones. Flagged systems still get spectra; qa.csv records it.
#   python mlmd_spectra.py --check <dipole txt> <their spectrum txt> <T>   reproduce one 3_calc_IR.py output
import argparse, glob, json, os, sys
import numpy as np
from scipy import fftpack, signal
from scipy.constants import physical_constants as pc

FRACTION, SCALE, FMIN, FMAX = 0.07987, 0.9578, 300.0, 3800.0
VEC_FLAG, MAG_FLAG, POLAR, SHIFT_WINDOW_FS = 0.2, 0.15, 0.25, 1000.0


def ir_sigma(mu, T, dt_fs=1.0, qm=0):
    """mu: (N, 3) dipole in e*A every dt_fs. Returns scaled wavenumbers (cm^-1) and sigma (cm^2), 300-3800 cm^-1,
    exactly as calc_MD_IR in the authors' 3_calc_IR.py (before its normalisation)."""
    boltz = pc["Boltzmann constant"][0]; c = pc["speed of light in vacuum"][0]; hbar = pc["reduced Planck constant"][0]
    e_c = pc["elementary charge"][0]; AA_0 = pc["Angstrom star"][0]; eps0 = pc["vacuum electric permittivity"][0]
    timestep = dt_fs * 1e-15
    n = len(mu)
    d = np.asarray(mu, np.float64) * e_c * AA_0
    acf = 0
    for k in range(3):
        x = d[:, k]
        sh = np.zeros(n * 2 if n % 2 == 0 else n * 2 - 1)
        sh[n // 2:n // 2 + n] = x
        acf = acf + signal.fftconvolve(sh, x[::-1], mode="same")[-n:] / np.arange(n, 0, -1)
    acf = acf[:int(n * FRACTION)]
    lineshape = fftpack.dct(acf, type=1)[1:] * timestep
    f_hz = np.linspace(0, 0.5 / timestep, len(acf))[1:]
    wn = f_hz / (100 * c)
    w = f_hz * 2 * np.pi
    x = hbar * w / (boltz * T)
    field = w * (1.0 - np.exp(-x))
    qc = {0: 1.0, 1: 2 / (1.0 + np.exp(-x)), 2: x / (2 * np.tanh(x / 2)), 3: x / (1.0 - np.exp(-x))}[qm]
    sigma = lineshape * field * (10000 / (6 * hbar * c * eps0)) * qc
    sigma[sigma < 0] = 0.0
    wn = wn * SCALE
    m = (wn >= FMIN) & (wn <= FMAX)
    return wn[m], sigma[m]


def dipole_shift(mu, dt_fs=1.0):
    """(vec_shift, mag_shift, mu_med, flag) over 1 ps windows; see the header."""
    w = max(1, int(round(SHIFT_WINDOW_FS / dt_fs))); n = len(mu) // w
    x = np.asarray(mu[:n * w], np.float64).reshape(n, w, 3)
    m = x.mean(1); a = np.linalg.norm(x, axis=2).mean(1)
    vec = float(np.linalg.norm(m - np.median(m, axis=0), axis=1).max())
    mag = float(np.abs(a - np.median(a)).max()); med = float(np.median(a))
    return vec, mag, med, int(mag > MAG_FLAG or (med < POLAR and vec > VEC_FLAG))


def save_txt(path, f, y):
    np.savetxt(path, np.column_stack((f, y)), fmt="%.1f %.4e", header="Freq_MD (cm^-1) Inten_MD (Normalized intensity)")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--check":
        dip, theirs, T = sys.argv[2], sys.argv[3], float(sys.argv[4])
        t, *mu = np.loadtxt(dip, skiprows=1, unpack=True)
        f, s = ir_sigma(np.column_stack(mu), T, dt_fs=t[1] - t[0])
        y = s / s.max()
        F, Y = np.loadtxt(theirs, unpack=True)
        fr = np.round(f, 1); yr = np.array([float(f"{v:.4e}") for v in y])
        print(f"{len(F)} points vs {len(f)}; max|dfreq| {np.abs(F - fr).max():.2e}; max|dI| {np.abs(Y - yr).max():.2e} "
              f"(after the authors' %.1f/%.4e rounding)")
        sys.exit(0 if len(F) == len(f) and np.abs(F - fr).max() == 0 and np.abs(Y - yr).max() <= 1e-4 else 1)
    ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("out"); ap.add_argument("--qm", type=int, default=0)
    ap.add_argument("--units", default=None, help="comma-separated unit ids: only these (testing)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    groups, skipped = {}, []
    for meta in sorted(glob.glob(f"{a.run}/units/*/meta.json")):
        d = os.path.dirname(meta)
        if not os.path.exists(f"{d}/DONE") or (a.units and os.path.basename(d) not in a.units.split(",")):
            continue
        m = json.load(open(meta)); fl = m["flags"]
        dt_fs = m["settings"]["dt_fs"] * m["settings"]["frame_every"]
        mu = np.load(f"{d}/dipole.npy")         # one unit at a time (a few hundred MB); keep only spectra
        for k, s in enumerate(m["systems"]):
            if fl["nonfinite"][k] or fl["hot"][k] or fl["max_neighbours"][k] > 25 or fl["edge_overflow"]:
                skipped.append((s["species"], s["T"], s["replica"])); continue
            x = np.ascontiguousarray(mu[:, k])
            groups.setdefault((s["species"], s["T"]), []).append(
                (s["replica"], ir_sigma(x, s["T"], dt_fs, a.qm), dipole_shift(x, dt_fs), m["id"]))
        del mu
    qa = []
    for (sp, T), reps in sorted(groups.items()):
        reps.sort(key=lambda r: r[0])
        shift = [r[2] for r in reps]
        qa += [(sp, T, r[0], r[3], *(round(v, 4) for v in x[:3]), x[3]) for r, x in zip(reps, shift)]
        f = reps[0][1][0]; sig = np.array([r[1][1] for r in reps]); mean = sig.mean(0); mean /= mean.max()
        tag = f"{sp}_{T:g}K"
        np.savez(f"{a.out}/{tag}.npz", freq=f, sigma=sig, mean=mean, replicas=[r[0] for r in reps], qm=a.qm,
                 dipole_shift=np.array(shift))
        save_txt(f"{a.out}/{tag}_qm{a.qm}.txt", f, mean)
    with open(f"{a.out}/qa.csv", "w") as fo:
        fo.write("species,T,replica,unit,vec_shift,mag_shift,mu_med,shift_flag\n")
        fo.writelines(",".join(map(str, q)) + "\n" for q in qa)
    print(f"{len(groups)} species x T spectra in {a.out}; {len(skipped)} flagged systems skipped; "
          f"{sum(q[-1] for q in qa)} of {len(qa)} systems flagged by the dipole-shift check (qa.csv)")
    for x in skipped[:20]:
        print("   skipped", x)


if __name__ == "__main__":
    main()
