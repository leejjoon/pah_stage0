# Why do PAHFIT and local integration disagree on the Orion Bar templates? (reports/nc_sr_v4, section 2.) Runs in
# container C.
#   python slice/orion_gap.py <emission h5> <out dir> [--template ...] [--energies 6,10] [--iter 4000]
# Per template (CDS J/A+A/685/A75, thinned 4x, gas lines removed with an 11-pixel running median), 5.2-19 um:
#   1. Continuum estimates under the bands: PAHFIT's (classic.yaml; modified blackbodies + starlight), the local
#      spline and the local straight segments of measure.py. Printed: flux above each continuum per window, and the
#      level of each continuum at 7.7 and 12 um relative to the template.
#   2. Best-fit library mixtures (non-negative least squares, relative weights 3 %):
#        A  every species of the emission file at the given photon energies (variant tdep), smoothed to the MRS
#           resolving power, plus PAHFIT's continuum basis
#        B  A plus a dictionary of broad Gaussians (FWHM 1.5-2 um) at 6-9, 10.5-14.5 and 15.5-18.5 um ("plateaus")
#      Printed per window: share of the template taken by continuum, library, plateaus and residual, and the rms of
#      the relative residual. The plateau integrals of fit B are given relative to the 11.2 um band of the library
#      part (window 10.9-11.7 um): the numbers behind measure.py --plateau.
# Output: orion_gap.npz (all curves), orion_gap.json (the tables), fig_orion_gap.png.
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure as M  # noqa: E402

WIN = [("5.2-6.0", 5.2, 6.0), ("6.0-6.6", 6.0, 6.6), ("6.6-7.0", 6.6, 7.0), ("7.0-9.0", 7.0, 9.0), ("9.0-10.5", 9.0, 10.5),
       ("10.5-10.9", 10.5, 10.9), ("10.9-11.7", 10.9, 11.7), ("11.7-13.9", 11.7, 13.9), ("13.9-15.5", 13.9, 15.5),
       ("15.5-18.5", 15.5, 18.5)]
PLAT = [(c, 1.5) for c in np.arange(6.0, 9.01, 0.5)] + [(c, 2.0) for c in np.arange(10.5, 14.51, 1.0)] + [(c, 1.5) for c in np.arange(15.5, 18.51, 0.75)]
REG = {"6-9": (5.0, 9.8), "10-15": (9.8, 15.2), "15-18": (15.2, 20.0)}             # plateau centres per region


def nnls_pg(A, b, it):
    """Non-negative least squares by accelerated projected gradient with restart (columns scaled to unit norm)."""
    s = np.linalg.norm(A, axis=0); s[s == 0] = 1; A = A / s
    G = A.T @ A; h = A.T @ b; Lc = np.linalg.eigvalsh(G)[-1]
    x = np.zeros(A.shape[1]); y = x.copy(); t = 1.0
    for _ in range(it):
        xn = np.maximum(y - (G @ y - h) / Lc, 0)
        if (y - xn) @ (xn - x) > 0:
            t = 1.0; y = xn
        else:
            tn = 0.5 * (1 + np.sqrt(1 + 4 * t * t)); y = xn + (t - 1) / tn * (xn - x); t = tn
        x = xn
    return x / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emission"); ap.add_argument("out")
    ap.add_argument("--template", default="/data/pdrs4all/J_A+A_685_A75/sptemp.tsv")
    ap.add_argument("--energies", default="6,10"); ap.add_argument("--iter", type=int, default=4000)
    a = ap.parse_args()
    import h5py, astropy.units as u
    from astropy.nddata import StdDevUncertainty
    from specutils import Spectrum1D
    from pahfit.model import Model
    from scipy.ndimage import median_filter
    from scipy.interpolate import CubicSpline
    import warnings; warnings.filterwarnings("ignore")
    os.makedirs(a.out, exist_ok=True)
    wav, tmpl = M.read_templates(a.template); seg = M.segment_of(wav)
    sel = (wav >= 5.2) & (wav <= 19.0)
    h = h5py.File(a.emission, "r"); grid = h["grid"][:]; E = list(h["energies"][:]); sp = [s.decode() for s in h["species"][:]]
    cols = []; names = []
    for e in [float(x) for x in a.energies.split(",")]:
        S = h["tdep"][:, E.index(e), :]
        for i in range(S.shape[0]):
            cols.append(M.to_mrs(grid, S[i], wav, seg)[sel]); names.append((sp[i], e))
    Lb = np.array(cols).T; print(f"library basis: {Lb.shape[1]} spectra ({len(sp)} species x {a.energies} eV), {sel.sum()} pixels", flush=True)
    ft = Model.from_yaml("classic.yaml").features
    temps = sorted(float(t) for t in ft["temperature"]["val"][np.asarray(ft["kind"]) == "dust_continuum"]) + [5000.0]
    w = wav[sel]; Cb = np.stack([M.mbb(w, T) / M.mbb(11.25, T) for T in temps], 1)
    Pb = np.stack([np.exp(-0.5 * ((w - c) / (fw / 2.3548)) ** 2) for c, fw in PLAT], 1)
    cation = np.array([n.endswith("_q1") for n, _ in names])
    out = {"wav": w}; tab = {}
    for t in M.TEMPLATES:
        f0 = tmpl[t]; f = median_filter(f0, 11, mode="nearest"); fs = f[sel]; sig = 0.03 * fs; T = tab[t] = {}
        # 1. continua
        s = Spectrum1D(spectral_axis=wav * u.um, flux=f0 * u.MJy / u.sr, uncertainty=StdDevUncertainty(0.01 * f0)); s.meta["instrument"] = M.SEG
        m = Model.from_yaml("classic.yaml"); m.guess(s); m.fit(s, verbose=False, maxiter=10000)
        kind = np.asarray(m.features["kind"]); cm = (kind == "dust_continuum") | (kind == "starlight")
        cP = m.tabulate(instrumentname=M.SEG, wavelengths=wav, feature_mask=cm).flux.value[sel]
        dP = m.tabulate(instrumentname=M.SEG, wavelengths=wav, feature_mask=kind == "dust_feature").flux.value[sel]
        x = np.array([0.5 * (lo + hi) for lo, hi in M.LANCH]); y = np.array([np.median(f[(wav >= lo) & (wav <= hi)]) for lo, hi in M.LANCH])
        cL = np.interp(w, x, y); cS = CubicSpline(x, y, bc_type="natural")(w)
        out.update({f"{t}_flux": fs, f"{t}_cont_pahfit": cP, f"{t}_cont_lin": cL, f"{t}_cont_spl": cS, f"{t}_pahfit_dust": dP})
        T["cont_level"] = {f"{name}@{x0}": float(np.interp(x0, w, c) / np.interp(x0, w, fs)) for name, c in (("pahfit", cP), ("linear", cL), ("spline", cS))
                           for x0 in (6.2, 7.7, 8.6, 11.25, 12.7, 17.0)}
        T["above"] = {n: {name: M.win(w, fs - c, lo, hi) for name, c in (("pahfit", cP), ("linear", cL), ("spline", cS))} for n, lo, hi in WIN}
        T["pahfit_dust"] = {n: M.win(w, dP, lo, hi) for n, lo, hi in WIN}
        # 2. library mixtures
        for tag, blocks in (("A", [Lb, Cb]), ("B", [Lb, Cb, Pb])):
            A = np.hstack(blocks); xx = nnls_pg(A / sig[:, None], fs / sig, a.iter)
            nl, nc = Lb.shape[1], Cb.shape[1]
            lib = Lb @ xx[:nl]; con = Cb @ xx[nl:nl + nc]; pl = Pb @ xx[nl + nc:] if tag == "B" else np.zeros_like(lib); res = fs - lib - con - pl
            out.update({f"{t}_{tag}_lib": lib, f"{t}_{tag}_cont": con, f"{t}_{tag}_plat": pl})
            R = T[tag] = {"rms_rel": float(np.sqrt(np.mean((res / fs) ** 2))), "n_species_used": int((xx[:nl] > 1e-6 * xx[:nl].max()).sum()),
                          "cation_share_of_library_flux": float((Lb[:, cation] @ xx[:nl][cation]).sum() / lib.sum())}
            R["windows"] = {n: dict(cont=M.win(w, con, lo, hi) / M.win(w, fs, lo, hi), lib=M.win(w, lib, lo, hi) / M.win(w, fs, lo, hi),
                                    plat=M.win(w, pl, lo, hi) / M.win(w, fs, lo, hi), res=M.win(w, res, lo, hi) / M.win(w, fs, lo, hi),
                                    rms_rel=float(np.sqrt(np.mean((res[(w >= lo) & (w <= hi)] / fs[(w >= lo) & (w <= hi)]) ** 2)))) for n, lo, hi in WIN}
            if tag == "B":
                b112 = M.win(w, lib, 10.9, 11.7); pk = lib[(w > 11.0) & (w < 11.6)].max(); xp = xx[nl + nc:]
                R["plateau"] = {}
                for rn, (lo, hi) in REG.items():
                    k = [i for i, (c, _) in enumerate(PLAT) if lo <= c < hi]; comp = Pb[:, k] @ xp[k]; tot = M.win(w, comp, w[0], w[-1])
                    cen = float((w * comp / w ** 2).sum() / (comp / w ** 2).sum()) if comp.sum() > 0 else float("nan")
                    R["plateau"][rn] = dict(integral_over_lib112=tot / b112, peak_over_lib112_peak=float(comp.max() / pk), centroid_um=cen,
                                            amplitudes={f"{PLAT[i][0]:.2f}": float(xp[i] / pk) for i in k})
            print(f"{t} fit {tag}: rms of the relative residual {R['rms_rel']:.3f}; {R['n_species_used']} spectra used; cations carry "
                  f"{R['cation_share_of_library_flux']:.2f} of the library flux", flush=True)
    np.savez_compressed(os.path.join(a.out, "orion_gap.npz"), **out); json.dump(tab, open(os.path.join(a.out, "orion_gap.json"), "w"), indent=1)
    # ---- tables
    print("\ncontinuum level / template flux at 6.2, 7.7, 8.6, 11.25, 12.7, 17.0 um")
    for t in M.TEMPLATES:
        for name in ("pahfit", "linear", "spline"):
            print(f"  {t:5s} {name:7s} " + " ".join(f"{tab[t]['cont_level'][f'{name}@{x0}']:.2f}" for x0 in (6.2, 7.7, 8.6, 11.25, 12.7, 17.0)))
    print("\nflux above each continuum per window, relative to the same above PAHFIT's continuum in 10.9-11.7 (PAHFIT / linear / spline); last: PAHFIT dust features")
    for t in M.TEMPLATES:
        ref = tab[t]["above"]["10.9-11.7"]["pahfit"]
        for n, *_ in WIN:
            v = tab[t]["above"][n]; print(f"  {t:5s} {n:10s} {v['pahfit'] / ref:6.2f} {v['linear'] / ref:6.2f} {v['spline'] / ref:6.2f}   {tab[t]['pahfit_dust'][n] / ref:6.2f}")
    for tag in ("A", "B"):
        print(f"\nfit {tag}: share of the template per window: continuum / library / plateaus / residual (rms of the relative residual)")
        for t in M.TEMPLATES:
            for n, *_ in WIN:
                v = tab[t][tag]["windows"][n]; print(f"  {t:5s} {n:10s} {v['cont']:5.2f} {v['lib']:5.2f} {v['plat']:5.2f} {v['res']:+6.3f}  ({v['rms_rel']:.3f})")
    print("\nfit B plateaus: integral / library 11.2 um band integral, peak / library 11.2 um peak, centroid (um)")
    for t in M.TEMPLATES:
        print(f"  {t:5s} " + "; ".join(f"{rn}: {v['integral_over_lib112']:.2f}, {v['peak_over_lib112_peak']:.3f}, {v['centroid_um']:.1f}" for rn, v in tab[t]["B"]["plateau"].items()))
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(M.TEMPLATES), 2, figsize=(15, 3.0 * len(M.TEMPLATES)), sharex=True)
    for i, t in enumerate(M.TEMPLATES):
        fs = out[f"{t}_flux"]; ax = axs[i][0]
        ax.plot(w, fs, color="k", lw=0.6, label="template (lines removed)")
        for name, col, lab in (("cont_pahfit", "#d1495b", "PAHFIT continuum"), ("cont_lin", "#1f6feb", "local, straight"), ("cont_spl", "#2a9d8f", "local, spline")):
            ax.plot(w, out[f"{t}_{name}"], color=col, lw=1, label=lab)
        ax.set_ylim(0, 1.15 * fs[(w > 10.9) & (w < 11.7)].max()); ax.set_ylabel(f"{t}\nMJy/sr"); ax.spines[["top", "right"]].set_visible(False)
        ax = axs[i][1]
        ax.plot(w, fs - out[f"{t}_B_cont"], color="k", lw=0.6, label="template − continuum (fit B)")
        ax.plot(w, out[f"{t}_A_lib"] + out[f"{t}_A_cont"] - out[f"{t}_B_cont"], color="#9aa0a6", lw=0.8, label="fit A (library + continuum)")
        ax.plot(w, out[f"{t}_B_lib"], color="#1f6feb", lw=0.8, label="fit B: library")
        ax.fill_between(w, 0, out[f"{t}_B_plat"], color="#edae49", alpha=0.6, label="fit B: broad components")
        ax.set_ylim(0, 1.15 * (fs - out[f"{t}_B_cont"])[(w > 7) & (w < 12)].max()); ax.spines[["top", "right"]].set_visible(False)
        if i == 0:
            axs[0][0].legend(frameon=False, fontsize=7); axs[0][1].legend(frameon=False, fontsize=7)
    axs[-1][0].set_xlabel("wavelength (µm)"); axs[-1][1].set_xlabel("wavelength (µm)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_orion_gap.png"), dpi=120)


if __name__ == "__main__":
    main()
