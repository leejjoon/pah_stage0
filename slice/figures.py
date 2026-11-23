# Figures for the neutral prototype report. Runs in container C.
#   python slice/figures.py <species csv> <emission h5> <pop h5> <regress out dir> <fig dir> [--template sptemp.tsv]
# fig_emission.png   three species (small, medium, large) at 6 eV: temperature rules against the harmonic reference
# fig_fits.png       three PAHFIT fits of population spectra (lowest, median, highest reduced chi-square)
# fig_regress.png    held-out test rows: random forest and the chosen formulas against the true label, per target
import argparse, csv, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species"); ap.add_argument("emission"); ap.add_argument("pop"); ap.add_argument("reg"); ap.add_argument("out")
    ap.add_argument("--template", default="/data/pdrs4all/J_A+A_685_A75/sptemp.tsv")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import h5py
    h = h5py.File(a.emission, "r"); names = [s.decode() for s in h["species"][:]]; grid = h["grid"][:]; en = list(h["energies"][:])
    meta = {r["species"]: r for r in csv.DictReader(open(a.species))}
    nc = np.array([int(meta[n]["nC"]) for n in names]); j = en.index(6.0)
    pick = [int(np.argmin(abs(nc - t) + 1e-3 * np.arange(len(nc)))) for t in (32, 54, 96)]
    fig, axs = plt.subplots(3, 1, figsize=(9, 7.5), sharex=True)
    for ax, i in zip(axs, pick):
        w = 1e4 / grid; m = (w >= 5) & (w <= 20)
        ax.plot(w[m], h["harm"][i, j][m] * 1e3, color="0.55", lw=1, label="harmonic lines, Gaussian 15 cm$^{-1}$")
        ax.plot(w[m], h["tdep"][i, j][m] * 1e3, color="C0", lw=1, label="D1 temperature rules")
        ax.set_ylabel("meV per cm$^{-1}$ per photon"); ax.set_title(f"{names[i]} (N$_C$ = {nc[i]}), 6 eV, T$_{{max}}$ = {h['tmax'][i, j]:.0f} K", fontsize=10)
    axs[0].legend(frameon=False, fontsize=8); axs[-1].set_xlabel("wavelength (um)"); axs[-1].set_xscale("log")
    axs[-1].set_xticks([5, 6, 7, 8, 10, 12, 15, 20]); axs[-1].set_xticklabels(["5", "6", "7", "8", "10", "12", "15", "20"])
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_emission.png"), dpi=130); plt.close(fig)

    feats = list(csv.DictReader(open(os.path.join(a.reg, "features.csv"))))
    ok = [r for r in feats if r["fit_ok"] == "1"]; ok.sort(key=lambda r: float(r["chi2_red"]))
    sel = [ok[0], ok[len(ok) // 2], ok[-1]]
    import astropy.units as u
    from astropy.nddata import StdDevUncertainty
    from specutils import Spectrum1D
    from pahfit.model import Model
    wav, tmpl = measure.read_templates(a.template); seg = measure.segment_of(wav)
    ft = Model.from_yaml("classic.yaml").features
    temps = sorted(float(t) for t in ft["temperature"]["val"][np.asarray(ft["kind"]) == "dust_continuum"])
    cshape, _ = measure.continuum_shape(wav, tmpl["APDR"], temps)
    P = h5py.File(a.pop, "r")
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for ax, r in zip(axs, sel):
        k = int(r["pop"]); rng = np.random.default_rng([0, k])
        pah = measure.to_mrs(P["grid"][:], P["tdep"][k], wav, seg); pah *= 100.0 / pah[(wav > 11.0) & (wav < 11.6)].max()
        cf = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0)))); rng.uniform(1.5, 3.5); noise = rng.normal(0, 1, wav.size)
        tot = pah + 30.0 * cf * cshape; sig = float(np.interp(11.25, wav, tot) / 50.0); flux = tot + sig * noise
        s = Spectrum1D(spectral_axis=wav * u.um, flux=flux * u.MJy / u.sr, uncertainty=StdDevUncertainty(sig * np.ones_like(flux)))
        s.meta["instrument"] = measure.SEG
        m = Model.from_yaml("classic.yaml"); m.guess(s); m.fit(s, verbose=False, maxiter=10000)
        dust = np.asarray(m.features["kind"]) == "dust_feature"
        mod = m.tabulate(instrumentname=measure.SEG, wavelengths=wav).flux.value
        pm = m.tabulate(instrumentname=measure.SEG, wavelengths=wav, feature_mask=dust).flux.value
        ax.plot(wav, flux, color="0.6", lw=0.5, label="synthetic spectrum"); ax.plot(wav, mod, "C3", lw=1, label="PAHFIT model")
        ax.plot(wav, pah, "C0", lw=0.8, label="input PAH emission"); ax.plot(wav, pm, "C1", lw=0.8, ls="--", label="PAHFIT dust features")
        ax.set_ylabel("MJy/sr (11.2 peak = 100)")
        ax.set_title(f"population {k}: log N$_C$ = {float(r['logNc']):.2f}, E = {float(r['E_eV']):.1f} eV, reduced $\\chi^2$ = {float(r['chi2_red']):.2f}", fontsize=10)
    axs[0].legend(frameon=False, fontsize=8, ncol=2); axs[-1].set_xlabel("wavelength (um)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_fits.png"), dpi=130); plt.close(fig)

    res = json.load(open(os.path.join(a.reg, "results.json"))); tg = list(res["targets"])
    fig, axs = plt.subplots(1, len(tg), figsize=(5.2 * len(tg), 4.8), squeeze=False)
    for ax, t in zip(axs[0], tg):
        rows = [r for r in csv.DictReader(open(os.path.join(a.reg, f"pred_{t}.csv"))) if r["split"] == "test"]
        y = np.array([float(r["y"]) for r in rows])
        for col, c, mk in (("rf", "0.6", "o"), ("full", "C0", "."), ("compact", "C1", ".")):
            if col in rows[0]:
                key = "rf_primary" if col == "rf" else f"formula_{col}"
                lab = "random forest" if col == "rf" else f"formula, complexity {res['targets'][t][key]['complexity']}"
                ax.plot(y, [float(r[col]) for r in rows], mk, color=c, ms=3, alpha=0.6, label=f"{lab} (RMSE {res['targets'][t][key]['test']:.3f})")
        ax.plot([y.min(), y.max()], [y.min(), y.max()], "k-", lw=0.8)
        pad = 0.25 * (y.max() - y.min()); ax.set_ylim(y.min() - pad, y.max() + pad)
        ax.set_xlabel(f"true {t}"); ax.set_ylabel(f"predicted {t}"); ax.legend(frameon=False, fontsize=8); ax.set_title(f"test set, n = {len(y)}", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "fig_regress.png"), dpi=130); plt.close(fig)
    print("wrote", os.listdir(a.out))


if __name__ == "__main__":
    main()
