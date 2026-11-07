# PAHFIT closure test at the pinned commit: synthetic MIRI-like spectrum in PAHFIT's own basis,
# fit with classic.yaml + jwst MRS segments, recover complex strengths, a few Monte Carlo refits.
import os, time, warnings, numpy as np, astropy.units as u
from astropy.nddata import StdDevUncertainty
from specutils import Spectrum1D
from pahfit.model import Model
warnings.filterwarnings("ignore")
MAXIT = int(os.environ.get("MAXIT", "1000")); print("maxiter", MAXIT)
SEG = [f"jwst.miri.mrs.ch{c}.{b}" for c in (1, 2, 3) for b in "ABC"] + ["jwst.miri.mrs.ch4.A"]
wav = np.exp(np.arange(np.log(5.0), np.log(20.0), 1 / 6000.0))   # ~R 3000 Nyquist, 5-20 um
print("grid points", wav.size)

def spec(flux, unc):
    s = Spectrum1D(spectral_axis=wav * u.um, flux=flux * u.MJy / u.sr,
                   uncertainty=StdDevUncertainty(unc * np.ones_like(flux)))
    s.meta["instrument"] = SEG
    return s

def drude(x, x0, fwhm, a):
    g = fwhm / x0
    return a * g**2 / ((x / x0 - x0 / x)**2 + g**2)

# 1) seed spectrum (numpy, not PAHFIT) -> fit -> this fitted model becomes the truth
seed = 20 * (wav / 15.0)**3 + sum(drude(wav, x0, w, a) for x0, w, a in
        [(6.22, .19, 60), (7.6, .45, 90), (8.6, .34, 50), (11.25, .2, 120), (12.7, .45, 45), (17.0, 1.1, 20)])
t = time.time(); truth = Model.from_yaml("classic.yaml"); s0 = spec(seed, 0.02 * seed.max())
truth.guess(s0); truth.fit(s0, verbose=False); print(f"seed fit {time.time()-t:.1f} s:", str(truth.fitter.message)[:90])
f = truth.features
print("power dtype fields:", f["power"].dtype.names)
assert not any("11.0" in n for n in f["name"]), "classic unexpectedly has an 11.0 feature"
print("features near 11 um:", [str(n) for n, w in zip(f["name"], f["wavelength"]["val"]) if 10.5 < w < 11.5])
clean = truth.tabulate(instrumentname=SEG, wavelengths=wav).flux.value

BANDS = {"6.2": ("name", "PAH_6.2"), "7.7": ("group", "PAH_7.7_cmp"), "8.6": ("name", "PAH_8.6"),
         "11.2": ("group", "PAH_11.3_cmp"), "12.7": ("group", "PAH_12.6_cmp")}
def strengths(m):
    ft = m.features; p = np.asarray(ft["power"]["val"].filled(np.nan), float)
    return {b: float(np.nansum(p[ft[col] == key])) for b, (col, key) in BANDS.items()}
true = strengths(truth)

# 2) noisy realisation at S/N 50 at the 11.2 um peak, fresh model, fit
sig = clean[np.argmin(abs(wav - 11.25))] / 50
rng = np.random.default_rng(1)
def fit_once(flux):
    m = Model.from_yaml("classic.yaml"); s = spec(flux, sig); m.guess(s)
    t = time.time(); m.fit(s, verbose=False, maxiter=MAXIT); dt = time.time() - t
    resid = (flux - m.tabulate(instrumentname=SEG, wavelengths=wav).flux.value) / sig
    nfree = sum(np.isfinite(m.features[c]["val"]).sum() for c in ("power",))  # rough
    return m, dt, float((resid**2).sum() / (wav.size - nfree))
obs = clean + rng.normal(0, sig, wav.size)
m, dt, chi2 = fit_once(obs); rec = strengths(m)
print(f"fit time {dt:.1f} s, reduced chi2 ~ {chi2:.3f}, converged msg: {str(m.fitter.message)[:90]}")
for b in BANDS: print(f"  {b:>5}: true {true[b]:.4g}  recovered {rec[b]:.4g}  ratio {rec[b]/true[b]:.3f}")
print(f"  6.2/11.2 true {true['6.2']/true['11.2']:.4f} recovered {rec['6.2']/rec['11.2']:.4f}")

# 3) Monte Carlo refits: perturb the observed spectrum by its noise, refit, ratio inside each realisation
r = []
for k in range(int(os.environ.get("NMC", "5"))):
    mk, dtk, _ = fit_once(obs + rng.normal(0, sig, wav.size)); print("   ", str(mk.fitter.message)[:70]); sk = strengths(mk); r.append(sk["6.2"] / sk["11.2"])
    print(f"  MC {k}: {dtk:.1f} s, 6.2/11.2 = {r[-1]:.4f}")
print(f"MC 6.2/11.2 mean {np.mean(r):.4f} std {np.std(r, ddof=1):.4f}")
