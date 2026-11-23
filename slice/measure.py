# Instrument model and PAHFIT band measurement (prototype stages 4-5; plan 5.5-5.6). Runs in container C.
#   python slice/measure.py <pop h5> <out csv> [--variant tdep|harm] [--mode main|clean|oob] [--n all] [--jobs 48]
#                           [--template /data/pdrs4all/J_A+A_685_A75/sptemp.tsv] [--seed 0]
#   python slice/measure.py orion <out csv> [--template ...] [--nmc 30] [--jobs 48]
# Instrument: MIRI MRS, 5-20 um. Each population spectrum is smoothed to the resolving power of the MRS segment
# (PAHFIT's jwst.yaml pack, the same model the fit uses) and sampled on the wavelength grid of the Orion Bar
# templates (CDS J/A+A/685/A75), thinned 4x (step 0.00184 um) to keep a fit at a few seconds.
# Modes:
#   main   + dust continuum (modified blackbodies, nu^2 emissivity, at PAHFIT's continuum temperatures, with weights
#            fitted to the atomic-PDR template at feature-free wavelengths), scaled to 0.3 x the 11.2 um peak at
#            11.2 um times a random factor 0.5-2 (log-uniform); Gaussian noise with S/N 50 per pixel at the 11.2 peak
#   clean  no continuum, no noise (injection-recovery, plan 5.9 control 1)
#   oob    as main, but the continuum is a power law in wavelength (index 1.5-3.5), which PAHFIT's basis lacks
#   orion  the five templates themselves, block-averaged 4x; 1 % relative uncertainty assumed (the table has none),
#          plus --nmc Monte Carlo refits with that noise
# PAHFIT: classic.yaml, maxiter 10000, fitter message recorded. Output per spectrum: P_<feature> (PAHFIT power of
# every dust feature), Win_<band> / Wfit_<band> (integrals of F_nu d nu of the input PAH-only spectrum and of
# PAHFIT's PAH-only model over the same windows, same units), reduced chi-square, convergence flag.
import argparse, csv, os, sys, time, warnings
import numpy as np

SEG = [f"jwst.miri.mrs.ch{c}.{b}" for c in (1, 2, 3) for b in "ABC"] + ["jwst.miri.mrs.ch4.A"]
WINDOWS = {"6.2": (6.0, 6.6), "7.7": (7.0, 8.15), "8.6": (8.4, 8.9), "11.2": (10.9, 11.7), "12.0": (11.75, 12.25),
           "12.7": (12.3, 13.1), "13.5": (13.2, 13.9)}
ANCHORS = [(5.45, 5.55), (5.88, 5.95), (9.3, 9.55), (10.1, 10.4), (14.6, 14.9), (15.0, 15.4), (18.0, 18.5), (19.2, 20.0)]
HC_K = 1.438776877
TEMPLATES = ["HII", "APDR", "DF3", "DF2", "DF1"]


def read_templates(path, thin=4):
    rows = []
    for line in open(path):
        p = line.split()
        if len(p) == 7 and p[0].isdigit():
            rows.append([float(x) for x in p[1:]])
    d = np.array(rows); d = d[(d[:, 0] >= 5.0) & (d[:, 0] <= 20.0)]
    n = len(d) // thin * thin
    d = d[:n].reshape(-1, thin, 6).mean(1)
    return d[:, 0], {t: d[:, i + 1] for i, t in enumerate(TEMPLATES)}


def mbb(wav, T):
    nu = 1e4 / wav
    return nu ** 5 / np.expm1(HC_K * nu / T)


def continuum_shape(wav, apdr, temps):
    from scipy.optimize import nnls
    a = np.array([np.median(apdr[(wav >= lo) & (wav <= hi)]) for lo, hi in ANCHORS])
    x = np.array([0.5 * (lo + hi) for lo, hi in ANCHORS])
    B = np.stack([mbb(x, T) / mbb(11.25, T) for T in temps], 1)
    w, _ = nnls(B / a[:, None], np.ones(len(a)))                                    # relative residuals
    c = sum(wi * mbb(wav, T) / mbb(11.25, T) for wi, T in zip(w, temps))
    return c / np.interp(11.25, wav, c), w


def segment_of(wav):
    from pahfit import instrument
    r = [instrument.wave_range(s) for s in SEG]
    cut = [0.5 * (r[i][1] + r[i + 1][0]) for i in range(len(SEG) - 1)]
    return np.searchsorted(cut, wav)


def to_mrs(grid, spec, wav, seg):
    """Smooth a spectrum on the wavenumber grid to each segment's resolving power, sample at wav."""
    from pahfit import instrument
    from scipy.ndimage import gaussian_filter1d
    out = np.zeros(wav.size); dg = grid[1] - grid[0]
    for i, s in enumerate(SEG):
        m = seg == i
        if not m.any():
            continue
        wc = np.median(wav[m]); fw = float(np.asarray(instrument.fwhm(s, wc, as_bounded=False)).ravel()[0])
        sg = (fw / wc) * (1e4 / wc) / 2.3548 / dg                                   # pixels
        out[m] = np.interp(1e4 / wav[m], grid, gaussian_filter1d(spec, sg))
    return out


def win(wav, f, lo, hi):
    m = (wav >= lo) & (wav <= hi)
    return float(np.trapezoid(f[m] / wav[m] ** 2, wav[m]))                           # int F_nu d nu, up to a constant


def fit(job):
    key, wav, flux, sig, pah, extra = job
    import astropy.units as u
    from astropy.nddata import StdDevUncertainty
    from specutils import Spectrum1D
    from pahfit.model import Model
    warnings.filterwarnings("ignore")
    s = Spectrum1D(spectral_axis=wav * u.um, flux=flux * u.MJy / u.sr, uncertainty=StdDevUncertainty(sig * np.ones_like(flux)))
    s.meta["instrument"] = SEG
    row = dict(key=key, **extra)
    try:
        m = Model.from_yaml("classic.yaml"); m.guess(s); t = time.time()
        m.fit(s, verbose=False, maxiter=10000); row["fit_s"] = round(time.time() - t, 2)
        msg = str(m.fitter.message); row["message"] = msg[:80].replace(",", ";").replace("\n", " ")
        row["converged"] = int("maxfev" not in msg and "Number of calls" not in msg)      # MINPACK: stopped on tolerances
        ft = m.features; p = np.asarray(ft["power"]["val"].filled(np.nan), float); dust = np.asarray(ft["kind"]) == "dust_feature"
        for n, v in zip(np.asarray(ft["name"])[dust], p[dust]):
            row[f"P_{n}"] = v
        mod = m.tabulate(instrumentname=SEG, wavelengths=wav).flux.value
        row["chi2_red"] = float((((flux - mod) / sig) ** 2).sum() / (wav.size - np.isfinite(p).sum()))
        pm = m.tabulate(instrumentname=SEG, wavelengths=wav, feature_mask=dust).flux.value
        for b, (lo, hi) in WINDOWS.items():
            row[f"Wfit_{b}"] = win(wav, pm, lo, hi)
            if pah is not None:
                row[f"Win_{b}"] = win(wav, pah, lo, hi)
    except Exception as e:                                                            # a failed fit is recorded, not dropped
        row.update(message=f"EXCEPTION {type(e).__name__}: {e}"[:80].replace(",", ";"), converged=0)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pop"); ap.add_argument("out")
    ap.add_argument("--variant", default="tdep"); ap.add_argument("--mode", default="main"); ap.add_argument("--n", default="all")
    ap.add_argument("--jobs", type=int, default=48); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--template", default="/data/pdrs4all/J_A+A_685_A75/sptemp.tsv"); ap.add_argument("--nmc", type=int, default=30)
    a = ap.parse_args()
    from joblib import Parallel, delayed
    from pahfit.model import Model
    wav, tmpl = read_templates(a.template)
    jobs = []
    if a.pop == "orion":
        for t in TEMPLATES:
            f = tmpl[t]; sg = 0.01 * f
            for k in range(a.nmc + 1):
                rng = np.random.default_rng([a.seed, TEMPLATES.index(t), k])
                jobs.append((f"{t}:{k}", wav, f + (rng.normal(0, sg) if k else 0), np.median(sg), None, dict(template=t, mc=k)))
        # PAHFIT takes one uncertainty per pixel; pass the per-pixel 1 % through a wrapper below
        jobs = [(k, w, f, 0.01 * tmpl[e["template"]], p, e) for k, w, f, s, p, e in jobs]
    else:
        import h5py
        h = h5py.File(a.pop, "r"); grid = h["grid"][:]; S = h[a.variant]
        n = S.shape[0] if a.n == "all" else int(a.n)
        seg = segment_of(wav)
        ft = Model.from_yaml("classic.yaml").features
        temps = sorted(float(t) for t in ft["temperature"]["val"][np.asarray(ft["kind"]) == "dust_continuum"])
        cshape, cw = continuum_shape(wav, tmpl["APDR"], temps)
        print("continuum: modified blackbodies at", temps, "K; weights at 11.25 um", np.round(cw / cw.sum(), 3), flush=True)
        for k in range(n):
            rng = np.random.default_rng([a.seed, k])
            pah = to_mrs(grid, S[k], wav, seg)
            pk = pah[(wav > 11.0) & (wav < 11.6)].max(); pah = pah * (100.0 / pk)   # 11.2 um peak = 100 MJy/sr
            cf = float(np.exp(rng.uniform(np.log(0.5), np.log(2.0)))); idx = float(rng.uniform(1.5, 3.5))
            noise = rng.normal(0, 1, wav.size)
            if a.mode == "clean":
                flux, sig, extra = pah, 0.1, dict(cont_factor=0.0)
            else:
                c = cshape if a.mode == "main" else (wav / 11.25) ** idx
                tot = pah + 0.3 * 100.0 * cf * c
                sig = float(np.interp(11.25, wav, tot) / 50.0)
                flux, extra = tot + sig * noise, dict(cont_factor=round(cf, 4))
                if a.mode == "oob":
                    extra["pl_index"] = round(idx, 3)
            jobs.append((k, wav, flux, sig, pah, dict(pop=k, variant=a.variant, mode=a.mode, sigma=round(sig, 4), **extra)))
    print(f"{len(jobs)} fits, {wav.size} points each, {a.jobs} workers", flush=True)
    t0 = time.time()
    rows = Parallel(n_jobs=a.jobs, batch_size=1, verbose=0)(delayed(fit)(j) for j in jobs)
    cols = []
    for r in rows:
        r.pop("key", None)
        cols += [c for c in r if c not in cols]
    with open(a.out, "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)
    ok = [r for r in rows if r.get("converged")]
    ts = [r["fit_s"] for r in rows if "fit_s" in r]
    print(f"done in {time.time() - t0:.0f} s wall; converged {len(ok)}/{len(rows)}; fit time median {np.median(ts):.1f} s, max {max(ts):.1f} s; "
          f"reduced chi2 median {np.median([r['chi2_red'] for r in rows if 'chi2_red' in r]):.2f}")
    from collections import Counter
    print("messages:", Counter(r["message"] for r in rows).most_common(4))


if __name__ == "__main__":
    main()
