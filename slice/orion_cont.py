# Orion Bar templates refitted with PAHFIT packs whose continuum has more freedom under the 6-9 um bands (version 5,
# 2026-11-27). The question: PAHFIT's classic pack gives larger 6.2, 7.7 and 8.6 um bands than local integration
# (reports/nc_sr_v4). Does a continuum that is free to rise under these bands take the difference?
#   python slice/orion_cont.py <out dir> [--nmc 10] [--jobs 48]          (container C; PAHFIT)
# Packs (written to <out dir>/packs):
#   classic  PAHFIT's classic.yaml: starlight at 5,000 K and modified blackbodies at 300-35 K
#   hot      classic + modified blackbodies at 1,500, 1,000, 700, 500 and 400 K, which peak at 2-8 um
#   plat     classic + two broad Drude profiles (6.6 and 7.8 um, FWHM 1.5 um, fixed shape, free amplitude) counted as
#            continuum, not as bands: the 5-10 um plateau of the spline decompositions
#   hotplat  both
#   sub      classic, fitted to the spectrum minus the monotone local continuum (Lpch of measure.py): the local
#            continuum forced on PAHFIT. Its chi-square against classic says whether the data can tell the two apart.
#   anc5     classic, with the two blue anchors of the local continuum (5.45-5.55 and 5.88-5.95 um, next to the 5.25 and
#            5.7 um bands, which the harmonic library does not have) replaced by one at 5.03-5.12 um: how much the local
#            integrals depend on the blue anchor
# Further columns: anchor_pah_<um>, the share of the flux in each anchor window of the local continuum that PAHFIT
# assigns to band profiles (Drude wings and broad components); LpchMod_<band>, the local integral (monotone continuum)
# of PAHFIT's own band model, i.e. what local integration keeps of the PAHFIT bands.
# Output: feat_orion_cont.csv, rows like those of measure.py for the templates (so regress_v2.py --orion reads them),
# with template = <name>@<pack>, the plateau powers as C_<name>, and per fit: chi-square, and the continuum share of the
# flux at 6.2, 7.7, 8.6 and 11.2 um (continuum = model - bands - lines; for plat it includes the two broad profiles).
# orion_cont.json holds the summary table.
import argparse, csv, json, os, sys, time, warnings
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure as M  # noqa: E402

PACKS = {"classic": ([], []), "hot": ([1500, 1000, 700, 500, 400], []), "plat": ([], [(6.6, 1.5), (7.8, 1.5)]),
         "hotplat": ([1500, 1000, 700, 500, 400], [(6.6, 1.5), (7.8, 1.5)]), "sub": ([], []), "anc5": ([], [])}
ANC5 = [(5.03, 5.12)] + M.LANCH[2:]
PEAK = {"6.2": (6.15, 6.30), "7.7": (7.5, 7.75), "8.6": (8.5, 8.7), "11.2": (11.15, 11.35)}


def write_packs(d):
    import yaml
    from importlib import resources
    base = yaml.safe_load(open(str(resources.files("pahfit") / "packs/science" / "classic.yaml")))
    os.makedirs(d, exist_ok=True); out = {}
    for name, (temps, plat) in PACKS.items():
        p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in base.items()}
        p["dust_cont"] = dict(base["dust_cont"], temperature=temps + list(base["dust_cont"]["temperature"]))
        for w, fw in plat:
            p[f"CONT_{w}"] = dict(kind="dust_feature", wavelength=w, fwhm=fw)
        out[name] = os.path.join(d, f"{name}.yaml"); yaml.safe_dump(p, open(out[name], "w"), sort_keys=False)
    return out


def fit(job):
    key, pack, wav, flux, sig, nir, extra = job
    import astropy.units as u
    from astropy.nddata import StdDevUncertainty
    from scipy.ndimage import median_filter
    from specutils import Spectrum1D
    from pahfit.model import Model
    warnings.filterwarnings("ignore")
    row = dict(key=key, **extra)
    from scipy.interpolate import PchipInterpolator
    fm = median_filter(flux, 11, mode="nearest")
    x = np.array([0.5 * (lo + hi) for lo, hi in M.LANCH]); y = np.array([np.median(fm[(wav >= lo) & (wav <= hi)]) for lo, hi in M.LANCH])
    pch = PchipInterpolator(x, y, extrapolate=True)(wav); off = pch if extra["pack"] == "sub" else 0.0
    s = Spectrum1D(spectral_axis=wav * u.um, flux=(flux - off) * u.MJy / u.sr, uncertainty=StdDevUncertainty(sig))
    s.meta["instrument"] = M.SEG
    row.update(M.local_bands(wav, fm, ANC5 if extra["pack"] == "anc5" else M.LANCH, M.LBANDS))
    if nir is not None:
        row.update(M.local_bands(nir[0], median_filter(nir[1], 9, mode="nearest"), M.NANCH, {"3.3": M.NBAND}))
    try:
        m = Model.from_yaml(pack); m.guess(s); t = time.time()
        m.fit(s, verbose=False, maxiter=10000); row["fit_s"] = round(time.time() - t, 2)
        msg = str(m.fitter.message); row["message"] = msg[:80].replace(",", ";").replace("\n", " ")
        row["converged"] = int("maxfev" not in msg and "Number of calls" not in msg)
        ft = m.features; p = np.asarray(ft["power"]["val"].filled(np.nan), float); names = np.asarray(ft["name"]).astype(str)
        kind = np.asarray(ft["kind"]).astype(str); isc = np.char.startswith(names, "CONT_")
        band = (kind == "dust_feature") & ~isc
        for n, v in zip(names[band], p[band]):
            row[f"P_{n}"] = v
        for n, v in zip(names[isc], p[isc]):
            row[f"C_{n[5:]}"] = v
        tab = lambda mask=None: m.tabulate(instrumentname=M.SEG, wavelengths=wav, feature_mask=mask).flux.value  # noqa: E731
        pm = tab(band); ln = tab(kind == "line"); mod = tab() + off; cont = mod - pm - ln
        row["chi2_red"] = float((((flux - mod) / sig) ** 2).sum() / (wav.size - np.isfinite(p).sum()))
        row["n_par"] = int(np.isfinite(p).sum()); row["rms_rel"] = float(np.sqrt(np.mean((fm / median_filter(mod, 11, mode="nearest") - 1) ** 2)))
        k69 = (wav >= 5.3) & (wav <= 10.2); row["chi2_5_10"] = float((((flux - mod) / sig)[k69] ** 2).mean())
        row.update({f"anchor_pah_{0.5 * (lo + hi):.2f}": float(np.median(pm[(wav >= lo) & (wav <= hi)]) / np.median(fm[(wav >= lo) & (wav <= hi)]))
                    for lo, hi in M.LANCH})
        row.update({k_.replace("Lpch_", "LpchMod_"): v for k_, v in M.local_bands(wav, pm, M.LANCH, M.LBANDS).items() if k_.startswith("Lpch_")})
        for b, (lo, hi) in M.WINDOWS.items():
            row[f"Wfit_{b}"] = M.win(wav, pm, lo, hi)
        for b, (lo, hi) in PEAK.items():
            k = (wav >= lo) & (wav <= hi)
            row[f"cshare_{b}"] = float(np.median(cont[k]) / np.median(fm[k]))
        # the same share for the monotone local continuum, for comparison
        for b, (lo, hi) in PEAK.items():
            k = (wav >= lo) & (wav <= hi); row[f"cshare_pch_{b}"] = float(np.median(pch[k]) / np.median(fm[k]))
        if extra["mc"] == 0:
            row["_curves"] = dict(wav=wav, flux=fm, model=mod, cont=cont, bands=pm, pch=pch)
    except Exception as e:
        row.update(message=f"EXCEPTION {type(e).__name__}: {e}"[:80].replace(",", ";"), converged=0)
    return row


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--nmc", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=48); ap.add_argument("--packs", default=",".join(PACKS))
    ap.add_argument("--template", default="/data/pdrs4all/J_A+A_685_A75/sptemp.tsv")
    ap.add_argument("--nir-template", default="/data/pdrs4all/J_A+A_685_A74/t-g395h.tsv")
    a = ap.parse_args()
    from joblib import Parallel, delayed
    os.makedirs(a.out, exist_ok=True); packs = write_packs(os.path.join(a.out, "packs"))
    wav, tmpl = M.read_templates(a.template); nt = M.read_nir_templates(a.nir_template) if os.path.exists(a.nir_template) else None
    jobs = []
    for pk in a.packs.split(","):
        for ti, t in enumerate(M.TEMPLATES):
            f = tmpl[t]; sg = 0.01 * f
            for k in range(a.nmc + 1):                                                  # the noise draws of measure.py orion
                rng = np.random.default_rng([0, ti, k]); fl = f + (rng.normal(0, sg) if k else 0); nir = None
                if nt:
                    wn, fn, en_ = nt[t]; r2 = np.random.default_rng([0, 100 + ti, k]); nir = (wn, fn + (r2.normal(0, 1, fn.size) * en_ if k else 0))
                jobs.append((f"{t}@{pk}:{k}", packs[pk], wav, fl, sg, nir, dict(template=f"{t}@{pk}", pack=pk, mc=k)))
    print(f"{len(jobs)} fits, {a.jobs} workers", flush=True); t0 = time.time()
    rows = Parallel(n_jobs=a.jobs, batch_size=1)(delayed(fit)(j) for j in jobs)
    curves = {r["template"]: r.pop("_curves") for r in rows if "_curves" in r}
    np.savez_compressed(os.path.join(a.out, "orion_cont_curves.npz"), **{f"{t}|{k}": v for t, c in curves.items() for k, v in c.items()})
    cols = []
    for r in rows:
        r.pop("key", None); cols += [c for c in r if c not in cols]
    with open(os.path.join(a.out, "feat_orion_cont.csv"), "w", newline="") as fo:
        w = csv.DictWriter(fo, fieldnames=cols, restval=""); w.writeheader(); w.writerows(rows)
    S = {}
    b112 = lambda r: r["P_PAH_11.3a"] + r["P_PAH_11.3b"]  # noqa: E731
    print(f"done in {time.time() - t0:.0f} s; converged {sum(r.get('converged', 0) for r in rows)}/{len(rows)}")
    print("template@pack     chi2  rms%  | continuum share of the flux at 6.2 7.7 8.6 11.2 | PAHFIT 6.2/11.2 7.7/11.2 8.6/11.2 | Lpch same | 7.7a share | C/7.7")
    for r in rows:
        if r["mc"] or "chi2_red" not in r:
            continue
        p77 = r["P_PAH_7.7a"] + r["P_PAH_7.7b"] + r["P_PAH_7.7c"]; cc = sum(v for c, v in r.items() if c.startswith("C_"))
        e = dict(chi2_red=r["chi2_red"], chi2_5_10=r["chi2_5_10"], anchor_pah={c[11:]: r[c] for c in r if c.startswith("anchor_pah_")},
                 lpch_of_model={b: r[f"LpchMod_{b}"] / r["LpchMod_11.2"] for b in ("6.2", "7.7", "8.6")},
                 kept={b: r[f"LpchMod_{b}"] / r[f"Wfit_{b}"] for b in ("6.2", "7.7", "8.6", "11.2")}, rms_rel=r["rms_rel"], converged=r["converged"], n_par=r["n_par"],
                 cshare={b: r[f"cshare_{b}"] for b in PEAK}, cshare_pch={b: r[f"cshare_pch_{b}"] for b in PEAK},
                 pahfit={"6.2": r["P_PAH_6.2"] / b112(r), "7.7": p77 / b112(r), "8.6": r["P_PAH_8.6"] / b112(r)},
                 lpch={b: r[f"Lpch_{b}"] / r["Lpch_11.2"] for b in ("6.2", "7.7", "8.6")},
                 share_77a=r["P_PAH_7.7a"] / p77, plateau_over_77=cc / p77)
        S[r["template"]] = e
        print(f"{r['template']:16s} {e['chi2_red']:6.1f} {100 * e['rms_rel']:5.2f} | " + " ".join(f"{e['cshare'][b]:.2f}" for b in PEAK)
              + "  (Lpch " + " ".join(f"{e['cshare_pch'][b]:.2f}" for b in PEAK) + ") | " + " ".join(f"{v:5.2f}" for v in e["pahfit"].values())
              + " | " + " ".join(f"{v:5.2f}" for v in e["lpch"].values()) + f" | {e['share_77a']:.2f} | {e['plateau_over_77']:.2f} | chi2(5.3-10.2) {e['chi2_5_10']:.1f} | Lpch of the band model "
              + " ".join(f"{v:5.2f}" for v in e["lpch_of_model"].values()) + " | anchors " + " ".join(f"{v:.2f}" for v in e["anchor_pah"].values()))
    json.dump(S, open(os.path.join(a.out, "orion_cont.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
