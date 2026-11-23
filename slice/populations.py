# Population spectra and labels (prototype stage 3; plan 5.4, extended to a variable photon energy).
#   python slice/populations.py <species csv> <emission h5> <out h5> [--n 2000] [--seed 0] [--nc 35,110] [--ev 3,12]
# Latin hypercube over
#   logNc0   centre of a log-normal size distribution of fixed 0.2 dex width, log10(35)..log10(110) by default
#   E        energy of the absorbed photon, 3-12 eV (one energy per population; spectra interpolated linearly between
#            the energies of the emission file)
#   f        cation number fraction, 0-1; drawn only when the species table holds charged rows (same structure_id,
#            charge 1). With neutrals only, f = 0 for every population.
# Weights (plan 5.4): the weight of a size Nc is the log-normal density times the width of that size's bin on the
# ladder of sizes present in the pool, renormalised (so the truncation at both ends is accounted for). Structures of
# the same Nc share that weight with random Dirichlet(1) shares, drawn per population, so that the structure mix
# varies between populations instead of being one fixed average. Neutral and cation of a structure get (1 - f) and
# f. Absorption is taken proportional to Nc: emitted weight = number weight x Nc.
# --tilt S (default 0 = off): a fourth hypercube axis s in [-S, S] tilts the structure mix inside every size,
# shares x exp(s z), with z the species' share of trio + quartet H minus its linear trend with log Nc, in units of its
# scatter. In the pool, edge structure is correlated with size (quartet H and N-containing species sit at the small
# end), so a regression can read size off the edge-structure bands; the tilt makes the mix vary at fixed size.
# Labels: logNc (number-weighted mean of log10 Nc, the size target), sig_logNc, logNc_abs (absorption-weighted),
# E_eV, f_cation, cation_emission_share (6-15 um), solo_frac (sum w Nc n_solo / sum w Nc n_H) and likewise duo, trio,
# quartet; own_frac and N_frac (number fractions of our own high-spin species and of N-containing species);
# share112 = emission in 860-925 cm^-1 over emission in 5-20 um (the 11.2 um denominator check).
import argparse, csv
import numpy as np
import h5py
from scipy.stats import qmc

SIG = 0.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species"); ap.add_argument("emission"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nc", default="35,110"); ap.add_argument("--ev", default="3,12")
    ap.add_argument("--tilt", type=float, default=0.0)
    a = ap.parse_args()
    meta = {r["species"]: r for r in csv.DictReader(open(a.species)) if r["status"] == "use"}
    h = h5py.File(a.emission, "r")
    names = [s.decode() for s in h["species"][:]]; grid = h["grid"][:]; en = h["energies"][:]
    S = {v: h[v][:] for v in ("tdep", "harm")}
    M = [meta[n] for n in names]
    nc = np.array([int(m["nC"]) for m in M]); q = np.array([int(m["charge"]) for m in M])
    sid = np.array([m["structure_id"] for m in M]); nH = np.array([int(m["nH"]) for m in M])
    adj = {k: np.array([int(m[k]) for m in M]) for k in ("solo", "duo", "trio", "quartet")}
    own = np.array([m["source"] == "own" for m in M]); hasN = np.array([int(m["nN"]) > 0 for m in M])
    charged = bool((q != 0).any())
    # structures: one per (structure_id, source); partners in other charge states share the structure
    skey = np.array([f"{s}|{m['source']}" for s, m in zip(sid, M)]); ust, sidx = np.unique(skey, return_inverse=True)
    snc = np.array([nc[sidx == k][0] for k in range(len(ust))])
    ladder = np.unique(snc); lg = np.log10(ladder)
    edges = np.concatenate([[lg[0] - (lg[1] - lg[0]) / 2], (lg[1:] + lg[:-1]) / 2, [lg[-1] + (lg[-1] - lg[-2]) / 2]])
    width = np.diff(edges)
    irr = (adj["trio"] + adj["quartet"]) / nH; sirr = np.array([irr[sidx == k][0] for k in range(len(ust))])
    c = np.polyfit(np.log10(snc), sirr, 1); z = sirr - np.polyval(c, np.log10(snc)); z = z / z.std()
    lo, hi = [np.log10(float(x)) for x in a.nc.split(",")]; e0, e1 = [float(x) for x in a.ev.split(",")]
    u = qmc.LatinHypercube(d=4 if a.tilt else 3, seed=a.seed).random(a.n); rng = np.random.default_rng(a.seed + 1)
    dg = grid[1] - grid[0]; miri = (grid >= 500) & (grid <= 2000); w112 = (grid >= 860) & (grid < 925); w615 = (grid >= 1e4 / 15) & (grid <= 1e4 / 6)
    out = {v: np.zeros((a.n, grid.size), np.float32) for v in S}; lab = []
    for k in range(a.n):
        mu = lo + (hi - lo) * u[k, 0]; E = e0 + (e1 - e0) * u[k, 1]; f = u[k, 2] if charged else 0.0
        tilt = a.tilt * (2 * u[k, 3] - 1) if a.tilt else 0.0
        wl = np.exp(-0.5 * ((lg - mu) / SIG) ** 2) * width; wl /= wl.sum()
        ws = np.zeros(len(ust))
        for n_, w_ in zip(ladder, wl):
            m = np.where(snc == n_)[0]; sh = rng.dirichlet(np.ones(len(m))) * np.exp(tilt * z[m]); ws[m] = w_ * sh / sh.sum()
        w = ws[sidx] * np.where(q == 0, 1 - f, f)                                  # number weight of each species row
        if not charged:
            w = ws[sidx]
        w = w / w.sum(); wa = w * nc; wa = wa / wa.sum()                            # absorbed-photon weights
        j = min(np.searchsorted(en, E) - 1, len(en) - 2); j = max(j, 0); t = (E - en[j]) / (en[j + 1] - en[j])
        row = dict(pop=k, logNc0=mu, E_eV=E, f_cation=f, tilt=tilt)
        for v in S:
            sp = np.tensordot(wa, (1 - t) * S[v][:, j, :] + t * S[v][:, j + 1, :], axes=1)
            out[v][k] = sp
            row[f"share112_{v}"] = sp[w112].sum() / sp[miri].sum()
            if charged:
                spc = np.tensordot(wa * (q != 0), (1 - t) * S[v][:, j, :] + t * S[v][:, j + 1, :], axes=1)
                row[f"cation_emission_share_{v}"] = spc[w615].sum() / sp[w615].sum()
        lm = (w * np.log10(nc)).sum()
        row.update(logNc=lm, sig_logNc=np.sqrt((w * (np.log10(nc) - lm) ** 2).sum()), logNc_abs=(wa * np.log10(nc)).sum(),
                   own_frac=w[own].sum(), N_frac=w[hasN].sum(),
                   **{f"{a_}_frac": (w * nc * adj[a_]).sum() / (w * nc * nH).sum() for a_ in adj})
        lab.append(row)
    with h5py.File(a.out, "w") as o:
        o["grid"] = grid
        for v in S:
            o[v] = out[v]
        o.attrs["seed"] = a.seed; o.attrs["species_table"] = a.species; o.attrs["n_species"] = len(names)
        o.attrs["units"] = "eV per cm^-1 per absorbed photon (population mean)"
    cols = list(lab[0])
    with open(a.out.replace(".h5", "_labels.csv"), "w", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=cols); wr.writeheader()
        wr.writerows({c: (r[c] if c == "pop" else f"{r[c]:.6g}") for c in cols} for r in lab)
    L = {c: np.array([r[c] for r in lab]) for c in cols}
    print(f"{a.n} populations from {len(names)} species ({len(ust)} structures, {len(ladder)} sizes, Nc {ladder[0]}-{ladder[-1]}); "
          f"charged rows: {'yes' if charged else 'no (f = 0)'}")
    print(f"  logNc realised {L['logNc'].min():.3f}..{L['logNc'].max():.3f} (requested centre {lo:.3f}..{hi:.3f}); "
          f"width {np.median(L['sig_logNc']):.3f} dex median; corr(logNc, E) = {np.corrcoef(L['logNc'], L['E_eV'])[0, 1]:+.3f}")
    print(f"  solo_frac {L['solo_frac'].min():.3f}..{L['solo_frac'].max():.3f}, corr with logNc {np.corrcoef(L['logNc'], L['solo_frac'])[0, 1]:+.3f}; "
          f"own_frac median {np.median(L['own_frac']):.3f}; N_frac median {np.median(L['N_frac']):.3f}")
    for v in S:
        s = L[f"share112_{v}"]
        print(f"  11.2 um share of 5-20 um emission, {v}: min {s.min():.4f}, median {np.median(s):.4f}, max {s.max():.4f}")


if __name__ == "__main__":
    main()
