# Population spectra and labels (prototype stage 3; plan 5.4, extended to a variable photon energy).
#   python slice/populations.py <species csv> <emission h5> <out h5> [--design v1|v2] [--n 2000] [--seed 0]
#                               [--nc 35,110] [--ev 3,12] [--fixed logNc0,E,...]
# Every population has a log-normal size distribution of fixed 0.2 dex width, centred on logNc0 (log10(35)..log10(110)
# by default), one photon energy E (3-12 eV; spectra interpolated linearly between the energies of the emission file)
# and, when the species table holds charged rows (same structure_id, charge 1), a cation number fraction f in 0-1
# (with neutrals only, f = 0). Neutral and cation of a structure get (1 - f) and f of the structure's weight.
# Absorption is taken proportional to Nc: emitted weight = number weight x Nc.
#
# --design v1 (default; the 2026-11-23 prototype). Latin hypercube over logNc0, E (and f). The weight of a size Nc is
#   the log-normal density times the width of that size's bin on the ladder of sizes present in the pool; structures
#   of the same Nc share it with random Dirichlet(1) shares. --tilt S adds an axis s in [-S, S] that tilts the
#   structure mix inside every size, shares x exp(s z), z = the species' share of trio + quartet H minus its linear
#   trend with log Nc, in units of its scatter.
#
# --design v2 (2026-11-24): the structure mix no longer follows size. In the pool, edge structure and species class
#   are tied to size (quartet H, N-containing and own species sit at particular sizes), and the v1 size formula read
#   size off the 12.0 and 12.7 um bands. v2 draws more hypercube axes and imposes them on the weights:
#     edge   target shares of solo, trio and quartet H (--edge name:lo:hi,...), as the bands see them: sum w Nc n_x /
#            sum w Nc n_H (the labels solo_frac, trio_frac, quartet_frac); 0.22-0.38, 0.08-0.28 and 0.03-0.10 by
#            default; duo is the rest
#     pN     number fraction of N-containing species, 0-0.15
#     pOwn   number fraction of our own high-spin species, 0-0.20
#   Size bins are 0.05 dex wide, merged until each holds at least 8 C/H species of PAHdb; a bin's weight is the
#   log-normal mass inside it. Inside a bin the three classes (C/H, N, own) take fixed shares: a class that exists only
#   in some bins gets, in those bins, its target fraction divided by the log-normal mass of those bins (at most 0.55
#   for N and 0.35 for own), so that the population total meets the target wherever the pool allows. Inside each bin
#   and class, species get Dirichlet(1) shares times the log-normal density. Then every species is tilted by
#   exp(s . x_i) within its bin and class (x_i = Nc (n_x - target n_H) of the molecule), with one vector s per population, solved
#   by Newton iteration so that the population's mean shares equal the targets. Bin weights and class shares are
#   untouched by the tilt. The targets hold for the population, not for each size bin: no pair of targets can be
#   met in all bins of this pool (11 of 13 at best; quartet H exists almost only below 37 C), so inside a population
#   edge structure still varies with size, and the per-molecule means (*_num) stay tied to size. Three further axes are drawn for the
#   continuum of measure.py and stored as labels: cont_type (orion, pl, mbb), cont_level (0.5-2, log-uniform) and
#   cont_par (0-1).
# --fixed: comma-separated logNc0 centres (as Nc) and one photon energy, e.g. "40,60,90@6": populations on a regular
#   grid of f (0, 0.1, ..., 1) at fixed size distributions, v1 weights with equal shares (the analytic-endpoint
#   control, plan 5.9 control 2; endpoint.py).
# Labels: logNc (number-weighted mean of log10 Nc, the size target), sig_logNc, logNc_abs (absorption-weighted),
# E_eV, f_cation, cation_emission_share (6-15 um), irr_num (mean over molecules of irr), solo_frac (sum w Nc n_solo /
# sum w Nc n_H) and likewise duo, trio, quartet; own_frac and N_frac (number fractions); share112 = emission in
# 860-925 cm^-1 over emission in 5-20 um (the 11.2 um denominator check).
import argparse, csv
import numpy as np
import h5py
from scipy.stats import qmc
from scipy.special import erf

SIG = 0.2
CONT_TYPES = ["orion", "pl", "mbb"]


def size_bins(lg, is_ch, step=0.05, nmin=8):
    """Edges of size bins in log10 Nc: `step` wide, merged from the small end until each holds >= nmin C/H structures;
    a short last bin joins its neighbour."""
    top = lg.max() + 1e-6; edges = [lg.min() - 1e-9]; x = edges[0] + step
    while x < top:
        if ((lg >= edges[-1]) & (lg < x) & is_ch).sum() >= nmin:
            edges.append(x)
        x += step
    if ((lg >= edges[-1]) & is_ch).sum() >= nmin or len(edges) == 1:
        edges.append(top)
    else:
        edges[-1] = top
    return np.array(edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species"); ap.add_argument("emission"); ap.add_argument("out")
    ap.add_argument("--n", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nc", default="35,110"); ap.add_argument("--ev", default="3,12")
    ap.add_argument("--tilt", type=float, default=0.0); ap.add_argument("--design", default="v1"); ap.add_argument("--fixed")
    ap.add_argument("--edge", default="solo:0.22:0.38,trio:0.08:0.28,quartet:0.03:0.10")
    a = ap.parse_args()
    meta = {r["species"]: r for r in csv.DictReader(open(a.species)) if r["status"] == "use"}
    h = h5py.File(a.emission, "r")
    names = [s.decode() for s in h["species"][:]]; grid = h["grid"][:]; en = h["energies"][:]
    sel = np.array([n in meta for n in names]); names = [n for n in names if n in meta]   # the table may use fewer species
    S = {v: h[v][:][sel] for v in h if h[v].ndim == 3 and h[v].shape[2] == grid.size}
    M = [meta[n] for n in names]
    nc = np.array([int(m["nC"]) for m in M]); q = np.array([int(m["charge"]) for m in M])
    sid = np.array([m["structure_id"] for m in M]); nH = np.array([int(m["nH"]) for m in M])
    adj = {k: np.array([int(m[k]) for m in M]) for k in ("solo", "duo", "trio", "quartet")}
    own = np.array([m["source"] == "own" for m in M]); hasN = np.array([int(m["nN"]) > 0 for m in M])
    charged = bool((q != 0).any())
    # structures: one per (structure_id, source); partners in other charge states share the structure
    skey = np.array([f"{s}|{m['source']}" for s, m in zip(sid, M)]); ust, sidx = np.unique(skey, return_inverse=True)
    first = np.array([np.where(sidx == k)[0][0] for k in range(len(ust))])
    snc = nc[first]; slg = np.log10(snc)
    ladder = np.unique(snc); lg = np.log10(ladder)
    edges = np.concatenate([[lg[0] - (lg[1] - lg[0]) / 2], (lg[1:] + lg[:-1]) / 2, [lg[-1] + (lg[-1] - lg[-2]) / 2]])
    width = np.diff(edges)
    irr = (adj["trio"] + adj["quartet"]) / np.maximum(nH, 1); sirr = irr[first]
    c = np.polyfit(slg, sirr, 1); z = sirr - np.polyval(c, slg); z = z / z.std()
    scl = np.where(own[first], 2, np.where(hasN[first], 1, 0))                         # class: 0 C/H, 1 N, 2 own
    num = {k_: adj[k_] / np.maximum(nH, 1) for k_ in adj}
    lo, hi = [np.log10(float(x)) for x in a.nc.split(",")]; e0, e1 = [float(x) for x in a.ev.split(",")]
    rng = np.random.default_rng(a.seed + 1)
    if a.fixed:
        cen, Efix = a.fixed.split("@"); fs = np.linspace(0, 1, 11) if charged else [0.0]
        plan = [dict(mu=np.log10(float(x)), E=float(Efix), f=float(f)) for x in cen.split(",") for f in fs]
        a.n = len(plan); design = "fixed"
    else:
        design = a.design
        EDGE = [(x.split(":")[0], float(x.split(":")[1]), float(x.split(":")[2])) for x in a.edge.split(",")]; ne = len(EDGE)
        d = (4 if a.tilt else 3) if design == "v1" else ne + 8
        u = qmc.LatinHypercube(d=d, seed=a.seed).random(a.n)
        plan = [dict(mu=lo + (hi - lo) * u[k, 0], E=e0 + (e1 - e0) * u[k, 1], f=u[k, 2] if charged else 0.0) for k in range(a.n)]
    if design == "v2":
        be = size_bins(slg, scl == 0); nb = len(be) - 1; sbin = np.clip(np.searchsorted(be, slg, side="right") - 1, 0, nb - 1)
        has = np.array([[((sbin == b) & (scl == k)).any() for k in range(3)] for b in range(nb)])
        cid = sbin * 3 + scl; ncell = nb * 3
        SA = np.stack([snc * adj[n_][first] for n_, _, _ in EDGE], 1) / 1e3; SH = snc * nH[first] / 1e3   # Nc n_x, Nc n_H
        print(f"v2 size bins (Nc): {np.round(10 ** be).astype(int).tolist()}; C/H, N, own structures per bin: "
              f"{np.bincount(cid, minlength=ncell).reshape(nb, 3).tolist()}; edge targets {EDGE}")
    dg = grid[1] - grid[0]; miri = (grid >= 500) & (grid <= 2000); w112 = (grid >= 860) & (grid < 925); w615 = (grid >= 1e4 / 15) & (grid <= 1e4 / 6)
    out = {v: np.zeros((a.n, grid.size), np.float32) for v in S}; lab = []; miss = 0.0
    for k in range(a.n):
        mu, E, f = plan[k]["mu"], plan[k]["E"], plan[k]["f"]
        row = dict(pop=k, logNc0=mu, E_eV=E, f_cation=f)
        ws = np.zeros(len(ust))
        if design in ("v1", "fixed"):
            tilt = a.tilt * (2 * u[k, 3] - 1) if (a.tilt and design == "v1") else 0.0
            wl = np.exp(-0.5 * ((lg - mu) / SIG) ** 2) * width; wl /= wl.sum()
            for n_, w_ in zip(ladder, wl):
                m = np.where(snc == n_)[0]
                sh = (rng.dirichlet(np.ones(len(m))) if design == "v1" else np.ones(len(m))) * np.exp(tilt * z[m]); ws[m] = w_ * sh / sh.sum()
            row["tilt"] = tilt
        else:
            tgt = np.array([lo_ + (hi_ - lo_) * u[k, 3 + i_] for i_, (_, lo_, hi_) in enumerate(EDGE)])
            pN = 0.15 * u[k, ne + 3]; pO = 0.20 * u[k, ne + 4]
            ct = CONT_TYPES[min(int(3 * u[k, ne + 5]), 2)]
            row.update(**{f"{n_}_target": t_ for (n_, _, _), t_ in zip(EDGE, tgt)}, pN=pN, pOwn=pO, cont_type=ct,
                       cont_level=float(np.exp(np.log(0.5) + np.log(4.0) * u[k, ne + 6])), cont_par=u[k, ne + 7])
            cdf = 0.5 * (1 + erf((be - mu) / SIG / np.sqrt(2))); Mb = np.diff(cdf); Mb /= Mb.sum()
            aN = min(pN / max(Mb[has[:, 1]].sum(), 1e-9), 0.55); aO = min(pO / max(Mb[has[:, 2]].sum(), 1e-9), 0.35)
            share = np.stack([np.zeros(nb), aN * has[:, 1], aO * has[:, 2]], 1); share[:, 0] = 1 - share.sum(1)
            Mc = (Mb[:, None] * share).ravel()                                        # weight of every (bin, class) cell
            base = rng.exponential(size=len(ust)) * np.exp(-0.5 * ((slg - mu) / SIG) ** 2)   # Dirichlet(1) shares x density
            X = SA - tgt[None, :] * SH[:, None]                                         # zero mean <=> sum w Nc n_x / sum w Nc n_H = target

            def tilted(sv):
                e = X @ sv; e0 = np.full(ncell, -np.inf); np.maximum.at(e0, cid, e)
                wv = base * np.exp(e - e0[cid]); Z = np.bincount(cid, wv, ncell)
                return Mc[cid] * wv / Z[cid], (Mc * (np.log(np.where(Z > 0, Z, 1)) + np.where(np.isfinite(e0), e0, 0))).sum()
            sv = np.zeros(ne); wv, phi = tilted(sv)
            for it in range(60):                                                      # Newton with backtracking (convex problem)
                mom = wv @ X
                if np.abs(mom).max() < 1e-5:
                    break
                mcell = np.stack([np.bincount(cid, wv * X[:, i_], ncell) for i_ in range(ne)], 1)
                J = (X * wv[:, None]).T @ X - (mcell / np.where(Mc > 0, Mc, 1)[:, None]).T @ mcell
                step = -np.linalg.solve(J + 1e-7 * np.eye(ne), mom); lam = 1.0
                while lam > 1e-4:
                    s2 = np.clip(sv + lam * step, -150, 150); w2, p2 = tilted(s2)
                    if p2 <= phi + 1e-12:
                        break
                    lam *= 0.5
                if lam <= 1e-4:
                    break
                sv, wv, phi = s2, w2, p2
            miss += np.abs(wv @ X).max(); ws = wv / wv.sum()
            row["n_eff"] = float(np.exp(-(ws[ws > 0] * np.log(ws[ws > 0])).sum()))
        w = ws[sidx] * np.where(q == 0, 1 - f, f) if charged else ws[sidx]          # number weight of each species row
        w = w / w.sum(); wa = w * nc; wa = wa / wa.sum()                            # absorbed-photon weights
        j = min(np.searchsorted(en, E) - 1, len(en) - 2); j = max(j, 0); t = (E - en[j]) / (en[j + 1] - en[j])
        for v in S:
            sp = np.tensordot(wa, (1 - t) * S[v][:, j, :] + t * S[v][:, j + 1, :], axes=1)
            out[v][k] = sp
            if v in ("tdep", "harm"):
                row[f"share112_{v}"] = sp[w112].sum() / sp[miri].sum()
                if charged:
                    spc = np.tensordot(wa * (q != 0), (1 - t) * S[v][:, j, :] + t * S[v][:, j + 1, :], axes=1)
                    row[f"cation_emission_share_{v}"] = spc[w615].sum() / sp[w615].sum()
        lm = (w * np.log10(nc)).sum()
        row.update(logNc=lm, sig_logNc=np.sqrt((w * (np.log10(nc) - lm) ** 2).sum()), logNc_abs=(wa * np.log10(nc)).sum(),
                   own_frac=w[own].sum(), N_frac=w[hasN].sum(), irr_num=(w * irr).sum(),
                   **{f"{a_}_num": (w * num[a_]).sum() for a_ in adj},
                   **{f"{a_}_frac": (w * nc * adj[a_]).sum() / (w * nc * nH).sum() for a_ in adj})
        lab.append(row)
    with h5py.File(a.out, "w") as o:
        o["grid"] = grid
        for v in S:
            o[v] = out[v]
        o.attrs["seed"] = a.seed; o.attrs["species_table"] = a.species; o.attrs["n_species"] = len(names); o.attrs["design"] = design
        o.attrs["units"] = "eV per cm^-1 per absorbed photon (population mean)"
    cols = list(lab[0])
    with open(a.out.replace(".h5", "_labels.csv"), "w", newline="") as fo:
        wr = csv.DictWriter(fo, fieldnames=cols); wr.writeheader()
        wr.writerows({c: (r[c] if c in ("pop", "cont_type") else f"{r[c]:.6g}") for c in cols} for r in lab)
    L = {c: np.array([r[c] for r in lab]) for c in cols if c != "cont_type"}
    print(f"{a.n} populations (design {design}) from {len(names)} species ({len(ust)} structures, {len(ladder)} sizes, Nc {ladder[0]}-{ladder[-1]}); "
          f"charged rows: {'yes' if charged else 'no (f = 0)'}; variants {list(S)}")
    print(f"  logNc realised {L['logNc'].min():.3f}..{L['logNc'].max():.3f} (requested centre {lo:.3f}..{hi:.3f}); "
          f"width {np.median(L['sig_logNc']):.3f} dex median")
    if a.n > 20:
        cc = lambda x, y: np.corrcoef(L[x], L[y])[0, 1]
        print("  correlation with logNc: " + ", ".join(f"{c} {cc('logNc', c):+.3f}" for c in
              ("E_eV", "irr_num", "solo_frac", "duo_frac", "trio_frac", "quartet_frac", "N_frac", "own_frac") if L[c].std() > 0))
        if design == "v2":
            print("  correlation with logNc, per-molecule shares: " + ", ".join(f"{c} {cc('logNc', c + '_num'):+.3f}" for c in adj if L[c + "_num"].std() > 0))
            r = lambda x, y: np.sqrt(np.mean((L[x] - L[y]) ** 2))
            print("  targets met (rms of realised - target): " + ", ".join(f"{n_} {r(n_ + '_frac', n_ + '_target'):.4f}" for n_, _, _ in EDGE)
                  + f", N {r('N_frac', 'pN'):.4f}, own {r('own_frac', 'pOwn'):.4f}; mean largest miss {miss / a.n:.5f}; "
                  f"effective number of species per population: median {np.median(L['n_eff']):.0f}, min {L['n_eff'].min():.0f}")
            print("  ranges: " + ", ".join(f"{c} {L[c].min():.3f}..{L[c].max():.3f}" for c in ("solo_frac", "duo_frac", "trio_frac", "quartet_frac", "N_frac", "own_frac"))
                  + f"; continuum types { {t: sum(r_['cont_type'] == t for r_ in lab) for t in CONT_TYPES} }")
            print("  correlation of the edge shares with E and with each other: " + ", ".join(
                f"{x}-{y} {cc(x, y):+.2f}" for x, y in (("E_eV", "solo_frac"), ("E_eV", "trio_frac"), ("solo_frac", "trio_frac"), ("N_frac", "solo_frac"), ("own_frac", "solo_frac")) if L[x].std() > 0 and L[y].std() > 0))
    for v in ("tdep", "harm"):
        if v in S:
            s = L[f"share112_{v}"]
            print(f"  11.2 um share of 5-20 um emission, {v}: min {s.min():.4f}, median {np.median(s):.4f}, max {s.max():.4f}")


if __name__ == "__main__":
    main()
