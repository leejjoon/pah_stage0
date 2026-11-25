# Zero-new-DFT diagnostics for a dipole model (coordinator 2026-11-24 17:03, feat-dipole step 1): does the static dipole
# error matter for the vibrational (dmu/dQ) intensities that G5 and the finite-T spectra use?
#   python dipole_diag.py corr <eval.json> <g5 dir> --dipole polar [--ref-flags csv] [--json out.json]
#   python dipole_diag.py fd-dft <g5 dir> <out dir> --pairs A,B [--workers 6 --threads 8]   (needs pyscf)
#       DFT-FD G5 ceiling: B3LYP/6-31G* dipole derivatives at the force model's minimum, scored on its modes
#   python dipole_diag.py paired <relabel dir> [--json]   dipole-model slope error on identical DFT MD frames
# eval.json: d1/mlff_dipole_eval.py output (frames: pair, ref/pred in D); g5 dir: d1/dipole_g5.py run output with
# dmu_<dipole>. Per held-out pair and charge: static MAE (all held-out frames of the pair, D) vs the G5 band-integrated
# scores (dipole_g5.score: share deviations of the 3.3-12.7 um bands, band-pass). Prints the rank correlation of static
# MAE with the worst tested-band |share dev|, the named worst static cations against the < 0.20 D cations, and, per band,
# absolute errors in km/mol next to the relative ones, split by PAHdb band strength (< 5 km/mol vs >= 5), so fractional
# blow-ups on tiny bands are separated from real intensity errors.
import argparse, collections, csv, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dipole_g5 import BANDS, band_of, intensities, score  # noqa: E402

WORST = ["C60H20_10211", "C60H30_3608", "C62H20_10176", "C92H24_6477"]


def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def corr(a):
    ev = json.load(open(a.eval))["frames"]
    st = collections.defaultdict(list)
    for key, fr in ev.items():
        q, split = key.split("|")
        if split == "train":
            continue
        for r in fr:
            st[(r["pair"], int(q[1]))].append(np.linalg.norm(np.subtract(r["pred"], r["ref"])))
    flags = {int(r["pahdb_v4_uid"]) for r in csv.DictReader(open(a.ref_flags))} if a.ref_flags else set()
    rows = []
    for f in sorted(glob.glob(f"{a.g5}/*_q[01].npz")):
        d = np.load(f); info = json.loads(str(d["info"])); key = f"dmu_{a.dipole}"
        if key not in d or info["uid"] in flags:
            continue
        I = intensities(d, d[key]); s = score(d, I)
        Ip, fp = d["inten_pahdb"], d["freq_pahdb"]; bands = np.array([band_of(x) for x in fp], dtype=object)
        km = {b: (float(I[bands == b].sum()), float(Ip[bands == b].sum())) for b in BANDS}
        tested = [abs(s["band_rel_dev"][b]) for b in s["bands_tested"]]
        mu = st.get((info["pair"], info["charge"]), [])
        rows.append(dict(pair=info["pair"], q=info["charge"], n_atoms=len(d["masses"]), split=info["split"],
                         static_mae=float(np.mean(mu)) if mu else np.nan, n_frames=len(mu), band_pass=s["band_pass"],
                         worst_dev=max(tested) if tested else np.nan, median_dev=float(np.median(tested)) if tested else np.nan,
                         total_ratio=s["total_ratio"], band_rel_dev=s["band_rel_dev"], band_km=km))
    out = dict(rows=rows)
    for q in (0, 1):
        R = [r for r in rows if r["q"] == q and np.isfinite(r["static_mae"])]
        x = np.array([r["static_mae"] for r in R]); y = np.array([r["worst_dev"] for r in R])
        ok = np.isfinite(y)
        print(f"--- q{q}: {len(R)} pairs; Spearman(static MAE, worst tested-band |share dev|) = {spearman(x[ok], y[ok]):+.2f}; "
              f"Spearman(static MAE, n_atoms) = {spearman(x, np.array([r['n_atoms'] for r in R])):+.2f}")
        out[f"q{q}_spearman"] = spearman(x[ok], y[ok])
        # size-matched: within >= 60 atoms only
        big = [r for r in R if r["n_atoms"] >= 60]
        if len(big) > 4:
            xb = np.array([r["static_mae"] for r in big]); yb = np.array([r["worst_dev"] for r in big]); ob = np.isfinite(yb)
            print(f"    >=60 atoms only ({len(big)}): Spearman {spearman(xb[ob], yb[ob]):+.2f}")
            out[f"q{q}_spearman_ge60"] = spearman(xb[ob], yb[ob])
        if q == 1:
            groups = {"worst static (named)": [r for r in R if r["pair"] in WORST],
                      "static < 0.20 D": [r for r in R if r["static_mae"] < 0.20],
                      "static < 0.20 D, >= 60 atoms": [r for r in R if r["static_mae"] < 0.20 and r["n_atoms"] >= 60]}
            for g, G in groups.items():
                if not G:
                    continue
                print(f"    {g:30s} n={len(G):2d}  static {np.mean([r['static_mae'] for r in G]):.3f} D  band-pass "
                      f"{np.mean([r['band_pass'] for r in G]):.2f}  median worst |dev| {np.median([r['worst_dev'] for r in G]):.2f}  "
                      f"median total ratio {np.median([r['total_ratio'] for r in G]):.2f}")
                out[f"q1_group|{g}"] = dict(n=len(G), pairs=[r["pair"] for r in G], band_pass=float(np.mean([r["band_pass"] for r in G])),
                                           median_worst_dev=float(np.median([r["worst_dev"] for r in G])))
            for r in sorted(R, key=lambda r: -r["static_mae"])[:8]:
                print(f"      {r['pair']:14s} {r['n_atoms']:3d} at  static {r['static_mae']:.3f} D  pass {int(r['band_pass'])}  "
                      f"worst |dev| {r['worst_dev']:.2f}  " + " ".join(f"{b}:{r['band_rel_dev'][b]:+.2f}" for b in BANDS
                                                                    if np.isfinite(r['band_rel_dev'][b])))
        print(f"    per band: median |share dev| | abs error km/mol (median, model - PAHdb) | split by PAHdb band < 5 / >= 5 km/mol")
        for b in BANDS:
            e = [(r["band_km"][b][0], r["band_km"][b][1], r["band_rel_dev"][b]) for r in R]
            e = [t for t in e if np.isfinite(t[2])]
            m, p, rel = map(np.array, zip(*e))
            small = p < 5
            def cell(s):
                return (f"n={s.sum():2d} |rel| {np.median(np.abs(rel[s])):.2f} |abs| {np.median(np.abs(m[s] - p[s])):6.2f} km/mol"
                        if s.any() else "n= 0")
            print(f"      {b:5s} PAHdb median {np.median(p):7.1f} km/mol  all: {cell(np.ones_like(small))}   "
                  f"<5: {cell(small)}   >=5: {cell(~small)}")
            out[f"q{q}_band|{b}"] = dict(pahdb_median_km=float(np.median(p)), n_small=int(small.sum()),
                                         rel_all=float(np.median(np.abs(rel))), abs_all=float(np.median(np.abs(m - p))),
                                         rel_small=float(np.median(np.abs(rel[small]))) if small.any() else None,
                                         rel_big=float(np.median(np.abs(rel[~small]))) if (~small).any() else None,
                                         abs_big=float(np.median(np.abs(m[~small] - p[~small]))) if (~small).any() else None)
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=float)


_DFT = {}


def _dft_init(sym, charge, mult, threads):
    from pyscf import lib
    lib.num_threads(threads); _DFT.update(sym=sym, charge=charge, mult=mult, dm=None)


def _dft_dipole(pos):
    """B3LYP(VWN3)/6-31G* (6D) dipole about the isotopic COM, e*A, with the label settings of d1/dft_seed_run.py."""
    from pyscf import gto, dft
    from g16_io import MASS
    sym, q, mult = _DFT["sym"], _DFT["charge"], _DFT["mult"]
    mol = gto.M(atom=list(zip(sym, pos)), basis="6-31g*", cart=True, charge=q, spin=mult - 1, unit="Angstrom", verbose=0)
    mf = (dft.RKS if mult == 1 else dft.UKS)(mol, xc="b3lypg"); mf.grids.atom_grid = (99, 590)
    mf.conv_tol = 1e-9; mf.conv_tol_grad = 1e-6; mf.max_cycle = 200
    mf.kernel(dm0=_DFT["dm"])
    if not mf.converged:
        mf = mf.newton(); mf.kernel(mf.mo_coeff, mf.mo_occ)
    _DFT["dm"] = mf.make_rdm1()
    m = np.array([MASS[s] for s in sym]); com = (m[:, None] * pos).sum(0) / m.sum()
    mu = np.asarray(mf.dip_moment(unit="Debye", verbose=0)) / 4.80320471 - q * com
    return mu, bool(mf.converged)


def fd_dft(a):
    """dmu_dft (3N x 3, e*A/A) by central differences (step --h A, as dipole_g5.py) of the DFT dipole at the force model's
    minimum stored in each g5 npz; written to a copy of the npz in <out>, so dipole_g5.py report scores 'dft' (the G5
    ceiling with the force model's modes) next to the model predictors on the same species."""
    import multiprocessing as mp
    from dipole_g5 import displaced, fd
    os.makedirs(a.out, exist_ok=True)
    for pair in a.pairs.split(","):
        for q in (0, 1):
            src = f"{a.g5}/{pair}_q{q}.npz"; dst = f"{a.out}/{pair}_q{q}.npz"
            if os.path.exists(dst) and "dmu_dft" in np.load(dst):
                continue
            d = dict(np.load(src)); n = len(d["masses"]); sym = ["H" if x < 2 else "C" for x in d["masses"]]
            geo = displaced(d["pos"], a.h); t0 = time.time()
            chunks = [geo[i::a.workers] for i in range(a.workers)]          # each worker reuses its density
            ctx = mp.get_context("spawn")
            with ctx.Pool(a.workers, initializer=_dft_init, initargs=(sym, q, 1 if q == 0 else 2, a.threads)) as pool:
                res = pool.map(_dft_chunk, chunks)
            mus = [None] * len(geo); ok = True
            for i, r in enumerate(res):
                for k, (mu, conv) in enumerate(r):
                    mus[i + k * a.workers] = mu; ok &= conv
            d["dmu_dft"] = fd(mus, n, a.h); d["dft_all_converged"] = ok
            np.savez(dst, **d)
            print(f"{pair} q{q}: {len(geo)} DFT dipoles, {time.time() - t0:.0f} s, converged {ok}", flush=True)


def _dft_chunk(geos):
    return [_dft_dipole(np.asarray(p)) for p in geos]


def paired(a):
    """Dipole model vs DFT dipoles on identical DFT MD frames (mlmd_polar.py relabel output <dir>/{dft,model}/): per run
    the model/DFT band-strength ratio; its change from the lowest to the highest T is the dipole model's own error in
    the % per 100 K slope (force-model dynamics play no part)."""
    import re, argparse as _ap
    from mlmd_tslopes import series, nve_T, spectrum, strengths
    o = _ap.Namespace(dt_fs=0, nve_ps_max=0); rows = []
    for p in sorted(glob.glob(f"{a.dir}/dft/*.npz")):
        T = int(re.search(r"_(\d+)K_", p).group(1)); st = {}
        for s in ("dft", "model"):
            z = np.load(p.replace("/dft/", f"/{s}/")); mu, dt = series(z, o); st[s] = strengths(*spectrum(mu, nve_T(z), dt))
        rows.append((T, nve_T(np.load(p)), st))
    Ts = sorted({r[0] for r in rows}); lo, hi = Ts[0], Ts[-1]; out = {}
    print(f"--- {a.dir}: {len(rows)} runs; model/DFT band strength on the same frames")
    for w in ("oop", "1000-1300", "1300-1700", "inplane", "CHstr"):
        r = {T: np.array([s["model"][w] / s["dft"][w] for t, _, s in rows if t == T]) for T in (lo, hi)}
        Tm = {T: np.mean([x for t, x, _ in rows if t == T]) for T in (lo, hi)}
        m0, m1 = r[lo].mean(), r[hi].mean()
        e0, e1 = (r[T].std(ddof=1) / np.sqrt(len(r[T])) if len(r[T]) > 1 else np.nan for T in (lo, hi))
        dT = (Tm[hi] - Tm[lo]) / 100; s = 100 * (m1 / m0 - 1) / dT; es = 100 * np.hypot(e1 / m0, m1 * e0 / m0 ** 2) / dT
        out[w] = dict(ratio_lo=float(m0), se_lo=float(e0), ratio_hi=float(m1), se_hi=float(e1), slope_error_pct=float(s),
                      slope_error_se=float(es))
        print(f"  {w:10s} {lo} K {m0:.3f} +- {e0:.3f}   {hi} K {m1:.3f} +- {e1:.3f}   slope error {s:+.2f} +- {es:.2f} %/100 K")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    pp = sp.add_parser("paired"); pp.add_argument("dir"); pp.add_argument("--json")
    c = sp.add_parser("corr"); c.add_argument("eval"); c.add_argument("g5"); c.add_argument("--dipole", default="polar")
    c.add_argument("--ref-flags"); c.add_argument("--json")
    f = sp.add_parser("fd-dft"); f.add_argument("g5"); f.add_argument("out"); f.add_argument("--pairs", required=True)
    f.add_argument("--h", type=float, default=0.005); f.add_argument("--workers", type=int, default=6)
    f.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    {"corr": corr, "fd-dft": fd_dft, "paired": paired}[a.cmd](a)


if __name__ == "__main__":
    main()
