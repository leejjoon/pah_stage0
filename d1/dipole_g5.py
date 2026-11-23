# Intensity half of G5 (docs/plan/charged_mlff_proposal.md §5.7: relative intensities within 20 % for bands > 10 % of
# the strongest, vs PAHdb v4.00 B3LYP/6-31G*, the level of our labels) for dipole models, plus band-integrated
# strengths of the 6.2, 7.7, 8.6 and 11.2 um bands. Harmonic: the force model's minimum and analytic Hessian, modes
# paired with PAHdb's within irreps (mlff_accept.sym_resolve, as `symfreq`; sorted pairing where unresolved), so
# frequency errors of the force model do not enter the intensity comparison. Dipole derivatives by central finite
# differences (step --h A) of each dipole predictor about the centre of mass:
#   xtb      GFN2-xTB (tblite, multiplicity 1/2 as d1/mlff_qa.py), mu - q R_COM
#   <name>   a mace-torch dipole model (COM-centred input, float64), one file per charge
# Derivatives are stored per predictor, so sums (xTB + a Delta-xTB model) are formed in `report`.
#   python dipole_g5.py run <force model> <species.csv,...> <xyz dir,...> <pahdb.xml[.gz]> <out dir>
#       [--splits validation,final_test,transfer_test] [--pairs A,B] [--xtb]
#       [--dipole name:model_type:q0_model:q1_model ...] [--workers 32] [--device cuda] [--h 0.005]
#   python dipole_g5.py report <out dir> <out.json> --combo xtb --combo xtb+dxtb_dip --combo diel ...
#       [--pahdb <xml>] [--ref-flags pahdb_v4_replacements_v1.csv]
# report, primary metric = band-integrated strengths: model and PAHdb intensities summed over the 3.3, 6.2, 7.7, 8.6,
# 11.2 and 12.7 um windows, both placed at PAHdb's (paired) frequency, so intensity the force model's modes redistribute
# among nearby modes of a window does not count as error. Per band: share of the six-band sum relative to PAHdb's
# (G5 is about relative intensities; species passes if every band carrying > 10 % is within 20 %) and the absolute
# ratio. Secondary: per-mode shares of the strong (> 10 % of max) PAHdb modes.
# --pahdb: each reference's basis from its Gaussian route comment, results grouped by it (PAHdb v4.00: 10,703 of
# 10,749 species B3LYP/6-31G*, 14 4-31G; all 128 held-out references here 6-31G*, the label level).
# --ref-flags (gh200-containers d1/pahdb_v4_replacements_v1.csv): references flagged there (non-Kekule neutrals listed
# as closed-shell singlets, intensities ~7x too high) are left out of the summary. Their own high-spin spectra
# (d1/own_spectra, UB3LYP/6-31G* triplets) are not a reference for the singlet q0 surface the models describe.
# run writes <out dir>/<pair>_q<c>.npz: PAHdb freq (unscaled) / intensity / labels, model freq, mass-weighted mode
# vectors (PAHdb order), masses, and dmu_<predictor> (3N x 3, e*A/A). species.csv and xyz dir lists pair up in order
# (species_v2.csv with mlff_seed/xyz, species_transfer2.csv with transfer2/xyz).
import argparse, csv, glob, gzip, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

KM_MOL = 974.8645          # km/mol per (e/amu^1/2)^2: N_A e^2 / (12 eps0 c^2 amu) / 1e3 (as d1/pyscf_freq_fd.py)
DEBYE_EA = 0.2081943
SCALE = 0.9578             # harmonic scale factor (d1/mlmd_spectra.py), only to place modes in the band windows
BANDS = {"3.3": (3.1, 3.6), "6.2": (6.0, 6.5), "7.7": (7.0, 8.2), "8.6": (8.2, 8.9), "11.2": (10.8, 11.6),
         "12.7": (12.4, 13.2)}   # um, at the scaled PAHdb frequency of each (paired) mode
MASS = {"H": 1.00782503, "C": 12.0}
_XTB = {}


def _xtb_init(num, charge, mult):
    os.environ["OMP_NUM_THREADS"] = "1"
    _XTB.update(num=num, charge=charge, mult=mult)


def xtb_dipole(pos):
    """GFN2-xTB dipole about the COM (e*A); pos in A. Returns (mu, converged flag)."""
    from tblite.interface import Calculator
    num, q = _XTB["num"], _XTB["charge"]
    c = Calculator("GFN2-xTB", num, pos / 0.52917721067, charge=q, uhf=_XTB["mult"] - 1)
    c.set("verbosity", 0)
    try:
        r = c.singlepoint()
    except Exception:
        return np.full(3, np.nan), False
    m = np.array([1.00782503 if z == 1 else 12.0 for z in num])
    return np.asarray(r.get("dipole")) * 0.52917721067 - q * (m[:, None] * pos).sum(0) / m.sum(), True


def displaced(pos, h):
    out = []
    for i in range(len(pos)):
        for x in range(3):
            for s in (1, -1):
                p = pos.copy(); p[i, x] += s * h; out.append(p)
    return out


def fd(mus, n, h):
    mus = np.asarray(mus).reshape(n, 3, 2, 3)
    return ((mus[:, :, 0] - mus[:, :, 1]) / (2 * h)).reshape(3 * n, 3)     # d mu_c / d x_(i,x)


def run(a):
    from mlff_accept import calculator, atoms_from, relax, pahdb_scan, sym_resolve, CM1
    import multiprocessing as mp
    jobs = []
    for sp, xd in zip(a.species.split(","), a.xyz.split(",")):
        for r in csv.DictReader(open(sp)):
            if r["split"] in a.splits.split(",") and (not a.pairs or r["pair"] in a.pairs.split(",")):
                jobs += [(r["pair"], q, int(r[f"uid_q{q}"]), r["split"], xd) for q in (0, 1)]
    path = a.pahdb
    if path.endswith(".gz"):
        plain = path[:-3]                    # decompressed once, next to the .gz
        if not os.path.exists(plain):
            with gzip.open(path, "rt") as f, open(plain, "w") as g:
                g.write(f.read())
        path = plain
    os.makedirs(a.out, exist_ok=True)
    ref = pahdb_scan(path, [u for _, _, u, _, _ in jobs], labels=True)
    fcalc = {q: calculator(a.force_model, q, a.device) for q in (0, 1)}
    dips = []
    for spec in a.dipole:
        name, mtype, m0, m1 = spec.split(":")
        from mace.calculators import MACECalculator
        dips.append((name, {0: MACECalculator(model_paths=m0, device=a.device, default_dtype="float64", model_type=mtype),
                            1: MACECalculator(model_paths=m1, device=a.device, default_dtype="float64", model_type=mtype)}))
    ctx = mp.get_context("spawn")
    for pair, q, uid, split, xd in jobs:
        out = f"{a.out}/{pair}_q{q}.npz"
        t0 = time.time()
        at = atoms_from(f"{xd}/{pair}_q{q}.xyz")
        n = len(at); sym = at.get_chemical_symbols(); m = np.array([MASS[s] for s in sym]); mm = np.repeat(m, 3)
        if os.path.exists(out):              # add missing predictors at the stored minimum and modes
            res = dict(np.load(out)); info = json.loads(str(res["info"])); at.positions = res["pos"]
            if (not a.xtb or "dmu_xtb" in res) and all(f"dmu_{k}" in res for k, _ in dips):
                continue
        else:
            at.calc = fcalc[q]
            rel = relax(at, a.fmax)
            at.positions -= (m[:, None] * at.positions).sum(0) / m.sum()
            H = np.asarray(fcalc[q].get_hessian(at)).reshape(3 * n, 3 * n); H = 0.5 * (H + H.T)
            Hmw = H / np.sqrt(np.outer(mm, mm))
            R, labels = ref[uid]
            f_sym, info, W = sym_resolve(at.positions, sym, m, Hmw, labels, vectors=True)
            if f_sym is None:                    # sorted pairing over all vibrations
                w, V = np.linalg.eigh(Hmw)
                o = np.argsort(np.abs(w))[6:]; o = o[np.argsort(w[o])]
                W, f_sym = V[:, o], np.sign(w[o]) * np.sqrt(np.abs(w[o])) * CM1
                info = dict(info, pairing="sorted")
            else:
                info = dict(info, pairing="irrep")
            info = dict(info, pair=pair, charge=q, uid=uid, split=split, **rel)
            res = dict(freq_pahdb=R[:, 0], inten_pahdb=R[:, 1], labels=np.array(labels), freq_model=f_sym, modes=W,
                       masses=m, pos=at.positions.copy(), info=json.dumps(info))
        geo = displaced(at.positions, a.h)
        if a.xtb and "dmu_xtb" not in res:
            mult = 1 if q == 0 else 2
            with ctx.Pool(a.workers, initializer=_xtb_init, initargs=(at.numbers, q, mult)) as pool:
                r = pool.map(xtb_dipole, geo, chunksize=4)
            ok = np.array([x[1] for x in r])
            res["dmu_xtb"] = fd([x[0] for x in r], n, a.h); res["xtb_failed"] = int((~ok).sum())
        for name, cs in dips:
            if f"dmu_{name}" in res:
                continue
            mus = []
            for p in geo:
                b = at.copy(); b.positions = p - (m[:, None] * p).sum(0) / m.sum(); b.info["charge"] = q
                b.calc = cs[q]; mus.append(np.asarray(cs[q].get_property("dipole", b)))
            res[f"dmu_{name}"] = fd(mus, n, a.h)
        np.savez(out, **res)
        print(f"{pair:14s} q{q} {split:13s} n={n:3d} {info.get('sym', '')} {info['pairing']:6s} "
              f"xtb_failed={res.get('xtb_failed', '-')} {time.time() - t0:.0f} s", flush=True)


def intensities(d, dmu):
    """km/mol per mode (PAHdb order): dmu (3N x 3, e*A/A) projected on mass-weighted modes."""
    dq = (d["modes"] / np.sqrt(np.repeat(d["masses"], 3))[:, None]).T @ dmu      # e / amu^1/2
    return KM_MOL * (dq ** 2).sum(1)


def band_of(f_unscaled):
    lam = 1e4 / (f_unscaled * SCALE)
    for b, (lo, hi) in BANDS.items():
        if lo <= lam < hi:
            return b
    return None


def score(d, I):
    Ip = d["inten_pahdb"]; fp = d["freq_pahdb"]
    strong = Ip > 0.1 * Ip.max()
    fm, fpd = I[strong] / I[strong].sum(), Ip[strong] / Ip[strong].sum()
    mode_dev = np.abs(fm / fpd - 1)
    bands = np.array([band_of(f) for f in fp], dtype=object)
    Bp = {b: Ip[bands == b].sum() for b in BANDS}; Bm = {b: I[bands == b].sum() for b in BANDS}
    tp, tm = sum(Bp.values()), sum(Bm.values())
    rel = {b: (Bm[b] / tm) / (Bp[b] / tp) - 1 if tp > 0 and tm > 0 and Bp[b] > 0 else np.nan for b in BANDS}
    big = [b for b in BANDS if tp > 0 and Bp[b] / tp > 0.1]
    r62 = (Bm["6.2"] / Bm["11.2"]) / (Bp["6.2"] / Bp["11.2"]) - 1 if min(Bp["6.2"], Bp["11.2"], Bm["11.2"]) > 0 else np.nan
    return dict(n_strong=int(strong.sum()), mode_pass=float((mode_dev <= 0.2).mean()), mode_median_dev=float(np.median(mode_dev)),
                band_rel_dev={b: float(v) for b, v in rel.items()}, band_pass=all(abs(rel[b]) <= 0.2 for b in big),
                bands_tested=big, total_ratio=float(I.sum() / Ip.sum()), r62_112_dev=float(r62),
                band_abs_ratio={b: float(Bm[b] / Bp[b]) if Bp[b] > 0 else None for b in BANDS})


def pahdb_basis(path):
    import re
    out = {}
    for m in re.finditer(r'<specie uid="(\d+)">(.*?)</specie>', open(path).read(), re.S):
        for c in re.findall(r"<comment[^>]*>([^<]*)</comment>", m.group(2)):
            r = re.search(r"(?i)/\s*([0-9a-z\-+*(),]+g[*()dp,]*)", c)
            if r:
                out[int(m.group(1))] = r.group(1).lower(); break
    return out


def report(a):
    basis = pahdb_basis(a.pahdb[:-3] if a.pahdb.endswith(".gz") else a.pahdb) if a.pahdb else {}
    flags = {int(r["pahdb_v4_uid"]): r["action"] for r in csv.DictReader(open(a.ref_flags))} if a.ref_flags else {}
    rows = []
    for f in sorted(glob.glob(f"{a.out}/*_q[01].npz")):
        d = np.load(f, allow_pickle=False); info = json.loads(str(d["info"]))
        for combo in a.combo:
            keys = [f"dmu_{k}" for k in combo.split("+")]
            if not all(k in d for k in keys):
                continue
            I = intensities(d, sum(d[k] for k in keys))
            rows.append(dict(combo=combo, pair=info["pair"], charge=info["charge"], split=info["split"], uid=info["uid"],
                             basis=basis.get(info["uid"], "?"), ref_flag=flags.get(info["uid"], ""),
                             n_atoms=len(d["masses"]), pairing=info["pairing"], **score(d, I)))
    json.dump(rows, open(a.json, "w"), indent=1)
    dropped = sorted({(r["pair"], r["charge"], r["ref_flag"]) for r in rows if r["ref_flag"]})
    print(f"G5 intensity half. Primary: band-integrated 3.3/6.2/7.7/8.6/11.2/12.7 um shares of their sum; a species passes"
          f"\nif every band > 10 % is within 20 % (band-pass). |dev| = median |share/share_PAHdb - 1|, abs = median"
          f" model/PAHdb band strength.\nSecondary: strong PAHdb modes (> 10 % of max) within 20 %.\n"
          f"Left out (--ref-flags): {dropped}")
    for grp in sorted({r["basis"] for r in rows}):
      print(f"-- PAHdb reference basis {grp}")
      for combo in a.combo:
        for q in (0, 1):
            R = [r for r in rows if r["combo"] == combo and r["charge"] == q and r["basis"] == grp and not r["ref_flag"]]
            if not R:
                continue
            big = [r for r in R if r["n_atoms"] >= 80]
            dev = lambda b: np.nanmedian([abs(r["band_rel_dev"][b]) for r in R])
            print(f"{combo:18s} q{q} n={len(R):3d}  modes within 20 %: {np.mean([r['mode_pass'] for r in R]):.2f}"
                  f"  species band-pass {np.mean([r['band_pass'] for r in R]):.2f} (>=80 atoms {np.mean([r['band_pass'] for r in big]) if big else float('nan'):.2f}, {len(big)})"
                  f"  |dev| " + " ".join(f"{b}:{dev(b):.2f}" for b in BANDS) +
                  f"  abs " + " ".join(f"{b}:{np.nanmedian([r['band_abs_ratio'][b] if r['band_abs_ratio'][b] is not None else np.nan for r in R]):.2f}" for b in BANDS) +
                  f"  6.2/11.2 {np.nanmedian([abs(r['r62_112_dev']) for r in R]):.2f}  total {np.median([r['total_ratio'] for r in R]):.2f}")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    for k in ("force_model", "species", "xyz", "pahdb", "out"):
        r.add_argument(k)
    r.add_argument("--splits", default="validation,final_test,transfer_test"); r.add_argument("--pairs", default="")
    r.add_argument("--xtb", action="store_true"); r.add_argument("--dipole", action="append", default=[])
    r.add_argument("--workers", type=int, default=32); r.add_argument("--device", default="cuda")
    r.add_argument("--h", type=float, default=0.005); r.add_argument("--fmax", type=float, default=1e-4)
    p = sub.add_parser("report"); p.add_argument("out"); p.add_argument("json"); p.add_argument("--combo", action="append", required=True)
    p.add_argument("--pahdb", default=""); p.add_argument("--ref-flags", default="")
    a = ap.parse_args()
    run(a) if a.cmd == "run" else report(a)


if __name__ == "__main__":
    main()
