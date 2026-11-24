# Backfill of the 61-100 C transfer_test set after the open-shell exclusion (docs/plan/charged_mlff_plan.md,
# 2026-11-15/16), without xTB: frames are the PAHdb minimum plus Gaussian random displacements ("naive" sampler),
# not the MD/NMS families of mlff_sample.py. Labels are written with mlff_sample.py's own helpers.
#   python transfer_backfill.py candidates <pahdb.xml[.gz]> <species_v2.csv> <exclusions.csv> <out dir>
#        [--per-bin 8] [--sigmas 0.02,0.03,0.04,0.05] [--seed 0]
#     per 10-C bin (61-70 ... 91-100): unused PAHdb neutral/cation pairs (RB3LYP/UB3LYP, both 6-31G*, carbon skeleton
#     not in species_v2.csv, one pair per skeleton, Mai et al. uids excluded), --per-bin chosen by farthest-point
#     sampling in edge-type fractions, seeded with the kept transfer_test pairs of that bin. Writes a seed directory
#     that slurm_dft_seed.sbatch reads as SEED_DIR: frames/ (empty), frames_transfer/<pair>_q{0,1}.extxyz (min + one
#     disp frame per sigma), gjf_transfer/manifest.json (stability_test 0), xyz/, and candidates.csv.
#   python transfer_backfill.py select <out dir> <bs screen dir> [--per-bin 4] [--max-dE-mHa 1.0]
#     drops candidates whose neutral minimum is RKS->UKS unstable with E_RKS - E_BS >= --max-dE-mHa (the Phase 1
#     criterion), then farthest-point selects --per-bin per bin from the rest (seeded as above). Writes
#     candidates.csv (with screen results and the decision) and species_transfer2.csv (species_v2 columns + sampler).
import argparse, csv, gzip, json, os, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlff_species  # noqa: E402
from g16_io import MASS  # noqa: E402
from mlff_sample import bond_list, centred, graph, labels, write_frame  # noqa: E402
from mlff_species import SYM, graph_hash, read_pahdb  # noqa: E402
from mlff_splits import FIELDS, skeleton  # noqa: E402

MAI_UIDS = {618, 641}                       # Mai et al. geometries (CC BY-NC-SA): never used
BINS = [(61, 70), (71, 80), (81, 90), (91, 100)]
EDGES = ("solo", "duo", "trio", "quartet")


def feat(e):
    return np.array([e[k] for k in EDGES], float) / max(1, sum(e.values()))


def fps(cand, seeds, k, rng):
    """Farthest-point picks of k from cand (dicts with 'f'), measured against seeds and earlier picks."""
    cand, sel = list(cand), []
    while cand and len(sel) < k:
        ref = [s["f"] for s in seeds + sel]
        if not ref:
            i = int(rng.integers(len(cand)))
        else:
            F, G = np.array([c["f"] for c in cand]), np.array(ref)
            i = int(np.argmax(np.linalg.norm(F[:, None] - G[None], axis=-1).min(1)))
        sel.append(cand.pop(i))
    return sel


def kept_transfer(species, excl):
    out = []
    for r in species:
        if r["split"] == "transfer_test" and r["pair"] not in excl:
            out.append(dict(pair=r["pair"], nC=int(r["nC"]), f=feat({k: int(r[f"n_{k}"]) for k in EDGES})))
    return out


def cmd_candidates(a):
    if a.xml.endswith(".gz"):
        mlff_species.open = lambda p, m="r": gzip.open(p, m)
    species = list(csv.DictReader(open(a.species)))
    excl = {r["pair"] for r in csv.DictReader(open(a.exclusions))}
    used = {r["skeleton"] for r in species}
    by = defaultdict(dict)
    for s in read_pahdb(a.xml):
        by[graph_hash(s["Z"], s["P"])].setdefault(s["charge"], s)
    pool, seen = defaultdict(list), set()
    for d in by.values():
        n, c = d.get(0), d.get(1)
        if not (n and c and n["method"] == "RB3LYP" and c["method"] == "UB3LYP" and 60 < n["nC"] <= 100):
            continue
        if n["uid"] in MAI_UIDS or c["uid"] in MAI_UIDS or n["basis"].lower() != "6-31g*" or c["basis"].lower() != "6-31g*":
            continue
        sk = skeleton(n["Z"], n["P"])
        if sk in used or sk in seen:
            continue
        seen.add(sk)
        b = next(i for i, (lo, hi) in enumerate(BINS) if lo <= n["nC"] <= hi)
        pool[b].append(dict(n=n, c=c, skeleton=sk, f=feat(n["edges"]), pair=f"{n['formula']}_{n['uid']}"))
    rng = np.random.default_rng(a.seed)
    kept = kept_transfer(species, excl)
    chosen = []
    for b, (lo, hi) in enumerate(BINS):
        seeds = [k for k in kept if lo <= k["nC"] <= hi]
        chosen += [dict(p, bin=f"{lo}-{hi}") for p in fps(sorted(pool[b], key=lambda p: p["pair"]), seeds, a.per_bin, rng)]
        print(f"bin {lo}-{hi}: pool {len(pool[b])}, kept transfer pairs {len(seeds)}, candidates {a.per_bin}")
    assert not any(p["n"]["uid"] in MAI_UIDS or p["c"]["uid"] in MAI_UIDS for p in chosen)
    D = a.outdir
    for sub in ("frames", "frames_transfer", "gjf_transfer", "xyz"):
        os.makedirs(f"{D}/{sub}", exist_ok=True)
    sig = [float(x) for x in a.sigmas.split(",")]
    manifest = {}
    for p in chosen:
        for q, r, mult in ((0, p["n"], 1), (1, p["c"], 2)):
            tag = f"{p['pair']}_q{q}"
            sym = [SYM[z] for z in r["Z"]]; p0 = np.asarray(r["P"], float)
            with open(f"{D}/xyz/{tag}.xyz", "w") as g:
                g.write(f"{len(sym)}\nPAHdb v4.00 uid {r['uid']} charge {q} {r['method']}/{r['basis']}\n")
                g.writelines(f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, (x, y, z) in zip(sym, p0))
            mass = np.array([MASS[s] for s in sym]); ref = {"bonds": bond_list(sym, p0), "graph": graph(sym, p0)}
            prng = np.random.default_rng([a.seed, r["uid"]])
            base = dict(charge=q, mult=mult, source=f"{tag}.xyz")
            frames = [(p0, "min", None)] + [(p0 + prng.normal(0, s, p0.shape), "disp", s) for s in sig]
            with open(f"{D}/frames_transfer/{tag}.extxyz", "w") as f:
                for k, (pk, ct, s) in enumerate(frames):
                    info = {**base, "config_type": ct, "sample_T": "none", **labels(sym, pk, ref), "sampler": "naive",
                            "disp_sigma": "none" if s is None else f"{s:.3f}"}
                    write_frame(f, sym, centred(pk, mass), info)
                    manifest[f"{tag}_f{k:03d}"] = {**{x: str(v) for x, v in info.items()}, "frame": k,
                                                   "basis": "6-31G*", "stability_test": 0}
    json.dump(manifest, open(f"{D}/gjf_transfer/manifest.json", "w"), indent=0)
    with open(f"{D}/candidates.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair", "bin", "nC", "formula", "uid_q0", "uid_q1", "basis_q0", "basis_q1", "skeleton"]
                   + [f"n_{k}" for k in EDGES] + ["solo_frac"])
        for p in chosen:
            e = p["n"]["edges"]
            w.writerow([p["pair"], p["bin"], p["n"]["nC"], p["n"]["formula"], p["n"]["uid"], p["c"]["uid"],
                        p["n"]["basis"], p["c"]["basis"], p["skeleton"]] + [e[k] for k in EDGES]
                       + [f"{e['solo'] / max(1, sum(e.values())):.3f}"])
    print(f"{len(chosen)} candidates, {len(manifest)} frames -> {D}")


def cmd_select(a):
    D = a.outdir
    cand = list(csv.DictReader(open(f"{D}/candidates.csv")))
    species = list(csv.DictReader(open(a.species)))
    excl = {r["pair"] for r in csv.DictReader(open(a.exclusions))}
    kept = kept_transfer(species, excl)
    rng = np.random.default_rng(a.seed)
    for c in cand:
        r = json.load(open(f"{a.screen}/{c['pair']}_q0_f000.json"))
        dE = r["bs_dE_Ha"] * 1e3 if r.get("bs_guess") else None
        c.update(stability=r["stability"], bs_dE_mHa="" if dE is None else f"{dE:.3f}",
                 bs_s2="" if dE is None else f"{r['bs_s2']:.3f}",
                 open_shell=int(r["stability"] != "stable" and (dE is None or dE >= a.max_dE_mHa)))
        c["f"] = feat({k: int(c[f"n_{k}"]) for k in EDGES})
    sel = []
    for lo, hi in BINS:
        ok = [c for c in cand if c["bin"] == f"{lo}-{hi}" and not c["open_shell"]]
        pick = fps(ok, [k for k in kept if lo <= k["nC"] <= hi], a.per_bin, rng)
        sel += pick
        print(f"bin {lo}-{hi}: {sum(c['bin'] == f'{lo}-{hi}' for c in cand)} screened, {len(ok)} passed, {len(pick)} selected")
    names = {c["pair"] for c in sel}
    for c in cand:
        c["selected"] = int(c["pair"] in names)
    cols = [k for k in cand[0] if k != "f"]
    with open(f"{D}/candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader(); w.writerows(cand)
    with open(f"{D}/species_transfer2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["sampler"], extrasaction="ignore"); w.writeheader()
        for c in sel:
            w.writerow(dict(c, split="transfer_test", sampler="naive", uid_qm1="", uid_q2="", mult_q2=""))
    open(f"{D}/selected_pairs.txt", "w").write(",".join(sorted(names)) + "\n")
    print(f"{len(sel)} pairs -> {D}/species_transfer2.csv, {D}/selected_pairs.txt")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates"); c.add_argument("xml"); c.add_argument("species"); c.add_argument("exclusions")
    c.add_argument("outdir"); c.add_argument("--per-bin", type=int, default=8)
    c.add_argument("--sigmas", default="0.02,0.03,0.04,0.05"); c.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("select"); s.add_argument("outdir"); s.add_argument("screen")
    s.add_argument("--species", required=True); s.add_argument("--exclusions", required=True)
    s.add_argument("--per-bin", type=int, default=4); s.add_argument("--max-dE-mHa", type=float, default=1.0)
    s.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"candidates": cmd_candidates, "select": cmd_select}[a.cmd](a)


if __name__ == "__main__":
    main()
