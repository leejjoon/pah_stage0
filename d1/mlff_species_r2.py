# Active-learning round 2 (docs/plan/charged_mlff_plan.md, Phase 3): new training species instead of more frames of
# the old ones. v3's remaining cation error is per species (small five-ring, CH2/sp3 and non-planar molecules whose
# skeletons are not in train), so round 2 adds every unused small PAHdb skeleton.
#   python mlff_species_r2.py candidates <pahdb.xml> <species csv,...> <out dir> [--cmax 25] [--cmin 1]
#        [--exclude-uids d1/pahdb_v4_exclude_nonkekule_v1.csv]
#       closed-shell neutrals (RB3LYP/6-31G*) with --cmin..--cmax C whose carbon skeleton (mlff_splits.skeleton) is in none of
#       the given species tables (all splits), one per skeleton, Mai et al. uids excluded. All go to train. Neutral uids listed in --exclude-uids
#       (column `uid`) are skipped: PAHdb v4 lists its non-Kekule species as closed-shell singlets, but their ground
#       state is a triplet (the list comes from branch gh200-containers, commit 7de3421; the smallest has 30 C, so it
#       matters from AL round 3 on). Writes
#       <out>/species_r2.csv (species_v2 columns + features: rings, five-/seven-rings, sp3 C, out-of-plane RMS),
#       <out>/xyz/<pair>_q0.xyz (PAHdb geometry) and the cation: <out>/xyz/<pair>_q1.xyz when PAHdb has a UB3LYP/6-31G*
#       doublet of the same structure, otherwise <out>/xyz_start/<pair>_q1.xyz (the neutral geometry, to optimise).
#   python mlff_species_r2.py select <pool dir> <out dir> [--n 100] [--oop-min 0.2]
#       AL round 3, track 1: from a `candidates` pool keep the non-planar pairs (oop_rms > --oop-min A) and take --n of
#       them, ranked: first every pair with a five- or seven-ring or an sp3 C (the motifs behind v3's cation error;
#       taken whether planar or not), then farthest-point sampling (standardised nC, oop_rms, n_ring and the
#       solo/duo/trio/quartet shares of the H atoms), starting from the pair nearest the pool mean. The rank is a
#       priority: a shorter run takes the first k. Writes <out>/species_r3.csv (pool columns + rank) and copies the
#       pairs' xyz/ and xyz_start/ files.
#   python mlff_species_r2.py opt <out dir> [--only PAIR,...] [--charge 1]       (container D, one GPU)
#       B3LYP/6-31G* optimisation of each xyz_start/<pair>_q<charge>.xyz (charge 1: UKS doublet; charge 0: RKS
#       singlet, for hand-built species without a PAHdb neutral, mlff_species_own.py) with geomeTRIC, SCF settings as in
#       dft_seed_run.py label() (b3lypg, cart 6D, (99, 590) grid, conv_tol 1e-9); internal UKS stability test with
#       re-optimisation (Gaussian Stable=Opt) at the start and at the end (if the end point was unstable, the
#       optimisation is repeated once from the stable solution). Writes xyz/<pair>_q1.xyz and opt/<pair>_q1.json
#       (energy, <S^2>, steps, max force, time). Finished pairs are skipped.
import argparse, csv, glob, json, os, shutil, sys, time
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

MAI_UIDS = {618, 641}
EDGES = ("solo", "duo", "trio", "quartet")


def cycles(adj, L):
    out = set()

    def dfs(s, cur, path):
        for nb in adj[cur]:
            if nb == s and len(path) == L:
                out.add(frozenset(path))
            elif nb > s and nb not in path and len(path) < L:
                dfs(s, nb, path + [nb])
    for s in adj:
        dfs(s, s, [s])
    return out


def features(Z, P):
    Z = np.array(Z); P = np.asarray(P, float); n = len(Z)
    d = np.linalg.norm(P[:, None] - P[None], axis=-1)
    A = (d < np.where((Z[:, None] == 6) & (Z[None] == 6), 1.75, 1.25)) & ~np.eye(n, dtype=bool)
    c = [i for i in range(n) if Z[i] == 6]; adj = {i: [j for j in c if A[i, j]] for i in c}
    E = sum(len(v) for v in adj.values()) // 2
    return dict(n_ring=E - len(c) + 1, n_ring5=len(cycles(adj, 5)), n_ring7=len(cycles(adj, 7)),
                n_sp3=sum(1 for i in c if A[i].sum() == 4),
                oop_rms=round(float(np.linalg.svd(P - P.mean(0))[1][-1] / np.sqrt(n)), 3))


def write_xyz(path, Z, P, comment):
    from mlff_species import SYM
    with open(path, "w") as g:
        g.write(f"{len(Z)}\n{comment}\n")
        g.writelines(f"{SYM[z]} {x:.6f} {y:.6f} {zz:.6f}\n" for z, (x, y, zz) in zip(Z, P))


def cmd_candidates(a):
    from mlff_species import graph_hash, read_pahdb
    from mlff_splits import FIELDS, skeleton
    used = set()
    bad_uid = {int(r["uid"]) for r in csv.DictReader(open(a.exclude_uids))} if a.exclude_uids else set()
    for f in a.species.split(","):
        used |= {r["skeleton"] for r in csv.DictReader(open(f))}
    by = defaultdict(dict)
    for s in read_pahdb(a.xml):
        by[graph_hash(s["Z"], s["P"])].setdefault(s["charge"], s)
    rows, seen = [], set()
    for d in by.values():
        n, c = d.get(0), d.get(1)
        if not n or n["nC"] > a.cmax or n["nC"] < a.cmin or int(n["uid"]) in bad_uid or n["method"] != "RB3LYP" or n["basis"].lower() != "6-31g*" or n["uid"] in MAI_UIDS:
            continue
        sk = skeleton(n["Z"], n["P"])
        if sk in used or sk in seen:
            continue
        seen.add(sk)
        if c and (c["method"] != "UB3LYP" or c["basis"].lower() != "6-31g*" or c["uid"] in MAI_UIDS):
            c = None
        rows.append((n, c, sk))
    rows.sort(key=lambda r: (r[0]["nC"], r[0]["uid"]))
    os.makedirs(f"{a.out}/xyz", exist_ok=True); os.makedirs(f"{a.out}/xyz_start", exist_ok=True)
    extra = ["n_ring", "n_ring5", "n_ring7", "n_sp3", "oop_rms", "q1_source"]
    with open(f"{a.out}/species_r2.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(FIELDS + extra)
        for n, c, sk in rows:
            pair = f"{n['formula']}_{n['uid']}"
            write_xyz(f"{a.out}/xyz/{pair}_q0.xyz", n["Z"], n["P"], f"PAHdb v4.00 uid {n['uid']} charge 0 RB3LYP/6-31g*")
            if c:
                write_xyz(f"{a.out}/xyz/{pair}_q1.xyz", c["Z"], c["P"], f"PAHdb v4.00 uid {c['uid']} charge 1 UB3LYP/6-31g*")
            else:
                write_xyz(f"{a.out}/xyz_start/{pair}_q1.xyz", n["Z"], n["P"],
                          f"start: neutral geometry PAHdb v4.00 uid {n['uid']}, to optimise as the cation")
            ft = features(n["Z"], n["P"]); e = n["edges"]
            w.writerow([pair, n["formula"], n["nC"], "train", sk, n["uid"], c["uid"] if c else "", "", "", "",
                        "6-31g*", "6-31g*" if c else "6-31g*(opt)", e["solo"], e["duo"], e["trio"], e["quartet"]]
                       + [ft[k] for k in extra[:-1]] + ["pahdb" if c else "opt"])
    print(f"{len(rows)} pairs ({a.cmin}-{a.cmax} C, skeletons new, {len(bad_uid)} uids excluded); cation from PAHdb {sum(1 for r in rows if r[1])}, "
          f"to optimise {sum(1 for r in rows if not r[1])} -> {a.out}/species_r2.csv")


def cmd_select(a):
    rows = list(csv.DictReader(open(f"{a.pool}/species_r2.csv")))
    motif = lambda r: int(r["n_ring5"]) + int(r["n_ring7"]) + int(r["n_sp3"]) > 0
    pool = [r for r in rows if float(r["oop_rms"]) > a.oop_min or motif(r)]

    def feat(r):
        e = np.array([float(r[f"n_{k}"]) for k in EDGES]); w = e * np.array([1, 2, 3, 4])   # H atoms per edge type
        return [float(r["nC"]), float(r["oop_rms"]), float(r["n_ring"])] + list(w / max(w.sum(), 1))
    X = np.array([feat(r) for r in pool]); X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1)
    order = sorted([i for i, r in enumerate(pool) if motif(r)], key=lambda i: int(pool[i]["nC"]))
    if not order:
        order = [int(np.argmin((X ** 2).sum(1)))]
    d = np.min([((X - X[i]) ** 2).sum(1) for i in order], axis=0)
    while len(order) < min(a.n, len(pool)):
        i = int(np.argmax(d)); order.append(i); d = np.minimum(d, ((X - X[i]) ** 2).sum(1))
    os.makedirs(f"{a.out}/xyz", exist_ok=True); os.makedirs(f"{a.out}/xyz_start", exist_ok=True)
    with open(f"{a.out}/species_r3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, list(rows[0]) + ["rank"]); w.writeheader()
        for k, i in enumerate(order, 1):
            r = pool[i]; w.writerow({**r, "rank": k})
            for sub in ("xyz", "xyz_start"):
                for g in glob.glob(f"{a.pool}/{sub}/{r['pair']}_q*.xyz"):
                    shutil.copy(g, f"{a.out}/{sub}/")
    sel = [pool[i] for i in order]
    print(f"pool {len(rows)}, non-planar or motif {len(pool)}; selected {len(sel)} (motif {sum(map(motif, sel))}), "
          f"nC {min(int(r['nC']) for r in sel)}-{max(int(r['nC']) for r in sel)}, cation from PAHdb "
          f"{sum(r['q1_source'] == 'pahdb' for r in sel)} -> {a.out}/species_r3.csv")


def cmd_opt(a):
    import cupy
    from pyscf import gto
    from pyscf.geomopt.geometric_solver import kernel as geom_kernel
    from gpu4pyscf import dft
    from g16_io import HB_TO_EVA
    os.makedirs(f"{a.out}/opt", exist_ok=True)
    only = set(filter(None, a.only.split(",")))
    os.makedirs(f"{a.out}/xyz", exist_ok=True)
    files = sorted(glob.glob(f"{a.out}/xyz_start/*_q{a.charge}.xyz"), key=os.path.getsize)
    for fn in files:
        tag = os.path.basename(fn)[:-4]; pair = tag[:-3]
        if (only and pair not in only) or os.path.exists(f"{a.out}/xyz/{tag}.xyz"):
            continue
        L = open(fn).read().split("\n"); nat = int(L[0])
        sym = [l.split()[0] for l in L[2:2 + nat]]; pos = np.array([[float(x) for x in l.split()[1:4]] for l in L[2:2 + nat]])
        t0 = time.time(); log = dict(pair=pair, rounds=[])

        def build(p):
            mol = gto.M(atom=list(zip(sym, p)), basis="6-31g*", cart=True, charge=a.charge, spin=a.charge % 2,
                        unit="Angstrom", verbose=0)
            mf = (dft.UKS if a.charge % 2 else dft.RKS)(mol, xc="b3lypg"); mf.grids.atom_grid = (99, 590)
            mf.conv_tol = 1e-9; mf.conv_tol_grad = 1e-6; mf.max_cycle = 200
            return mf

        def stable(mf, dm0=None):
            e = mf.kernel(dm0=dm0); k = 0
            if not mf.converged:
                mf = mf.newton(); e = mf.kernel(mf.mo_coeff, mf.mo_occ)
            while True:
                mo, _, st, _ = mf.stability(internal=True, external=False, return_status=True)
                if st or k == 5:
                    return mf, float(e), k, bool(st)
                k += 1; e = mf.kernel(dm0=mf.make_rdm1(mo, mf.mo_occ))

        p = pos; dm0 = None
        for rnd in range(2):
            mf, e0, k0, st0 = stable(build(p), dm0)
            dm_start = mf.make_rdm1()
            mf2 = build(p); mf2.kernel(dm0=dm_start)
            conv, mol_eq = geom_kernel(mf2, maxsteps=200)
            p = mol_eq.atom_coords(unit="Angstrom")
            mf3, e1, k1, st1 = stable(build(p), mf2.make_rdm1())
            g = mf3.nuc_grad_method().kernel(); g = np.asarray(g.get() if hasattr(g, "get") else g)
            log["rounds"].append(dict(start_reopt=k0, start_stable=st0, converged=bool(conv), end_reopt=k1, end_stable=st1,
                                      energy_Ha=e1, s2=float(mf3.spin_square()[0]) if a.charge % 2 else 0.0,
                                      max_force_eVA=float(np.abs(g).max() * HB_TO_EVA)))
            if k1 == 0:
                break
            dm0 = mf3.make_rdm1()            # end point was unstable: optimise again from the stable solution
        r = log["rounds"][-1]
        log.update(time_s=round(time.time() - t0, 1), ok=bool(r["converged"] and r["end_reopt"] == 0),
                   gpu_mem_gb=round(cupy.get_default_memory_pool().total_bytes() / 1e9, 3))
        json.dump(log, open(f"{a.out}/opt/{tag}.json", "w"), indent=1)
        if log["ok"]:
            from mlff_species import SYM
            Z = [{v: k for k, v in SYM.items()}[s] for s in sym]
            write_xyz(f"{a.out}/xyz/{tag}.xyz", Z, p, f"{'U' if a.charge % 2 else 'R'}B3LYP/6-31G* optimised (gpu4pyscf b3lypg + geomeTRIC), "
                      f"E {r['energy_Ha']:.8f} Ha, <S^2> {r['s2']:.4f}, start {os.path.basename(fn)}")
        print(f"{tag}: {'ok' if log['ok'] else 'NOT OK'} E {r['energy_Ha']:.8f} <S2> {r['s2']:.3f} "
              f"max|F| {r['max_force_eVA']:.4f} eV/A, rounds {len(log['rounds'])}, {log['time_s']:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("candidates"); c.add_argument("xml"); c.add_argument("species"); c.add_argument("out")
    c.add_argument("--cmax", type=int, default=25); c.add_argument("--cmin", type=int, default=1)
    c.add_argument("--exclude-uids", default="")
    s = sub.add_parser("select"); s.add_argument("pool"); s.add_argument("out")
    s.add_argument("--n", type=int, default=100); s.add_argument("--oop-min", type=float, default=0.2)
    o = sub.add_parser("opt"); o.add_argument("out"); o.add_argument("--only", default="")
    o.add_argument("--charge", type=int, default=1)
    a = ap.parse_args()
    {"candidates": cmd_candidates, "select": cmd_select, "opt": cmd_opt}[a.cmd](a)


if __name__ == "__main__":
    main()
